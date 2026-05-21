#!/usr/bin/env python3
"""
Run the full monrad pipeline (stages 1-5) against one real acquisition.

Usage:
    run_pipeline.py --telescope <dir> --probe <dir> [--out <dir>]
                    [--z-tel Z0 Z1 Z2]

Arguments:
    --telescope  Directory containing telescope *_header.txt, *_GPS.bin, *.bin
    --probe      Directory containing probe *_header.txt, *_GPS.bin, *.bin
    --out        Output directory for summary.txt (default: ./pipeline_out)
    --z-tel      Telescope plane z-coordinates in mm, top to bottom
                 (default: 0 400 800)

Expected console output (example values):

    === Stage 1: Time reconstruction ===
      Telescope  12345 events   GOOD  11200   DEGRADED    900   UNTRUSTED   245
      Probe       8765 events   GOOD   8500   DEGRADED    200   UNTRUSTED    65
      tel/probe ratio: 1.41  (telescope hardware filter requires ≥2-plane ribbon coincidence)

    === Stage 4: Telescope alignment ===
      Plane 0   delta_x =  +0.12 mm   delta_y =  -0.05 mm   rot_z =  +3.00e-04 rad
      Plane 1   delta_x =  -0.08 mm   delta_y =  +0.11 mm   rot_z =  -1.00e-04 rad
      Plane 2   delta_x =  +0.04 mm   delta_y =  +0.02 mm   rot_z =  +2.00e-04 rad
      needs_correction: True
      Symmetry check (Plane 0 vs Plane 2):
        |delta_x[0] - delta_x[2]| =  0.00 mm   |delta_y[0] - delta_y[2]| =  0.00 mm
        delta_x[1]/delta_x[0] = -0.50   (expected -0.50: algorithm identity for z=[0,400,800])
      Note: for evenly-spaced z the two-plane predictor always gives
        delta[0]=delta[2], delta[1]=-delta[0]/2 (measures curvature only).

    === Stage 2: Coincidence search ===
      Coincidences     :    523
      Mean cluster size:   2.00

    === Stage 3: Hit quality (coincidence survivors) ===
      523 coincidences x 3 telescope planes = 1569 readings
      Plane 0    golden 314   cluster 32   unresolved 4   invalid 2   missing 1
      Plane 1    golden 315   cluster 30   unresolved 5   invalid 1   missing 1
      Plane 2    golden 312   cluster 32   unresolved 3   invalid 2   missing 0
      523 coincidences x 1 probe plane = 523 readings
      Probe      golden 410   cluster 80   unresolved 20   invalid 13   missing 0

    === Stage 5: Probe pose fit ===
      t_x   =  +51.3 ±  1.2 mm
      t_y   =  -29.7 ±  1.1 mm
      theta =  +17.1 ±  0.3 deg
      z_p   = +301.4 ±  4.7 mm
      n_inliers = 487

If stage 5 has too few coincidences:

    === Stage 5: Probe pose fit ===
      SKIPPED — too few coincidences survived to fit pose; check
      telescope/probe spatial overlap and coincidence window setting.
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from monrad.stage1 import (
    Quality,
    find_file_pairs,
    load_header_params,
    reconstruct_stream,
)
from monrad.stage2 import coincidence_stream
from monrad.stage3 import decode_position
from monrad.stage4 import AlignmentAccumulator
from monrad.stage5 import PoseFitter

_HIT_QUALITIES = ('golden', 'cluster', 'unresolved', 'invalid', 'missing')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='monrad pipeline smoke test')
    p.add_argument('--telescope', required=True, type=Path,
                   metavar='DIR', help='Telescope acquisition directory')
    p.add_argument('--probe', required=True, type=Path,
                   metavar='DIR', help='Probe acquisition directory')
    p.add_argument('--out', default=Path('./pipeline_out'), type=Path,
                   metavar='DIR', help='Output directory (default: ./pipeline_out)')
    p.add_argument('--z-tel', nargs=3, type=float, default=[0., 400., 800.],
                   metavar=('Z0', 'Z1', 'Z2'),
                   help='Telescope plane z-coords in mm (default: 0 400 800)')
    return p.parse_args()


def _load_detector(
    d: Path, label: str,
) -> tuple[object, int, list[Path], list[Path]]:
    headers = list(d.glob('*_header*.txt'))
    if not headers:
        sys.exit(f'ERROR: no *_header.txt found in {d} ({label})')
    utc0, f0 = load_header_params(headers[0])
    gps_paths, pos_paths = find_file_pairs(d)
    if not gps_paths:
        sys.exit(
            f'ERROR: no matching *_GPS.bin / *.bin pairs found in {d} ({label})'
        )
    return utc0, f0, gps_paths, pos_paths


def _fmt_q(q: Counter) -> str:
    return (
        f'GOOD {q[Quality.GOOD]:>6}   '
        f'DEGRADED {q[Quality.DEGRADED]:>6}   '
        f'UNTRUSTED {q[Quality.UNTRUSTED]:>6}'
    )


def _emit(lines: list[str], msg: str = '') -> None:
    print(msg)
    lines.append(msg)


def main() -> None:
    args = _parse_args()
    tel_dir: Path = args.telescope
    prb_dir: Path = args.probe
    out_dir: Path = args.out
    z_tel = np.array(args.z_tel)

    lines: list[str] = []

    # ── Load both detectors ──────────────────────────────────────────────
    tel_utc0, tel_f0, tel_gps, tel_pos = _load_detector(tel_dir, 'telescope')
    prb_utc0, prb_f0, prb_gps, prb_pos = _load_detector(prb_dir, 'probe')

    # ── Pass 1a: telescope alignment (stage 4) + telescope event quality ─
    accum = AlignmentAccumulator(z_tel=z_tel)
    tel_q: Counter = Counter()
    for ev, ref in reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0):
        tel_q[ev.quality] += 1
        hits = decode_position(ref, tel_pos, n_cols=3)
        accum.add(hits)
    alignment = accum.flush()

    # ── Pass 1b: probe event quality (stage 1 only) ──────────────────────
    prb_q: Counter = Counter()
    for ev, _ref in reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0):
        prb_q[ev.quality] += 1

    # ── Print stage 1 ────────────────────────────────────────────────────
    tel_total = sum(tel_q.values())
    prb_total = sum(prb_q.values())
    ratio_str = f'{tel_total/prb_total:.3f}' if prb_total else 'N/A'
    _emit(lines, '=== Stage 1: Time reconstruction ===')
    _emit(lines, f'  Telescope  {tel_total:>6} events   {_fmt_q(tel_q)}')
    _emit(lines, f'  Probe      {prb_total:>6} events   {_fmt_q(prb_q)}')
    _emit(lines,
          f'  tel/probe ratio: {ratio_str}'
          f'  (telescope hardware filter requires >=2-plane ribbon coincidence)')
    _emit(lines)

    # ── Print stage 4 ────────────────────────────────────────────────────
    _emit(lines, '=== Stage 4: Telescope alignment ===')
    for k, pc in enumerate(alignment.planes):
        _emit(lines,
              f'  Plane {k}   '
              f'delta_x = {pc.delta_x:+7.2f} mm   '
              f'delta_y = {pc.delta_y:+7.2f} mm   '
              f'rot_z = {pc.rotation_z:+.2e} rad')
    _emit(lines, f'  needs_correction: {alignment.needs_correction}')
    # Symmetry / spacing check.
    # Sort columns by z to identify the physical outer and middle planes,
    # regardless of column order in the *.bin file.
    dx = [pc.delta_x for pc in alignment.planes]
    dy = [pc.delta_y for pc in alignment.planes]
    sorted_idx = list(np.argsort(z_tel))           # col indices in z order
    i_lo, i_mid, i_hi = sorted_idx
    dz_lo  = float(z_tel[i_mid] - z_tel[i_lo])
    dz_hi  = float(z_tel[i_hi]  - z_tel[i_mid])
    span   = float(z_tel[i_hi]  - z_tel[i_lo])
    evenly_spaced = abs(dz_lo - dz_hi) < 1e-6 * span
    # For evenly-spaced planes the two-plane predictor is a mathematical
    # identity: delta[outer_lo] = delta[outer_hi], delta[mid] = -delta[outer]/2.
    asym_x = abs(dx[i_lo] - dx[i_hi])
    asym_y = abs(dy[i_lo] - dy[i_hi])
    _emit(lines,
          f'  Symmetry check (outer planes: col {i_lo} z={z_tel[i_lo]:.0f} mm'
          f' vs col {i_hi} z={z_tel[i_hi]:.0f} mm):')
    _emit(lines,
          f'    |delta_x[{i_lo}] - delta_x[{i_hi}]| = {asym_x:5.2f} mm'
          f'   |delta_y[{i_lo}] - delta_y[{i_hi}]| = {asym_y:5.2f} mm')
    if evenly_spaced:
        ratio_dx = dx[i_mid] / dx[i_lo] if abs(dx[i_lo]) > 1e-9 else float('nan')
        _emit(lines,
              f'    delta_x[mid={i_mid}]/delta_x[outer={i_lo}] = {ratio_dx:+.2f}'
              f'   (expected -0.50: algorithm identity for evenly-spaced z)')
        _emit(lines,
              f'  Note: z spacing is even ({dz_lo:.0f} mm gaps); '
              f'middle column is col {i_mid} (z={z_tel[i_mid]:.0f} mm). '
              f'Two-plane predictor measures curvature only.')
    else:
        t_lo = (z_tel[i_lo] - z_tel[i_mid]) / (z_tel[i_hi] - z_tel[i_mid])
        t_hi = (z_tel[i_hi] - z_tel[i_lo]) / (z_tel[i_mid] - z_tel[i_lo])
        _emit(lines,
              f'  Note: z=[{z_tel[0]:.0f},{z_tel[1]:.0f},{z_tel[2]:.0f}] mm '
              f'(uneven: dz_lo={dz_lo:.0f} mm, dz_hi={dz_hi:.0f} mm); '
              f'extrapolation factors t_lo={t_lo:.3f}, t_hi={t_hi:.3f}. '
              f'delta[outer_lo]≠delta[outer_hi] expected.')
    _emit(lines)

    # ── Pass 2: coincidence search (stage 2) + hit quality (stage 3)
    #           + pose fit (stage 5) ─────────────────────────────────────
    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    fitter = PoseFitter(
        tel_z=z_tel,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
    )

    n_coinc = 0
    total_cluster_size = 0
    tel_hit_q: list[Counter] = [Counter(), Counter(), Counter()]
    prb_hit_q: Counter = Counter()
    _pos_paths = {0: tel_pos, 1: prb_pos}
    _n_cols    = {0: 3,       1: 1}

    for cluster in coincidence_stream(
        [tel_stream, prb_stream], detector_ids=[0, 1],
    ):
        n_coinc += 1
        total_cluster_size += len(cluster)
        for det_id, _ev, ref in cluster:
            hits = decode_position(
                ref, _pos_paths[det_id], n_cols=_n_cols[det_id],
            )
            if det_id == 0:
                for plane_idx, h in enumerate(hits):
                    tel_hit_q[plane_idx][
                        h.quality if h is not None else 'missing'
                    ] += 1
            else:
                for h in hits:
                    prb_hit_q[h.quality if h is not None else 'missing'] += 1
        fitter.add(cluster)

    pose = fitter.flush()

    # ── Print stage 2 ────────────────────────────────────────────────────
    mean_sz = (total_cluster_size / n_coinc) if n_coinc else 0.0
    _emit(lines, '=== Stage 2: Coincidence search ===')
    _emit(lines, f'  Coincidences     : {n_coinc:>6}')
    _emit(lines, f'  Mean cluster size: {mean_sz:>6.2f}')
    _emit(lines)

    # ── Print stage 3 ────────────────────────────────────────────────────
    _emit(lines, '=== Stage 3: Hit quality (coincidence survivors) ===')
    _emit(lines, f'  {n_coinc} coincidences x 3 telescope planes = {n_coinc*3} readings')
    for k, q in enumerate(tel_hit_q):
        parts = '   '.join(f'{qn} {q[qn]}' for qn in _HIT_QUALITIES)
        _emit(lines, f'  Plane {k}    {parts}')
    _emit(lines, f'  {n_coinc} coincidences x 1 probe plane = {n_coinc} readings')
    prb_parts = '   '.join(f'{q} {prb_hit_q[q]}' for q in _HIT_QUALITIES)
    _emit(lines, f'  Probe      {prb_parts}')
    _emit(lines)

    # ── Print stage 5 ────────────────────────────────────────────────────
    _emit(lines, '=== Stage 5: Probe pose fit ===')
    if pose is None:
        _emit(lines, '  SKIPPED — too few coincidences survived to fit pose; check')
        _emit(lines,
              '  telescope/probe spatial overlap and coincidence window setting.')
    else:
        sigma_tx    = math.sqrt(abs(pose.cov[0, 0]))
        sigma_ty    = math.sqrt(abs(pose.cov[1, 1]))
        sigma_theta = math.sqrt(abs(pose.cov[2, 2]))
        sigma_zp    = math.sqrt(abs(pose.cov[3, 3]))
        _emit(lines, f'  t_x   = {pose.t_x:+7.1f} ± {sigma_tx:.1f} mm')
        _emit(lines, f'  t_y   = {pose.t_y:+7.1f} ± {sigma_ty:.1f} mm')
        _emit(lines,
              f'  theta = {math.degrees(pose.theta):+7.1f} '
              f'± {math.degrees(sigma_theta):.1f} deg')
        _emit(lines, f'  z_p   = {pose.z_p:+7.1f} ± {sigma_zp:.1f} mm')
        _emit(lines, f'  n_inliers = {pose.n_inliers}')

    # ── Write summary.txt ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / 'summary.txt'
    summary.write_text('\n'.join(lines) + '\n')
    print(f'\nSummary written to {summary}')


if __name__ == '__main__':
    main()
