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
                delta_z =  +0.00 mm   tilt_x =  +0.00e+00 rad   tilt_y =  +0.00e+00 rad
      Plane 1   delta_x =  -0.08 mm   delta_y =  +0.11 mm   rot_z =  -1.00e-04 rad
                delta_z =  +1.20 mm   tilt_x =  +4.00e-03 rad   tilt_y =  -2.00e-03 rad
      Plane 2   delta_x =  +0.04 mm   delta_y =  +0.02 mm   rot_z =  +2.00e-04 rad
                delta_z =  +0.00 mm   tilt_x =  +0.00e+00 rad   tilt_y =  +0.00e+00 rad
      needs_correction: True
      (delta_z/tilt_x/tilt_y are fitted for the middle plane only; outer
       planes carry these degrees of freedom degenerate with track slope.)
      Symmetry check (Plane 0 vs Plane 2):
        |delta_x[0] - delta_x[2]| =  0.00 mm   |delta_y[0] - delta_y[2]| =  0.00 mm
        delta_x[1]/delta_x[0] = -0.50   (expected -0.50: algorithm identity for z=[0,400,800])
      Note: for evenly-spaced z the two-plane predictor always gives
        delta[0]=delta[2], delta[1]=-delta[0]/2 (measures curvature only).

    === Stage 2: Coincidence search ===
      Coincidences     :    523
      Mean cluster size:   2.00

    === Stage 3: Pose-fit gate funnel (combinatorial path) ===
      (counts come from PoseFitter._decode_cluster's own DecodeReport, not a
      separate re-derivation)
      rejected: ambiguous_cluster            12   survivors -> 511
      rejected: zero_candidate_plane          8   survivors -> 503
      rejected: no_anchor_plane               3   survivors -> 500
      rejected: chi2_track_cut                9   survivors -> 491
      rejected: probe_quality                 4   survivors -> 487
      accepted (fed to pose optimizer)         487

      Telescope per-plane candidate-count distribution:
        Plane 0    invalid(0) 8   resolved(1) 460   ambiguous(2+) 55
        Plane 1    invalid(0) 6   resolved(1) 455   ambiguous(2+) 62
        Plane 2    invalid(0) 9   resolved(1) 462   ambiguous(2+) 52
      Probe hit quality (coincidences that reached probe decode):
        golden 410   cluster 91
      Winning-triple χ² (accepted + probe_quality-rejected): mean=0.421  std=0.612  n=500
      Winning-triple per-plane quality (candidate the χ² search picked):
        Plane 0    golden 460   cluster 40
        Plane 1    golden 455   cluster 45
        Plane 2    golden 462   cluster 38
      Winning-triple per-plane TOT score (tot_x + tot_y, ribbon×fiber):
        Plane 0    mean=412.3  std=58.1  n=500
        Plane 1    mean=405.7  std=61.4  n=500
        Plane 2    mean=409.0  std=59.9  n=500

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
from datetime import datetime
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
from monrad.stage5 import GATE_ORDER, DecodeReport, PoseFitter

_CAND_BUCKETS = ("invalid(0)", "resolved(1)", "ambiguous(2+)")
# Probe Hit.quality values (stage3.Hit), in canonical order, for the probe
# hit-quality table.  decode_position can yield any of these for the probe
# plane, so the table must list them all — printing only golden/cluster
# silently drops the unresolved/invalid rejections.
_PRB_QUALITY_ORDER = ("golden", "cluster", "efficiency", "unresolved", "invalid")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="monrad pipeline smoke test")
    p.add_argument(
        "--telescope",
        required=True,
        type=Path,
        metavar="DIR",
        help="Telescope acquisition directory",
    )
    p.add_argument(
        "--probe",
        required=True,
        type=Path,
        metavar="DIR",
        help="Probe acquisition directory",
    )
    p.add_argument(
        "--out",
        default=Path("./pipeline_out"),
        type=Path,
        metavar="DIR",
        help="Output directory (default: ./pipeline_out)",
    )
    p.add_argument(
        "--z-tel",
        nargs=3,
        type=float,
        default=[0.0, 400.0, 800.0],
        metavar=("Z0", "Z1", "Z2"),
        help="Telescope plane z-coords in mm (default: 0 400 800)",
    )
    p.add_argument(
        "--tot-thresh",
        type=int,
        default=1,
        metavar="N",
        help="Minimum number of the 16 rows in which a bit must fire "
        "to be kept in the OR mask (default: 1 = plain OR). "
        "Values 2-4 filter single-row cross-talk spikes.",
    )
    p.add_argument(
        "--tot-weights",
        action="store_true",
        default=False,
        help="Weight cluster centroids by per-bit TOT counts "
        "(ribbon_count × fiber_count). No effect on golden hits.",
    )
    return p.parse_args()


def _load_detector(
    d: Path,
    label: str,
) -> tuple[datetime, int, list[Path], list[Path]]:
    headers = list(d.glob("*_header*.txt"))
    if not headers:
        sys.exit(f"ERROR: no *_header.txt found in {d} ({label})")
    utc0, f0 = load_header_params(headers[0])
    gps_paths, pos_paths = find_file_pairs(d)
    if not gps_paths:
        sys.exit(f"ERROR: no matching *_GPS.bin / *.bin pairs found in {d} ({label})")
    return utc0, f0, gps_paths, pos_paths


def _fmt_q(q: Counter) -> str:
    return (
        f"GOOD {q[Quality.GOOD]:>6}   "
        f"DEGRADED {q[Quality.DEGRADED]:>6}   "
        f"UNTRUSTED {q[Quality.UNTRUSTED]:>6}"
    )


def _emit(lines: list[str], msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def _cand_bucket(n: int) -> str:
    if n == 0:
        return "invalid(0)"
    if n == 1:
        return "resolved(1)"
    return "ambiguous(2+)"


def main() -> None:
    args = _parse_args()
    tel_dir: Path = args.telescope
    prb_dir: Path = args.probe
    out_dir: Path = args.out
    z_tel = np.array(args.z_tel)
    tot_thresh: int = args.tot_thresh
    tot_weights: bool = args.tot_weights

    lines: list[str] = []

    # ── Run configuration ────────────────────────────────────────────────
    # Record the input data directories and the telescope plane
    # z-coordinates used: the alignment and pose fits depend on them, and
    # columns are not always stored in z order (the middle plane is
    # argsort(z)[1], not necessarily column 1).
    z_str = "  ".join(f"{zz:g}" for zz in args.z_tel)
    _emit(lines, "=== Run configuration ===")
    _emit(lines, f"  Telescope data: {tel_dir}")
    _emit(lines, f"  Probe data:     {prb_dir}")
    _emit(lines, f"  Telescope plane z (mm): {z_str}")
    _emit(lines)

    # ── Load both detectors ──────────────────────────────────────────────
    tel_utc0, tel_f0, tel_gps, tel_pos = _load_detector(tel_dir, "telescope")
    prb_utc0, prb_f0, prb_gps, prb_pos = _load_detector(prb_dir, "probe")

    # ── Pass 1a: telescope alignment (stage 4) + telescope event quality ─
    accum = AlignmentAccumulator(z_tel=z_tel)
    tel_q: Counter = Counter()
    for ev, ref in reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0):
        tel_q[ev.quality] += 1
        hits = decode_position(
            ref, tel_pos, n_cols=3, tot_thresh=tot_thresh, tot_weights=tot_weights
        )
        accum.add(hits)
    alignment = accum.flush()

    # ── Pass 1b: probe event quality (stage 1 only) ──────────────────────
    prb_q: Counter = Counter()
    for ev, _ref in reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0):
        prb_q[ev.quality] += 1

    # ── Print stage 1 ────────────────────────────────────────────────────
    tel_total = sum(tel_q.values())
    prb_total = sum(prb_q.values())
    ratio_str = f"{tel_total / prb_total:.3f}" if prb_total else "N/A"
    _emit(lines, "=== Stage 1: Time reconstruction ===")
    _emit(lines, f"  Telescope  {tel_total:>6} events   {_fmt_q(tel_q)}")
    _emit(lines, f"  Probe      {prb_total:>6} events   {_fmt_q(prb_q)}")
    _emit(
        lines,
        f"  tel/probe ratio: {ratio_str}"
        f"  (telescope hardware filter requires >=2-plane ribbon coincidence)",
    )
    _emit(lines)

    # ── Print stage 4 ────────────────────────────────────────────────────
    _emit(lines, "=== Stage 4: Telescope alignment ===")
    for k, pc in enumerate(alignment.planes):
        _emit(
            lines,
            f"  Plane {k}   "
            f"delta_x = {pc.delta_x:+7.2f} mm   "
            f"delta_y = {pc.delta_y:+7.2f} mm   "
            f"rot_z = {pc.rotation_z:+.2e} rad",
        )
        _emit(
            lines,
            f"            "
            f"delta_z = {pc.delta_z:+7.2f} mm   "
            f"tilt_x = {pc.tilt_x:+.2e} rad   "
            f"tilt_y = {pc.tilt_y:+.2e} rad",
        )
    _emit(lines, f"  needs_correction: {alignment.needs_correction}")
    # delta_z/tilt_x/tilt_y are fitted for the middle plane only (k=1);
    # the outer planes leave them at 0 (degenerate with track slope).
    _emit(
        lines,
        "  (delta_z/tilt_x/tilt_y are fitted for the middle plane only; outer"
        " planes carry these degrees of freedom degenerate with track slope.)",
    )
    # Symmetry / spacing check.
    # Sort columns by z to identify the physical outer and middle planes,
    # regardless of column order in the *.bin file.
    dx = [pc.delta_x for pc in alignment.planes]
    dy = [pc.delta_y for pc in alignment.planes]
    sorted_idx = list(np.argsort(z_tel))  # col indices in z order
    i_lo, i_mid, i_hi = sorted_idx
    dz_lo = float(z_tel[i_mid] - z_tel[i_lo])
    dz_hi = float(z_tel[i_hi] - z_tel[i_mid])
    span = float(z_tel[i_hi] - z_tel[i_lo])
    evenly_spaced = abs(dz_lo - dz_hi) < 1e-6 * span
    # For evenly-spaced planes the two-plane predictor is a mathematical
    # identity: delta[outer_lo] = delta[outer_hi], delta[mid] = -delta[outer]/2.
    asym_x = abs(dx[i_lo] - dx[i_hi])
    asym_y = abs(dy[i_lo] - dy[i_hi])
    _emit(
        lines,
        f"  Symmetry check (outer planes: col {i_lo} z={z_tel[i_lo]:.0f} mm"
        f" vs col {i_hi} z={z_tel[i_hi]:.0f} mm):",
    )
    _emit(
        lines,
        f"    |delta_x[{i_lo}] - delta_x[{i_hi}]| = {asym_x:5.2f} mm"
        f"   |delta_y[{i_lo}] - delta_y[{i_hi}]| = {asym_y:5.2f} mm",
    )
    if evenly_spaced:
        ratio_dx = dx[i_mid] / dx[i_lo] if abs(dx[i_lo]) > 1e-9 else float("nan")
        _emit(
            lines,
            f"    delta_x[mid={i_mid}]/delta_x[outer={i_lo}] = {ratio_dx:+.2f}"
            f"   (expected -0.50: algorithm identity for evenly-spaced z)",
        )
        _emit(
            lines,
            f"  Note: z spacing is even ({dz_lo:.0f} mm gaps); "
            f"middle column is col {i_mid} (z={z_tel[i_mid]:.0f} mm). "
            f"Two-plane predictor measures curvature only.",
        )
    else:
        t_lo = (z_tel[i_lo] - z_tel[i_mid]) / (z_tel[i_hi] - z_tel[i_mid])
        t_hi = (z_tel[i_hi] - z_tel[i_lo]) / (z_tel[i_mid] - z_tel[i_lo])
        _emit(
            lines,
            f"  Note: z=[{z_tel[0]:.0f},{z_tel[1]:.0f},{z_tel[2]:.0f}] mm "
            f"(uneven: dz_lo={dz_lo:.0f} mm, dz_hi={dz_hi:.0f} mm); "
            f"extrapolation factors t_lo={t_lo:.3f}, t_hi={t_hi:.3f}. "
            f"delta[outer_lo]≠delta[outer_hi] expected.",
        )
    _emit(lines)

    # ── Pass 2: coincidence search (stage 2) + hit quality (stage 3)
    #           + pose fit (stage 5) ─────────────────────────────────────
    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    # Gate-funnel and per-plane candidate-count stats, collected straight
    # from PoseFitter._decode_cluster's own DecodeReport — not a separate
    # re-derivation of its logic (see DecodeReport's docstring in stage5.py).
    gate_counts: Counter = Counter()
    cand_dist: list[Counter] = [Counter(), Counter(), Counter()]
    prb_q_at_decode: Counter = Counter()
    # The winning triple is reported for every cluster that reached the χ²
    # search, both ones that then passed the <_CHI2_TRACK cut ("accepted"/
    # "probe_quality") and ones that didn't ("chi2_track_cut" — the best of
    # a noisy candidate search, can be huge/low-quality). Track pass/fail
    # separately throughout: conflating them buries the post-cut population
    # (which should look like genuine tracks) under the much larger,
    # much noisier rejected pool.
    win_quality = {
        "pass": [Counter(), Counter(), Counter()],
        "fail": [Counter(), Counter(), Counter()],
    }
    tot_n = {"pass": [0, 0, 0], "fail": [0, 0, 0]}
    tot_sum = {"pass": [0.0, 0.0, 0.0], "fail": [0.0, 0.0, 0.0]}
    tot_sumsq = {"pass": [0.0, 0.0, 0.0], "fail": [0.0, 0.0, 0.0]}
    chi2_n = {"pass": 0, "fail": 0}
    chi2_sum = {"pass": 0.0, "fail": 0.0}
    chi2_sumsq = {"pass": 0.0, "fail": 0.0}

    def _on_decode(r: DecodeReport) -> None:
        gate_counts[r.reason] += 1
        if r.cand_counts is not None:
            for k in range(3):
                cand_dist[k][_cand_bucket(r.cand_counts[k])] += 1
        if r.prb_quality is not None:
            prb_q_at_decode[r.prb_quality] += 1
        bucket = "fail" if r.reason == "chi2_track_cut" else "pass"
        if r.winning_quality is not None:
            for k in range(3):
                win_quality[bucket][k][r.winning_quality[k]] += 1
        if r.winning_tot is not None:
            for k in range(3):
                tot = r.winning_tot[k][0] + r.winning_tot[k][1]
                tot_n[bucket][k] += 1
                tot_sum[bucket][k] += tot
                tot_sumsq[bucket][k] += tot * tot
        if r.chi2 is not None:
            chi2_n[bucket] += 1
            chi2_sum[bucket] += r.chi2
            chi2_sumsq[bucket] += r.chi2 * r.chi2

    fitter = PoseFitter(
        tel_z=z_tel,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
        on_decode=_on_decode,
    )

    n_coinc = 0
    total_cluster_size = 0

    for cluster in coincidence_stream(
        [tel_stream, prb_stream],
        detector_ids=[0, 1],
    ):
        n_coinc += 1
        total_cluster_size += len(cluster)
        fitter.add(cluster)

    pose = fitter.flush()

    # ── Print stage 2 ────────────────────────────────────────────────────
    mean_sz = (total_cluster_size / n_coinc) if n_coinc else 0.0
    _emit(lines, "=== Stage 2: Coincidence search ===")
    _emit(lines, f"  Coincidences     : {n_coinc:>6}")
    _emit(lines, f"  Mean cluster size: {mean_sz:>6.2f}")
    _emit(lines)

    # ── Print stage 3 ────────────────────────────────────────────────────
    _emit(lines, "=== Stage 3: Pose-fit gate funnel (combinatorial path) ===")
    _emit(
        lines,
        "  (counts come from PoseFitter._decode_cluster's own DecodeReport, "
        "not a separate re-derivation)",
    )
    running = n_coinc
    for reason in GATE_ORDER:
        n = gate_counts[reason]
        running -= n
        _emit(lines, f"  rejected: {reason:<22} {n:>7}   survivors -> {running}")
    # Catch-all: any reason _decode_cluster emits that is neither a known gate
    # nor "accepted" (e.g. a gate added to _decode_cluster but not to
    # GATE_ORDER).  Without this the funnel would silently drop those counts
    # and the survivor arithmetic would no longer reconcile.
    other = sum(
        n for r, n in gate_counts.items() if r not in GATE_ORDER and r != "accepted"
    )
    if other:
        running -= other
        _emit(
            lines, f"  rejected: {'gate_other':<22} {other:>7}   survivors -> {running}"
        )
    _emit(
        lines,
        f"  accepted (fed to pose optimizer)         {gate_counts['accepted']:>7}",
    )
    _emit(lines)
    _emit(lines, "  Telescope per-plane candidate-count distribution:")
    for k in range(3):
        parts = "   ".join(f"{label} {cand_dist[k][label]}" for label in _CAND_BUCKETS)
        _emit(lines, f"    Plane {k}    {parts}")
    _emit(lines, "  Probe hit quality (coincidences that reached probe decode):")
    # List every quality that actually occurred, in canonical order, so the
    # rejected buckets (unresolved / invalid) aren't hidden and the counts
    # reconcile with the number of probe decodes.
    prb_parts = "   ".join(
        f"{q} {prb_q_at_decode[q]}" for q in _PRB_QUALITY_ORDER if prb_q_at_decode[q]
    )
    _emit(lines, f"    {prb_parts or '(none)'}")
    for bucket, label in (
        ("pass", "passed cut (accepted + probe_quality-rejected)"),
        ("fail", "failed cut (chi2_track_cut, best of a noisy search)"),
    ):
        n = chi2_n[bucket]
        if n:
            mean = chi2_sum[bucket] / n
            var = max(chi2_sumsq[bucket] / n - mean * mean, 0.0)
            _emit(
                lines,
                f"  Winning-triple χ², {label}: "
                f"mean={mean:.3f}  std={math.sqrt(var):.3f}  n={n}",
            )
    for bucket, label in (
        ("pass", "passed cut"),
        ("fail", "failed cut"),
    ):
        _emit(lines, f"  Winning-triple per-plane quality, {label}:")
        for k in range(3):
            parts = "   ".join(
                f"{q} {win_quality[bucket][k][q]}" for q in ("golden", "cluster")
            )
            _emit(lines, f"    Plane {k}    {parts}")
        _emit(lines, f"  Winning-triple per-plane TOT score, {label} (tot_x + tot_y):")
        for k in range(3):
            n = tot_n[bucket][k]
            if n:
                mean_t = tot_sum[bucket][k] / n
                var_t = max(tot_sumsq[bucket][k] / n - mean_t * mean_t, 0.0)
                _emit(
                    lines,
                    f"    Plane {k}    mean={mean_t:.1f}  std={math.sqrt(var_t):.1f}  "
                    f"n={n}",
                )
    _emit(lines)

    # ── Print stage 5 ────────────────────────────────────────────────────
    _emit(lines, "=== Stage 5: Probe pose fit ===")
    if pose is None:
        _emit(lines, "  SKIPPED — too few coincidences survived to fit pose; check")
        _emit(
            lines, "  telescope/probe spatial overlap and coincidence window setting."
        )
    else:
        sigma_tx = math.sqrt(abs(pose.cov[0, 0]))
        sigma_ty = math.sqrt(abs(pose.cov[1, 1]))
        sigma_theta = math.sqrt(abs(pose.cov[2, 2]))
        sigma_zp = math.sqrt(abs(pose.cov[3, 3]))
        _emit(lines, f"  t_x   = {pose.t_x:+7.1f} ± {sigma_tx:.1f} mm")
        _emit(lines, f"  t_y   = {pose.t_y:+7.1f} ± {sigma_ty:.1f} mm")
        _emit(
            lines,
            f"  theta = {math.degrees(pose.theta):+7.1f} "
            f"± {math.degrees(sigma_theta):.1f} deg",
        )
        _emit(lines, f"  z_p   = {pose.z_p:+7.1f} ± {sigma_zp:.1f} mm")
        _emit(lines, f"  n_inliers = {pose.n_inliers}")

    # ── Write summary.txt ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.txt"
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {summary}")


if __name__ == "__main__":
    main()
