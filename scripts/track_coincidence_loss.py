"""Instrument the Stage 2 -> Stage 5 funnel: count where coincidences are lost.

Replays pass 2 of run_pipeline.py (coincidence search + pose fit) against real
data, but tallies *why* each Stage 2 coincidence fails to become a Stage 5
inlier.  The gates live inside PoseFitter._decode_cluster; rather than
re-implementing them here (which previously drifted out of sync when
_decode_cluster's algorithm changed), this hooks PoseFitter.on_decode and
tallies the DecodeReport it emits, so this script can never go stale again.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monrad.timing import (  # noqa: E402
    find_file_pairs,
    load_header_params,
    reconstruct_stream,
)
from monrad.coincidence import coincidence_stream  # noqa: E402
from monrad.reconstruction import decode_position  # noqa: E402
from monrad.alignment import AlignmentAccumulator  # noqa: E402
from monrad.pose import GATE_ORDER, DecodeReport, PoseFitter  # noqa: E402

Z_TEL = np.array([0.0, -1340.0, -670.0])
TEL_DIR = Path("data/0_testLab_20210723/Base")
PRB_DIR = Path("data/0_testLab_20210723/Probe_0")


def _load(d: Path) -> tuple[datetime, int, list[Path], list[Path]]:
    headers = list(d.glob("*_header*.txt"))
    utc0, f0 = load_header_params(headers[0])
    gps, pos = find_file_pairs(d)
    return utc0, f0, gps, pos


def main() -> None:
    tel_utc0, tel_f0, tel_gps, tel_pos = _load(TEL_DIR)
    prb_utc0, prb_f0, prb_gps, prb_pos = _load(PRB_DIR)

    # Stage 4 alignment (same as the pipeline).
    accum = AlignmentAccumulator(z_tel=Z_TEL)
    for _ev, ref in reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0):
        accum.add(decode_position(ref, tel_pos, n_cols=3, tot_thresh=1))
    alignment = accum.flush()

    reasons: Counter = Counter()
    cand_ambiguous_planes: Counter = Counter()  # per cluster: 0/1/2/3 planes len>1
    # The winning triple's chi2 is reported both for clusters that passed the
    # <_CHI2_TRACK cut ("accepted"/"probe_quality") and ones that didn't
    # ("chi2_track_cut" — the best of a noisy candidate search). Keep the
    # two separate throughout, or the much larger/noisier rejected pool
    # buries the post-cut stats (e.g. chi2 should sit near 0 post-cut, but
    # averaged with the rejected pool it looks like it's mean=147).
    chi2_n = {"pass": 0, "fail": 0}
    chi2_sum = {"pass": 0.0, "fail": 0.0}

    def _on_decode(r: DecodeReport) -> None:
        reasons[r.reason] += 1
        if r.cand_counts is not None:
            n_ambiguous = sum(1 for n in r.cand_counts if n > 1)
            cand_ambiguous_planes[n_ambiguous] += 1
        bucket = "fail" if r.reason == "chi2_track_cut" else "pass"
        if r.chi2 is not None:
            chi2_n[bucket] += 1
            chi2_sum[bucket] += r.chi2

    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        tot_thresh=1,
        on_decode=_on_decode,
    )

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    n_coinc = 0
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        n_coinc += 1
        fitter.add(cluster)
    pose = fitter.flush()

    clean = reasons["accepted"]
    n_inliers = pose.n_inliers if pose else 0
    ransac = clean - n_inliers

    print("\n=== Coincidence loss funnel (Stage 2 -> Stage 5) ===")
    print(f"  Stage 2 coincidences found            : {n_coinc:>8}")
    running = n_coinc
    for reason in GATE_ORDER:
        n = reasons[reason]
        pct = 100.0 * n / n_coinc if n_coinc else 0.0
        running -= n
        print(f"  - {reason:<28}: {n:>8}  ({pct:5.2f}%)   survivors -> {running}")
    # Catch-all for any reason that is neither a known gate nor "accepted"
    # (e.g. a gate added to _decode_cluster but not to GATE_ORDER), so the
    # funnel never silently loses counts and survivors stay reconciled.
    other = sum(
        n for r, n in reasons.items() if r not in GATE_ORDER and r != "accepted"
    )
    if other:
        pct = 100.0 * other / n_coinc if n_coinc else 0.0
        running -= other
        print(
            f"  - {'gate_other':<28}: {other:>8}  ({pct:5.2f}%)   survivors -> {running}"
        )
    print(f"  Clean coincidences fed to fit         : {clean:>8}")
    print(f"  - RANSAC Mahalanobis outliers          : {ransac:>8}")
    print(f"  Stage 5 inliers (final)               : {n_inliers:>8}")
    print("\n  Telescope ambiguous-plane count per cluster (candidates>1):")
    for n_amb in sorted(cand_ambiguous_planes):
        print(f"    {n_amb} plane(s) ambiguous: {cand_ambiguous_planes[n_amb]:>8}")
    for bucket, label in (
        ("pass", "passed cut (accepted + probe_quality-rejected)"),
        ("fail", "failed cut (chi2_track_cut, best of a noisy search)"),
    ):
        n = chi2_n[bucket]
        if n:
            print(
                f"\n  Winning-triple chi2, {label}: mean={chi2_sum[bucket] / n:.3f}  n={n}"
            )
    if n_coinc:
        print(
            f"\n  Overall survival: {n_inliers}/{n_coinc} = "
            f"{100.0 * n_inliers / n_coinc:.4f}%"
        )


if __name__ == "__main__":
    main()
