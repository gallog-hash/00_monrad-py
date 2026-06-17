"""Instrument the Stage 2 -> Stage 5 funnel: count where coincidences are lost.

Replays pass 2 of run_pipeline.py (coincidence search + pose fit) against real
data, but tallies *why* each Stage 2 coincidence fails to become a Stage 5
inlier.  The five gates live inside PoseFitter._decode_cluster + fit_probe_pose
(stage5.py); we subclass PoseFitter to count them without altering behaviour.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monrad.stage1 import (  # noqa: E402
    find_file_pairs,
    load_header_params,
    reconstruct_stream,
)
from monrad.stage2 import coincidence_stream  # noqa: E402
from monrad.stage3 import (  # noqa: E402
    decode_position,
    disambiguate_telescope_hits,
)
from monrad.stage4 import AlignmentAccumulator  # noqa: E402
from monrad.stage5 import _CHI2_TRACK, PoseFitter, _tel_line_fit  # noqa: E402

Z_TEL = np.array([0.0, -1340.0, -670.0])
TEL_DIR = Path("data/0_testLab_20210723/Base")
PRB_DIR = Path("data/0_testLab_20210723/Probe_0")


class CountingPoseFitter(PoseFitter):
    """PoseFitter that tallies each rejection gate in _decode_cluster."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reasons: Counter = Counter()

    def _decode_cluster(self, cluster):
        tel_refs = [r for det_id, _e, r in cluster if det_id == self.tel_id]
        prb_refs = [r for det_id, _e, r in cluster if det_id == self.prb_id]
        if len(tel_refs) != 1 or len(prb_refs) != 1:
            self.reasons["gate1_ambiguous_cluster"] += 1
            return None
        tel_ref, prb_ref = tel_refs[0], prb_refs[0]

        tel_hits = decode_position(
            tel_ref,
            self.tel_pos_paths,
            n_cols=3,
            tot_thresh=self.tot_thresh,
            tot_weights=self.tot_weights,
        )
        fail_before = any(h.quality not in ("golden", "cluster") for h in tel_hits)
        tel_hits = disambiguate_telescope_hits(tel_hits, self.tel_z)
        fail_after = any(h.quality not in ("golden", "cluster") for h in tel_hits)
        if fail_before and not fail_after:
            self.reasons["recovered_by_disambiguation"] += 1
        if fail_after:
            self.reasons["gate2_tel_hit_quality"] += 1
            return None

        corr = self.alignment
        x_arr = np.array([tel_hits[k].x_mm - corr.planes[k].delta_x for k in range(3)])
        y_arr = np.array([tel_hits[k].y_mm - corr.planes[k].delta_y for k in range(3)])
        sigma_x_arr = np.array([tel_hits[k].sigma_x for k in range(3)])
        sigma_y_arr = np.array([tel_hits[k].sigma_y for k in range(3)])
        z_arr = self.alignment.corrected_z_tel(self.tel_z)
        z_x_arr = z_arr + np.array([corr.planes[k].tilt_y * x_arr[k] for k in range(3)])
        z_y_arr = z_arr + np.array([corr.planes[k].tilt_x * y_arr[k] for k in range(3)])
        _ax, _bx, _ay, _by, _cx, _cy, chi2_line = _tel_line_fit(
            x_arr, y_arr, z_x_arr, sigma_x_arr, sigma_y_arr, z_y_arr=z_y_arr
        )
        if chi2_line >= _CHI2_TRACK:
            self.reasons["gate3_track_chi2"] += 1
            return None

        prb_hits = decode_position(
            prb_ref,
            self.prb_pos_paths,
            n_cols=1,
            tot_thresh=self.tot_thresh,
            tot_weights=self.tot_weights,
        )
        if prb_hits[0].quality not in ("golden", "cluster"):
            self.reasons["gate4_probe_hit_quality"] += 1
            return None

        co = super()._decode_cluster(cluster)
        if co is None:
            self.reasons["gate_other"] += 1
        else:
            self.reasons["accepted_clean"] += 1
        return co


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

    fitter = CountingPoseFitter(
        tel_z=Z_TEL,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        tot_thresh=1,
    )

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    n_coinc = 0
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        n_coinc += 1
        fitter.add(cluster)
    pose = fitter.flush()

    r = fitter.reasons
    clean = r["accepted_clean"]
    n_inliers = pose.n_inliers if pose else 0
    ransac = clean - n_inliers

    print("\n=== Coincidence loss funnel (Stage 2 -> Stage 5) ===")
    print(f"  Stage 2 coincidences found            : {n_coinc:>8}")
    rows = [
        ("gate1: ambiguous cluster (not 1 tel+1 prb)", r["gate1_ambiguous_cluster"]),
        ("gate2: telescope hit quality (any plane)  ", r["gate2_tel_hit_quality"]),
        ("gate3: telescope track chi2 cut           ", r["gate3_track_chi2"]),
        ("gate4: probe hit quality                  ", r["gate4_probe_hit_quality"]),
        ("gate_other                                ", r["gate_other"]),
    ]
    running = n_coinc
    for label, n in rows:
        pct = 100.0 * n / n_coinc if n_coinc else 0.0
        running -= n
        print(f"  - {label}: {n:>8}  ({pct:5.2f}%)   survivors -> {running}")
    print(f"  Clean coincidences fed to fit         : {clean:>8}")
    print(f"  - gate5: RANSAC Mahalanobis outliers      : {ransac:>8}")
    print(f"  Stage 5 inliers (final)               : {n_inliers:>8}")
    print(
        f"\n  (of which recovered by two-plane disambiguation at gate2: "
        f"{r['recovered_by_disambiguation']})"
    )
    if n_coinc:
        print(
            f"\n  Overall survival: {n_inliers}/{n_coinc} = "
            f"{100.0 * n_inliers / n_coinc:.4f}%"
        )


if __name__ == "__main__":
    main()
