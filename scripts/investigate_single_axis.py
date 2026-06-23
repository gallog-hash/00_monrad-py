"""Measure how often a telescope plane is 'unresolved' on a SINGLE axis only.

A plane decodes as 'unresolved' if either the x OR the y axis fails
(stage3.decode_position).  When only one axis fails, the other axis was
cleanly resolved but its coordinate is currently discarded, so the plane
cannot be recovered by disambiguate_telescope_hits (the recovery path needs
both axes re-derived from candidate lists).

This script classifies, over the real coincidence telescope events, every
'unresolved' plane as:
  - both : both x and y unresolved (not recoverable by keeping one axis)
  - x_only / y_only : only that axis unresolved (the recoverable pool)

and reports, per coincidence, the "one bad plane" topology (exactly one
unresolved plane, the other two golden/cluster) that gate2 currently drops —
split by whether the bad plane is single-axis (extension could recover) or
both-axis.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monrad.timing import (  # noqa: E402
    find_file_pairs,
    load_header_params,
    reconstruct_stream,
)
from monrad.coincidence import coincidence_stream  # noqa: E402
from monrad.reconstruction import decode_position  # noqa: E402

TEL_DIR = Path("data/0_testLab_20210723/Base")
PRB_DIR = Path("data/0_testLab_20210723/Probe_0")


def _load(d: Path) -> tuple[datetime, int, list[Path], list[Path]]:
    headers = list(d.glob("*_header*.txt"))
    utc0, f0 = load_header_params(headers[0])
    gps, pos = find_file_pairs(d)
    return utc0, f0, gps, pos


def _classify_plane(h) -> str:
    """golden | cluster | invalid | unres_both | unres_x_only | unres_y_only."""
    if h.quality != "unresolved":
        return h.quality  # golden / cluster / invalid
    x_unres = h.candidates_x is not None
    y_unres = h.candidates_y is not None
    if x_unres and y_unres:
        return "unres_both"
    if x_unres:
        return "unres_x_only"  # y was resolved
    return "unres_y_only"  # x was resolved


def main() -> None:
    tel_utc0, tel_f0, tel_gps, tel_pos = _load(TEL_DIR)
    prb_utc0, prb_f0, prb_gps, prb_pos = _load(PRB_DIR)

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    plane_cls = Counter()  # per-plane-reading classification
    coinc_topology = Counter()  # per-coincidence topology
    # The "one bad plane" recoverable topology, split by single/both axis:
    one_bad_single = 0
    one_bad_both = 0
    n_coinc = 0

    _resolved = ("golden", "cluster")
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        tel_refs = [r for det_id, _e, r in cluster if det_id == 0]
        if len(tel_refs) != 1:
            continue
        n_coinc += 1
        hits = decode_position(tel_refs[0], tel_pos, n_cols=3, tot_thresh=1)
        cls = [_classify_plane(h) for h in hits]
        for c in cls:
            plane_cls[c] += 1

        resolved = [c in _resolved for c in cls]
        n_res = sum(resolved)
        coinc_topology[f"{n_res}_resolved"] += 1

        # Exactly one plane blocks gate2 (the other two are usable references).
        if n_res == 2:
            bad = next(c for c, r in zip(cls, resolved) if not r)
            if bad in ("unres_x_only", "unres_y_only"):
                one_bad_single += 1
            elif bad == "unres_both":
                one_bad_both += 1

    print("\n=== Per-plane reading classification (coincidence tel events) ===")
    total_readings = sum(plane_cls.values())
    for k in [
        "golden",
        "cluster",
        "unres_x_only",
        "unres_y_only",
        "unres_both",
        "invalid",
    ]:
        n = plane_cls[k]
        print(f"  {k:14}: {n:>8}  ({100.0 * n / total_readings:5.2f}%)")
    unres_total = (
        plane_cls["unres_x_only"] + plane_cls["unres_y_only"] + plane_cls["unres_both"]
    )
    unres_single = plane_cls["unres_x_only"] + plane_cls["unres_y_only"]
    if unres_total:
        print(
            f"  -> of {unres_total} unresolved readings, "
            f"{unres_single} ({100.0 * unres_single / unres_total:.1f}%) "
            f"are single-axis (one axis was cleanly resolved)"
        )

    print("\n=== Per-coincidence telescope topology ===")
    print(f"  coincidences: {n_coinc}")
    for k in ["3_resolved", "2_resolved", "1_resolved", "0_resolved"]:
        n = coinc_topology[k]
        print(f"  {k:12}: {n:>8}  ({100.0 * n / n_coinc:5.2f}%)")

    print("\n=== 'One bad plane' topology (2 good refs + 1 unresolved) ===")
    one_bad = one_bad_single + one_bad_both
    print(f"  total one-bad-plane coincidences : {one_bad}")
    print(f"  - bad plane single-axis (recoverable by extension): {one_bad_single}")
    print(f"  - bad plane both-axis  (needs full re-derivation)  : {one_bad_both}")
    print(
        "\n  (single-axis ones are the additional pool the extension targets, on top\n"
        "   of the both-axis ones the current disambiguation already attempts.)"
    )


if __name__ == "__main__":
    main()
