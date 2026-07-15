"""Tests for the time-varying alignment schedule (monrad.monitor.io).

Covers AlignmentSchedule's step-function lookup, the directory loader
(per-file z_tel guard, empty-dir error, filename-order independence), the
integer-ns window-start clock, and the _cluster_tel_time pre-decode helper.
All synthetic; no real detector files required.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from monrad.alignment import AlignmentCorrection, PlaneCorrection, save_alignment
from monrad.monitor.io import (
    AlignmentSchedule,
    _cluster_tel_time,
    _parse_window_label,
    load_alignment_schedule,
)
from monrad.synthetic.generate import Z_TEL
from monrad.timing import PosRef, Quality, TimedEvent, _utc_to_ns

_REF = PosRef(file_idx=0, row_offset=0)


def _ev(t_ns: int) -> TimedEvent:
    return TimedEvent(t_ns=t_ns, evt_seq=0, quality=Quality.GOOD)


_Z = np.array(Z_TEL, dtype=float)


def _corr(delta_x: float) -> AlignmentCorrection:
    """A distinct correction, tagged by the first plane's delta_x so tests can
    tell which window a lookup resolved to."""
    planes = [
        PlaneCorrection(delta_x, 0.0, 0.0, 0.0, 0.0, 0.0),
        PlaneCorrection(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        PlaneCorrection(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    return AlignmentCorrection(planes, False)


def _schedule(labels: list[str]) -> AlignmentSchedule:
    """Build a schedule directly from window labels, one distinct correction
    each (delta_x = window index)."""
    starts = np.array(
        [_utc_to_ns(_parse_window_label(lbl)) for lbl in labels], np.int64
    )
    corrs = [_corr(float(i)) for i in range(len(labels))]
    return AlignmentSchedule(starts_ns=starts, corrections=corrs, labels=labels)


def _write_window(
    dir_: Path,
    label: str,
    delta_x: float,
    *,
    z_tel=_Z,
    utc_start_ns: int | None = None,
    utc_end_ns: int | None = None,
) -> None:
    save_alignment(
        _corr(delta_x),
        dir_ / f"alignment_{label}.json",
        date=label,
        z_tel=z_tel,
        files=[f"{label}.bin"],
        n_events=100,
        utc_start_ns=utc_start_ns,
        utc_end_ns=utc_end_ns,
    )


# ── AlignmentSchedule.at (step function) ─────────────────────────────────────


def test_before_first_window_clamps_to_first():
    sch = _schedule(["20230418_060000", "20230418_120000"])
    t = _utc_to_ns(datetime(2023, 4, 18, 5, 0, 0))  # before window 0
    assert sch.at(t) is sch.corrections[0]


def test_boundary_crossing_switches():
    sch = _schedule(["20230418_060000", "20230418_120000"])
    start1 = int(sch.starts_ns[1])
    # side="right": exactly at the boundary and just after -> window 1.
    assert sch.at(start1) is sch.corrections[1]
    assert sch.at(start1 + 1) is sch.corrections[1]
    # just before -> still window 0.
    assert sch.at(start1 - 1) is sch.corrections[0]


def test_gap_holds_last_correction():
    # windows at 06:00 and 18:00 with nothing in between; a coincidence at
    # 12:00 (a gap) must hold the last correction that started (window 0).
    sch = _schedule(["20230418_060000", "20230418_180000"])
    t = _utc_to_ns(datetime(2023, 4, 18, 12, 0, 0))
    assert sch.at(t) is sch.corrections[0]


def test_single_window_passthrough():
    sch = _schedule(["20230418"])
    for dt in (
        datetime(2020, 1, 1),
        datetime(2023, 4, 18, 23, 59),
        datetime(2030, 1, 1),
    ):
        assert sch.at(_utc_to_ns(dt)) is sch.corrections[0]


def test_at_order_independent():
    # searchsorted is stateless: querying out of time order still returns the
    # correct step each time (no monotonic-pointer bug).
    sch = _schedule(["20230418_000000", "20230418_060000", "20230418_120000"])
    early = _utc_to_ns(datetime(2023, 4, 18, 3, 0, 0))  # window 0
    late = _utc_to_ns(datetime(2023, 4, 18, 13, 0, 0))  # window 2
    mid = _utc_to_ns(datetime(2023, 4, 18, 7, 0, 0))  # window 1
    assert sch.at(late) is sch.corrections[2]
    assert sch.at(early) is sch.corrections[0]
    assert sch.at(mid) is sch.corrections[1]
    assert sch.at(late) is sch.corrections[2]  # again, after going backwards


# ── load_alignment_schedule (directory) ──────────────────────────────────────


def test_load_schedule_two_windows(tmp_path: Path):
    _write_window(tmp_path, "20230418_060000", 1.0)
    _write_window(tmp_path, "20230418_120000", 2.0)
    sch = load_alignment_schedule(tmp_path, expect_z_tel=_Z)
    assert sch.labels == ["20230418_060000", "20230418_120000"]
    assert sch.corrections[0].planes[0].delta_x == 1.0
    assert sch.corrections[1].planes[0].delta_x == 2.0


def test_load_schedule_sorted_ascending(tmp_path: Path):
    # Written (and hence globbed) in a mixed order; loader must sort by start.
    _write_window(tmp_path, "20230418_120000", 2.0)
    _write_window(tmp_path, "20230418_000000", 0.0)
    _write_window(tmp_path, "20230418_060000", 1.0)
    sch = load_alignment_schedule(tmp_path, expect_z_tel=_Z)
    assert list(sch.starts_ns) == sorted(sch.starts_ns)
    assert sch.labels == ["20230418_000000", "20230418_060000", "20230418_120000"]
    # delta_x tags follow the labels, confirming corrections track the sort.
    assert [c.planes[0].delta_x for c in sch.corrections] == [0.0, 1.0, 2.0]


def test_load_schedule_keys_on_utc_start_ns(tmp_path: Path):
    """When present, the schedule keys on the true UTC ``utc_start_ns`` field,
    not on the (DAQ-local) file-name label -- the whole point of the fix."""
    # labels say 06:00 / 12:00, but the stored UTC starts are 2h earlier
    # (a DAQ that names files in UTC+2), so keying must follow utc_start_ns.
    off = 2 * 3600 * 1_000_000_000
    s0 = _utc_to_ns(datetime(2023, 4, 18, 6, 0, 0)) - off
    s1 = _utc_to_ns(datetime(2023, 4, 18, 12, 0, 0)) - off
    _write_window(tmp_path, "20230418_060000", 1.0, utc_start_ns=s0)
    _write_window(tmp_path, "20230418_120000", 2.0, utc_start_ns=s1)
    sch = load_alignment_schedule(tmp_path, expect_z_tel=_Z)
    assert list(sch.starts_ns) == [s0, s1]
    # a coincidence at real UTC 10:30 (past s1) gets the second window...
    assert sch.at(_utc_to_ns(datetime(2023, 4, 18, 10, 30, 0))) is sch.corrections[1]
    # ...whereas label-as-UTC keying would have kept the first (10:30 < 12:00).


def test_load_schedule_legacy_fallback_to_label(tmp_path: Path):
    """A file with no utc_start_ns (pre-fix) falls back to label-as-UTC."""
    _write_window(tmp_path, "20230418_060000", 1.0)  # no utc_start_ns
    sch = load_alignment_schedule(tmp_path, expect_z_tel=_Z)
    assert int(sch.starts_ns[0]) == _utc_to_ns(datetime(2023, 4, 18, 6, 0, 0))


def test_save_alignment_writes_utc_bounds(tmp_path: Path):
    start = _utc_to_ns(datetime(2023, 4, 18, 6, 0, 0))
    end = _utc_to_ns(datetime(2023, 4, 18, 12, 0, 0))
    path = tmp_path / "alignment_20230418_060000.json"
    _write_window(tmp_path, "20230418_060000", 1.0, utc_start_ns=start, utc_end_ns=end)
    import json

    payload = json.loads(path.read_text())
    assert payload["utc_start_ns"] == start
    assert payload["utc_end_ns"] == end
    assert payload["utc_start"].startswith("2023-04-18T06:00:00")
    # load_alignment still works (ignores the extra provenance).
    from monrad.alignment import load_alignment

    assert load_alignment(path, expect_z_tel=_Z).planes[0].delta_x == 1.0


def test_load_schedule_z_tel_mismatch_raises(tmp_path: Path):
    _write_window(tmp_path, "20230418_060000", 1.0)  # saved with _Z
    _write_window(
        tmp_path, "20230418_120000", 2.0, z_tel=np.array([0.0, -1340.0, -670.0])
    )
    with pytest.raises(ValueError, match="z-order-dependent"):
        load_alignment_schedule(tmp_path, expect_z_tel=_Z)


def test_load_schedule_empty_dir_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no alignment_.*json"):
        load_alignment_schedule(tmp_path, expect_z_tel=_Z)


# ── integer-ns clock ─────────────────────────────────────────────────────────


def test_label_to_ns_is_exact_integer():
    # The schedule keys must be the same integer clock that stamps
    # Coincidence.t_ns -- not a float timestamp() that would lose ns precision.
    label = "20230418_060000"
    dt = _parse_window_label(label)
    got = _utc_to_ns(dt)
    expected = int((dt - datetime(1970, 1, 1)).total_seconds()) * 1_000_000_000
    assert got == expected
    assert isinstance(got, int)


def test_parse_window_label_both_forms():
    assert _parse_window_label("20230418") == datetime(2023, 4, 18, 0, 0, 0)
    assert _parse_window_label("20230418_063000") == datetime(2023, 4, 18, 6, 30, 0)


# ── _cluster_tel_time (pre-decode helper) ────────────────────────────────────


def test_cluster_tel_time_reads_telescope_entry():
    cluster = [(0, _ev(12345), _REF), (1, _ev(99999), _REF)]
    assert _cluster_tel_time(cluster, tel_id=0) == 12345


def test_cluster_tel_time_none_when_ambiguous():
    # Two telescope entries (or zero) -> None; the decode rejects it anyway.
    two_tel = [(0, _ev(1), _REF), (0, _ev(2), _REF)]
    no_tel = [(1, _ev(1), _REF)]
    assert _cluster_tel_time(two_tel, tel_id=0) is None
    assert _cluster_tel_time(no_tel, tel_id=0) is None
