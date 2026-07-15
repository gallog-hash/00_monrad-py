"""Tests for the daily alignment calibration + hardware-drift monitor.

Covers monrad.alignment.io (JSON round-trip + z_tel guard) and
monrad.monitor.align (day grouping/selection, whole-subset fit,
needs_correction flagging, idempotent history CSV, and the monitor drivers'
--alignment reuse path).  All synthetic; no real detector files required.
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from monrad.alignment import (
    AlignmentCorrection,
    PlaneCorrection,
    load_alignment,
    save_alignment,
)
from monrad.monitor.align import (
    compute_daily_alignment,
    group_by_day,
    select_day_files,
)
from monrad.monitor.io import DetectorFiles, load_detector
from monrad.monitor.timeseries import monitor_probe
from monrad.synthetic.generate import Z_TEL, generate

_Z = np.array(Z_TEL, dtype=float)


# ── serialization ────────────────────────────────────────────────────────────


def _sample_correction(needs: bool = False) -> AlignmentCorrection:
    planes = [
        PlaneCorrection(0.1, -0.2, 1e-4, 0.0, 0.0, 0.0),
        PlaneCorrection(0.3, 0.4, -2e-4, 1.5, 3e-4, -4e-4),
        PlaneCorrection(-0.5, 0.6, 5e-5, 0.0, 0.0, 0.0),
    ]
    return AlignmentCorrection(planes, needs)


def test_save_load_roundtrip(tmp_path: Path):
    corr = _sample_correction(needs=True)
    path = tmp_path / "align.json"
    save_alignment(
        corr, path, date="20230418", z_tel=_Z, files=["a.bin"], n_events=1234
    )
    loaded = load_alignment(path)
    assert loaded.needs_correction == corr.needs_correction
    assert len(loaded.planes) == 3
    for got, want in zip(loaded.planes, corr.planes):
        assert got == pytest.approx(want)


def test_load_z_tel_match_ok(tmp_path: Path):
    path = tmp_path / "align.json"
    save_alignment(
        _sample_correction(), path, date="20230418", z_tel=_Z, files=[], n_events=0
    )
    # Same z_tel → no error.
    load_alignment(path, expect_z_tel=_Z)


def test_load_z_tel_mismatch_raises(tmp_path: Path):
    path = tmp_path / "align.json"
    save_alignment(
        _sample_correction(), path, date="20230418", z_tel=_Z, files=[], n_events=0
    )
    with pytest.raises(ValueError, match="z-order-dependent"):
        load_alignment(path, expect_z_tel=np.array([0.0, -1340.0, -670.0]))


def test_load_bad_schema_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "nope", "planes": [], "needs_correction": false}')
    with pytest.raises(ValueError, match="schema"):
        load_alignment(path)


# ── day grouping / selection ─────────────────────────────────────────────────


def _fake_detector(names: list[str]) -> DetectorFiles:
    """A DetectorFiles whose paths only need valid names for grouping tests."""
    gps = [Path(f"{n}_GPS.bin") for n in names]
    pos = [Path(f"{n}.bin") for n in names]
    return DetectorFiles(datetime(2023, 4, 18), 100_000_000, gps, pos)


def test_group_by_day_orders_ascending():
    det = _fake_detector(["20230419_000000", "20230418_100000", "20230418_090000"])
    days = group_by_day(det)
    assert list(days) == ["20230418", "20230419"]  # day-ascending
    # Within a day, earliest-first (input already sorted by find_file_pairs).
    assert [p[0].name for p in days["20230418"]] == [
        "20230418_090000_GPS.bin",
        "20230418_100000_GPS.bin",
    ]


def test_select_earliest_day_and_n_files():
    det = _fake_detector(
        [
            "20230418_090000",
            "20230418_093000",
            "20230418_100000",
            "20230419_080000",
        ]
    )
    day, gps, pos = select_day_files(det, date=None, n_files=2)
    assert day == "20230418"
    assert [p.name for p in gps] == [
        "20230418_090000_GPS.bin",
        "20230418_093000_GPS.bin",
    ]
    assert len(pos) == 2


def test_select_explicit_date():
    det = _fake_detector(["20230418_090000", "20230419_080000"])
    day, gps, _ = select_day_files(det, date="20230419", n_files=3)
    assert day == "20230419"
    assert [p.name for p in gps] == ["20230419_080000_GPS.bin"]


def test_select_missing_date_raises():
    det = _fake_detector(["20230418_090000"])
    with pytest.raises(ValueError, match="no files for date"):
        select_day_files(det, date="20991231", n_files=3)


# ── whole-subset fit + artifacts (end-to-end on synthetic data) ──────────────


def _gen_day(dest: Path, day: str, hhmmss: str, **kw) -> None:
    """Generate one synthetic acquisition and copy its telescope files into
    *dest*, stamped with the given day/time so grouping sees the right date."""
    src = dest.parent / f"gen_{day}_{hhmmss}"
    start = datetime.strptime(day + hhmmss, "%Y%m%d%H%M%S")
    info = generate(src, start_utc=start, n_tracks=800, **kw)
    dest.mkdir(parents=True, exist_ok=True)
    for f in Path(info["tel_dir"]).iterdir():
        shutil.copy(f, dest / f.name)


def test_compute_aligned_needs_correction_false(tmp_path: Path):
    tel = tmp_path / "tel"
    _gen_day(tel, "20230418", "192100")
    out = tmp_path / "out"
    corr = compute_daily_alignment(tel, _Z, out_dir=out, make_plots=False)
    assert corr.needs_correction is False
    assert (out / "alignment_20230418.json").exists()
    assert (out / "alignment_history.csv").exists()


def test_compute_misaligned_needs_correction_true(tmp_path: Path):
    tel = tmp_path / "tel"
    # 6 mm offset on the middle plane ≫ the 1 mm mechanical limit.
    _gen_day(tel, "20230418", "192100", plane_offsets={1: (6.0, 0.0)})
    out = tmp_path / "out"
    corr = compute_daily_alignment(tel, _Z, out_dir=out, make_plots=False)
    assert corr.needs_correction is True


def test_n_files_limits_files_used(tmp_path: Path):
    tel = tmp_path / "tel"
    _gen_day(tel, "20230418", "192100")
    _gen_day(tel, "20230418", "192600")
    _gen_day(tel, "20230418", "193100")
    out = tmp_path / "out"
    compute_daily_alignment(tel, _Z, n_files=2, out_dir=out, make_plots=False)
    import json

    payload = json.loads((out / "alignment_20230418.json").read_text())
    assert len(payload["files"]) == 2


def test_default_picks_earliest_day(tmp_path: Path):
    tel = tmp_path / "tel"
    _gen_day(tel, "20230418", "192100")
    _gen_day(tel, "20230419", "080000")
    out = tmp_path / "out"
    compute_daily_alignment(tel, _Z, out_dir=out, make_plots=False)
    assert (out / "alignment_20230418.json").exists()
    assert not (out / "alignment_20230419.json").exists()


def _read_history(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_history_idempotent_same_day(tmp_path: Path):
    tel = tmp_path / "tel"
    _gen_day(tel, "20230418", "192100")
    out = tmp_path / "out"
    compute_daily_alignment(tel, _Z, out_dir=out, make_plots=False)
    compute_daily_alignment(tel, _Z, out_dir=out, make_plots=False)
    rows = _read_history(out / "alignment_history.csv")
    assert len(rows) == 1  # re-running the same day must not duplicate


def test_history_two_days_sorted(tmp_path: Path):
    tel = tmp_path / "tel"
    _gen_day(tel, "20230419", "080000")
    _gen_day(tel, "20230418", "192100")
    out = tmp_path / "out"
    # Calibrate the later day first, then the earlier one.
    compute_daily_alignment(tel, _Z, date="20230419", out_dir=out, make_plots=False)
    compute_daily_alignment(tel, _Z, date="20230418", out_dir=out, make_plots=False)
    rows = _read_history(out / "alignment_history.csv")
    assert [r["date"] for r in rows] == ["20230418", "20230419"]  # sorted by date


# ── monitor reuse path ───────────────────────────────────────────────────────


def test_monitor_probe_uses_saved_alignment(tmp_path: Path):
    src = tmp_path / "gen"
    info = generate(src, t_x=300.0, t_y=350.0, z_p=300.0, n_tracks=5000, seed=42)
    out_align = tmp_path / "align"
    compute_daily_alignment(info["tel_dir"], _Z, out_dir=out_align, make_plots=False)
    det = load_detector(info["tel_dir"])
    json_path = out_align / f"alignment_{det.gps_paths[0].name[:8]}.json"
    assert json_path.exists()

    results = monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=150.0,
        z_tel=_Z,
        n_probe_ch=30,
        alignment_path=json_path,
        make_plots=False,
    )
    assert len(results) >= 1
    for r in results:
        assert abs(r.z_p - 300.0) < 5 * r.sigma_zp


def test_monitor_probe_alignment_z_tel_mismatch_raises(tmp_path: Path):
    src = tmp_path / "gen"
    info = generate(src, n_tracks=800, seed=1)
    out_align = tmp_path / "align"
    compute_daily_alignment(info["tel_dir"], _Z, out_dir=out_align, make_plots=False)
    det = load_detector(info["tel_dir"])
    json_path = out_align / f"alignment_{det.gps_paths[0].name[:8]}.json"
    with pytest.raises(ValueError, match="z-order-dependent"):
        monitor_probe(
            info["tel_dir"],
            info["probe_dir"],
            z_tel=np.array([0.0, -1340.0, -670.0]),
            n_probe_ch=30,
            alignment_path=json_path,
            make_plots=False,
        )
