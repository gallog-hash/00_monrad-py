"""
Tests for stage 1 — per-detector time reconstruction.

Run against the synthetic dataset produced by monrad.synthetic.generate().
All events use an ideal 100 MHz clock with no drift, so every tick maps
to an exact integer nanosecond count and all events should be GOOD.
"""

import struct
import pytest
from datetime import datetime

from monrad.timing import (
    Quality,
    _utc_to_ns,
    load_header_params,
    find_file_pairs,
    reconstruct,
    reconstruct_stream,
)
from monrad.synthetic import generate, F0

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS = 1000


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Generate one set of telescope + probe files for the whole module."""
    out = tmp_path_factory.mktemp("synth")
    result = generate(
        out_dir=out,
        t_x=50.0,
        t_y=-30.0,
        theta=0.29671,
        z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
    )
    return result, out


# ── batch fixtures (deprecated reconstruct()) ─────────────────────


@pytest.fixture(scope="module")
def tel_batch(synth):
    """Run stage 1 (batch) on the telescope detector."""
    result, out = synth
    tel_dir = out / "telescope"
    header = next(tel_dir.glob("*_header.txt"))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(tel_dir)
    with pytest.warns(DeprecationWarning):
        events, pos_map = reconstruct(gps, pos, utc0, f0)
    return events, pos_map, utc0, f0


@pytest.fixture(scope="module")
def prb_batch(synth):
    """Run stage 1 (batch) on the probe detector."""
    result, out = synth
    prb_dir = out / "probe"
    header = next(prb_dir.glob("*_header.txt"))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(prb_dir)
    with pytest.warns(DeprecationWarning):
        events, pos_map = reconstruct(gps, pos, utc0, f0)
    return events, pos_map


# ── streaming fixtures ────────────────────────────────────────────


@pytest.fixture(scope="module")
def tel(synth):
    """Run stage 1 (streaming) on the telescope detector."""
    result, out = synth
    tel_dir = out / "telescope"
    header = next(tel_dir.glob("*_header.txt"))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(tel_dir)
    events_and_refs = list(reconstruct_stream(gps, pos, utc0, f0))
    events = [ev for ev, _ in events_and_refs]
    pos_index = [ref for _, ref in events_and_refs]
    return events, pos_index, utc0, f0


@pytest.fixture(scope="module")
def prb(synth):
    """Run stage 1 (streaming) on the probe detector."""
    result, out = synth
    prb_dir = out / "probe"
    header = next(prb_dir.glob("*_header.txt"))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(prb_dir)
    events_and_refs = list(reconstruct_stream(gps, pos, utc0, f0))
    events = [ev for ev, _ in events_and_refs]
    pos_index = [ref for _, ref in events_and_refs]
    return events, pos_index


# ── helpers ───────────────────────────────────────────────────────


def _expected_ns(utc0_ns: int, i: int, f0: int) -> int:
    """
    Expected t_ns for telescope event i.

    synth.py places events at tick = (f0//20) + i*(f0//10),
    which—with the ideal PPS chain—maps to
      t_ns = utc0_ns + tick * 1_000_000_000 // f0.
    """
    dt = f0 // 10  # ticks between events
    toff = dt // 2  # offset from PPS edge
    tick = toff + i * dt
    return utc0_ns + tick * 1_000_000_000 // f0


# ── utility tests ─────────────────────────────────────────────────


class TestUtcToNs:
    def test_epoch_is_zero(self):
        assert _utc_to_ns(datetime(1970, 1, 1)) == 0

    def test_one_second(self):
        assert _utc_to_ns(datetime(1970, 1, 1, 0, 0, 1)) == 1_000_000_000

    def test_microsecond_precision(self):
        t = datetime(1970, 1, 1, 0, 0, 0, 500_000)
        assert _utc_to_ns(t) == 500_000_000


class TestLoadHeaderParams:
    def test_utc0_matches_start(self, synth):
        _, out = synth
        header = next((out / "telescope").glob("*_header.txt"))
        utc0, _ = load_header_params(header)
        assert utc0 == _START_UTC

    def test_f0_matches_synth(self, synth):
        _, out = synth
        header = next((out / "telescope").glob("*_header.txt"))
        _, f0 = load_header_params(header)
        assert f0 == F0


class TestFindFilePairs:
    def test_returns_one_pair(self, synth):
        _, out = synth
        gps, pos = find_file_pairs(out / "telescope")
        assert len(gps) == 1 and len(pos) == 1

    def test_gps_suffix(self, synth):
        _, out = synth
        gps, _ = find_file_pairs(out / "telescope")
        assert gps[0].name.endswith("_GPS.bin")

    def test_pos_not_gps_suffix(self, synth):
        _, out = synth
        _, pos = find_file_pairs(out / "telescope")
        assert not pos[0].name.endswith("_GPS.bin")
        assert pos[0].name.endswith(".bin")


# ── batch-API telescope tests ─────────────────────────────────────


class TestTelescopeEvents:
    def test_count(self, tel_batch):
        events, _, _, _ = tel_batch
        assert len(events) == _N_TRACKS

    def test_all_good(self, tel_batch):
        events, _, _, _ = tel_batch
        bad = [e for e in events if e.quality != Quality.GOOD]
        assert not bad, f"{len(bad)} events not GOOD"

    def test_monotonic(self, tel_batch):
        events, _, _, _ = tel_batch
        for i in range(1, len(events)):
            assert events[i].t_ns >= events[i - 1].t_ns, (
                f"Non-monotonic at i={i}: {events[i].t_ns} < {events[i - 1].t_ns}"
            )

    def test_timestamps_exact(self, tel_batch):
        events, _, utc0, f0 = tel_batch
        utc0_ns = _utc_to_ns(utc0)
        for i, ev in enumerate(events):
            exp = _expected_ns(utc0_ns, i, f0)
            assert ev.t_ns == exp, f"evt {i}: got {ev.t_ns}, expected {exp}"

    def test_evt_seq_contiguous(self, tel_batch):
        events, _, _, _ = tel_batch
        for i, ev in enumerate(events):
            assert ev.evt_seq == i

    def test_spacing_100ms(self, tel_batch):
        events, _, _, _ = tel_batch
        for i in range(1, len(events)):
            diff = events[i].t_ns - events[i - 1].t_ns
            assert diff == 100_000_000, f"Spacing at i={i}: {diff} ns, want 100 ms"


class TestTelescopePosMap:
    def test_coverage(self, tel_batch):
        _, pos_map, _, _ = tel_batch
        assert set(pos_map) == set(range(_N_TRACKS))

    def test_file_idx_zero(self, tel_batch):
        _, pos_map, _, _ = tel_batch
        assert all(ref.file_idx == 0 for ref in pos_map.values())

    def test_row_offsets(self, tel_batch):
        _, pos_map, _, _ = tel_batch
        for i in range(_N_TRACKS):
            assert pos_map[i].row_offset == i * 16, (
                f"evt {i}: row_offset={pos_map[i].row_offset}, want {i * 16}"
            )

    def test_no_split_blocks(self, tel_batch):
        _, pos_map, _, _ = tel_batch
        assert all(ref.split_rows == 0 for ref in pos_map.values())


# ── batch-API probe tests ─────────────────────────────────────────


class TestProbeEvents:
    def test_count(self, synth, prb_batch):
        result, _ = synth
        events, _ = prb_batch
        assert len(events) == result["n_coincidences"]

    def test_all_good(self, prb_batch):
        events, _ = prb_batch
        bad = [e for e in events if e.quality != Quality.GOOD]
        assert not bad, f"{len(bad)} probe events not GOOD"

    def test_monotonic(self, prb_batch):
        events, _ = prb_batch
        for i in range(1, len(events)):
            assert events[i].t_ns >= events[i - 1].t_ns

    def test_timestamps_match_telescope(self, synth, tel_batch, prb_batch):
        result, _ = synth
        tel_evts, _, utc0, f0 = tel_batch
        prb_evts, _ = prb_batch
        utc0_ns = _utc_to_ns(utc0)
        coinc_idx = sorted(result["probe_hits"])
        for j, prb_ev in enumerate(prb_evts):
            i = coinc_idx[j]
            exp = _expected_ns(utc0_ns, i, f0)
            assert prb_ev.t_ns == exp, (
                f"probe evt {j} (tel idx {i}): got {prb_ev.t_ns}, expected {exp}"
            )

    def test_evt_seq_contiguous(self, prb_batch):
        events, _ = prb_batch
        for j, ev in enumerate(events):
            assert ev.evt_seq == j


class TestProbePosMap:
    def test_coverage(self, synth, prb_batch):
        result, _ = synth
        _, pos_map = prb_batch
        n = result["n_coincidences"]
        assert set(pos_map) == set(range(n))

    def test_no_split_blocks(self, prb_batch):
        _, pos_map = prb_batch
        assert all(ref.split_rows == 0 for ref in pos_map.values())

    def test_row_offsets(self, synth, prb_batch):
        result, _ = synth
        _, pos_map = prb_batch
        for i in range(result["n_coincidences"]):
            assert pos_map[i].row_offset == i * 16


# ── streaming-API tests ───────────────────────────────────────────


class TestReconstructStream:
    """
    Same assertions as the batch-API test classes, but exercising
    reconstruct_stream() via the streaming fixtures.

    The streaming fixture returns (events, pos_index) where pos_index
    is a plain list: pos_index[evt_seq] == PosRef.
    """

    # telescope event assertions

    def test_tel_count(self, tel):
        events, _, _, _ = tel
        assert len(events) == _N_TRACKS

    def test_tel_all_good(self, tel):
        events, _, _, _ = tel
        bad = [e for e in events if e.quality != Quality.GOOD]
        assert not bad, f"{len(bad)} events not GOOD"

    def test_tel_monotonic(self, tel):
        events, _, _, _ = tel
        for i in range(1, len(events)):
            assert events[i].t_ns >= events[i - 1].t_ns

    def test_tel_timestamps_exact(self, tel):
        events, _, utc0, f0 = tel
        utc0_ns = _utc_to_ns(utc0)
        for i, ev in enumerate(events):
            exp = _expected_ns(utc0_ns, i, f0)
            assert ev.t_ns == exp, f"evt {i}: got {ev.t_ns}, expected {exp}"

    def test_tel_evt_seq_contiguous(self, tel):
        events, _, _, _ = tel
        for i, ev in enumerate(events):
            assert ev.evt_seq == i

    def test_tel_spacing_100ms(self, tel):
        events, _, _, _ = tel
        for i in range(1, len(events)):
            diff = events[i].t_ns - events[i - 1].t_ns
            assert diff == 100_000_000

    # telescope pos_index assertions

    def test_tel_pos_file_idx_zero(self, tel):
        _, pos_index, _, _ = tel
        assert all(ref.file_idx == 0 for ref in pos_index)

    def test_tel_pos_row_offsets(self, tel):
        _, pos_index, _, _ = tel
        for i, ref in enumerate(pos_index):
            assert ref.row_offset == i * 16

    def test_tel_pos_no_split_blocks(self, tel):
        _, pos_index, _, _ = tel
        assert all(ref.split_rows == 0 for ref in pos_index)

    # probe event assertions

    def test_prb_count(self, synth, prb):
        result, _ = synth
        events, _ = prb
        assert len(events) == result["n_coincidences"]

    def test_prb_all_good(self, prb):
        events, _ = prb
        bad = [e for e in events if e.quality != Quality.GOOD]
        assert not bad, f"{len(bad)} probe events not GOOD"

    def test_prb_monotonic(self, prb):
        events, _ = prb
        for i in range(1, len(events)):
            assert events[i].t_ns >= events[i - 1].t_ns

    def test_prb_timestamps_match_telescope(self, synth, tel, prb):
        result, _ = synth
        _, _, utc0, f0 = tel
        prb_evts, _ = prb
        utc0_ns = _utc_to_ns(utc0)
        coinc_idx = sorted(result["probe_hits"])
        for j, prb_ev in enumerate(prb_evts):
            i = coinc_idx[j]
            exp = _expected_ns(utc0_ns, i, f0)
            assert prb_ev.t_ns == exp

    def test_prb_evt_seq_contiguous(self, prb):
        events, _ = prb
        for j, ev in enumerate(events):
            assert ev.evt_seq == j

    def test_prb_pos_no_split_blocks(self, prb):
        _, pos_index = prb
        assert all(ref.split_rows == 0 for ref in pos_index)

    def test_prb_pos_row_offsets(self, synth, prb):
        result, _ = synth
        _, pos_index = prb
        for i, ref in enumerate(pos_index):
            assert ref.row_offset == i * 16


# ── adversarial tests ─────────────────────────────────────────────

# Binary helpers shared by adversarial tests


def _gps_rec(tick: int, gen: int, is_pps: bool) -> int:
    return tick | (gen << 52) | ((1 << 63) if is_pps else 0)


def _write_gps(path, records: list[int]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for r in records:
            f.write(struct.pack("<Q", r))


def _write_pos(path, n_events: int, n_cols: int = 1) -> None:
    """Write a minimal pos file: n_events golden-hit blocks."""
    n_rows = n_events * 16
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n_rows))
        f.write(struct.pack("<I", n_cols))
        for evt in range(n_events):
            gen = evt % 2048
            # minimal golden hit: one fiber + one ribbon bit per axis
            word = (
                (1 << 0)  # Y ribbon bit 0
                | (1 << 10)  # Y fiber bit 0
                | (1 << 32)  # X ribbon bit 0
                | (1 << 42)  # X fiber bit 0
                | (gen << 52)
            )
            for _ in range(16):
                for _ in range(n_cols):
                    f.write(struct.pack("<Q", word))


def test_pre_pps1_events(tmp_path):
    """
    Events before PPS_1 are back-extrapolated using the PPS_1→PPS_2
    interval and yielded with Quality.DEGRADED once PPS_2 is seen.
    Events in PPS_1→PPS_2 are GOOD.
    """
    f0 = 100_000_000
    utc0 = datetime(2023, 1, 1, 0, 0, 0)
    utc0_ns = _utc_to_ns(utc0)

    # utc0 / PPS_1 at tick 0, PPS_2 at tick f0, PPS_3 at tick 2*f0
    pps1_tick = 0
    pps2_tick = f0
    pps3_tick = 2 * f0

    # 10 events before PPS_1 at ticks 5e5, 10e5, ..., 50e5
    n_pre = 10
    pre_ticks = [(i + 1) * 500_000 for i in range(n_pre)]  # 5ms apart

    # 2 events between PPS_1 and PPS_2
    n_mid = 2
    mid_ticks = [f0 // 4, f0 // 2]  # 250ms, 500ms after PPS_1

    total_events = n_pre + n_mid

    # Build GPS stream: pre events, PPS_1, mid events, PPS_2, PPS_3
    records: list[int] = []
    gen = 0
    for tick in pre_ticks:
        records.append(_gps_rec(tick, gen, False))
        gen = (gen + 1) % 2048
    records.append(_gps_rec(pps1_tick, gen, True))
    for tick in mid_ticks:
        records.append(_gps_rec(tick, gen, False))
        gen = (gen + 1) % 2048
    records.append(_gps_rec(pps2_tick, gen, True))
    records.append(_gps_rec(pps3_tick, gen, True))

    gps_path = tmp_path / "test_GPS.bin"
    pos_path = tmp_path / "test.bin"
    _write_gps(gps_path, records)
    _write_pos(pos_path, total_events)

    results = list(reconstruct_stream([gps_path], [pos_path], utc0, f0))
    assert len(results) == total_events

    # pre-PPS_1 events: first n_pre, all DEGRADED
    # The back_iv has c0=pps1_tick=0, n0=0, dc=f0, dn=1
    # t_ns = utc0_ns + 0 + (tick - 0) * 1e9 * 1 // f0
    for i, (ev, ref) in enumerate(results[:n_pre]):
        assert ev.quality == Quality.DEGRADED, (
            f"pre-PPS_1 event {i} should be DEGRADED, got {ev.quality}"
        )
        expected = utc0_ns + pre_ticks[i] * 1_000_000_000 // f0
        assert ev.t_ns == expected, (
            f"pre-PPS_1 event {i}: got {ev.t_ns}, expected {expected}"
        )
        assert ev.evt_seq == i

    # PPS_1→PPS_2 events: next n_mid, all GOOD (trusted interval)
    for j, (ev, ref) in enumerate(results[n_pre:]):
        assert ev.quality == Quality.GOOD, (
            f"mid event {j} should be GOOD, got {ev.quality}"
        )
        expected = utc0_ns + mid_ticks[j] * 1_000_000_000 // f0
        assert ev.t_ns == expected
        assert ev.evt_seq == n_pre + j


def test_stream_exhaustion_mid_buffer(tmp_path):
    """
    When the last GPS file ends after the last event record but before
    a closing PPS, the buffered events are forward-extrapolated with
    Quality.DEGRADED.
    """
    f0 = 100_000_000
    utc0 = datetime(2023, 1, 1, 0, 0, 0)

    # PPS_1 at 0, PPS_2 at f0
    # 3 events in PPS_1→PPS_2, then 3 events after PPS_2 with no PPS_3
    pps1_tick = 0
    pps2_tick = f0

    mid_ticks = [f0 // 4, f0 // 2, f0 * 3 // 4]  # 3 GOOD events
    tail_ticks = [f0 + f0 // 4, f0 + f0 // 2, f0 + f0 * 3 // 4]  # 3 DEGRADED

    records: list[int] = []
    gen = 0
    records.append(_gps_rec(pps1_tick, gen, True))
    for tick in mid_ticks:
        records.append(_gps_rec(tick, gen, False))
        gen = (gen + 1) % 2048
    records.append(_gps_rec(pps2_tick, gen, True))
    for tick in tail_ticks:
        records.append(_gps_rec(tick, gen, False))
        gen = (gen + 1) % 2048
    # No PPS_3 — stream ends here

    n_events = len(mid_ticks) + len(tail_ticks)
    gps_path = tmp_path / "test_GPS.bin"
    pos_path = tmp_path / "test.bin"
    _write_gps(gps_path, records)
    _write_pos(pos_path, n_events)

    results = list(reconstruct_stream([gps_path], [pos_path], utc0, f0))
    assert len(results) == n_events

    for ev, _ in results[: len(mid_ticks)]:
        assert ev.quality == Quality.GOOD

    for ev, _ in results[len(mid_ticks) :]:
        assert ev.quality == Quality.DEGRADED


def test_back_to_back_untrusted(tmp_path):
    """
    Events inside three consecutive untrusted PPS intervals receive
    Quality.UNTRUSTED.  Events in trusted intervals before and after
    the bad stretch are unaffected (Quality.GOOD).
    """
    f0 = 100_000_000
    utc0 = datetime(2023, 1, 1, 0, 0, 0)
    # Introduce a 1% clock error to make intervals untrusted
    # (PPS_TAU default = 1e-4, so 1e-2 >> 1e-4)
    bad_dc = f0 + f0 // 100  # one bad second

    # Layout:
    #   PPS_1 at 0, PPS_2 at f0     → good startup interval
    #   1 good event at f0//2
    #   PPS_3 at 2*f0+bad_dc        → untrusted (interval spans bad_dc)
    #   1 bad  event inside that interval
    #   PPS_4 at 3*f0+2*bad_dc      → untrusted
    #   1 bad  event
    #   PPS_5 at 4*f0+3*bad_dc      → untrusted
    #   1 bad  event
    #   PPS_6 at 5*f0+3*bad_dc      → good (1 clean second after PPS_5)
    #   1 good event

    p1 = 0
    p2 = f0
    p3 = p2 + bad_dc
    p4 = p3 + bad_dc
    p5 = p4 + bad_dc
    p6 = p5 + f0

    good1_tick = f0 // 2  # between PPS_1 and PPS_2
    bad1_tick = (p2 + p3) // 2  # between PPS_2 and PPS_3
    bad2_tick = (p3 + p4) // 2
    bad3_tick = (p4 + p5) // 2
    good2_tick = p5 + f0 // 2  # between PPS_5 and PPS_6

    records: list[int] = []
    gen = 0

    records.append(_gps_rec(p1, gen, True))
    records.append(_gps_rec(good1_tick, gen, False))
    gen = (gen + 1) % 2048
    records.append(_gps_rec(p2, gen, True))
    records.append(_gps_rec(bad1_tick, gen, False))
    gen = (gen + 1) % 2048
    records.append(_gps_rec(p3, gen, True))
    records.append(_gps_rec(bad2_tick, gen, False))
    gen = (gen + 1) % 2048
    records.append(_gps_rec(p4, gen, True))
    records.append(_gps_rec(bad3_tick, gen, False))
    gen = (gen + 1) % 2048
    records.append(_gps_rec(p5, gen, True))
    records.append(_gps_rec(good2_tick, gen, False))
    gen = (gen + 1) % 2048
    records.append(_gps_rec(p6, gen, True))

    n_events = 5
    gps_path = tmp_path / "test_GPS.bin"
    pos_path = tmp_path / "test.bin"
    _write_gps(gps_path, records)
    _write_pos(pos_path, n_events)

    results = list(reconstruct_stream([gps_path], [pos_path], utc0, f0))
    assert len(results) == n_events

    ev0, _ = results[0]  # good1 — in trusted PPS_1→PPS_2
    ev1, _ = results[1]  # bad1  — in untrusted PPS_2→PPS_3
    ev2, _ = results[2]  # bad2  — in untrusted PPS_3→PPS_4
    ev3, _ = results[3]  # bad3  — in untrusted PPS_4→PPS_5
    ev4, _ = results[4]  # good2 — in trusted PPS_5→PPS_6

    assert ev0.quality == Quality.GOOD, f"expected GOOD,      got {ev0.quality}"
    assert ev1.quality == Quality.UNTRUSTED, f"expected UNTRUSTED, got {ev1.quality}"
    assert ev2.quality == Quality.UNTRUSTED, f"expected UNTRUSTED, got {ev2.quality}"
    assert ev3.quality == Quality.UNTRUSTED, f"expected UNTRUSTED, got {ev3.quality}"
    assert ev4.quality == Quality.GOOD, f"expected GOOD,      got {ev4.quality}"
