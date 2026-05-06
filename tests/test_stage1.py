"""
Tests for stage 1 — per-detector time reconstruction.

Run against the synthetic dataset produced by monrad.synth.generate().
All events use an ideal 100 MHz clock with no drift, so every tick maps
to an exact integer nanosecond count and all events should be GOOD.
"""

import pytest
from datetime import datetime

from monrad.stage1 import (
    Quality,
    PosRef,
    _utc_to_ns,
    load_header_params,
    find_file_pairs,
    reconstruct,
)
from monrad.synth import generate, F0

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS  = 1000


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    """Generate one set of telescope + probe files for the whole module."""
    out = tmp_path_factory.mktemp('synth')
    result = generate(
        out_dir=out,
        t_x=50.0, t_y=-30.0,
        theta=0.29671, z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
    )
    return result, out


@pytest.fixture(scope='module')
def tel(synth):
    """Run stage 1 on the telescope detector."""
    result, out = synth
    tel_dir = out / 'telescope'
    header  = next(tel_dir.glob('*_header.txt'))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(tel_dir)
    events, pos_map = reconstruct(gps, pos, utc0, f0)
    return events, pos_map, utc0, f0


@pytest.fixture(scope='module')
def prb(synth):
    """Run stage 1 on the probe detector."""
    result, out = synth
    prb_dir = out / 'probe'
    header  = next(prb_dir.glob('*_header.txt'))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(prb_dir)
    events, pos_map = reconstruct(gps, pos, utc0, f0)
    return events, pos_map


# ── helpers ───────────────────────────────────────────────────────────

def _expected_ns(utc0_ns: int, i: int, f0: int) -> int:
    """
    Expected t_ns for telescope event i.

    synth.py places events at tick = (f0//20) + i*(f0//10),
    which—with the ideal PPS chain—maps to
      t_ns = utc0_ns + tick * 1_000_000_000 // f0.
    """
    dt   = f0 // 10   # ticks between events
    toff = dt  // 2   # offset from PPS edge
    tick = toff + i * dt
    return utc0_ns + tick * 1_000_000_000 // f0


# ── utility tests ─────────────────────────────────────────────────────

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
        header = next((out / 'telescope').glob('*_header.txt'))
        utc0, _ = load_header_params(header)
        assert utc0 == _START_UTC

    def test_f0_matches_synth(self, synth):
        _, out = synth
        header = next((out / 'telescope').glob('*_header.txt'))
        _, f0 = load_header_params(header)
        assert f0 == F0


class TestFindFilePairs:
    def test_returns_one_pair(self, synth):
        _, out = synth
        gps, pos = find_file_pairs(out / 'telescope')
        assert len(gps) == 1 and len(pos) == 1

    def test_gps_suffix(self, synth):
        _, out = synth
        gps, _ = find_file_pairs(out / 'telescope')
        assert gps[0].name.endswith('_GPS.bin')

    def test_pos_not_gps_suffix(self, synth):
        _, out = synth
        _, pos = find_file_pairs(out / 'telescope')
        assert not pos[0].name.endswith('_GPS.bin')
        assert pos[0].name.endswith('.bin')


# ── telescope stage-1 tests ───────────────────────────────────────────

class TestTelescopeEvents:
    def test_count(self, tel):
        events, _, _, _ = tel
        assert len(events) == _N_TRACKS

    def test_all_good(self, tel):
        events, _, _, _ = tel
        bad = [e for e in events if e.quality != Quality.GOOD]
        assert not bad, f'{len(bad)} events not GOOD'

    def test_monotonic(self, tel):
        events, _, _, _ = tel
        for i in range(1, len(events)):
            assert events[i].t_ns >= events[i - 1].t_ns, (
                f'Non-monotonic at i={i}: '
                f'{events[i].t_ns} < {events[i-1].t_ns}'
            )

    def test_timestamps_exact(self, tel):
        events, _, utc0, f0 = tel
        utc0_ns = _utc_to_ns(utc0)
        for i, ev in enumerate(events):
            exp = _expected_ns(utc0_ns, i, f0)
            assert ev.t_ns == exp, (
                f'evt {i}: got {ev.t_ns}, expected {exp}'
            )

    def test_evt_seq_contiguous(self, tel):
        events, _, _, _ = tel
        for i, ev in enumerate(events):
            assert ev.evt_seq == i

    def test_spacing_100ms(self, tel):
        events, _, _, _ = tel
        for i in range(1, len(events)):
            diff = events[i].t_ns - events[i - 1].t_ns
            assert diff == 100_000_000, (
                f'Spacing at i={i}: {diff} ns, want 100 ms'
            )


class TestTelescopePosMap:
    def test_coverage(self, tel):
        _, pos_map, _, _ = tel
        assert set(pos_map) == set(range(_N_TRACKS))

    def test_file_idx_zero(self, tel):
        _, pos_map, _, _ = tel
        assert all(ref.file_idx == 0 for ref in pos_map.values())

    def test_row_offsets(self, tel):
        _, pos_map, _, _ = tel
        for i in range(_N_TRACKS):
            assert pos_map[i].row_offset == i * 16, (
                f'evt {i}: row_offset={pos_map[i].row_offset}, '
                f'want {i * 16}'
            )

    def test_no_split_blocks(self, tel):
        _, pos_map, _, _ = tel
        assert all(ref.split_rows == 0 for ref in pos_map.values())


# ── probe stage-1 tests ───────────────────────────────────────────────

class TestProbeEvents:
    def test_count(self, synth, prb):
        result, _ = synth
        events, _ = prb
        assert len(events) == result['n_coincidences']

    def test_all_good(self, prb):
        events, _ = prb
        bad = [e for e in events if e.quality != Quality.GOOD]
        assert not bad, f'{len(bad)} probe events not GOOD'

    def test_monotonic(self, prb):
        events, _ = prb
        for i in range(1, len(events)):
            assert events[i].t_ns >= events[i - 1].t_ns

    def test_timestamps_match_telescope(self, synth, tel, prb):
        result, _   = synth
        tel_evts, _, utc0, f0 = tel
        prb_evts, _ = prb
        utc0_ns     = _utc_to_ns(utc0)
        coinc_idx   = sorted(result['probe_hits'])
        for j, prb_ev in enumerate(prb_evts):
            i   = coinc_idx[j]
            exp = _expected_ns(utc0_ns, i, f0)
            assert prb_ev.t_ns == exp, (
                f'probe evt {j} (tel idx {i}): '
                f'got {prb_ev.t_ns}, expected {exp}'
            )

    def test_evt_seq_contiguous(self, prb):
        events, _ = prb
        for j, ev in enumerate(events):
            assert ev.evt_seq == j


class TestProbePosMap:
    def test_coverage(self, synth, prb):
        result, _ = synth
        _, pos_map = prb
        n = result['n_coincidences']
        assert set(pos_map) == set(range(n))

    def test_no_split_blocks(self, prb):
        _, pos_map = prb
        assert all(ref.split_rows == 0 for ref in pos_map.values())

    def test_row_offsets(self, synth, prb):
        result, _ = synth
        _, pos_map = prb
        for i in range(result['n_coincidences']):
            assert pos_map[i].row_offset == i * 16
