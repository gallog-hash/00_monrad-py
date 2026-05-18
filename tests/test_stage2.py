"""
Tests for stage 2 — sliding-window coincidence search.

Run against the synthetic dataset produced by monrad.synth.generate().
In the synthetic data every probe event sits at exactly the same clock
tick as its corresponding telescope event, so the 200 ns window yields
one cluster per coincidence and no spurious clusters.
"""

import pytest
from datetime import datetime

from monrad.stage1 import (
    _utc_to_ns,
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.stage2 import coincidence_stream
from monrad.synth import generate, F0

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS  = 1000
TEL_ID     = 0
PRB_ID     = 1


# ── fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    """Generate synthetic telescope + probe files for the whole module."""
    out = tmp_path_factory.mktemp('synth_stage2')
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
def clusters(synth):
    """
    Run the full stage-1 + stage-2 pipeline and return all clusters.
    """
    result, out = synth
    tel_dir = out / 'telescope'
    prb_dir = out / 'probe'

    tel_utc0, tel_f0 = load_header_params(
        next(tel_dir.glob('*_header.txt'))
    )
    prb_utc0, prb_f0 = load_header_params(
        next(prb_dir.glob('*_header.txt'))
    )

    tel_gps, tel_pos = find_file_pairs(tel_dir)
    prb_gps, prb_pos = find_file_pairs(prb_dir)

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    return list(
        coincidence_stream(
            [tel_stream, prb_stream],
            detector_ids=[TEL_ID, PRB_ID],
        )
    )


# ── helpers ───────────────────────────────────────────────────────

def _expected_ns(utc0_ns: int, i: int, f0: int) -> int:
    """Expected t_ns for telescope event i (same formula as test_stage1)."""
    dt   = f0 // 10
    toff = dt  // 2
    tick = toff + i * dt
    return utc0_ns + tick * 1_000_000_000 // f0


# ── tests ─────────────────────────────────────────────────────────

class TestCoincidenceStream:

    def test_cluster_count(self, synth, clusters):
        result, _ = synth
        assert len(clusters) == result['n_coincidences'], (
            f"expected {result['n_coincidences']} clusters, "
            f"got {len(clusters)}"
        )

    def test_each_cluster_spans_two_detectors(self, clusters):
        for j, cluster in enumerate(clusters):
            det_set = {det_id for det_id, _, _ in cluster}
            assert len(det_set) == 2, (
                f'cluster {j}: expected 2 distinct detectors, '
                f'got {len(det_set)}'
            )

    def test_no_same_detector_twice(self, clusters):
        for j, cluster in enumerate(clusters):
            det_ids = [det_id for det_id, _, _ in cluster]
            assert len(det_ids) == len(set(det_ids)), (
                f'cluster {j}: duplicate detector_id in cluster'
            )

    def test_both_detectors_present(self, clusters):
        for j, cluster in enumerate(clusters):
            det_set = {det_id for det_id, _, _ in cluster}
            assert TEL_ID in det_set, (
                f'cluster {j}: telescope (det {TEL_ID}) missing'
            )
            assert PRB_ID in det_set, (
                f'cluster {j}: probe (det {PRB_ID}) missing'
            )

    def test_telescope_timestamps(self, synth, clusters):
        """
        The telescope t_ns in each cluster must match the ground-truth
        timestamp for the corresponding coincident track.

        Clusters are yielded in time order; track indices in
        sorted(result['probe_hits']) map 1-to-1 to clusters.
        """
        result, _ = synth
        utc0_ns   = _utc_to_ns(_START_UTC)
        coinc_idx = sorted(result['probe_hits'])

        assert len(clusters) == len(coinc_idx)

        for j, cluster in enumerate(clusters):
            tel_evs = [
                ev for det_id, ev, _ in cluster if det_id == TEL_ID
            ]
            assert len(tel_evs) == 1, (
                f'cluster {j}: expected 1 telescope event, '
                f'got {len(tel_evs)}'
            )
            ev  = tel_evs[0]
            i   = coinc_idx[j]
            exp = _expected_ns(utc0_ns, i, F0)
            assert ev.t_ns == exp, (
                f'cluster {j} (tel track {i}): '
                f't_ns={ev.t_ns}, expected {exp}'
            )

    def test_cluster_entries_are_triples(self, clusters):
        """Each cluster element is a (int, TimedEvent, PosRef) triple."""
        for j, cluster in enumerate(clusters):
            for entry in cluster:
                assert len(entry) == 3, (
                    f'cluster {j}: entry is not a 3-tuple'
                )
                det_id, ev, ref = entry
                assert isinstance(det_id, int)
                # TimedEvent and PosRef are NamedTuples — check lengths
                assert len(ev)  == 3   # t_ns, evt_seq, quality
                assert len(ref) == 3   # file_idx, row_offset, split_rows
