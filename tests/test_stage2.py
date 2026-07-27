"""
Tests for stage 2 — sliding-window coincidence search.

Run against the synthetic dataset produced by monrad.synthetic.generate().
In the synthetic data every probe event sits at exactly the same clock
tick as its corresponding telescope event, so the 200 ns window yields
one cluster per coincidence and no spurious clusters.
"""

import pytest
from datetime import datetime

from monrad.timing import (
    _utc_to_ns,
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.coincidence import coincidence_stream
from monrad.synthetic import generate, F0

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS = 1000
TEL_ID = 0
PRB_ID = 1


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Generate synthetic telescope + probe files for the whole module."""
    out = tmp_path_factory.mktemp("synth_stage2")
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


@pytest.fixture(scope="module")
def clusters(synth):
    """
    Run the full stage-1 + stage-2 pipeline and return all clusters.
    """
    result, out = synth
    tel_dir = out / "telescope"
    prb_dir = out / "probe"

    tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))

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
    dt = f0 // 10
    toff = dt // 2
    tick = toff + i * dt
    return utc0_ns + tick * 1_000_000_000 // f0


# ── tests ─────────────────────────────────────────────────────────


class TestCoincidenceStream:
    def test_cluster_count(self, synth, clusters):
        result, _ = synth
        assert len(clusters) == result["n_coincidences"], (
            f"expected {result['n_coincidences']} clusters, got {len(clusters)}"
        )

    def test_each_cluster_spans_two_detectors(self, clusters):
        for j, cluster in enumerate(clusters):
            det_set = {det_id for det_id, _, _ in cluster}
            assert len(det_set) == 2, (
                f"cluster {j}: expected 2 distinct detectors, got {len(det_set)}"
            )

    def test_no_same_detector_twice(self, clusters):
        for j, cluster in enumerate(clusters):
            det_ids = [det_id for det_id, _, _ in cluster]
            assert len(det_ids) == len(set(det_ids)), (
                f"cluster {j}: duplicate detector_id in cluster"
            )

    def test_both_detectors_present(self, clusters):
        for j, cluster in enumerate(clusters):
            det_set = {det_id for det_id, _, _ in cluster}
            assert TEL_ID in det_set, f"cluster {j}: telescope (det {TEL_ID}) missing"
            assert PRB_ID in det_set, f"cluster {j}: probe (det {PRB_ID}) missing"

    def test_telescope_timestamps(self, synth, clusters):
        """
        The telescope t_ns in each cluster must match the ground-truth
        timestamp for the corresponding coincident track.

        Clusters are yielded in time order; track indices in
        sorted(result['probe_hits']) map 1-to-1 to clusters.
        """
        result, _ = synth
        utc0_ns = _utc_to_ns(_START_UTC)
        coinc_idx = sorted(result["probe_hits"])

        assert len(clusters) == len(coinc_idx)

        for j, cluster in enumerate(clusters):
            tel_evs = [ev for det_id, ev, _ in cluster if det_id == TEL_ID]
            assert len(tel_evs) == 1, (
                f"cluster {j}: expected 1 telescope event, got {len(tel_evs)}"
            )
            ev = tel_evs[0]
            i = coinc_idx[j]
            exp = _expected_ns(utc0_ns, i, F0)
            assert ev.t_ns == exp, (
                f"cluster {j} (tel track {i}): t_ns={ev.t_ns}, expected {exp}"
            )

    def test_cluster_entries_are_triples(self, clusters):
        """Each cluster element is a (int, TimedEvent, PosRef) triple."""
        for j, cluster in enumerate(clusters):
            for entry in cluster:
                assert len(entry) == 3, f"cluster {j}: entry is not a 3-tuple"
                det_id, ev, ref = entry
                assert isinstance(det_id, int)
                # TimedEvent and PosRef are NamedTuples — check lengths
                assert len(ev) == 3  # t_ns, evt_seq, quality
                assert len(ref) == 3  # file_idx, row_offset, split_rows


# ── unit tests on synthetic in-memory streams ─────────────────────
#
# The file-based synthetic data spaces every track ~10 ms apart, so the
# coincidence window never holds more than one (tel, prb) pair and the
# growing-window double-counting bug cannot show up there.  These tests
# feed hand-built streams directly to coincidence_stream() to exercise
# the high-rate transitive-closure behaviour required by DESIGN.md §5.1.


from monrad.timing import TimedEvent, PosRef, Quality  # noqa: E402


def _stream(events):
    """Build a stage-1-style iterator from (t_ns, evt_seq) tuples."""
    for i, (t_ns, seq) in enumerate(events):
        yield TimedEvent(t_ns, seq, Quality.GOOD), PosRef(0, i * 16)


class TestTransitiveClosure:
    """Each event must appear in exactly one emitted cluster (no re-yield)."""

    def _run(self, tel, prb, window_ns=200):
        return list(
            coincidence_stream(
                [_stream(tel), _stream(prb)],
                detector_ids=[TEL_ID, PRB_ID],
                window_ns=window_ns,
            )
        )

    def test_no_event_double_counted(self):
        # Three telescope events 50 ns apart, all within the window of a
        # single probe hit.  The buggy growing-window code yielded the probe
        # three times (paired with each telescope event in turn).
        tel = [(0, 0), (50, 1), (100, 2)]
        prb = [(40, 0)]
        clusters = self._run(tel, prb)
        # One transitive-closure cluster: 0,40,50,100 are all ≤200 ns apart.
        assert len(clusters) == 1
        # The single probe event must appear exactly once across all clusters.
        prb_count = sum(1 for cl in clusters for det_id, _, _ in cl if det_id == PRB_ID)
        assert prb_count == 1, f"probe event reported {prb_count}× (expected 1)"

    def test_clusters_are_disjoint(self):
        # Two well-separated coincidences (gap >> window between them).
        tel = [(0, 0), (60, 1), (10_000, 2), (10_050, 3)]
        prb = [(30, 0), (10_030, 1)]
        clusters = self._run(tel, prb)
        assert len(clusters) == 2
        # Every yielded event is unique by (det_id, evt_seq).
        seen = set()
        for cl in clusters:
            for det_id, ev, _ in cl:
                key = (det_id, ev.evt_seq)
                assert key not in seen, f"{key} reported twice"
                seen.add(key)

    def test_gap_just_over_window_splits(self):
        # A probe hit 201 ns after the telescope event — outside the 200 ns
        # window — must NOT form a coincidence.
        tel = [(0, 0)]
        prb = [(201, 0)]
        assert self._run(tel, prb) == []

    def test_gap_at_window_edge_joins(self):
        # Exactly window_ns apart is inside the (inclusive) window.
        tel = [(0, 0)]
        prb = [(200, 0)]
        clusters = self._run(tel, prb)
        assert len(clusters) == 1
        assert {d for d, _, _ in clusters[0]} == {TEL_ID, PRB_ID}


class TestWindowNsReachesCoincidenceStream:
    """``window_ns`` must be forwarded by the monitor drivers, not dropped.

    ``monrad.monitor.io.stream_coincidences`` used to call
    ``coincidence_stream`` with no ``window_ns`` at all, so a driver-level
    override could never reach stage 2.  These pin the wiring by capturing
    the kwarg the real call site passes.
    """

    def _capture(self, monkeypatch):
        from monrad.monitor import io as monitor_io

        seen: list = []

        def _fake(streams, detector_ids, window_ns=None):
            seen.append(window_ns)
            return iter(())

        monkeypatch.setattr(monitor_io, "coincidence_stream", _fake)
        return seen

    def _detector(self):
        """A DetectorFiles whose streams are never consumed (stream is faked)."""
        from monrad.monitor.io import DetectorFiles

        return DetectorFiles(gps_paths=[], pos_paths=[], utc0=0, f0=F0)

    def test_stream_coincidences_forwards_window_ns(self, monkeypatch):
        import numpy as np

        from monrad.alignment import AlignmentCorrection
        from monrad.monitor.io import stream_coincidences

        seen = self._capture(monkeypatch)
        det = self._detector()
        list(
            stream_coincidences(
                det,
                det,
                z_tel=np.array([0.0, 400.0, 800.0]),
                alignment=AlignmentCorrection.identity(),
                window_ns=137,
            )
        )
        assert seen == [137]

    def test_stream_coincidences_defaults_to_200(self, monkeypatch):
        import numpy as np

        from monrad.alignment import AlignmentCorrection
        from monrad.coincidence import WINDOW_NS_DEFAULT
        from monrad.monitor.io import stream_coincidences

        seen = self._capture(monkeypatch)
        det = self._detector()
        list(
            stream_coincidences(
                det,
                det,
                z_tel=np.array([0.0, 400.0, 800.0]),
                alignment=AlignmentCorrection.identity(),
            )
        )
        assert seen == [WINDOW_NS_DEFAULT] == [200]

    def test_build_cluster_stream_forwards_window_ns(self, monkeypatch):
        from monrad.monitor.io import build_cluster_stream

        seen = self._capture(monkeypatch)
        det = self._detector()
        list(build_cluster_stream(det, [det], window_ns=99))
        assert seen == [99]
