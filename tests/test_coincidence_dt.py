"""
Tests for probe-minus-telescope event-time tracking (Coincidence.dt_ns).

Pure instrumentation: stage 2 pairs a telescope event with a probe event on
time proximity alone, in either order, and nothing downstream gates on the
sign or magnitude of the difference.  These tests pin down that

  * dt_ns is recorded, with the right sign, for both orderings;
  * dt_ns is reported on the rejection paths too, not only on "accepted",
    so the timing distribution is not a geometrically selected subsample;
  * recording it did not turn into a gate — a probe that fires *after* the
    telescope is still accepted;
  * the per-window mean/sigma and the CSV columns carry it through.

See Coincidence.dt_ns for why the per-event sign is not physically
meaningful at this timing resolution.
"""

import csv
import math
from collections import Counter
from datetime import datetime, timezone

import pytest

from monrad.alignment import AlignmentCorrection
from monrad.monitor.timeseries import WindowResult, _window_dt_stats, _write_csv
from monrad.pose import (
    Coincidence,
    PoseFitter,
    dt_counter,
    dt_histogram,
    dt_stats,
)
from monrad.synthetic import Z_TEL
from monrad.timing import PosRef, Quality, TimedEvent

# A telescope triple exactly on a straight line through z=(0,400,800), so the
# chi2 cut passes and the cluster reaches "accepted" (same construction as
# TestChi2TrackOverride in test_stage5.py).
_CX_ONLINE = (10, 15, 20)
_CX_OFFSET = (10, 16, 20)  # middle plane off by one channel -> chi2 ~ 8
_CY = (5, 5, 5)
_CU_GOLDEN = 5


def _one_event_bin(tmp_path, name, words):
    """A *.bin holding exactly one event (16 identical rows)."""
    from monrad.synthetic import _write_pos_bin

    path = tmp_path / name
    _write_pos_bin(path, [words], n_cols=len(words))
    return path


def _cluster(tmp_path, name, t_tel_ns, t_prb_ns, cx=_CX_ONLINE):
    """
    A two-entry cluster whose telescope and probe events carry *different*
    timestamps, so the difference under test is non-trivial.

    Entries are listed in ascending t_ns, as coincidence_stream emits them
    (its clusters are in heap-pop order), to keep the fixture faithful to
    what the fitter really receives.
    """
    from monrad.synthetic import _ch_to_u64

    tel_words = [_ch_to_u64(cx[k], _CY[k], gen=0) for k in range(3)]
    prb_words = [_ch_to_u64(_CU_GOLDEN, _CU_GOLDEN, gen=0)]
    tel_path = _one_event_bin(tmp_path, f"{name}_tel.bin", tel_words)
    prb_path = _one_event_bin(tmp_path, f"{name}_prb.bin", prb_words)

    tel_entry = (0, TimedEvent(t_tel_ns, 0, Quality.GOOD), PosRef(0, 0, 0))
    prb_entry = (1, TimedEvent(t_prb_ns, 0, Quality.GOOD), PosRef(0, 0, 0))
    entries = sorted([tel_entry, prb_entry], key=lambda e: e[1].t_ns)
    return entries, [tel_path], [prb_path]


def _fitter(tel_paths, prb_paths, **kwargs):
    return PoseFitter(
        tel_z=Z_TEL,
        alignment=AlignmentCorrection.identity(),
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_paths,
        prb_pos_paths=prb_paths,
        **kwargs,
    )


class TestDtSign:
    """dt_ns = t_prb - t_tel, both orderings, on the accepted path."""

    @pytest.mark.parametrize(
        "t_tel, t_prb, expected",
        [
            (1000, 850, -150),  # probe first — expected sign, probe above
            (1000, 1150, +150),  # telescope first — still paired, not gated
            (1000, 1000, 0),  # same timestamp
        ],
    )
    def test_dt_recorded_with_sign(self, tmp_path, t_tel, t_prb, expected):
        cluster, tel_paths, prb_paths = _cluster(
            tmp_path, f"sign{expected}", t_tel, t_prb
        )
        fitter = _fitter(tel_paths, prb_paths)
        reports = []
        fitter.on_decode = reports.append

        co = fitter.decode_cluster(cluster)

        assert co is not None, "on-line triple + golden probe hit should accept"
        assert reports[-1].reason == "accepted"
        assert co.dt_ns == expected
        assert reports[-1].dt_ns == expected
        # The telescope time stays the coincidence's own timestamp — dt_ns is
        # an extra field, not a redefinition of t_ns.
        assert co.t_ns == t_tel

    def test_probe_after_telescope_is_not_rejected(self, tmp_path):
        """
        The physically "wrong" ordering must decode exactly like the right
        one.  A one-sided cut would show up here as an asymmetry.
        """
        early, tel_e, prb_e = _cluster(tmp_path, "early", 1000, 850)
        late, tel_l, prb_l = _cluster(tmp_path, "late", 1000, 1150)

        co_early = _fitter(tel_e, prb_e).decode_cluster(early)
        co_late = _fitter(tel_l, prb_l).decode_cluster(late)

        assert co_early is not None and co_late is not None
        assert co_early.dt_ns == -150 and co_late.dt_ns == +150
        # Identical geometry, so every fitted quantity must match; only the
        # timing fields differ.
        assert co_early._replace(dt_ns=0, t_ns=0) == co_late._replace(dt_ns=0, t_ns=0)


class TestDtOnRejectionPaths:
    """
    dt_ns must be reported for every cluster with a unique one-telescope/
    one-probe pairing — the unbiased timing population — not just accepted
    ones.
    """

    def test_reported_on_chi2_track_cut(self, tmp_path):
        cluster, tel_paths, prb_paths = _cluster(
            tmp_path, "chi2", 2000, 1900, cx=_CX_OFFSET
        )
        fitter = _fitter(tel_paths, prb_paths)
        reports = []
        fitter.on_decode = reports.append

        assert fitter.decode_cluster(cluster) is None
        assert reports[-1].reason == "chi2_track_cut"
        assert reports[-1].dt_ns == -100

    def test_none_for_ambiguous_cluster(self):
        """
        Two probe events: no unique pairing exists, so there is no difference
        to report.  Empty path lists prove the guard short-circuits before
        any position decode (reaching one would raise).
        """
        fitter = _fitter([], [])
        reports = []
        fitter.on_decode = reports.append

        cluster = [
            (0, TimedEvent(1000, 0, Quality.GOOD), PosRef(0, 0)),
            (1, TimedEvent(1050, 1, Quality.GOOD), PosRef(0, 16)),
            (1, TimedEvent(1100, 2, Quality.GOOD), PosRef(0, 32)),
        ]
        assert fitter.decode_cluster(cluster) is None
        assert reports[-1].reason == "ambiguous_cluster"
        assert reports[-1].dt_ns is None


class TestWindowDtStats:
    """Per-window mean/sigma of dt (monitor.timeseries._window_dt_stats)."""

    @staticmethod
    def _co(dt_ns):
        return Coincidence(
            a_x=0.0,
            b_x=0.0,
            a_y=0.0,
            b_y=0.0,
            cov_ab_x=(0.0, 0.0, 0.0),
            cov_ab_y=(0.0, 0.0, 0.0),
            u=0.0,
            v=0.0,
            sigma_prb_x=1.0,
            sigma_prb_y=1.0,
            dt_ns=dt_ns,
        )

    def test_empty_window_is_nan(self):
        mean, std = _window_dt_stats([])
        assert math.isnan(mean) and math.isnan(std)

    def test_single_coincidence_has_zero_spread(self):
        mean, std = _window_dt_stats([self._co(-40)])
        assert mean == pytest.approx(-40.0)
        assert std == pytest.approx(0.0)

    def test_mean_and_sample_sigma(self):
        # -30, -10, +10, +30 -> mean 0, sample sigma sqrt(2000/3)
        mean, std = _window_dt_stats([self._co(d) for d in (-30, -10, 10, 30)])
        assert mean == pytest.approx(0.0)
        assert std == pytest.approx(math.sqrt(2000.0 / 3.0))

    def test_negative_mean_survives_mixed_signs(self):
        """A real offset must not be cancelled by the sign scatter around it."""
        mean, std = _window_dt_stats([self._co(d) for d in (-150, -50, 50)])
        assert mean == pytest.approx(-50.0)
        assert std > 0.0


class TestCsvColumns:
    """dt columns reach pose_timeseries.csv."""

    def test_dt_columns_written(self, tmp_path):
        t0 = datetime(2021, 7, 23, 17, 8, tzinfo=timezone.utc)
        results = [
            WindowResult(
                utc_start=t0,
                utc_end=t0,
                n_inliers=42,
                t_x=1.0,
                sigma_tx=0.1,
                t_y=2.0,
                sigma_ty=0.1,
                z_p=300.0,
                sigma_zp=5.0,
                theta=0.3,
                sigma_theta=0.01,
                resid_rms=3.0,
                dt_mean_ns=-37.5,
                dt_std_ns=48.25,
            )
        ]
        path = tmp_path / "pose_timeseries.csv"
        _write_csv(results, path)

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert float(rows[0]["dt_mean_ns"]) == pytest.approx(-37.5)
        assert float(rows[0]["dt_std_ns"]) == pytest.approx(48.25)
        # Appended, so the pre-existing columns kept their meaning.
        assert float(rows[0]["resid_rms"]) == pytest.approx(3.0)


class TestDtStats:
    """monrad.pose.dt_stats — the shared summary used by both reports."""

    def test_empty(self):
        st = dt_stats(Counter())
        assert st.n == 0
        assert math.isnan(st.mean) and math.isnan(st.std) and math.isnan(st.median)

    def test_single_entry(self):
        st = dt_stats(Counter({-40: 1}))
        assert st.n == 1
        assert st.mean == pytest.approx(-40.0)
        assert st.std == pytest.approx(0.0)
        assert st.median == pytest.approx(-40.0)
        assert (st.min_ns, st.max_ns) == (-40, -40)

    def test_weighted_mean_and_sample_sigma(self):
        # Counter weights must behave exactly like repeated samples.
        hist = Counter({-30: 1, -10: 1, 10: 1, 30: 1})
        st = dt_stats(hist)
        assert st.n == 4
        assert st.mean == pytest.approx(0.0)
        assert st.std == pytest.approx(math.sqrt(2000.0 / 3.0))
        assert (st.min_ns, st.max_ns) == (-30, 30)

    def test_counter_weights_equal_repeated_samples(self):
        flat = [-20, -20, -20, 10, 60]
        st = dt_stats(Counter(flat))
        mean = sum(flat) / len(flat)
        var = sum((d - mean) ** 2 for d in flat) / (len(flat) - 1)
        assert st.n == len(flat)
        assert st.mean == pytest.approx(mean)
        assert st.std == pytest.approx(math.sqrt(var))

    @pytest.mark.parametrize(
        "flat, expected",
        [
            ([10], 10.0),
            ([10, 30], 20.0),  # even n -> mean of the two central values
            ([10, 20, 30], 20.0),
            ([-50, -10, -10, 40], -10.0),
            ([0, 0, 0, 100], 0.0),  # central pair both inside one key
        ],
    )
    def test_median_matches_flat_sample(self, flat, expected):
        assert dt_stats(Counter(flat)).median == pytest.approx(expected)

    def test_dt_counter_over_coincidences(self):
        cos = [TestWindowDtStats._co(d) for d in (-10, -10, 20)]
        assert dt_counter(cos) == Counter({-10: 2, 20: 1})


class TestDtHistogram:
    """monrad.pose.dt_histogram — text rows for the whole-run report."""

    def test_empty_gives_no_rows(self):
        assert dt_histogram(Counter()) == []

    def test_rejects_nonpositive_bin(self):
        with pytest.raises(ValueError, match="bin_ns"):
            dt_histogram(Counter({0: 1}), bin_ns=0)

    def test_bins_are_floor_aligned_through_zero(self):
        """
        A bin edge must fall on 0, so negative Δt bins down (not toward
        zero): -5 belongs to [-20, 0), not [0, +20).
        """
        rows = dt_histogram(Counter({-5: 1}), bin_ns=20)
        assert len(rows) == 1
        assert rows[0].startswith("[   -20,    +0)")

    def test_empty_interior_bins_are_kept(self):
        # -50 and +50 with 20 ns bins -> edges -60,-40,-20,0,+20,+40 = 6 rows,
        # four of them empty, so a gap stays visible instead of collapsing.
        rows = dt_histogram(Counter({-50: 3, 50: 3}), bin_ns=20)
        assert len(rows) == 6
        counts = [int(r.split(")")[1].split()[0]) for r in rows]
        assert counts == [3, 0, 0, 0, 0, 3]

    def test_bar_scales_to_the_peak(self):
        rows = dt_histogram(Counter({0: 10, 20: 5}), bin_ns=20, width=10)
        assert rows[0].endswith("#" * 10)
        assert rows[1].endswith("#" * 5)

    def test_counts_are_conserved(self):
        hist = Counter({-37: 4, -11: 9, 3: 2, 44: 1})
        rows = dt_histogram(hist, bin_ns=20)
        total = sum(int(r.split(")")[1].split()[0]) for r in rows)
        assert total == sum(hist.values())
