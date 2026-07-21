"""Correctness of the two-tier cut scan (``scripts/scan_geometric_cuts.py``).

The linchpin is :class:`TestReplayEquivalence`: the whole two-tier design is
only sound if a *replay* from a single loose decode pass reproduces a **live**
``PoseFitter`` run at the tighter settings exactly -- same funnel counts, same
accepted ``Coincidence`` list, same ``n_inliers``.  If that ever breaks, every
number the scan produces is measuring something other than the pipeline.

All tests run on synthetic data (``monrad.synthetic.generate``); no real
detector files are required.
"""

import csv
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scan_geometric_cuts as scan  # noqa: E402

from monrad.alignment import AlignmentCorrection  # noqa: E402
from monrad.coincidence import coincidence_stream  # noqa: E402
from monrad.monitor.io import DetectorFiles, load_detector  # noqa: E402
from monrad.pose import GATE_ORDER, PoseFitter, fit_probe_pose  # noqa: E402
from monrad.synthetic.generate import generate  # noqa: E402
from monrad.timing import reconstruct_stream  # noqa: E402

Z_TEL = np.array([0.0, 400.0, 800.0])
N_PROBE_CH = 30


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A folded synthetic acquisition + its Tier-A cache.

    Folding a subset of the telescope planes is what makes the scan
    interesting: it produces mirror-fold-ambiguous planes, so clusters spread
    across the anchor-count and chi2 range instead of all landing at chi2~0
    with 3 anchors.
    """
    out = tmp_path_factory.mktemp("scan")
    generate(
        out,
        n_tracks=8000,
        n_probe_ch=N_PROBE_CH,
        seed=7,
        fold=True,
        fold_planes={1},
        fold_crosstalk_rate=0.15,
        tel_cluster_widths={0: (2, 1), 2: (1, 2)},
    )
    tel = load_detector(out / "telescope")
    prb = load_detector(out / "probe")
    alignment = AlignmentCorrection.identity()
    cache = scan.decode_pass(
        tel,
        prb,
        z_tel=Z_TEL,
        alignment=alignment,
        min_anchor_planes=0,
        log_every=0,
    )
    return tel, prb, alignment, cache


def _live_run(tel, prb, alignment, *, chi2_track, min_anchor_planes):
    """Funnel counts + accepted coincidences from a real ``PoseFitter`` run."""
    counts: Counter = Counter()
    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel.pos_paths,
        prb_pos_paths=prb.pos_paths,
        min_anchor_planes=min_anchor_planes,
        chi2_track=chi2_track,
        on_decode=lambda r: counts.update([r.reason]),
    )
    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)
    coincs = []
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        co = fitter.decode_cluster(cluster)
        if co is not None:
            coincs.append(co)
    return counts, coincs


class TestReplayEquivalence:
    """Offline replay == live PoseFitter, gate for gate and event for event."""

    @pytest.mark.parametrize(
        "chi2_track,min_anchor_planes",
        [
            (4.0, 1),  # the shipped default
            (4.0, 0),  # anchor gate disabled
            (4.0, 2),
            (4.0, 3),
            (1.0, 1),  # much tighter chi2
            (37.0, 1),  # the MATLAB-equivalent looseness
            (1000.0, 1),  # effectively no chi2 cut
        ],
    )
    def test_matches_live_fitter(self, dataset, chi2_track, min_anchor_planes):
        tel, prb, alignment, cache = dataset
        live_counts, live_coincs = _live_run(
            tel,
            prb,
            alignment,
            chi2_track=chi2_track,
            min_anchor_planes=min_anchor_planes,
        )
        rep = scan.replay(
            cache, chi2_track=chi2_track, min_anchor_planes=min_anchor_planes
        )

        for gate in (*GATE_ORDER, "accepted"):
            assert rep.counts[gate] == live_counts[gate], (
                f"gate {gate!r} diverged at chi2={chi2_track}, "
                f"anchor={min_anchor_planes}"
            )
        assert rep.coincs == live_coincs

    def test_funnel_is_exhaustive(self, dataset):
        """Every cached cluster lands in exactly one funnel slot."""
        _tel, _prb, _alignment, cache = dataset
        rep = scan.replay(cache, chi2_track=4.0, min_anchor_planes=1)
        assert rep.n_raw == len(cache)
        assert set(rep.counts) <= {*GATE_ORDER, "accepted"}

    def test_inliers_match_live_pose_fit(self, dataset):
        """The full stage-5 fit agrees, not just the decode funnel."""
        tel, prb, alignment, cache = dataset
        _counts, live_coincs = _live_run(
            tel, prb, alignment, chi2_track=4.0, min_anchor_planes=1
        )
        z_corr = alignment.corrected_z_tel(Z_TEL)
        live_pose = fit_probe_pose(live_coincs, z_corr, alignment)
        row = scan.evaluate(
            cache,
            chi2_track=4.0,
            min_anchor_planes=1,
            min_fit=3,  # compare against a bare fit_probe_pose, which has no floor
            alignment=alignment,
        )
        assert row["n_accepted"] == len(live_coincs)
        assert row["n_inliers"] == live_pose.n_inliers
        assert row["z_p"] == pytest.approx(live_pose.z_p)


class TestReplayGuards:
    """The replay refuses configurations the cache cannot honestly reproduce."""

    def test_rejects_looser_anchor_than_decoded(self, dataset):
        _tel, _prb, _alignment, cache = dataset
        # Tightening is always allowed; the guard only fires on loosening.
        scan.replay(cache, chi2_track=4.0, min_anchor_planes=2)
        cache.meta["min_anchor_planes"] = 2
        with pytest.raises(ValueError, match="min_anchor_planes"):
            scan.replay(cache, chi2_track=4.0, min_anchor_planes=1)
        cache.meta["min_anchor_planes"] = 0  # restore for the other tests

    def test_rejects_looser_chi2_than_decoded(self, dataset):
        _tel, _prb, _alignment, cache = dataset
        cache.meta["chi2_track"] = 4.0
        try:
            with pytest.raises(ValueError, match="chi2_track"):
                scan.replay(cache, chi2_track=10.0, min_anchor_planes=1)
        finally:
            cache.meta["chi2_track"] = None  # None == the inf the decode used


class TestCacheRoundTrip:
    def test_npz_round_trip_is_exact(self, dataset, tmp_path):
        _tel, _prb, alignment, cache = dataset
        path = tmp_path / "cache.npz"
        cache.save(path)
        loaded = scan.ClusterCache.load(path)
        assert loaded.meta == cache.meta
        assert len(loaded) == len(cache)
        before = scan.replay(cache, chi2_track=4.0, min_anchor_planes=1)
        after = scan.replay(loaded, chi2_track=4.0, min_anchor_planes=1)
        assert after.counts == before.counts
        assert after.coincs == before.coincs

    def test_subset_replays_as_a_row_slice(self, dataset):
        """Time-binning the cache must not change any per-cluster verdict."""
        _tel, _prb, _alignment, cache = dataset
        full = scan.replay(cache, chi2_track=4.0, min_anchor_planes=1)
        mask = np.zeros(len(cache), dtype=bool)
        mask[::2] = True
        first = scan.replay(cache.subset(mask), chi2_track=4.0, min_anchor_planes=1)
        second = scan.replay(cache.subset(~mask), chi2_track=4.0, min_anchor_planes=1)
        assert first.counts + second.counts == full.counts

    def test_cluster_width_inverts_the_sigma_encoding(self, dataset):
        """``cluster_width`` recovers integer widths from the stored sigmas."""
        _tel, _prb, _alignment, cache = dataset
        w = cache.cluster_width
        finite = w[np.isfinite(w)]
        assert finite.size > 0
        assert np.all(finite >= 1)
        assert np.array_equal(finite, np.rint(finite))


class TestResidualsAndMetrics:
    def test_max_resid_cut_is_monotone_in_acceptance(self, dataset):
        """A looser absolute-mm cut can only accept more clusters."""
        _tel, _prb, _alignment, cache = dataset
        accepted = [
            scan.replay(cache, min_anchor_planes=1, max_resid_mm=m).counts["accepted"]
            for m in (2.0, 5.0, 10.0, 20.0, 50.0)
        ]
        assert accepted == sorted(accepted)

    def test_chi2_cut_is_monotone_in_acceptance(self, dataset):
        _tel, _prb, _alignment, cache = dataset
        accepted = [
            scan.replay(cache, chi2_track=c, min_anchor_planes=1).counts["accepted"]
            for c in (0.5, 2.0, 4.0, 37.0, 1000.0)
        ]
        assert accepted == sorted(accepted)

    def test_residuals_are_consistent_with_chi2(self, dataset):
        """Cached residuals reproduce the cached chi2 via the stored sigmas.

        chi2 = sum over planes and axes of (r / sigma)^2 -- the same weighted
        sum ``_tel_line_fit`` computes.  This is what licenses comparing the
        sigma-adaptive cut against an absolute-mm one offline.
        """
        _tel, _prb, _alignment, cache = dataset
        ok = np.isfinite(cache.chi2) & np.all(
            np.isfinite(cache.resid.reshape(len(cache), -1)), axis=1
        )
        assert np.count_nonzero(ok) > 10
        recomputed = np.sum((cache.resid[ok] / cache.cand_sigma[ok]) ** 2, axis=(1, 2))
        assert np.allclose(recomputed, cache.chi2[ok], rtol=1e-9, atol=1e-9)

    def test_purity_of_a_clean_fit_is_high(self, dataset):
        """The synthetic tracks all cross the probe, so the pedestal is ~empty."""
        _tel, _prb, alignment, cache = dataset
        row = scan.evaluate(
            cache,
            chi2_track=4.0,
            min_anchor_planes=1,
            probe_size_mm=N_PROBE_CH * scan.STRIP_MM,
            alignment=alignment,
        )
        assert row["fit"] == "ok"
        assert row["purity"] > 0.8

    def test_skipped_fit_reports_nan_metrics(self, dataset):
        """Below ``min_fit`` the fit is skipped, not silently degraded."""
        _tel, _prb, alignment, cache = dataset
        row = scan.evaluate(
            cache,
            chi2_track=4.0,
            min_anchor_planes=1,
            min_fit=10**9,
            alignment=alignment,
        )
        assert row["fit"] == "skipped"
        assert row["n_inliers"] == 0
        assert np.isnan(row["z_p"])


class TestMahalPatch:
    def test_mahal_cut_is_restored_after_evaluate(self, dataset):
        """``evaluate`` patches ``optimize._MAHAL_CUT``; it must always restore it."""
        from monrad.pose import optimize as pose_optimize

        _tel, _prb, alignment, cache = dataset
        before = pose_optimize._MAHAL_CUT
        scan.evaluate(
            cache,
            chi2_track=4.0,
            min_anchor_planes=1,
            mahal_cut=2.0,
            alignment=alignment,
        )
        assert pose_optimize._MAHAL_CUT == before
        with pytest.raises(ValueError):
            scan.evaluate(
                cache,
                chi2_track=4.0,
                min_anchor_planes=-1,  # replay() rejects this before any fit
                alignment=alignment,
            )
        assert pose_optimize._MAHAL_CUT == before

    def test_tighter_mahal_cut_keeps_fewer_inliers(self, dataset):
        _tel, _prb, alignment, cache = dataset
        kw = dict(chi2_track=4.0, min_anchor_planes=1, alignment=alignment)
        loose = scan.evaluate(cache, mahal_cut=6.0, **kw)["n_inliers"]
        tight = scan.evaluate(cache, mahal_cut=2.0, **kw)["n_inliers"]
        assert tight <= loose


class TestWindowSlicing:
    """Per-detector filename-window selection (the probe DAQ lags ~25 s)."""

    def _det(self, names: list[str]) -> DetectorFiles:
        return DetectorFiles(
            utc0=datetime(2021, 7, 23, 11, 40),
            f0=100,
            gps_paths=[Path(f"{n}_GPS.bin") for n in names],
            pos_paths=[Path(f"{n}.bin") for n in names],
        )

    def test_slice_is_half_open(self):
        det = self._det(
            ["20210723_235500", "20210724_000000", "20210724_055500", "20210724_060000"]
        )
        sliced = scan.slice_detector(
            det,
            scan._parse_window_bound("20210724_000000"),
            scan._parse_window_bound("20210724_060000"),
        )
        assert [p.name for p in sliced.pos_paths] == [
            "20210724_000000.bin",
            "20210724_055500.bin",
        ]

    def test_pairing_check_accepts_the_probe_lag(self):
        tel = self._det(["20210724_000032", "20210724_000532"])
        prb = self._det(["20210724_000058", "20210724_000558"])
        assert scan.check_window_pairing(tel, prb) == []

    def test_pairing_check_flags_an_extra_batch(self):
        tel = self._det(["20210724_000032", "20210724_000532"])
        prb = self._det(["20210724_000058"])
        problems = scan.check_window_pairing(tel, prb)
        assert problems and "2 file pairs but probe has 1" in problems[0]

    def test_pairing_check_flags_a_shifted_batch(self):
        tel = self._det(["20210724_000032", "20210724_000532"])
        prb = self._det(["20210724_000058", "20210724_001058"])
        problems = scan.check_window_pairing(tel, prb)
        assert problems and "differ by" in problems[0]

    def test_window_bound_accepts_both_forms(self):
        assert scan._parse_window_bound("20210724") == datetime(2021, 7, 24)
        assert scan._parse_window_bound("20210724_061500") == datetime(
            2021, 7, 24, 6, 15
        )
        with pytest.raises(ValueError):
            scan._parse_window_bound("2021-07-24")


class TestReanchoring:
    """Mid-acquisition slices must be re-anchored, and the shift *measured*.

    Stage 1 anchors a stream's first PPS to the header ``utc0`` and counts PPS
    from there, so a slice starting mid-run comes back timed as though it were
    the start of the run.  The two detectors' slices do not start at the same
    instant, so the skew differs between them and every coincidence is lost --
    silently.  These tests pin the detection and the correction.
    """

    def test_count_matches_finds_the_true_shift(self):
        rng = np.random.default_rng(0)
        t_tel = np.sort(rng.integers(0, 10**12, size=5000)).astype(np.int64)
        true_shift = -3 * 10**9
        t_prb = np.sort(t_tel - true_shift + rng.integers(-50, 50, size=t_tel.size))
        counts = {s: scan.count_matches(t_tel, t_prb, s * 10**9) for s in range(-5, 6)}
        assert max(counts, key=lambda s: counts[s]) == -3
        assert counts[-3] > 20 * max(n for s, n in counts.items() if s != -3)

    def test_shift_calibration_confidence(self):
        sharp = scan.ShiftCalibration(
            best_shift_ns=3 * 10**9, nominal_shift_ns=2 * 10**9, counts={2: 2, 3: 4120}
        )
        assert sharp.is_confident
        flat = scan.ShiftCalibration(
            best_shift_ns=0, nominal_shift_ns=0, counts={-1: 9, 0: 11, 1: 8}
        )
        assert not flat.is_confident
        tiny = scan.ShiftCalibration(
            best_shift_ns=0, nominal_shift_ns=0, counts={-1: 0, 0: 4, 1: 0}
        )
        assert not tiny.is_confident  # too few matches to call

    def test_periodic_data_is_reported_as_inconclusive(self, dataset):
        """Perfectly regular event times alias, and must not be called confident.

        ``synthetic.generate`` emits tracks on an exact 0.1 s grid, so shifting
        the probe by a whole second maps every event onto a *different*
        telescope event and reproduces the true match count exactly.  There is
        genuinely no information to pick a shift from, and the calibration has
        to say so rather than pick the first maximum -- which is what makes it
        safe to rely on for real (Poisson-timed) acquisitions.
        """
        tel, prb, _alignment, _cache = dataset
        _tel_a, _prb_a, cal = scan.reanchor_window(
            tel, prb, full_tel=tel, full_prb=prb, search_s=2
        )
        assert cal is not None
        assert len(set(cal.counts.values())) <= 2  # aliased: all shifts tie
        assert not cal.is_confident

    def test_reanchor_without_measuring_uses_file_names_only(self, dataset):
        tel, prb, _alignment, _cache = dataset
        tel_a, prb_a, cal = scan.reanchor_window(
            tel, prb, full_tel=tel, full_prb=prb, measure=False
        )
        assert cal is None
        assert (tel_a.utc0, prb_a.utc0) == (tel.utc0, prb.utc0)

    def test_file_name_elapsed_is_relative_to_the_full_acquisition(self, dataset):
        tel, _prb, _alignment, _cache = dataset
        assert scan.file_name_elapsed(tel, tel) == timedelta(0)


class TestBinSeries:
    def test_bins_partition_the_cache(self, dataset):
        _tel, _prb, alignment, cache = dataset
        rows = scan.bin_series(
            cache,
            bin_s=0.05,
            chi2_track=4.0,
            min_anchor_planes=1,
            alignment=alignment,
        )
        assert len(rows) > 1
        assert sum(r["n_raw"] for r in rows) == len(cache)
        assert [r["bin_index"] for r in rows] == sorted(r["bin_index"] for r in rows)


class TestCli:
    """End-to-end: decode -> replay -> plots, through ``main()``."""

    def _argv(self, dataset, out: Path) -> list[str]:
        tel, prb, _alignment, _cache = dataset
        return [
            "--telescope",
            str(tel.pos_paths[0].parent),
            "--probe",
            str(prb.pos_paths[0].parent),
            "--z-tel",
            *[str(z) for z in Z_TEL],
            "--n-probe-ch",
            str(N_PROBE_CH),
            "--out",
            str(out),
            "--label",
            "synth",
            "--min-fit",
            "3",
            "--chi2-grid",
            "4",
            "37",
            "--anchor-grid",
            "1",
            "--max-resid-mm-grid",
            "10",
            "--rigidity-grid",
            "none",
            "--off-probe-grid",
            "none",
            "--mahal-grid",
            "4",
            "--log-every",
            "0",
        ]

    def test_full_run_writes_cache_grid_and_figures(self, dataset, tmp_path):
        out = tmp_path / "report"
        assert scan.main(self._argv(dataset, out)) == 0

        cache_files = list(out.glob("cache_synth_*.npz"))
        assert len(cache_files) == 1
        assert (out / "scan_grid_synth.csv").exists()
        assert (out / "funnel_synth.csv").exists()

        rows = list(csv.DictReader((out / "scan_grid_synth.csv").open()))
        assert rows
        assert {r["stage"] for r in rows} == {"B1", "B2", "B3"}

        figs = list((out / "figures_synth").glob("*.png"))
        names = {f.name for f in figs}
        assert "sigma_vs_n.png" in names
        assert "anomaly_bins.png" in names
        assert any(n.startswith("footprint_") for n in names)

    def test_stages_are_resumable(self, dataset, tmp_path):
        """``--stage replay`` reuses an earlier ``--stage decode``'s cache."""
        out = tmp_path / "staged"
        argv = self._argv(dataset, out)
        assert scan.main(argv + ["--stage", "decode"]) == 0
        assert not (out / "scan_grid_synth.csv").exists()
        assert scan.main(argv + ["--stage", "replay"]) == 0
        assert (out / "scan_grid_synth.csv").exists()

    def test_replay_without_a_cache_is_an_error(self, dataset, tmp_path):
        out = tmp_path / "empty"
        with pytest.raises(SystemExit):
            scan.main(self._argv(dataset, out) + ["--stage", "replay"])

    def test_no_plots_skips_figures(self, dataset, tmp_path):
        out = tmp_path / "noplots"
        assert scan.main(self._argv(dataset, out) + ["--no-plots"]) == 0
        assert (out / "scan_grid_synth.csv").exists()
        assert not (out / "figures_synth").exists()
