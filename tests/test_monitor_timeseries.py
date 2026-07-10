"""Tests for monrad.monitor.timeseries (monitoring Step 2).

Two tiers:
- Structural / math tests use synthetic data (fast, no real data needed).
- A real-data consistency test uses data/0_testLab_20210723 (skipped if absent).
  It checks that windowed fits are internally consistent across windows (std of
  t_x/t_y values ≲ mean σ), not against absolute truth (which is unknown for
  real data).
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from monrad.monitor import timeseries as timeseries_mod
from monrad.monitor.io import centre_cov_2x2
from monrad.monitor.timeseries import WindowResult, _parse_args, monitor_probe
from monrad.pose import _MIN_COINCS
from monrad.synthetic.generate import Z_TEL, generate

# ── Real-data paths (skipped when absent) ────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data" / "0_testLab_20210723"
_TEL_DIR = _DATA_DIR / "Base"
_PRB_DIR = _DATA_DIR / "Probe_0"
_Z_TEL_REAL = [0.0, -1340.0, -670.0]  # physical z-order for this dataset


# ── Synthetic smoke tests ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    """Run monitor_probe on synthetic data with known pose.  Fast, for structure checks."""
    src = tmp_path_factory.mktemp("ts_synth")
    info = generate(
        src,
        t_x=300.0,
        t_y=350.0,
        theta=0.29671,
        z_p=300.0,
        n_probe_ch=30,
        n_tracks=5000,
        seed=42,
    )
    out = tmp_path_factory.mktemp("ts_out")
    results = monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=150.0,
        z_tel=np.array(Z_TEL, dtype=float),
        n_probe_ch=30,
        out_dir=out,
        make_plots=True,
    )
    return results, out


def test_synthetic_at_least_two_windows(synth_run):
    results, _ = synth_run
    assert len(results) >= 2, f"Expected ≥2 windows, got {len(results)}"


def test_synthetic_window_times_ordered(synth_run):
    results, _ = synth_run
    for i in range(1, len(results)):
        assert results[i].utc_start >= results[i - 1].utc_end
        assert results[i].utc_start < results[i].utc_end


def test_synthetic_positive_inliers(synth_run):
    results, _ = synth_run
    for r in results:
        assert r.n_inliers > 0


def test_synthetic_sigmas_finite_positive(synth_run):
    results, _ = synth_run
    for r in results:
        for s in (r.sigma_tx, r.sigma_ty, r.sigma_zp, r.sigma_theta):
            assert math.isfinite(s) and s > 0


def test_synthetic_zp_within_5sigma(synth_run):
    """z_p is the best-constrained parameter; each window should be within 5σ."""
    results, _ = synth_run
    Z_P_TRUE = 300.0
    for i, r in enumerate(results):
        assert abs(r.z_p - Z_P_TRUE) < 5 * r.sigma_zp, (
            f"window {i}: z_p={r.z_p:.2f} deviates >5σ from {Z_P_TRUE} "
            f"(σ_zp={r.sigma_zp:.3f})"
        )


def test_window_result_holds_only_scalars(synth_run):
    """WindowResult must not retain the full PoseResult (unbounded RAM growth).

    Every field should be a plain scalar (int/float/datetime), never a
    PoseResult or its inliers/outliers lists, so results accumulated over a
    long run stay bounded regardless of window count.
    """
    results, _ = synth_run
    assert results, "expected at least one window"
    for f in dataclasses.fields(WindowResult):
        assert f.type != "PoseResult", f"field {f.name} leaks a PoseResult"
    for r in results:
        for value in dataclasses.astuple(r):
            assert isinstance(value, (int, float)) or hasattr(value, "isoformat")


def test_synthetic_csv_rows(synth_run):
    results, out = synth_run
    csv_path = out / "pose_timeseries.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == len(results) + 1  # header + one row per window


def test_synthetic_plot_written(synth_run):
    _, out = synth_run
    assert (out / "pose_timeseries.png").exists()


# ── Count-based mode (default, --min-fit driven) tests ───────────────────────


@pytest.fixture(scope="module")
def count_run(tmp_path_factory):
    """Run monitor_probe in count-based mode (window_s omitted) on synthetic data."""
    src = tmp_path_factory.mktemp("ts_count")
    info = generate(
        src,
        t_x=300.0,
        t_y=350.0,
        theta=0.29671,
        z_p=300.0,
        n_probe_ch=30,
        n_tracks=5000,
        seed=42,
    )
    out = tmp_path_factory.mktemp("ts_count_out")
    results = monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=None,  # count-based mode
        z_tel=np.array(Z_TEL, dtype=float),
        n_probe_ch=30,
        out_dir=out,
        min_fit=30,
        make_plots=True,
    )
    return results, out, info


def test_count_based_produces_windows(count_run):
    results, _, _ = count_run
    assert len(results) >= 1


def test_count_based_batch_size_respected(count_run):
    """Each batch fits exactly min_fit coincidences, so inliers never exceed it."""
    results, _, _ = count_run
    for r in results:
        assert 0 < r.n_inliers <= 30


def test_count_based_times_strictly_ordered(count_run):
    """Batch timestamps come from the coincidences and advance monotonically."""
    results, _, _ = count_run
    for i, r in enumerate(results):
        assert r.utc_start < r.utc_end, f"batch {i}: start not before end"
    for i in range(1, len(results)):
        assert results[i].utc_start >= results[i - 1].utc_end


def test_count_based_sigmas_finite_positive(count_run):
    results, _, _ = count_run
    for r in results:
        for s in (r.sigma_tx, r.sigma_ty, r.sigma_zp, r.sigma_theta):
            assert math.isfinite(s) and s > 0


def test_count_based_csv_and_plot_written(count_run):
    results, out, _ = count_run
    csv_path = out / "pose_timeseries.csv"
    assert csv_path.exists()
    assert len(csv_path.read_text().splitlines()) == len(results) + 1
    assert (out / "pose_timeseries.png").exists()


def test_min_fit_controls_window_count(count_run):
    """Smaller min_fit yields at least as many batches on the same stream."""
    _, _, info = count_run

    def _n_windows(min_fit: int) -> int:
        return len(
            monitor_probe(
                info["tel_dir"],
                info["probe_dir"],
                window_s=None,
                z_tel=np.array(Z_TEL, dtype=float),
                min_fit=min_fit,
                make_plots=False,
            )
        )

    n_small = _n_windows(15)
    n_large = _n_windows(45)
    assert n_small >= n_large
    assert n_small >= 1


def test_min_fit_above_total_yields_no_windows(count_run):
    """A min_fit larger than the whole stream produces no fits."""
    _, _, info = count_run
    results = monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=None,
        z_tel=np.array(Z_TEL, dtype=float),
        min_fit=10_000_000,
        make_plots=False,
    )
    assert results == []


def test_trailing_batch_logs_warning(count_run, caplog):
    """A non-empty trailing batch at end-of-stream is logged, not silently
    dropped (mirrors the raw_cap-abandoned-window warning)."""
    _, _, info = count_run
    with caplog.at_level("WARNING", logger="monrad.monitor.timeseries"):
        results = monitor_probe(
            info["tel_dir"],
            info["probe_dir"],
            window_s=None,
            z_tel=np.array(Z_TEL, dtype=float),
            min_fit=10_000_000,
            make_plots=False,
        )
    assert results == []
    assert any("trailing window" in rec.message for rec in caplog.records)


def test_n_probe_ch_too_small_logs_warning(count_run, caplog):
    """Decoded hits landing outside the configured [0, n_probe_ch*10] mm
    footprint warn (both immediately and with an end-of-run summary), since a
    too-small --n-probe-ch silently biases the off-probe gate and the
    centre-covariance propagation."""
    _, _, info = count_run  # generated with the probe's real footprint = 30 ch
    with caplog.at_level("WARNING", logger="monrad.monitor.timeseries"):
        monitor_probe(
            info["tel_dir"],
            info["probe_dir"],
            window_s=None,
            z_tel=np.array(Z_TEL, dtype=float),
            n_probe_ch=20,  # smaller than the probe's real 30-channel footprint
            min_fit=30,
            make_plots=False,
        )
    assert any(
        "exceeds the configured footprint" in rec.message for rec in caplog.records
    )
    assert any("consider --n-probe-ch" in rec.message for rec in caplog.records)


def test_n_probe_ch_sufficient_logs_no_warning(count_run, caplog):
    """No footprint warning when --n-probe-ch matches (or exceeds) the
    probe's real footprint."""
    _, _, info = count_run
    with caplog.at_level("WARNING", logger="monrad.monitor.timeseries"):
        monitor_probe(
            info["tel_dir"],
            info["probe_dir"],
            window_s=None,
            z_tel=np.array(Z_TEL, dtype=float),
            n_probe_ch=30,  # matches the probe's real footprint
            min_fit=30,
            make_plots=False,
        )
    assert not any(
        "exceeds the configured footprint" in rec.message for rec in caplog.records
    )


def test_cli_min_fit_below_floor_rejected():
    """--min-fit under fit_probe_pose's hard minimum errors at parse time,
    not with an uncaught ValueError deep in the fit."""
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--telescope",
                "unused",
                "--probe",
                "unused",
                "--z-tel",
                "0",
                "--min-fit",
                str(_MIN_COINCS - 1),
            ]
        )


# ── Hybrid mode (both --window-s and --min-fit) tests ────────────────────────


def _run_hybrid(info, *, window_s, min_fit):
    return monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=window_s,
        z_tel=np.array(Z_TEL, dtype=float),
        min_fit=min_fit,
        make_plots=False,
    )


def test_hybrid_window_spans_at_least_window_s(count_run):
    """With --window-s given, every emitted window spans at least window_s."""
    _, _, info = count_run
    window_s = 150.0
    results = _run_hybrid(info, window_s=window_s, min_fit=30)
    assert len(results) >= 1
    for i, r in enumerate(results):
        span = (r.utc_end - r.utc_start).total_seconds()
        assert span >= window_s - 1e-6, f"window {i}: span {span:.3f}s < {window_s}s"


def test_hybrid_stretches_past_window_s_to_reach_min_fit(count_run):
    """A negligible window_s lets --min-fit drive the boundaries (windows stretch).

    With a 1 ms floor — far shorter than the time to gather ``min_fit`` sparse
    coincidences — the count bound binds, so the run reduces to count-based
    batching and every window spans far longer than window_s.
    """
    _, _, info = count_run
    tiny = 1e-3
    hybrid = _run_hybrid(info, window_s=tiny, min_fit=30)
    count_based = _run_hybrid(info, window_s=None, min_fit=30)
    assert len(hybrid) == len(count_based)
    assert len(hybrid) >= 1
    assert all((r.utc_end - r.utc_start).total_seconds() > tiny for r in hybrid)


def test_hybrid_min_fit_above_total_yields_no_windows(count_run):
    """A min_fit above the whole stream never closes a window, even with window_s."""
    _, _, info = count_run
    results = _run_hybrid(info, window_s=150.0, min_fit=10_000_000)
    assert results == []


# ── Residual-RMS diagnostic tests ────────────────────────────────────────────


def test_resid_rms_populated(count_run):
    """Every emitted window carries a finite, positive resid_rms."""
    results, _, _ = count_run
    assert results
    for r in results:
        assert math.isfinite(r.resid_rms) and r.resid_rms > 0


def test_gate_csv_has_resid_rms_column(count_run, tmp_path):
    """The CSV carries a resid_rms column with a numeric value per window."""
    _, _, info = count_run
    out = tmp_path / "gate_csv"
    results = monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=None,
        z_tel=np.array(Z_TEL, dtype=float),
        min_fit=30,
        out_dir=out,
        make_plots=False,
    )
    lines = (out / "pose_timeseries.csv").read_text().splitlines()
    header = lines[0].split(",")
    assert "resid_rms" in header
    col = header.index("resid_rms")
    assert len(lines) == len(results) + 1
    for line in lines[1:]:
        assert math.isfinite(float(line.split(",")[col]))


# ── centre_cov_2x2 unit tests ─────────────────────────────────────────────────


def test_centre_cov_identity_at_zero_lever():
    """At θ=0 and n_probe_ch=0 the centre coincides with the corner."""
    cov = np.diag([1.0, 2.0, 3.0, 4.0])
    cov_c = centre_cov_2x2(cov, theta=0.0, n_probe_ch=0)
    assert cov_c[0, 0] == pytest.approx(1.0)
    assert cov_c[1, 1] == pytest.approx(2.0)


def test_centre_cov_symmetric():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((4, 4))
    cov = A @ A.T
    cov_c = centre_cov_2x2(cov, theta=0.5, n_probe_ch=30)
    assert cov_c.shape == (2, 2)
    assert cov_c[0, 1] == pytest.approx(cov_c[1, 0])


def test_centre_cov_positive_diagonal():
    rng = np.random.default_rng(13)
    A = rng.standard_normal((4, 4))
    cov = A @ A.T + np.eye(4) * 0.1
    cov_c = centre_cov_2x2(cov, theta=math.pi / 6, n_probe_ch=30)
    assert cov_c[0, 0] > 0
    assert cov_c[1, 1] > 0


# ── Real-data consistency test (skipped when data absent) ────────────────────


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """Run monitor_probe on the testlab dataset, 30-min windows (~207 inliers each)."""
    out = tmp_path_factory.mktemp("ts_real_out")
    results = monitor_probe(
        _TEL_DIR,
        _PRB_DIR,
        window_s=1800.0,
        z_tel=np.array(_Z_TEL_REAL, dtype=float),
        n_probe_ch=30,
        out_dir=out,
        make_plots=True,
    )
    return results, out


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_multiple_windows(real_run):
    results, _ = real_run
    assert len(results) >= 2, f"Expected ≥2 windows, got {len(results)}"


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_window_times_ordered(real_run):
    results, _ = real_run
    for i in range(1, len(results)):
        assert results[i].utc_start >= results[i - 1].utc_end


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_sufficient_inliers(real_run):
    # Mahalanobis cut can reduce inliers below MIN_FIT; 3 is the optimizer minimum.
    results, _ = real_run
    for r in results:
        assert r.n_inliers >= 3


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_pose_in_active_area(real_run):
    """t_x, t_y (corner) must land within the 990 mm telescope active area."""
    results, _ = real_run
    ACTIVE_MM = 990.0
    N_PROBE_CH = 30
    STRIP_MM = 10.0
    probe_side = N_PROBE_CH * STRIP_MM  # 300 mm
    for r in results:
        assert -probe_side <= r.t_x <= ACTIVE_MM + probe_side, (
            f"t_x={r.t_x:.1f} outside plausible range"
        )
        assert -probe_side <= r.t_y <= ACTIVE_MM + probe_side, (
            f"t_y={r.t_y:.1f} outside plausible range"
        )


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_all_values_finite(real_run):
    """Every numerical field in every WindowResult must be finite."""
    results, _ = real_run
    for i, r in enumerate(results):
        for name, val in [
            ("t_x", r.t_x),
            ("sigma_tx", r.sigma_tx),
            ("t_y", r.t_y),
            ("sigma_ty", r.sigma_ty),
            ("z_p", r.z_p),
            ("sigma_zp", r.sigma_zp),
            ("theta", r.theta),
            ("sigma_theta", r.sigma_theta),
        ]:
            assert math.isfinite(val), f"window {i}: {name}={val} is not finite"


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_csv_written(real_run):
    results, out = real_run
    csv_path = out / "pose_timeseries.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == len(results) + 1


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_plot_written(real_run):
    _, out = real_run
    assert (out / "pose_timeseries.png").exists()


# ── Real-data: geometric gate + count-based min_fit interaction ──────────────


@pytest.fixture(scope="module")
def real_gated_run(tmp_path_factory):
    """Count-based mode with a lossy rigidity gate active.

    Regression for the raw-batch-growth fix: a raw batch sized to exactly
    ``min_fit`` can never clear ``min_fit`` survivors once a gate removes even
    one coincidence, so prior to the fix this exact combination (mirrors a
    real repro on this dataset) produced zero windows over the whole
    acquisition.
    """
    out = tmp_path_factory.mktemp("ts_real_gated_out")
    results = monitor_probe(
        _TEL_DIR,
        _PRB_DIR,
        window_s=None,
        z_tel=np.array(_Z_TEL_REAL, dtype=float),
        n_probe_ch=40,
        min_fit=100,
        max_rigidity_resid_mm=100.0,
        out_dir=out,
        make_plots=False,
    )
    return results, out


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_gated_count_based_produces_windows(real_gated_run):
    results, _ = real_gated_run
    assert len(results) >= 1


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_gated_window_times_ordered(real_gated_run):
    results, _ = real_gated_run
    for i in range(1, len(results)):
        assert results[i].utc_start >= results[i - 1].utc_end


@pytest.mark.skipif(not _TEL_DIR.exists(), reason="testlab data not available")
def test_real_gated_sufficient_inliers(real_gated_run):
    # Mahalanobis cut can reduce inliers below min_fit; 3 is the optimizer minimum.
    results, _ = real_gated_run
    for r in results:
        assert r.n_inliers >= 3


# ── Cold-start rigidity-gate bootstrap: throttled, not re-run every step ─────


def test_cold_start_bootstrap_fit_is_throttled(tmp_path, monkeypatch):
    """The rigidity gate's cold-start z_ref anchor is cached, not refit on
    every appended coincidence.

    Regression for the O(raw_cap) cold-start bootstrap-fit fan-out: before the
    fix, every single coincidence appended while growing a raw batch past
    ``min_fit`` (chasing gate survivors, with ``prev_pose`` still ``None``)
    triggered a full ``fit_probe_pose`` call just to get a throwaway z_ref
    anchor. An impossibly strict ``max_rigidity_resid_mm`` (drops everything)
    forces every window in this run to keep growing — the worst case for that
    fan-out — so the sequence of raw batch sizes at which fit_probe_pose was
    actually called directly reveals whether caching is in effect: with
    caching, consecutive calls within the same growth stretch must be at least
    COLD_START_REFIT_STRIDE coincidences apart, rather than one per append.
    """
    min_fit = 5
    stride = timeseries_mod.COLD_START_REFIT_STRIDE

    src = tmp_path / "src"
    info = generate(src, n_tracks=150, seed=7)

    real_fit_probe_pose = timeseries_mod.fit_probe_pose
    call_sizes = []

    def _counting_fit_probe_pose(coincs, *args, **kwargs):
        call_sizes.append(len(coincs))
        return real_fit_probe_pose(coincs, *args, **kwargs)

    monkeypatch.setattr(timeseries_mod, "fit_probe_pose", _counting_fit_probe_pose)

    monitor_probe(
        info["tel_dir"],
        info["probe_dir"],
        window_s=None,
        z_tel=np.array(Z_TEL, dtype=float),
        n_probe_ch=30,
        min_fit=min_fit,
        max_rigidity_resid_mm=0.0,  # drops every coincidence: worst-case growth
        make_plots=False,
    )

    assert call_sizes, "expected the cold-start bootstrap fit to run at least once"

    # Split the recorded call sizes into growth stretches: each stretch is one
    # contaminated window's raw batch growing from min_fit upward: a batch
    # reset (committed fit, raw-cap drop, or end-of-stream) starts a new
    # stretch, visible as the next recorded size dropping back to ~min_fit.
    stretches: list[list[int]] = []
    for size in call_sizes:
        if stretches and size > stretches[-1][-1]:
            stretches[-1].append(size)
        else:
            stretches.append([size])

    for stretch in stretches:
        assert stretch[0] == min_fit, (
            f"expected each growth stretch's first bootstrap call at "
            f"min_fit={min_fit} raw coincidences, got {stretch[0]}"
        )
        for prev, nxt in zip(stretch, stretch[1:]):
            assert nxt - prev >= stride, (
                f"cold-start bootstrap refit at {prev} -> {nxt} raw "
                f"coincidences, only {nxt - prev} apart (< stride={stride})"
            )
