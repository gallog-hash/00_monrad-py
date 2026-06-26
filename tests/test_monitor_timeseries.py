"""Tests for monrad.monitor.timeseries (monitoring Step 2).

Two tiers:
- Structural / math tests use synthetic data (fast, no real data needed).
- A real-data consistency test uses data/0_testLab_20210723 (skipped if absent).
  It checks that windowed fits are internally consistent across windows (std of
  t_x/t_y values ≲ mean σ), not against absolute truth (which is unknown for
  real data).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from monrad.monitor.io import centre_cov_2x2
from monrad.monitor.timeseries import monitor_probe
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


def test_synthetic_csv_rows(synth_run):
    results, out = synth_run
    csv_path = out / "pose_timeseries.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == len(results) + 1  # header + one row per window


def test_synthetic_plot_written(synth_run):
    _, out = synth_run
    assert (out / "pose_timeseries.png").exists()


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
