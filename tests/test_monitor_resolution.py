"""Tests for monrad.monitor.resolution (monitoring Step 1).

Runs a small synthetic σ(N, z_p) sweep and checks the headline behaviours:
σ falls like 1/√N, σ_eff grows with z_p, the covariance pull is ~unit, and the
CSV + diagnostic plots (σ-vs-N, σ_eff-vs-z_p, N_required, pull, and the
per-z_p χ²(θ) and residual histograms folded in from DESIGN.md §10) are
written.
"""

import csv
import math

import numpy as np
import pytest

from monrad.monitor import resolution as R


def test_n_required_inverts_sigma_eff():
    # σ_eff/√N = target  ⇒  N = (σ_eff/target)²
    assert R.n_required(4.0, 0.3) == pytest.approx((4.0 / 0.3) ** 2)
    assert R.n_required(2.9, 1.0) == pytest.approx(2.9**2)
    assert math.isnan(R.n_required(float("nan"), 0.3))


# ── Full sweep ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp("resolution")
    results = R.run_resolution_study(
        out,
        z_p_grid=[300.0, 1000.0],
        n_grid=[30, 100],
        n_repeats=8,
        n_tracks=8000,
        seed=7,
        targets=(0.3, 1.0),
        make_plots=True,
    )
    return out, results


def test_csv_outputs_written(sweep):
    out, _ = sweep
    sweep_csv = out / "resolution_sweep.csv"
    nreq_csv = out / "n_required.csv"
    assert sweep_csv.exists() and nreq_csv.exists()

    with open(sweep_csv) as fh:
        rows = list(csv.DictReader(fh))
    # 2 z_p × 2 N × 3 axes
    assert len(rows) == 2 * 2 * 3
    assert {r["axis"] for r in rows} == {"x", "y", "z"}
    for r in rows:
        assert float(r["sigma_cov"]) > 0

    with open(nreq_csv) as fh:
        nrows = list(csv.DictReader(fh))
    # 2 z_p × 3 axes × 2 targets
    assert len(nrows) == 2 * 3 * 2


def test_sigma_falls_with_n(sweep):
    """σ_cov shrinks as N grows (≈ 1/√N) for every axis and z_p."""
    _, results = sweep
    for zr in results:
        by_n = {c.n: c for c in zr.cells}
        for axis in R.AXES:
            assert by_n[100].sigma_cov[axis] < by_n[30].sigma_cov[axis]


def test_sigma_eff_grows_with_distance(sweep):
    """In-plane σ_eff increases with z_p (DESIGN §8.6 telescope-angle term)."""
    _, results = sweep
    seff = {zr.z_p: {a: R.fit_sigma_eff(zr.cells, a) for a in R.AXES} for zr in results}
    for axis in ("x", "y"):
        assert seff[1000.0][axis] > seff[300.0][axis]


def test_pull_is_approximately_unit(sweep):
    """z_p covariance pull std is order-unity (cov calibration sanity)."""
    _, results = sweep
    pulls = np.concatenate([c.pulls["z"] for zr in results for c in zr.cells])
    pulls = pulls[np.isfinite(pulls)]
    assert 0.4 < float(pulls.std()) < 2.5


def test_chi2_theta_unique_minimum_at_truth(sweep):
    """χ²(θ) has a single global minimum at the true mounting orientation.

    The 4-fold square-probe ambiguity (DESIGN §8.5) is an axis-relabeling
    ambiguity, not four equal θ-minima: once (u, v) are decoded and held fixed,
    the θ scan is unambiguous, so the θ±90° hypotheses fit far worse.
    """
    _, results = sweep
    zr = results[0]
    curve = zr.ref_pose.chi2_curve
    theta = curve[:, 0]
    chi2 = curve[:, 1]

    theta_true = zr.truth[2]
    i_min = int(np.argmin(chi2))
    # Global minimum sits at the true orientation (1° coarse-scan resolution).
    assert abs(
        math.atan2(
            math.sin(theta[i_min] - theta_true), math.cos(theta[i_min] - theta_true)
        )
    ) < math.radians(2.0)
    # The θ+90° hypothesis is dramatically worse — no competing equal minimum.
    i_90 = int(np.argmin(np.abs(theta - (theta[i_min] + math.pi / 2))))
    assert chi2[i_90] > 10.0 * chi2[i_min]


def test_diagnostic_plots_written(sweep):
    out, _ = sweep
    for name in (
        "sigma_vs_N.png",
        "sigma_eff_vs_zp.png",
        "n_required_vs_zp.png",
        "pull_hist.png",
    ):
        assert (out / name).exists(), name
    # per-z_p χ²(θ) and residual histograms
    for z in ("300", "1000"):
        assert (out / f"chi2_theta_z{z}.png").exists()
        assert (out / f"residuals_z{z}.png").exists()
