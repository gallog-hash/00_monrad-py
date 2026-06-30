"""Tests for monrad.monitor.resolution (monitoring Step 1).

Runs a small synthetic σ(N, z_p, r, φ) cylindrical sweep and checks the headline
behaviours: σ falls like 1/√N, σ_eff grows with z_p, the covariance pull is
~unit, the reconstructed probe centre matches truth, the azimuthal x↔y symmetry
holds (quadrant reduction), the geometry-normalized ρ/η helpers behave, and the
CSV + diagnostic plots (σ-vs-N, σ-vs-offset, σ-vs-azimuth, σ_eff-vs-z_p,
σ_eff-vs-ρ, N_required, pull, reconstructed-vs-truth map, and the per-z_p χ²(θ)
and residual histograms folded in from DESIGN.md §10) are written.
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


def test_interp_sigma_eff(tmp_path):
    """interp_sigma_eff reads n_required.csv, filters axis/offset, and interpolates."""
    csv_path = tmp_path / "n_required.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "z_p",
                "offset",
                "phi_deg",
                "rho",
                "eta",
                "axis",
                "sigma_eff",
                "sigma_eff_strip",
                "target_sigma",
                "N_required",
            ]
        )
        # Two z_p nodes for the on-axis z baseline; σ_eff repeats across targets.
        for z, seff in ((300.0, 5.0), (1000.0, 12.0)):
            for tgt in (0.3, 1.0):
                w.writerow([z, 0, 0, 0, 0, "z", seff, 0, tgt, 0])
            # A decoy off-axis row and a decoy axis the lookup must ignore.
            w.writerow([z, 150, 0, 0, 0, "z", 99.0, 0, 0.3, 0])
            w.writerow([z, 0, 0, 0, 0, "x", 99.0, 0, 0.3, 0])

    # Exact value at a grid node.
    assert R.interp_sigma_eff(csv_path, 300.0) == pytest.approx(5.0)
    assert R.interp_sigma_eff(csv_path, 1000.0) == pytest.approx(12.0)
    # Linear interpolation halfway between nodes.
    assert R.interp_sigma_eff(csv_path, 650.0) == pytest.approx(8.5)
    # Clamps past the grid ends (np.interp behaviour).
    assert R.interp_sigma_eff(csv_path, 0.0) == pytest.approx(5.0)
    assert R.interp_sigma_eff(csv_path, 5000.0) == pytest.approx(12.0)
    # No matching rows → clear error.
    with pytest.raises(ValueError):
        R.interp_sigma_eff(csv_path, 300.0, axis="nope")


def test_resolve_n_tracks():
    assert R._resolve_n_tracks(60000, 3) == [60000, 60000, 60000]
    assert R._resolve_n_tracks([60000], 3) == [60000, 60000, 60000]
    assert R._resolve_n_tracks([1, 2, 3], 3) == [1, 2, 3]
    with pytest.raises(ValueError):
        R._resolve_n_tracks([1, 2], 3)


def test_pose_offset_roundtrip():
    """_pose_for_offset and _probe_center are inverses; (r, φ) lands as expected."""
    theta = 0.29671
    for phi in (0.0, math.pi / 4, math.pi / 2):
        for r in (0.0, 150.0, 300.0):
            t_x, t_y = R._pose_for_offset(r, phi, theta, n_probe_ch=30)
            cx, cy = R._probe_center(t_x, t_y, theta, n_probe_ch=30)
            assert cx == pytest.approx(R.TEL_CENTER_MM + r * math.cos(phi))
            assert cy == pytest.approx(R.TEL_CENTER_MM + r * math.sin(phi))


def test_rho_eta_helpers():
    """ρ = z_p/L_tel and η = α/α_max are the geometry-normalized coordinates."""
    assert R.rho(R.L_TEL) == pytest.approx(1.0)
    assert R.rho(2 * R.L_TEL) == pytest.approx(2.0)
    assert R.eta(0.0, 1000.0) == 0.0
    assert R.eta(150.0, 0.0) == 0.0  # z_p ≤ 0 guard
    # η monotonically increases with offset magnitude at fixed z_p
    assert R.eta(300.0, 1000.0) > R.eta(150.0, 1000.0) > 0.0
    # master curve: σ_eff/σ_strip = √(1 + C_ρ·ρ²), unit at ρ=0
    assert R.design_sigma_eff_ratio(0.0) == pytest.approx(1.0)
    assert R.design_sigma_eff_ratio(1.0) > 1.0


# ── Full sweep ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp("resolution")
    results = R.run_resolution_study(
        out,
        z_p_grid=[300.0, 1000.0],
        offset_grid=[0.0, 150.0],
        phi_grid=[0.0, math.pi / 2],
        n_grid=[30, 100],
        n_repeats=8,
        n_tracks=8000,
        seed=7,
        targets=(0.3, 1.0),
        make_plots=True,
    )
    return out, results


def _baseline(results):
    return {g.z_p: g for g in results if abs(g.offset) < 1e-9}


def test_csv_outputs_written(sweep):
    out, results = sweep
    sweep_csv = out / "resolution_sweep.csv"
    nreq_csv = out / "n_required.csv"
    assert sweep_csv.exists() and nreq_csv.exists()

    with open(sweep_csv) as fh:
        rows = list(csv.DictReader(fh))
    # one row per (geometry-cell, axis)
    assert len(rows) == sum(len(g.cells) for g in results) * 3
    assert len(rows) > 0
    assert {r["axis"] for r in rows} == {"x", "y", "z"}
    assert {float(r["offset"]) for r in rows} == {0.0, 150.0}
    # azimuth quadrant: on-axis (φ=0) plus the two off-axis directions 0°, 90°
    assert {float(r["phi_deg"]) for r in rows} == {0.0, 90.0}
    # simulated + reconstructed position columns are present and populated
    for col in ("cx_true", "cy_true", "cx_fit", "cy_fit"):
        assert all(r[col] for r in rows)
    # geometry-normalized + radial/tangential columns are present and sane
    for r in rows:
        assert float(r["sigma_cov"]) > 0
        assert float(r["sigma_cov_strip"]) == pytest.approx(
            float(r["sigma_cov"]) / R.SIGMA_STRIP, rel=1e-4
        )
        assert float(r["sigma_rad"]) > 0 and float(r["sigma_tan"]) > 0
        assert float(r["rho"]) == pytest.approx(float(r["z_p"]) / R.L_TEL, rel=1e-4)

    with open(nreq_csv) as fh:
        nrows = list(csv.DictReader(fh))
    # one row per (geometry, axis, target)
    assert len(nrows) == len(results) * 3 * 2


def test_sigma_falls_with_n(sweep):
    """σ_cov shrinks as N grows (≈ 1/√N) for every axis and geometry."""
    _, results = sweep
    for g in results:
        by_n = {c.n: c for c in g.cells}
        if 30 not in by_n or 100 not in by_n:
            continue
        for axis in R.AXES:
            assert by_n[100].sigma_cov[axis] < by_n[30].sigma_cov[axis]


def test_sigma_eff_grows_with_distance(sweep):
    """In-plane σ_eff increases with z_p (DESIGN §8.6 telescope-angle term)."""
    _, results = sweep
    base = _baseline(results)
    seff = {
        z: {a: R.fit_sigma_eff(g.cells, a) for a in R.AXES} for z, g in base.items()
    }
    for axis in ("x", "y"):
        assert seff[1000.0][axis] > seff[300.0][axis]


def test_azimuth_isotropy(sweep):
    """Lab-frame σ_x, σ_y are ~independent of the offset azimuth φ.

    The square telescope's 4-fold symmetry means the *lab-frame* resolution does
    not care which direction the probe is offset — only the magnitude r matters.
    (The σ_x ≠ σ_y anisotropy is set by the fixed probe rotation θ, not by φ; the
    radial/tangential split merely swaps which one aligns with the offset.)  This
    azimuthal isotropy is what justifies sweeping only one quadrant φ ∈ [0, π/2].
    """
    _, results = sweep
    for z in {g.z_p for g in results}:
        for r in {g.offset for g in results if g.offset > 0}:
            by_phi = {
                g.phi: {c.n: c.sigma_cov for c in g.cells}
                for g in results
                if g.z_p == z and abs(g.offset - r) < 1e-9
            }
            if 0.0 not in by_phi or not any(
                abs(p - math.pi / 2) < 1e-9 for p in by_phi
            ):
                continue
            p90 = next(p for p in by_phi if abs(p - math.pi / 2) < 1e-9)
            for n in by_phi[0.0]:
                if n not in by_phi[p90]:
                    continue
                for axis in ("x", "y"):
                    assert by_phi[0.0][n][axis] == pytest.approx(
                        by_phi[p90][n][axis], rel=0.2
                    )


def test_reconstructed_center_matches_truth(sweep):
    """Mean reconstructed probe centre is close to the simulated one (unbiased)."""
    _, results = sweep
    for g in results:
        for c in g.cells:
            assert abs(c.cx_fit - c.cx_true) < 3.0
            assert abs(c.cy_fit - c.cy_true) < 3.0


def test_pull_is_approximately_unit(sweep):
    """z_p covariance pull std is order-unity (cov calibration sanity)."""
    _, results = sweep
    pulls = np.concatenate([c.pulls["z"] for g in results for c in g.cells])
    pulls = pulls[np.isfinite(pulls)]
    assert 0.4 < float(pulls.std()) < 2.5


def test_chi2_theta_unique_minimum_at_truth(sweep):
    """χ²(θ) has a single global minimum at the true mounting orientation.

    The 4-fold square-probe ambiguity (DESIGN §8.5) is an axis-relabeling
    ambiguity, not four equal θ-minima: once (u, v) are decoded and held fixed,
    the θ scan is unambiguous, so the θ±90° hypotheses fit far worse.
    """
    _, results = sweep
    g = _baseline(results)[300.0]
    curve = g.ref_pose.chi2_curve
    theta = curve[:, 0]
    chi2 = curve[:, 1]

    theta_true = g.truth[2]
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
        "sigma_vs_offset.png",
        "sigma_vs_azimuth.png",
        "sigma_eff_vs_zp.png",
        "sigma_eff_vs_rho.png",
        "n_required_vs_zp.png",
        "recon_vs_truth.png",
        "pull_hist.png",
    ):
        assert (out / name).exists(), name
    # per-z_p (on-axis) χ²(θ) and residual histograms
    for z in ("300", "1000"):
        assert (out / f"chi2_theta_z{z}.png").exists()
        assert (out / f"residuals_z{z}.png").exists()
