"""Probe-resolution characterization (monitoring Step 1).

Console script ``monrad-resolution``.  Answers "how many coincidences (hence
how much acquisition time) are needed for a sub-mm probe-position fix at a
given probe-telescope distance ``z_p`` and lateral offset from the telescope
axis?" by measuring, on synthetic data with known ground truth, how the pose
resolution ``(σ_x, σ_y, σ_z)`` scales with the inlier count ``N``, ``z_p`` and
the probe's lateral offset (DESIGN.md §8.6).

The sweep is a cylindrical grid ``(z_p, r, φ)``: axial distance ``z_p``, lateral
offset magnitude ``r`` and azimuth ``φ``.  For each geometry the expensive decode
is done **once** — generate a large synthetic acquisition, stream it through
stages 1-3, and decode every coincidence to a :class:`~monrad.pose.Coincidence`.
The cheap part (the pose fit) is then repeated over random subsamples of size
``N`` to build the σ(N) curve, validate the covariance calibration via a pull
test, and invert the ``σ_eff/√N`` law for the inlier budget at a sub-mm target.
The fitted probe centre is recorded alongside the truth so the report can
compare simulated vs reconstructed positions.

Geometry: the probe centre sits at ``r`` mm from the vertical telescope axis
(the centre of the 990 mm active area) along azimuth ``φ`` (``φ=0`` ⇒ +x,
``φ=90°`` ⇒ +y); ``r=0`` puts it on-axis.  The square telescope's 4-fold
symmetry means a single azimuth quadrant ``φ ∈ [0, π/2]`` covers all distinct
directions, so the on-axis ``r=0`` cell is run once and ``r>0`` is swept over
``r × φ``.

Geometry independence: results are reported both in absolute mm and in the
telescope-depth–normalized coordinates ``ρ = z_p/L_tel`` and normalized polar
angle ``η = α/α_max = (r/z_p)·(L_tel/active)``.  In those units
``σ_eff/σ_strip = √(1 + C_ρ·ρ²)`` is independent of the telescope's intra-plane
distance, so ``N_required`` measured here transfers to any plane spacing (polar
angle alone is insufficient — it must be normalized by ``α_max = active/L_tel``).

Two of the DESIGN.md §10 "still to implement" diagnostic plots ride along (the
module already stands up the matplotlib output and the residual substrate):
the χ²(θ) curve (§8.4 — the pose-fit consistency check) and the probe-plane
residual histograms (§8.7), each emitted for one full-statistics on-axis
reference fit per ``z_p``.

Note on the 4-fold ambiguity (DESIGN.md §8.5): a square probe is physically
ambiguous under 90° mounting rotations, but that is an *axis-relabeling*
ambiguity in interpreting the hardware, not a degeneracy of this optimizer.
Once the ``(u, v)`` channels are decoded, :func:`fit_probe_pose` holds them
fixed and scans only θ, so its χ²(θ) curve has a single sharp global minimum
at the true mounting orientation (the θ±90° hypotheses, which would also need
``(u, v) → (v, L−u)`` relabeling, fit far worse).  Fitted errors are therefore
scored directly against ground truth with no branch snapping.
"""

import argparse
import csv
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..alignment import AlignmentCorrection
from ..pose import Coincidence, PoseResult, fit_probe_pose
from ..synthetic import generate
from ..synthetic.generate import N_TEL, STRIP_MM, Z_TEL
from .io import centre_jacobian, load_detector, stream_coincidences

# Telescope angular resolution used in the DESIGN §8.6 σ_eff overlay:
# σ_eff² = σ_strip² + (TEL_ANG_SIGMA·z_p)².  3 mrad = σ_strip·√(2/3)/800 mm.
TEL_ANG_SIGMA = 3.0e-3  # rad
SIGMA_STRIP = STRIP_MM / math.sqrt(12.0)  # ≈ 2.9 mm single-strip resolution
ACTIVE_MM = N_TEL * STRIP_MM  # 990 mm — telescope active-area span
TEL_CENTER_MM = ACTIVE_MM / 2.0  # 495 mm — telescope axis (active-area centre)
L_TEL = float(Z_TEL.max() - Z_TEL.min())  # 800 mm — telescope depth (lever arm)
AXES = ("x", "y", "z")  # t_x, t_y, z_p — the three position components
DEFAULT_TARGETS = (0.3, 1.0)  # mm — sub-mm headline target, 1 mm for reference
DEFAULT_PHI_DEG = (0.0, 45.0, 90.0)  # azimuths of the offset direction (one quadrant)

# σ_eff/σ_strip = √(1 + C_RHO·ρ²) with ρ = z_p/L_tel — the telescope-independent
# master curve.  C_RHO = (σ_θ·L_tel/σ_strip)² ≈ 2/3 (the DESIGN §8.6 coefficient).
C_RHO = (TEL_ANG_SIGMA * L_TEL / SIGMA_STRIP) ** 2


def rho(z_p: float) -> float:
    """Dimensionless probe distance ρ = z_p/L_tel (telescope-depth units)."""
    return z_p / L_TEL


def eta(r: float, z_p: float) -> float:
    """Normalized polar angle η = α/α_max = (r/z_p)·(L_tel/active).

    ``α = atan(r/z_p)`` is the probe's off-axis angle; ``α_max ≈ active/L_tel`` is
    the telescope's footprint-limited slope acceptance (a track must cross all
    three planes within the active area).  η is the fraction of that acceptance,
    so the off-axis degradation collapses across telescope depths.  Returns 0 for
    the degenerate ``z_p ≤ 0`` (probe in the telescope plane).
    """
    if z_p <= 0:
        return 0.0
    return (r / z_p) * (L_TEL / ACTIVE_MM)


# ── Probe placement geometry ─────────────────────────────────────────────────


def _pose_for_offset(
    r: float, phi: float, theta: float, n_probe_ch: int
) -> tuple[float, float]:
    """Return ``(t_x, t_y)`` placing the probe centre at radius ``r``, azimuth ``phi``.

    The probe local origin is one corner; its centre sits at ``(L/2, L/2)`` in
    the rotated probe frame.  Solve for the corner translation that puts that
    centre at ``(TEL_CENTER_MM + r·cos φ, TEL_CENTER_MM + r·sin φ)`` — i.e. ``r``
    mm from the telescope axis along azimuth ``phi`` (``phi=0`` ⇒ along +x).
    """
    half = n_probe_ch * STRIP_MM / 2.0
    c, s = math.cos(theta), math.sin(theta)
    t_x = (TEL_CENTER_MM + r * math.cos(phi)) - half * (c - s)
    t_y = (TEL_CENTER_MM + r * math.sin(phi)) - half * (s + c)
    return t_x, t_y


def _probe_center(
    t_x: float, t_y: float, theta: float, n_probe_ch: int
) -> tuple[float, float]:
    """Physical ``(x, y)`` of the probe centre for a pose (inverse of above)."""
    half = n_probe_ch * STRIP_MM / 2.0
    c, s = math.cos(theta), math.sin(theta)
    return t_x + half * (c - s), t_y + half * (s + c)


# ── Coincidence decoding (the expensive part — done once per geometry) ───────


def decode_coincidences(
    out_dir: Path,
    *,
    z_p: float,
    n_tracks: int,
    seed: int,
    t_x: float,
    t_y: float,
    theta: float,
    n_probe_ch: int,
    z_tel: np.ndarray,
    alignment: AlignmentCorrection,
    tot_thresh: int,
    tot_weights: bool,
) -> tuple[list[Coincidence], dict]:
    """Generate one synthetic acquisition and decode all its coincidences.

    Returns ``(coincidences, info)`` where ``info`` is the
    :func:`monrad.synthetic.generate` metadata dict (carries ``pose`` and
    ``n_coincidences``).
    """
    info = generate(
        out_dir,
        t_x=t_x,
        t_y=t_y,
        theta=theta,
        z_p=z_p,
        n_probe_ch=n_probe_ch,
        n_tracks=n_tracks,
        seed=seed,
    )
    tel = load_detector(info["tel_dir"])
    prb = load_detector(info["probe_dir"])

    coincs = list(
        stream_coincidences(
            tel,
            prb,
            z_tel=z_tel,
            alignment=alignment,
            tot_thresh=tot_thresh,
            tot_weights=tot_weights,
        )
    )
    return coincs, info


# ── Sweep records ────────────────────────────────────────────────────────────


@dataclass
class CellResult:
    """Aggregated statistics for one (z_p, r, φ, N) cell, per position axis."""

    z_p: float
    offset: float  # radial offset magnitude r (mm)
    phi: float  # azimuth of the offset direction (rad); 0 ⇒ +x
    n: int
    n_repeats: int
    pool: int
    # Dimensionless geometry coordinates (telescope-depth normalized).
    rho: float = 0.0  # z_p / L_tel
    eta: float = 0.0  # α / α_max = (r/z_p)·(L_tel/active)
    # In-plane σ resolved along (radial) / perpendicular to (tangential) the
    # offset direction (cov-σ, mm).  NB the dominant σ_x≠σ_y anisotropy is set by
    # the fixed probe rotation θ, not by the offset; radial/tangential just swap
    # which lab axis they track as φ rotates.
    sigma_rad: float = 0.0
    sigma_tan: float = 0.0
    # Simulated vs reconstructed probe centre (mm).  cx/cy_true are the known
    # geometry; cx/cy_fit are the mean fitted centre over the repeats, with
    # cx/cy_fit_std the across-repeat scatter.
    cx_true: float = 0.0
    cy_true: float = 0.0
    cx_fit: float = 0.0
    cy_fit: float = 0.0
    cx_fit_std: float = 0.0
    cy_fit_std: float = 0.0
    # per-axis ("x"/"y"/"z") aggregates
    sigma_cov: dict[str, float] = field(default_factory=dict)
    sigma_emp: dict[str, float] = field(default_factory=dict)
    err_mean: dict[str, float] = field(default_factory=dict)
    pull_mean: dict[str, float] = field(default_factory=dict)
    pull_std: dict[str, float] = field(default_factory=dict)
    corr_t_zp: dict[str, float] = field(default_factory=dict)  # corr(t_axis, z_p)
    # raw per-repeat pulls, kept for the aggregate pull histogram
    pulls: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class GeomResult:
    """All sweep output for one (z_p, r, φ) geometry: cells + reference fit."""

    z_p: float
    offset: float  # radial offset magnitude r (mm)
    phi: float  # azimuth of the offset direction (rad)
    pool: int
    truth: tuple[float, float, float, float]  # (t_x, t_y, theta, z_p)
    cells: list[CellResult]
    # Full-statistics fit for the §10 diagnostic plots; None off the on-axis
    # baseline, where write_plots renders no χ²(θ)/residual figure.
    ref_pose: PoseResult | None


def _fit_subsample(
    sub: list[Coincidence],
    z_corr: np.ndarray,
    alignment: AlignmentCorrection,
    cx_true: float,
    cy_true: float,
    z_p_true: float,
    phi: float,
    n_probe_ch: int,
) -> tuple[
    dict[str, tuple[float, float, float]],
    tuple[float, float],
    tuple[float, float],
]:
    """Fit one subsample.

    Returns ``(per_axis, centre, radtan)`` where ``per_axis[a] = (error,
    sigma_cov, corr_with_zp)``, ``centre = (cx_fit, cy_fit)`` is the fitted
    probe centre, and ``radtan = (sigma_rad, sigma_tan)`` — the in-plane cov-σ
    resolved along and perpendicular to the offset azimuth ``phi``.

    For the x and y axes the statistics are reported at the physical **probe
    centre**, not the corner (pose parameter origin).  The centre covariance is
    propagated from the fitted 4×4 pose covariance through the corner→centre
    Jacobian :func:`centre_jacobian`.  The z_p axis is unchanged.  The probe
    pose is unambiguous in θ (see the module docstring), so the fitted
    parameters are scored directly against ground truth (``cx_true``,
    ``cy_true``, ``z_p_true`` are precomputed by the caller).
    """
    pose = fit_probe_pose(sub, z_corr, alignment)
    cov = pose.cov
    cx_fit, cy_fit = _probe_center(pose.t_x, pose.t_y, pose.theta, n_probe_ch)

    # Centre-referenced x/y: a single corner→centre Jacobian J supplies both the
    # 2×2 centre covariance (J cov Jᵀ) and the centre↔z_p cross-covariance
    # (J cov[:,3]), so the lever-arm convention lives only in centre_jacobian.
    J = centre_jacobian(pose.theta, n_probe_ch)
    cov_c = J @ cov @ J.T
    cov_cx_zp, cov_cy_zp = J @ cov[:, 3]

    sig_cx = math.sqrt(abs(cov_c[0, 0]))
    sig_cy = math.sqrt(abs(cov_c[1, 1]))
    sig_z = math.sqrt(abs(cov[3, 3]))

    err = {"x": cx_fit - cx_true, "y": cy_fit - cy_true, "z": pose.z_p - z_p_true}
    sig = {"x": sig_cx, "y": sig_cy, "z": sig_z}
    corr = {
        "x": cov_cx_zp / (sig_cx * sig_z) if sig_cx > 0 and sig_z > 0 else 0.0,
        "y": cov_cy_zp / (sig_cy * sig_z) if sig_cy > 0 and sig_z > 0 else 0.0,
        "z": 1.0,
    }
    per_axis = {a: (err[a], sig[a], corr[a]) for a in AXES}

    # Resolve the centre (x, y) block along (radial) / perpendicular to
    # (tangential) the offset direction: e_r=(cos φ, sin φ), e_t=(−sin φ, cos φ).
    cp, sp = math.cos(phi), math.sin(phi)
    var_rad = cp * cp * cov_c[0, 0] + 2 * sp * cp * cov_c[0, 1] + sp * sp * cov_c[1, 1]
    var_tan = sp * sp * cov_c[0, 0] - 2 * sp * cp * cov_c[0, 1] + cp * cp * cov_c[1, 1]
    radtan = (math.sqrt(abs(var_rad)), math.sqrt(abs(var_tan)))
    return per_axis, (cx_fit, cy_fit), radtan


def sweep_one_geometry(
    work_dir: Path,
    *,
    z_p: float,
    offset: float,
    phi: float,
    n_grid: list[int],
    n_repeats: int,
    n_tracks: int,
    seed: int,
    theta: float,
    n_probe_ch: int,
    z_tel: np.ndarray,
    alignment: AlignmentCorrection,
    tot_thresh: int,
    tot_weights: bool,
    rng: np.random.Generator,
) -> GeomResult | None:
    """Decode one (z_p, r, φ) geometry once, then fit σ(N) over subsamples.

    Returns ``None`` (with a warning) when the geometry yields fewer than 3
    coincidences — e.g. a far off-axis probe outside the track cone — so the
    driver can skip that grid cell instead of aborting the whole sweep.
    """
    t_x, t_y = _pose_for_offset(offset, phi, theta, n_probe_ch)
    coincs, info = decode_coincidences(
        work_dir,
        z_p=z_p,
        n_tracks=n_tracks,
        seed=seed,
        t_x=t_x,
        t_y=t_y,
        theta=theta,
        n_probe_ch=n_probe_ch,
        z_tel=z_tel,
        alignment=alignment,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    )
    pool = len(coincs)
    truth = info["pose"]
    z_corr = alignment.corrected_z_tel(z_tel)
    cx_true, cy_true = _probe_center(t_x, t_y, theta, n_probe_ch)

    if pool < 3:
        print(
            f"  z_p={z_p:6g} offset={offset:5g} mm φ={math.degrees(phi):3.0f}°  "
            f"SKIP — only {pool} coincidences (probe off the track cone; raise "
            f"--n-tracks if unexpected)"
        )
        return None

    # Reference full-statistics fit for the §10 χ²(θ) and residual plots, which
    # write_plots renders only for the on-axis baseline; skip it off-axis.
    ref_pose = fit_probe_pose(coincs, z_corr, alignment) if abs(offset) < 1e-9 else None

    cells: list[CellResult] = []
    for n in n_grid:
        if n > pool:
            print(
                f"  z_p={z_p:6g} offset={offset:5g} mm φ={math.degrees(phi):3.0f}°  "
                f"N={n:<5}  SKIP — only {pool} coincidences in pool (need N <= pool)"
            )
            continue
        errs = {a: [] for a in AXES}
        sigs = {a: [] for a in AXES}
        corrs = {a: [] for a in AXES}
        cx_fits: list[float] = []
        cy_fits: list[float] = []
        rads: list[float] = []
        tans: list[float] = []
        for _ in range(n_repeats):
            idx = rng.choice(pool, size=n, replace=False)
            sub = [coincs[i] for i in idx]
            per_axis, (cx_f, cy_f), (s_rad, s_tan) = _fit_subsample(
                sub, z_corr, alignment, cx_true, cy_true, truth[3], phi, n_probe_ch
            )
            for a in AXES:
                e, s, c = per_axis[a]
                errs[a].append(e)
                sigs[a].append(s)
                corrs[a].append(c)
            cx_fits.append(cx_f)
            cy_fits.append(cy_f)
            rads.append(s_rad)
            tans.append(s_tan)

        cell = CellResult(
            z_p=z_p,
            offset=offset,
            phi=phi,
            n=n,
            n_repeats=n_repeats,
            pool=pool,
            rho=rho(z_p),
            eta=eta(offset, z_p),
            sigma_rad=float(np.mean(rads)),
            sigma_tan=float(np.mean(tans)),
            cx_true=cx_true,
            cy_true=cy_true,
            cx_fit=float(np.mean(cx_fits)),
            cy_fit=float(np.mean(cy_fits)),
            cx_fit_std=float(np.std(cx_fits, ddof=1)) if len(cx_fits) > 1 else 0.0,
            cy_fit_std=float(np.std(cy_fits, ddof=1)) if len(cy_fits) > 1 else 0.0,
        )
        for a in AXES:
            e = np.array(errs[a])
            s = np.array(sigs[a])
            pull = e / np.where(s > 0, s, np.nan)
            cell.sigma_cov[a] = float(np.mean(s))
            cell.sigma_emp[a] = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
            cell.err_mean[a] = float(np.mean(e))
            cell.pull_mean[a] = float(np.nanmean(pull))
            cell.pull_std[a] = (
                float(np.nanstd(pull, ddof=1)) if len(pull) > 1 else float("nan")
            )
            cell.corr_t_zp[a] = float(np.mean(corrs[a]))
            cell.pulls[a] = pull
        cells.append(cell)
        print(
            f"  z_p={z_p:6g} offset={offset:5g} mm φ={math.degrees(phi):3.0f}°  "
            f"N={n:<5} pool={pool:<5} σ_cov(x,y,z)=({cell.sigma_cov['x']:.3f},"
            f"{cell.sigma_cov['y']:.3f},{cell.sigma_cov['z']:.3f}) mm  "
            f"σ(rad,tan)=({cell.sigma_rad:.3f},{cell.sigma_tan:.3f})  "
            f"pull_std(z)={cell.pull_std['z']:.2f}"
        )

    return GeomResult(
        z_p=z_p,
        offset=offset,
        phi=phi,
        pool=pool,
        truth=truth,
        cells=cells,
        ref_pose=ref_pose,
    )


# ── σ_eff/√N inversion ───────────────────────────────────────────────────────


def fit_sigma_eff(cells: list[CellResult], axis: str) -> float:
    """Fit σ(N) = σ_eff/√N (cov-σ) and return σ_eff for one axis.

    Closed-form least squares of σ_i against 1/√N_i through the origin:
    σ_eff = Σ(σ_i/√N_i) / Σ(1/N_i).
    """
    num = 0.0
    den = 0.0
    for cell in cells:
        if axis not in cell.sigma_cov:
            continue
        inv_sqrt_n = 1.0 / math.sqrt(cell.n)
        num += cell.sigma_cov[axis] * inv_sqrt_n
        den += 1.0 / cell.n
    return num / den if den > 0 else float("nan")


def n_required(sigma_eff: float, target: float) -> float:
    """Inlier count for cov-σ = target under the σ_eff/√N law: N = (σ_eff/target)²."""
    if not math.isfinite(sigma_eff) or target <= 0:
        return float("nan")
    return (sigma_eff / target) ** 2


def interp_sigma_eff(
    csv_path: Path | str,
    z_p: float,
    *,
    axis: str = "z",
    offset: float = 0.0,
) -> float:
    """Interpolate ``σ_eff(z_p)`` from a previously written ``n_required.csv``.

    Reads the resolution-study table (schema owned by this module — see
    :func:`write_n_required_csv`) with the :mod:`csv` reader, keeps the rows for
    the requested ``axis`` and lateral ``offset`` (the on-axis baseline,
    ``offset≈0``, by default), dedupes to one ``σ_eff`` per ``z_p`` (the value
    repeats across the ``target_sigma`` rows), and linearly interpolates with
    :func:`numpy.interp` (which clamps at the grid ends).

    Uses the on-axis baseline only; the offset dependence present in the table
    is ignored for this first cut.  A 2-D ``(z_p, offset)`` interpolation is the
    obvious later refinement.
    """
    by_zp: dict[float, float] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["axis"] != axis or abs(float(row["offset"]) - offset) > 1e-9:
                continue
            by_zp[float(row["z_p"])] = float(row["sigma_eff"])
    if not by_zp:
        raise ValueError(f"no rows for axis={axis!r} offset={offset} in {csv_path}")
    z_grid = np.array(sorted(by_zp))
    seff_grid = np.array([by_zp[z] for z in z_grid])
    return float(np.interp(z_p, z_grid, seff_grid))


def design_sigma_eff(z_p: float) -> float:
    """DESIGN.md §8.6 overlay: σ_eff = √(σ_strip² + (3 mrad·z_p)²)."""
    return math.sqrt(SIGMA_STRIP**2 + (TEL_ANG_SIGMA * z_p) ** 2)


def design_sigma_eff_ratio(rho_val: float) -> float:
    """Telescope-independent master curve: σ_eff/σ_strip = √(1 + C_RHO·ρ²)."""
    return math.sqrt(1.0 + C_RHO * rho_val**2)


# ── Helpers for the report / plots ──────────────────────────────────────────


def _is_baseline(g: GeomResult) -> bool:
    """True for the on-axis (offset ≈ 0) geometry used as the z_p baseline."""
    return abs(g.offset) < 1e-9


def _baseline_ref_n(results: list[GeomResult]) -> int | None:
    """Largest N present in any on-axis baseline cell (for σ-vs-offset slices)."""
    ns = [c.n for g in results if _is_baseline(g) for c in g.cells]
    return max(ns) if ns else None


# ── CSV output ───────────────────────────────────────────────────────────────


def write_sweep_csv(path: Path, results: list[GeomResult]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "z_p",
                "offset",
                "phi_deg",
                "rho",
                "eta",
                "N",
                "axis",
                "pool",
                "sigma_cov",
                "sigma_cov_strip",
                "sigma_rad",
                "sigma_tan",
                "sigma_emp",
                "err_mean",
                "pull_mean",
                "pull_std",
                "corr_t_zp",
                "cx_true",
                "cy_true",
                "cx_fit",
                "cy_fit",
                "cx_fit_std",
                "cy_fit_std",
            ]
        )
        for g in results:
            for cell in g.cells:
                for a in AXES:
                    w.writerow(
                        [
                            f"{cell.z_p:g}",
                            f"{cell.offset:g}",
                            f"{math.degrees(cell.phi):g}",
                            f"{cell.rho:.6g}",
                            f"{cell.eta:.6g}",
                            cell.n,
                            a,
                            cell.pool,
                            f"{cell.sigma_cov[a]:.6g}",
                            f"{cell.sigma_cov[a] / SIGMA_STRIP:.6g}",
                            f"{cell.sigma_rad:.6g}",
                            f"{cell.sigma_tan:.6g}",
                            f"{cell.sigma_emp[a]:.6g}",
                            f"{cell.err_mean[a]:.6g}",
                            f"{cell.pull_mean[a]:.6g}",
                            f"{cell.pull_std[a]:.6g}",
                            f"{cell.corr_t_zp[a]:.6g}",
                            f"{cell.cx_true:.6g}",
                            f"{cell.cy_true:.6g}",
                            f"{cell.cx_fit:.6g}",
                            f"{cell.cy_fit:.6g}",
                            f"{cell.cx_fit_std:.6g}",
                            f"{cell.cy_fit_std:.6g}",
                        ]
                    )


def write_n_required_csv(
    path: Path,
    results: list[GeomResult],
    targets: tuple[float, ...],
) -> dict[tuple[float, float, float, str], float]:
    """Write n_required.csv and return {(z_p, offset, phi, axis): sigma_eff}."""
    sigma_eff_map: dict[tuple[float, float, float, str], float] = {}
    with open(path, "w", newline="") as fh:
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
        for g in results:
            for a in AXES:
                seff = fit_sigma_eff(g.cells, a)
                sigma_eff_map[(g.z_p, g.offset, g.phi, a)] = seff
                for tgt in targets:
                    w.writerow(
                        [
                            f"{g.z_p:g}",
                            f"{g.offset:g}",
                            f"{math.degrees(g.phi):g}",
                            f"{rho(g.z_p):.6g}",
                            f"{eta(g.offset, g.z_p):.6g}",
                            a,
                            f"{seff:.6g}",
                            f"{seff / SIGMA_STRIP:.6g}",
                            f"{tgt:g}",
                            f"{n_required(seff, tgt):.6g}",
                        ]
                    )
    return sigma_eff_map


# ── Plots ────────────────────────────────────────────────────────────────────


def _plot_sigma_vs_n(results: list[GeomResult], path: Path) -> None:
    """σ vs N on the on-axis baseline, one line per z_p."""
    import matplotlib.pyplot as plt

    baseline = [g for g in results if _is_baseline(g)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ai, a in enumerate(AXES):
        ax = axs[ai]
        for g in baseline:
            ns = [c.n for c in g.cells]
            if not ns:
                continue
            sig = [c.sigma_cov[a] for c in g.cells]
            line = ax.plot(ns, sig, "o", label=f"z_p={g.z_p:g} mm")[0]
            seff = fit_sigma_eff(g.cells, a)
            n_line = np.array(sorted(ns))
            ax.plot(
                n_line, seff / np.sqrt(n_line), "-", color=line.get_color(), alpha=0.6
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (inliers)")
        ax.set_ylabel(f"σ_{a}  [mm]  (cov)")
        ax.set_title(f"σ_{a} vs N, on-axis  (lines: σ_eff/√N fit)")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_sigma_vs_offset(results: list[GeomResult], path: Path) -> None:
    """Radial σ vs offset magnitude r at fixed N — one subplot per z_p, line per φ."""
    import matplotlib.pyplot as plt

    n_ref = _baseline_ref_n(results)
    zps = sorted({g.z_p for g in results})
    phis = sorted({g.phi for g in results})
    ncol = max(1, len(zps))
    fig, axs = plt.subplots(1, ncol, figsize=(5 * ncol, 4.5), squeeze=False)
    for zi, z in enumerate(zps):
        ax = axs[0][zi]
        for ph in phis:
            # r=0 (on-axis) is φ-independent; include it on every φ series.
            pts = sorted(
                (g.offset, c.sigma_rad)
                for g in results
                if g.z_p == z and (abs(g.offset) < 1e-9 or abs(g.phi - ph) < 1e-9)
                for c in g.cells
                if c.n == n_ref
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", label=f"φ={math.degrees(ph):g}°")
        ax.set_xlabel("offset magnitude r  [mm]")
        ax.set_ylabel("σ_radial  [mm]  (cov)")
        ax.set_yscale("log")
        ax.set_title(f"σ_rad vs r at z_p={z:g} mm  (N={n_ref})")
        ax.grid(True, which="both", alpha=0.3)
    axs[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_sigma_vs_azimuth(results: list[GeomResult], path: Path) -> None:
    """Lab-frame σ_x, σ_y vs azimuth φ at a representative (z_p, r).

    The key azimuthal result: the lab-frame resolution is ~independent of the
    offset direction (flat lines) — only the offset magnitude matters.  The
    σ_x≠σ_y gap is the fixed probe-rotation (θ) anisotropy, not an azimuthal
    effect; σ_radial/σ_tangential are overlaid (dashed) for reference and simply
    cross over as the offset direction rotates between the lab axes.
    """
    import matplotlib.pyplot as plt

    n_ref = _baseline_ref_n(results)
    # Pick the (z_p, r>0) cell-set with the most azimuths present, largest r wins.
    by_zr: dict[tuple[float, float], list[GeomResult]] = {}
    for g in results:
        if abs(g.offset) < 1e-9 or not any(c.n == n_ref for c in g.cells):
            continue
        by_zr.setdefault((g.z_p, g.offset), []).append(g)
    if not by_zr:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("σ vs azimuth — no off-axis cells")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return
    (z_sel, r_sel), gs = max(by_zr.items(), key=lambda kv: (len(kv[1]), kv[0][1]))
    gs = sorted(gs, key=lambda g: g.phi)

    fig, ax = plt.subplots(figsize=(7, 5))
    phideg = [math.degrees(g.phi) for g in gs]
    cells = [next(c for c in g.cells if c.n == n_ref) for g in gs]
    ax.plot(phideg, [c.sigma_cov["x"] for c in cells], "o-", label="σ_x (lab)")
    ax.plot(phideg, [c.sigma_cov["y"] for c in cells], "s-", label="σ_y (lab)")
    ax.plot(phideg, [c.sigma_rad for c in cells], "^--", alpha=0.5, label="σ_radial")
    ax.plot(
        phideg, [c.sigma_tan for c in cells], "v--", alpha=0.5, label="σ_tangential"
    )
    ax.set_xlabel("azimuth φ  [deg]")
    ax.set_ylabel("σ  [mm]  (cov)")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"σ vs azimuth at z_p={z_sel:g} mm, r={r_sel:g} mm  (N={n_ref})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_sigma_eff_vs_rho(
    results: list[GeomResult],
    sigma_eff_map: dict[tuple[float, float, float, str], float],
    path: Path,
) -> None:
    """Telescope-independent master curve: σ_eff/σ_strip vs ρ = z_p/L_tel (on-axis)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    zps = sorted({g.z_p for g in results if _is_baseline(g)})
    rhos = [rho(z) for z in zps]
    for a in ("x", "y"):
        seff = [
            sigma_eff_map.get((z, 0.0, 0.0, a), float("nan")) / SIGMA_STRIP for z in zps
        ]
        ax.plot(rhos, seff, "o-", label=f"σ_eff,{a}/σ_strip (measured)")
    r_fine = np.linspace(min(rhos), max(rhos), 100) if rhos else np.array([0.0])
    ax.plot(
        r_fine,
        [design_sigma_eff_ratio(r) for r in r_fine],
        "k--",
        label="√(1 + C_ρ·ρ²)  (telescope-independent)",
    )
    ax.set_xlabel("ρ = z_p / L_tel")
    ax.set_ylabel("σ_eff / σ_strip")
    ax.set_title("Geometry-independent resolution master curve (on-axis)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_sigma_eff_vs_zp(
    results: list[GeomResult],
    sigma_eff_map: dict[tuple[float, float, float, str], float],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    zps = sorted({g.z_p for g in results if _is_baseline(g)})
    for a in AXES:
        seff = [sigma_eff_map.get((z, 0.0, 0.0, a), float("nan")) for z in zps]
        ax.plot(zps, seff, "o-", label=f"σ_eff,{a} (measured, on-axis)")
    z_fine = np.linspace(min(zps), max(zps), 100) if zps else np.array([0.0])
    ax.plot(
        z_fine,
        [design_sigma_eff(z) for z in z_fine],
        "k--",
        label="DESIGN §8.6: √(σ_strip²+(3mrad·z_p)²)",
    )
    ax.set_xlabel("z_p  [mm]")
    ax.set_ylabel("σ_eff  [mm]")
    ax.set_title("Effective single-coincidence resolution vs distance (on-axis)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_n_required_vs_zp(
    sigma_eff_map: dict[tuple[float, float, float, str], float],
    results: list[GeomResult],
    targets: tuple[float, ...],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    zps = sorted({g.z_p for g in results if _is_baseline(g)})
    for tgt in targets:
        # average N_required over the two in-plane axes (x, y), on-axis
        nreq = []
        for z in zps:
            vals = [
                n_required(sigma_eff_map.get((z, 0.0, 0.0, a), float("nan")), tgt)
                for a in ("x", "y")
            ]
            nreq.append(float(np.nanmean(vals)))
        ax.plot(zps, nreq, "o-", label=f"target σ_t = {tgt:g} mm")
    ax.set_xlabel("z_p  [mm]")
    ax.set_ylabel("N_required (mean of x,y, on-axis)")
    ax.set_yscale("log")
    ax.set_title("Inlier budget for a sub-mm in-plane fix vs distance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_recon_map(results: list[GeomResult], path: Path) -> None:
    """Simulated (×) vs reconstructed (○) probe centres over the grid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    active = N_TEL * STRIP_MM
    ax.plot(
        [0, active, active, 0, 0],
        [0, 0, active, active, 0],
        "-",
        color="steelblue",
        alpha=0.5,
        label="telescope footprint",
    )
    ax.plot(TEL_CENTER_MM, TEL_CENTER_MM, "k+", ms=12, label="telescope axis")
    seen_truth = seen_fit = False
    for g in results:
        if not g.cells:
            continue
        cell = max(g.cells, key=lambda c: c.n)  # best-statistics cell
        ax.plot(
            cell.cx_true,
            cell.cy_true,
            "x",
            color="crimson",
            ms=9,
            label="simulated" if not seen_truth else None,
        )
        ax.plot(
            cell.cx_fit,
            cell.cy_fit,
            "o",
            mfc="none",
            color="green",
            ms=9,
            label="reconstructed" if not seen_fit else None,
        )
        seen_truth = seen_fit = True
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("y  [mm]")
    ax.set_title("Simulated vs reconstructed probe centres (best-N cell per geometry)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_pull_hist(results: list[GeomResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    grid = np.linspace(-4, 4, 200)
    gauss = np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi)
    for ai, a in enumerate(AXES):
        ax = axs[ai]
        allp = np.concatenate(
            [c.pulls[a] for g in results for c in g.cells if a in c.pulls]
            or [np.array([])]
        )
        allp = allp[np.isfinite(allp)]
        if allp.size:
            ax.hist(allp, bins=40, range=(-4, 4), density=True, alpha=0.6)
            ax.set_title(f"pull {a}: mean={allp.mean():.2f} std={allp.std(ddof=1):.2f}")
        ax.plot(grid, gauss, "k--", label="N(0,1)")
        ax.set_xlabel(f"(fit − truth)/σ_cov  [{a}]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pull distributions (cov calibration: unit Gaussian if correct)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_chi2_theta(g: GeomResult, path: Path) -> None:
    """DESIGN §8.4/§10: χ²(θ) consistency check — single sharp minimum at θ_true.

    Plotted on a log-χ² axis so the global minimum is legible against the much
    larger off-orientation χ².  With the ``(u, v)`` channels decoded and held
    fixed, the θ scan is unambiguous (the θ±90° hypotheses would also need an
    axis relabeling, DESIGN §8.5): a healthy fit shows one deep well at the true
    mounting orientation.  A global minimum away from the expected orientation,
    or competing wells of comparable depth, flags a wiring or axis problem.
    """
    import matplotlib.pyplot as plt

    assert g.ref_pose is not None  # only called for on-axis baseline geometries
    curve = g.ref_pose.chi2_curve
    theta_true = math.degrees(g.truth[2])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(np.degrees(curve[:, 0]), curve[:, 1], "-")
    ax.axvline(
        theta_true,
        color="crimson",
        ls="--",
        alpha=0.7,
        label=f"θ_true={theta_true:.1f}°",
    )
    ax.set_xlabel("θ  [deg]")
    ax.set_ylabel("χ²(θ)  (min over t_x,t_y,z_p)")
    ax.set_title(f"χ²(θ) consistency check at z_p={g.z_p:g} mm, on-axis (DESIGN §8.4)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_residuals(g: GeomResult, path: Path) -> None:
    """DESIGN §8.7/§10: probe-plane x/y inlier residual histograms."""
    import matplotlib.pyplot as plt

    assert g.ref_pose is not None  # only called for on-axis baseline geometries
    pose = g.ref_pose
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, res, lbl in (
        (axs[0], pose.residuals_x, "x"),
        (axs[1], pose.residuals_y, "y"),
    ):
        if res.size:
            ax.hist(res, bins=40, alpha=0.7)
            ax.set_title(f"{lbl} residuals: mean={res.mean():.3f} std={res.std():.3f}")
        ax.axvline(0.0, color="k", ls="--", alpha=0.6)
        ax.set_xlabel(f"{lbl} residual at probe plane  [mm]")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Probe-plane residuals at z_p={g.z_p:g} mm, on-axis  (n={pose.n_inliers})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_plots(
    results: list[GeomResult],
    sigma_eff_map: dict[tuple[float, float, float, str], float],
    targets: tuple[float, ...],
    out_dir: Path,
) -> None:
    """Write every sweep + per-z_p diagnostic plot to out_dir."""
    _plot_sigma_vs_n(results, out_dir / "sigma_vs_N.png")
    _plot_sigma_vs_offset(results, out_dir / "sigma_vs_offset.png")
    _plot_sigma_vs_azimuth(results, out_dir / "sigma_vs_azimuth.png")
    _plot_sigma_eff_vs_zp(results, sigma_eff_map, out_dir / "sigma_eff_vs_zp.png")
    _plot_sigma_eff_vs_rho(results, sigma_eff_map, out_dir / "sigma_eff_vs_rho.png")
    _plot_n_required_vs_zp(
        sigma_eff_map, results, targets, out_dir / "n_required_vs_zp.png"
    )
    _plot_recon_map(results, out_dir / "recon_vs_truth.png")
    _plot_pull_hist(results, out_dir / "pull_hist.png")
    # χ²(θ) and residual histograms: one on-axis reference per z_p.
    for g in results:
        if not _is_baseline(g):
            continue
        tag = f"{g.z_p:g}".replace("-", "m").replace(".", "p")
        _plot_chi2_theta(g, out_dir / f"chi2_theta_z{tag}.png")
        _plot_residuals(g, out_dir / f"residuals_z{tag}.png")


# ── Driver ───────────────────────────────────────────────────────────────────


def _resolve_n_tracks(n_tracks: int | list[int], n_z: int) -> list[int]:
    """Broadcast a scalar track count, or validate a per-z_p list of length n_z."""
    if isinstance(n_tracks, int):
        return [n_tracks] * n_z
    if len(n_tracks) == 1:
        return list(n_tracks) * n_z
    if len(n_tracks) == n_z:
        return list(n_tracks)
    raise ValueError(
        f"n_tracks must be a single value or match the number of z_p values "
        f"({n_z}); got {len(n_tracks)}"
    )


def run_resolution_study(
    out_dir: Path,
    *,
    z_p_grid: list[float],
    n_grid: list[int],
    n_repeats: int,
    n_tracks: int | list[int],
    seed: int,
    offset_grid: list[float] | None = None,
    phi_grid: list[float] | None = None,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
    theta: float = 0.29671,
    n_probe_ch: int = 30,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> list[GeomResult]:
    """Run the full σ(N, z_p, r, φ) sweep and write CSVs + plots to out_dir.

    The lateral dimension is a 2-D polar grid: offset magnitude ``r`` (from
    ``offset_grid``) × azimuth ``φ`` (from ``phi_grid``, radians).  The on-axis
    ``r=0`` cell is φ-independent and run once per z_p.  By the square
    telescope's 4-fold symmetry only one azimuth quadrant ``φ ∈ [0, π/2]`` need
    be swept.

    ``n_tracks`` is either a single count applied to every z_p, or a list with
    one count per z_p (far planes are coincidence-starved and need many more
    tracks to fill the larger-N cells, so a single sweep can spend tracks where
    they matter without rerunning the cheap near-plane geometries).
    """
    if offset_grid is None:
        offset_grid = [0.0]
    if phi_grid is None:
        phi_grid = [0.0]
    # Lateral placements: on-axis once (φ irrelevant), then each r>0 × each φ.
    placements: list[tuple[float, float]] = [(0.0, 0.0)]
    placements += [(r, ph) for r in offset_grid if r > 0 for ph in phi_grid]
    n_tracks_per_z = _resolve_n_tracks(n_tracks, len(z_p_grid))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_tel = np.array(Z_TEL)
    # Synthetic data is generated perfectly aligned (no plane offsets), so the
    # identity correction is exactly right and avoids an extra telescope pass.
    alignment = AlignmentCorrection.identity()
    rng = np.random.default_rng(seed)

    results: list[GeomResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        k = 0
        for i, z_p in enumerate(z_p_grid):
            nt = n_tracks_per_z[i]
            for offset, phi in placements:
                print(
                    f"z_p = {z_p:g} mm  offset = {offset:g} mm  "
                    f"φ = {math.degrees(phi):g}°  (generating {nt} tracks)…"
                )
                work = Path(tmp) / f"g_{k}"
                g = sweep_one_geometry(
                    work,
                    z_p=z_p,
                    offset=offset,
                    phi=phi,
                    n_grid=n_grid,
                    n_repeats=n_repeats,
                    n_tracks=nt,
                    seed=seed + k,
                    theta=theta,
                    n_probe_ch=n_probe_ch,
                    z_tel=z_tel,
                    alignment=alignment,
                    tot_thresh=tot_thresh,
                    tot_weights=tot_weights,
                    rng=rng,
                )
                k += 1
                if g is not None:
                    results.append(g)

    write_sweep_csv(out_dir / "resolution_sweep.csv", results)
    sigma_eff_map = write_n_required_csv(out_dir / "n_required.csv", results, targets)
    if make_plots:
        write_plots(results, sigma_eff_map, targets, out_dir)

    _print_summary(results, sigma_eff_map, targets)
    print(f"\nWrote resolution study to {out_dir}")
    return results


def _print_summary(
    results: list[GeomResult],
    sigma_eff_map: dict[tuple[float, float, float, str], float],
    targets: tuple[float, ...],
) -> None:
    """Console N_required table (on-axis, with ρ) plus a σ_rad-vs-(r,φ) slice."""
    zps = sorted({g.z_p for g in results if _is_baseline(g)})
    for tgt in targets:
        print(f"\n  N_required (mean x,y, on-axis) for σ_t ≤ {tgt:g} mm:")
        for z in zps:
            vals = [
                n_required(sigma_eff_map.get((z, 0.0, 0.0, a), float("nan")), tgt)
                for a in ("x", "y")
            ]
            print(f"    z_p={z:7g} mm  (ρ={rho(z):.2f}) → N ≈ {np.nanmean(vals):.0f}")
    print(
        "    (N_required is a function of ρ=z_p/L_tel; scale distances by "
        "L_tel'/{0:g} mm for another telescope.)".format(L_TEL)
    )

    n_ref = _baseline_ref_n(results)
    phis = sorted({g.phi for g in results})
    offsets = sorted({g.offset for g in results if g.offset > 0})
    print(f"\n  σ_radial at N={n_ref}  (rows: z_p × r,  cols: azimuth φ):")
    header = "    z_p \\ (r, φ)  " + "".join(f"{math.degrees(p):>9g}°" for p in phis)
    print(header)
    for z in zps:
        for r in offsets:
            row = f"    z={z:g} r={r:g}".ljust(16)
            for p in phis:
                sig = [
                    c.sigma_rad
                    for g in results
                    if g.z_p == z and abs(g.offset - r) < 1e-9 and abs(g.phi - p) < 1e-9
                    for c in g.cells
                    if c.n == n_ref
                ]
                row += f"{sig[0]:>9.3f} " if sig else f"{'—':>9} "
            print(row)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="monrad-resolution",
        description=(
            "Characterize probe pose resolution σ(N, z_p, r, φ) on synthetic data."
        ),
    )
    p.add_argument(
        "--z",
        nargs="+",
        type=float,
        default=[0.0, 300.0, 1000.0, 3000.0, 5000.0],
        metavar="Z_P",
        help="Probe-telescope distances z_p in mm (default: 0 300 1000 3000 5000)",
    )
    p.add_argument(
        "--offset",
        nargs="+",
        type=float,
        default=[0.0, 150.0, 300.0, 450.0],
        metavar="MM",
        help="Offset MAGNITUDES r of the probe centre from the telescope axis, mm "
        "(default: 0 150 300 450). 0 = on-axis baseline; the direction of each "
        "offset is set by --phi.",
    )
    p.add_argument(
        "--phi",
        nargs="+",
        type=float,
        default=list(DEFAULT_PHI_DEG),
        metavar="DEG",
        help="Azimuths (deg) of the offset direction (default: 0 45 90). 0 = +x, "
        "90 = +y, 45 = diagonal. The square telescope's 4-fold symmetry means a "
        "single quadrant [0, 90] covers all distinct directions.",
    )
    p.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[30, 100, 300, 1000],
        metavar="N",
        help="Inlier subsample sizes (default: 30 100 300 1000)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Random subsamples per (z_p, r, φ, N) cell (default: 50)",
    )
    p.add_argument(
        "--n-tracks",
        nargs="+",
        type=int,
        default=[60000],
        metavar="N",
        help="Synthetic tracks generated per geometry (default: 60000). Either one "
        "value for every z_p, or one value per --z entry — far planes are "
        "coincidence-starved and need many more tracks to fill the larger-N cells, "
        "so e.g. '--z 0 300 1000 3000 5000 --n-tracks 60000 60000 60000 300000 "
        "600000' spends tracks only where they matter, in a single run.",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGETS),
        metavar="MM",
        help="Target σ_t values (mm) to invert for N_required (default: 0.3 1.0)",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./pipeline_out/resolution"),
        help="Output directory (default: ./pipeline_out/resolution)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib output (CSV only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_resolution_study(
        args.out,
        z_p_grid=args.z,
        offset_grid=args.offset,
        phi_grid=[math.radians(d) for d in args.phi],
        n_grid=args.n,
        n_repeats=args.repeats,
        n_tracks=args.n_tracks,  # list[int]: one value, or one per --z entry
        seed=args.seed,
        targets=tuple(args.targets),
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
