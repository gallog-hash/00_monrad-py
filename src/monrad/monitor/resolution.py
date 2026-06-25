"""Probe-resolution characterization (monitoring Step 1).

Console script ``monrad-resolution``.  Answers "how many coincidences (hence
how much acquisition time) are needed for a sub-mm probe-position fix at a
given probe-telescope distance ``z_p``?" by measuring, on synthetic data with
known ground truth, how the pose resolution ``(σ_x, σ_y, σ_z)`` scales with the
inlier count ``N`` and ``z_p`` (DESIGN.md §8.6).

For each ``z_p`` the expensive decode is done **once** — generate a large
synthetic acquisition, stream it through stages 1-3, and decode every
coincidence to a :class:`~monrad.pose.Coincidence`.  The cheap part (the pose
fit) is then repeated over random subsamples of size ``N`` to build the σ(N)
curve, validate the covariance calibration via a pull test, and invert the
``σ_eff/√N`` law for the inlier budget at a sub-mm target.

Two of the DESIGN.md §10 "still to implement" diagnostic plots ride along (the
module already stands up the matplotlib output and the residual substrate):
the χ²(θ) curve (§8.4 — the pose-fit consistency check) and the probe-plane
residual histograms (§8.7), each emitted for one full-statistics reference fit
per ``z_p``.

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
from ..coincidence import coincidence_stream
from ..pose import Coincidence, PoseFitter, PoseResult, fit_probe_pose
from ..synthetic import generate
from ..synthetic.generate import STRIP_MM, Z_TEL
from ..timing import reconstruct_stream
from .io import load_detector

# Telescope angular resolution used in the DESIGN §8.6 σ_eff overlay:
# σ_eff² = σ_strip² + (TEL_ANG_SIGMA·z_p)².  3 mrad = σ_strip·√(2/3)/800 mm.
TEL_ANG_SIGMA = 3.0e-3  # rad
SIGMA_STRIP = STRIP_MM / math.sqrt(12.0)  # ≈ 2.9 mm single-strip resolution
AXES = ("x", "y", "z")  # t_x, t_y, z_p — the three position components
DEFAULT_TARGETS = (0.3, 1.0)  # mm — sub-mm headline target, 1 mm for reference


# ── Coincidence decoding (the expensive part — done once per z_p) ────────────


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

    fitter = PoseFitter(
        tel_z=z_tel,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel.pos_paths,
        prb_pos_paths=prb.pos_paths,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    )
    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)

    coincs: list[Coincidence] = []
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        co = fitter.decode_cluster(cluster)
        if co is not None:
            coincs.append(co)
    return coincs, info


# ── Sweep records ────────────────────────────────────────────────────────────


@dataclass
class CellResult:
    """Aggregated statistics for one (z_p, N) cell, per position axis."""

    z_p: float
    n: int
    n_repeats: int
    pool: int
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
class ZpResult:
    """All sweep output for one z_p: per-N cells plus the reference fit."""

    z_p: float
    pool: int
    truth: tuple[float, float, float, float]  # (t_x, t_y, theta, z_p)
    cells: list[CellResult]
    ref_pose: PoseResult  # full-statistics fit, for the §10 diagnostic plots


def _fit_subsample(
    sub: list[Coincidence],
    z_corr: np.ndarray,
    alignment: AlignmentCorrection,
    truth: tuple[float, float, float, float],
) -> dict[str, tuple[float, float, float]]:
    """Fit one subsample and return per-axis (error, sigma_cov, corr_with_zp).

    The probe pose is unambiguous in θ (see the module docstring), so the
    fitted parameters are scored directly against ground truth.
    """
    pose = fit_probe_pose(sub, z_corr, alignment)
    t_x, t_y, _theta, z_p = truth
    cov = pose.cov
    err = {"x": pose.t_x - t_x, "y": pose.t_y - t_y, "z": pose.z_p - z_p}
    sig = {
        "x": math.sqrt(abs(cov[0, 0])),
        "y": math.sqrt(abs(cov[1, 1])),
        "z": math.sqrt(abs(cov[3, 3])),
    }
    sz = sig["z"]
    corr = {
        "x": cov[0, 3] / (sig["x"] * sz) if sig["x"] > 0 and sz > 0 else 0.0,
        "y": cov[1, 3] / (sig["y"] * sz) if sig["y"] > 0 and sz > 0 else 0.0,
        "z": 1.0,
    }
    return {a: (err[a], sig[a], corr[a]) for a in AXES}


def sweep_one_zp(
    work_dir: Path,
    *,
    z_p: float,
    n_grid: list[int],
    n_repeats: int,
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
    rng: np.random.Generator,
) -> ZpResult:
    """Decode one z_p once, then fit σ(N) over random subsamples of each N."""
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

    if pool < 3:
        raise RuntimeError(
            f"z_p={z_p:g} mm: only {pool} coincidences decoded "
            f"(need >= 3); raise --n-tracks or move the probe under the telescope."
        )

    # Reference full-statistics fit, reused for the §10 χ²(θ) and residual plots.
    ref_pose = fit_probe_pose(coincs, z_corr, alignment)

    cells: list[CellResult] = []
    for n in n_grid:
        if n > pool:
            print(
                f"  z_p={z_p:6g} mm  N={n:<5}  SKIP — only {pool} coincidences "
                f"in pool (need N <= pool)"
            )
            continue
        errs = {a: [] for a in AXES}
        sigs = {a: [] for a in AXES}
        corrs = {a: [] for a in AXES}
        for _ in range(n_repeats):
            idx = rng.choice(pool, size=n, replace=False)
            sub = [coincs[i] for i in idx]
            per_axis = _fit_subsample(sub, z_corr, alignment, truth)
            for a in AXES:
                e, s, c = per_axis[a]
                errs[a].append(e)
                sigs[a].append(s)
                corrs[a].append(c)

        cell = CellResult(z_p=z_p, n=n, n_repeats=n_repeats, pool=pool)
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
            f"  z_p={z_p:6g} mm  N={n:<5} pool={pool:<5} "
            f"σ_cov(x,y,z)=({cell.sigma_cov['x']:.3f},{cell.sigma_cov['y']:.3f},"
            f"{cell.sigma_cov['z']:.3f}) mm  "
            f"pull_std(z)={cell.pull_std['z']:.2f}"
        )

    return ZpResult(z_p=z_p, pool=pool, truth=truth, cells=cells, ref_pose=ref_pose)


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


def design_sigma_eff(z_p: float) -> float:
    """DESIGN.md §8.6 overlay: σ_eff = √(σ_strip² + (3 mrad·z_p)²)."""
    return math.sqrt(SIGMA_STRIP**2 + (TEL_ANG_SIGMA * z_p) ** 2)


# ── CSV output ───────────────────────────────────────────────────────────────


def write_sweep_csv(path: Path, results: list[ZpResult]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "z_p",
                "N",
                "axis",
                "pool",
                "sigma_cov",
                "sigma_emp",
                "err_mean",
                "pull_mean",
                "pull_std",
                "corr_t_zp",
            ]
        )
        for zr in results:
            for cell in zr.cells:
                for a in AXES:
                    w.writerow(
                        [
                            f"{cell.z_p:g}",
                            cell.n,
                            a,
                            cell.pool,
                            f"{cell.sigma_cov[a]:.6g}",
                            f"{cell.sigma_emp[a]:.6g}",
                            f"{cell.err_mean[a]:.6g}",
                            f"{cell.pull_mean[a]:.6g}",
                            f"{cell.pull_std[a]:.6g}",
                            f"{cell.corr_t_zp[a]:.6g}",
                        ]
                    )


def write_n_required_csv(
    path: Path,
    results: list[ZpResult],
    targets: tuple[float, ...],
) -> dict[tuple[float, str], float]:
    """Write n_required.csv and return {(z_p, axis): sigma_eff} for plotting."""
    sigma_eff_map: dict[tuple[float, str], float] = {}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z_p", "axis", "sigma_eff", "target_sigma", "N_required"])
        for zr in results:
            for a in AXES:
                seff = fit_sigma_eff(zr.cells, a)
                sigma_eff_map[(zr.z_p, a)] = seff
                for tgt in targets:
                    w.writerow(
                        [
                            f"{zr.z_p:g}",
                            a,
                            f"{seff:.6g}",
                            f"{tgt:g}",
                            f"{n_required(seff, tgt):.6g}",
                        ]
                    )
    return sigma_eff_map


# ── Plots ────────────────────────────────────────────────────────────────────


def _plot_sigma_vs_n(results: list[ZpResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ai, a in enumerate(AXES):
        ax = axs[ai]
        for zr in results:
            ns = [c.n for c in zr.cells]
            if not ns:
                continue
            sig = [c.sigma_cov[a] for c in zr.cells]
            line = ax.plot(ns, sig, "o", label=f"z_p={zr.z_p:g} mm")[0]
            seff = fit_sigma_eff(zr.cells, a)
            n_line = np.array(sorted(ns))
            ax.plot(
                n_line, seff / np.sqrt(n_line), "-", color=line.get_color(), alpha=0.6
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (inliers)")
        ax.set_ylabel(f"σ_{a}  [mm]  (cov)")
        ax.set_title(f"σ_{a} vs N  (lines: σ_eff/√N fit)")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_sigma_eff_vs_zp(
    results: list[ZpResult],
    sigma_eff_map: dict[tuple[float, str], float],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    zps = sorted({zr.z_p for zr in results})
    for a in AXES:
        seff = [sigma_eff_map.get((z, a), float("nan")) for z in zps]
        ax.plot(zps, seff, "o-", label=f"σ_eff,{a} (measured)")
    z_fine = np.linspace(min(zps), max(zps), 100) if zps else np.array([0.0])
    ax.plot(
        z_fine,
        [design_sigma_eff(z) for z in z_fine],
        "k--",
        label="DESIGN §8.6: √(σ_strip²+(3mrad·z_p)²)",
    )
    ax.set_xlabel("z_p  [mm]")
    ax.set_ylabel("σ_eff  [mm]")
    ax.set_title("Effective single-coincidence resolution vs distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_n_required_vs_zp(
    sigma_eff_map: dict[tuple[float, str], float],
    results: list[ZpResult],
    targets: tuple[float, ...],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    zps = sorted({zr.z_p for zr in results})
    for tgt in targets:
        # average N_required over the two in-plane axes (x, y)
        nreq = []
        for z in zps:
            vals = [
                n_required(sigma_eff_map.get((z, a), float("nan")), tgt)
                for a in ("x", "y")
            ]
            nreq.append(float(np.nanmean(vals)))
        ax.plot(zps, nreq, "o-", label=f"target σ_t = {tgt:g} mm")
    ax.set_xlabel("z_p  [mm]")
    ax.set_ylabel("N_required (mean of x,y)")
    ax.set_yscale("log")
    ax.set_title("Inlier budget for a sub-mm in-plane fix vs distance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_pull_hist(results: list[ZpResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    grid = np.linspace(-4, 4, 200)
    gauss = np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi)
    for ai, a in enumerate(AXES):
        ax = axs[ai]
        allp = np.concatenate(
            [c.pulls[a] for zr in results for c in zr.cells if a in c.pulls]
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


def _plot_chi2_theta(zr: ZpResult, path: Path) -> None:
    """DESIGN §8.4/§10: χ²(θ) consistency check — single sharp minimum at θ_true.

    Plotted on a log-χ² axis so the global minimum is legible against the much
    larger off-orientation χ².  With the ``(u, v)`` channels decoded and held
    fixed, the θ scan is unambiguous (the θ±90° hypotheses would also need an
    axis relabeling, DESIGN §8.5): a healthy fit shows one deep well at the true
    mounting orientation.  A global minimum away from the expected orientation,
    or competing wells of comparable depth, flags a wiring or axis problem.
    """
    import matplotlib.pyplot as plt

    curve = zr.ref_pose.chi2_curve
    theta_true = math.degrees(zr.truth[2])
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
    ax.set_title(f"χ²(θ) consistency check at z_p={zr.z_p:g} mm (DESIGN §8.4)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_residuals(zr: ZpResult, path: Path) -> None:
    """DESIGN §8.7/§10: probe-plane x/y inlier residual histograms."""
    import matplotlib.pyplot as plt

    pose = zr.ref_pose
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
    fig.suptitle(f"Probe-plane residuals at z_p={zr.z_p:g} mm  (n={pose.n_inliers})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_plots(
    results: list[ZpResult],
    sigma_eff_map: dict[tuple[float, str], float],
    targets: tuple[float, ...],
    out_dir: Path,
) -> None:
    """Write every sweep + per-z_p diagnostic plot to out_dir."""
    _plot_sigma_vs_n(results, out_dir / "sigma_vs_N.png")
    _plot_sigma_eff_vs_zp(results, sigma_eff_map, out_dir / "sigma_eff_vs_zp.png")
    _plot_n_required_vs_zp(
        sigma_eff_map, results, targets, out_dir / "n_required_vs_zp.png"
    )
    _plot_pull_hist(results, out_dir / "pull_hist.png")
    for zr in results:
        tag = f"{zr.z_p:g}".replace("-", "m").replace(".", "p")
        _plot_chi2_theta(zr, out_dir / f"chi2_theta_z{tag}.png")
        _plot_residuals(zr, out_dir / f"residuals_z{tag}.png")


# ── Driver ───────────────────────────────────────────────────────────────────


def run_resolution_study(
    out_dir: Path,
    *,
    z_p_grid: list[float],
    n_grid: list[int],
    n_repeats: int,
    n_tracks: int,
    seed: int,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
    t_x: float = 50.0,
    t_y: float = -30.0,
    theta: float = 0.29671,
    n_probe_ch: int = 30,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> list[ZpResult]:
    """Run the full σ(N, z_p) sweep and write CSVs + plots to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_tel = np.array(Z_TEL)
    # Synthetic data is generated perfectly aligned (no plane offsets), so the
    # identity correction is exactly right and avoids an extra telescope pass.
    alignment = AlignmentCorrection.identity()
    rng = np.random.default_rng(seed)

    results: list[ZpResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, z_p in enumerate(z_p_grid):
            print(f"z_p = {z_p:g} mm  (generating {n_tracks} tracks)…")
            work = Path(tmp) / f"zp_{i}"
            zr = sweep_one_zp(
                work,
                z_p=z_p,
                n_grid=n_grid,
                n_repeats=n_repeats,
                n_tracks=n_tracks,
                seed=seed + i,
                t_x=t_x,
                t_y=t_y,
                theta=theta,
                n_probe_ch=n_probe_ch,
                z_tel=z_tel,
                alignment=alignment,
                tot_thresh=tot_thresh,
                tot_weights=tot_weights,
                rng=rng,
            )
            results.append(zr)

    write_sweep_csv(out_dir / "resolution_sweep.csv", results)
    sigma_eff_map = write_n_required_csv(out_dir / "n_required.csv", results, targets)
    if make_plots:
        write_plots(results, sigma_eff_map, targets, out_dir)

    print(f"\nWrote resolution study to {out_dir}")
    for tgt in targets:
        print(f"  N_required (mean x,y) for σ_t ≤ {tgt:g} mm:")
        for zr in results:
            vals = [
                n_required(sigma_eff_map.get((zr.z_p, a), float("nan")), tgt)
                for a in ("x", "y")
            ]
            print(f"    z_p={zr.z_p:7g} mm → N ≈ {np.nanmean(vals):.0f}")
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="monrad-resolution",
        description="Characterize probe pose resolution σ(N, z_p) on synthetic data.",
    )
    p.add_argument(
        "--z",
        nargs="+",
        type=float,
        default=[0.0, 300.0, 1000.0, 5000.0],
        metavar="Z_P",
        help="Probe-telescope distances z_p in mm (default: 0 300 1000 5000)",
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
        help="Random subsamples per (z_p, N) cell (default: 50)",
    )
    p.add_argument(
        "--n-tracks",
        type=int,
        default=60000,
        help="Synthetic tracks generated per z_p (default: 60000). Must be large "
        "enough that decoded coincidences exceed max(--n), especially at large z_p.",
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
        n_grid=args.n,
        n_repeats=args.repeats,
        n_tracks=args.n_tracks,
        seed=args.seed,
        targets=tuple(args.targets),
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
