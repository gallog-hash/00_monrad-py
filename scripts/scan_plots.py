"""Matplotlib figures for ``scripts/scan_geometric_cuts.py``.

Two families:

*Cache figures* read a Tier-A ``ClusterCache`` directly and describe the
population the cuts act on -- where the winning triples' chi2 and per-plane mm
residuals actually sit, and how they relate to cluster width.  They are what
show the two candidate cut *shapes* (sigma-adaptive chi2 vs absolute mm) side
by side.

*Grid figures* read ``scan_grid_<label>.csv`` and describe what each cut
setting bought.  ``sigma_vs_n.png`` is the decision plot: real gain moves
down-and-right along a 1/sqrt(N) reference, junk moves up-and-right.

Follows the repo's plotting convention (``monrad.monitor.resolution``):
``Agg`` backend, ``dpi=110``, one figure per file, callers honour ``--no-plots``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 110

# Reference lines drawn on the residual/chi2 figures:
#   4.71 mm  -- the max residual chi2=4.0 corresponds to at cluster width 1
#               (memory `chi2-track-cut-in-mm`)
#   14.29 mm -- MATLAB's absolute ALIGNDIST
#   chi2=37  -- the sigma-adaptive threshold that matches ALIGNDIST at width 1
CHI2_EQUIV_MATLAB = 37.0
RESID_CHI2_4 = 4.71
RESID_ALIGNDIST = 14.29


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


# ── Cache figures ─────────────────────────────────────────────────────────


def chi2_hist(cache, path: Path, chi2_grid=(4.0, 37.0)) -> Path:
    """Winning-triple chi2 (log-log), stacked by how many planes were anchors.

    An "anchor" is a plane that decoded to a single candidate.  Splitting by
    anchor count separates the clean 3-anchor population from the searched
    ambiguous ones, which is where a looser chi2 would actually add events --
    and where pile-up can fabricate a low-chi2 track.
    """
    chi2 = cache.chi2
    n_anchor = np.count_nonzero(cache.cand_counts == 1, axis=1)
    ok = np.isfinite(chi2) & (chi2 > 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if np.any(ok):
        lo = max(chi2[ok].min(), 1e-3)
        bins = np.logspace(math.log10(lo), math.log10(chi2[ok].max()), 80)
        groups = [chi2[ok & (n_anchor == k)] for k in range(4)]
        ax.hist(
            groups,
            bins=bins,
            stacked=True,
            label=[f"{k} anchor plane(s)" for k in range(4)],
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    for c in sorted({*chi2_grid, CHI2_EQUIV_MATLAB}):
        ax.axvline(
            c,
            color="k",
            ls="--" if c != CHI2_EQUIV_MATLAB else ":",
            lw=0.8,
            alpha=0.7,
        )
        ax.text(c, ax.get_ylim()[1], f" {c:g}", va="top", fontsize=7, rotation=90)
    ax.set_xlabel("best-triple chi2")
    ax.set_ylabel("clusters")
    ax.set_title("Telescope track chi2 by anchor count (dotted = MATLAB-equivalent)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def chi2_vs_width(cache, path: Path) -> Path:
    """chi2 against the winning triple's largest cluster width, both cuts drawn.

    The plot that shows the two cut shapes directly.  A fixed chi2 is a
    *sigma-adaptive* mm cut -- because each plane's sigma is
    ``10 mm * width / sqrt(12)``, the same chi2 tolerates a wider absolute
    residual as the cluster widens.  An absolute-mm cut is a horizontal line in
    residual space and therefore a falling curve here; where the two cross is
    the width at which they agree.

    The overlaid mm curves are the chi2 that a *single* plane-axis sitting
    exactly at ``|r| = R`` would contribute, ``(R / sigma)^2`` -- an indicative
    boundary, not the exact cut.  The plotted chi2 is the sum over all three
    planes and both axes, so a cluster with several planes off-track sits above
    its curve.  The exact absolute-mm cut is what ``replay(max_resid_mm=...)``
    applies, and its measured yield is in the B2 rows of the grid CSV.
    """
    width = cache.max_cluster_width
    chi2 = cache.chi2
    ok = np.isfinite(chi2) & np.isfinite(width) & (chi2 > 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if np.any(ok):
        hb = ax.hexbin(
            width[ok],
            chi2[ok],
            yscale="log",
            gridsize=40,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        fig.colorbar(hb, ax=ax, label="clusters (log)")
    ax.axhline(4.0, color="tab:red", ls="--", lw=1.2, label="chi2 = 4 (shipped)")
    ax.axhline(
        CHI2_EQUIV_MATLAB,
        color="tab:orange",
        ls="--",
        lw=1.2,
        label=f"chi2 = {CHI2_EQUIV_MATLAB:g}",
    )
    # An absolute |r| <= R cut, expressed as the chi2 it corresponds to at each
    # width: chi2 ~ (r/sigma)^2 with sigma = STRIP_MM*width/sqrt(12).
    w = np.linspace(1, max(4.0, float(np.nanmax(width[ok])) if np.any(ok) else 4.0), 50)
    for R, style in ((RESID_CHI2_4, "-"), (RESID_ALIGNDIST, "-.")):
        ax.plot(
            w,
            (R / (10.0 * w / math.sqrt(12.0))) ** 2,
            style,
            color="k",
            lw=1.0,
            label=f"single-plane |r| = {R:g} mm",
        )
    ax.set_xlabel("max cluster width of the winning triple (channels)")
    ax.set_ylabel("best-triple chi2")
    ax.set_title("Cut shapes: sigma-adaptive chi2 vs absolute mm")
    ax.legend(fontsize=8)
    return _save(fig, path)


def residual_hist(cache, path: Path, middle_plane: int = 2) -> Path:
    """Per-plane fit residual (mm): the middle plane against the outer two.

    ``middle_plane`` defaults to index 2 -- on ``testLab_20210723`` the
    telescope's file columns are *not* in z order (``--z-tel 0 -1340 -670``), so
    the geometrically middle plane is the **third** column.  The middle plane's
    sigma carries about 4x the weight in the line fit, so it is where a
    mis-shaped cut bites first.
    """
    r = np.abs(cache.resid)  # (N, 3, 2)
    outer = [k for k in range(3) if k != middle_plane]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, 40, 80)
    ax.hist(
        _finite(r[:, middle_plane, :].ravel()),
        bins=bins,
        histtype="step",
        lw=1.6,
        label=f"middle plane (col {middle_plane})",
    )
    ax.hist(
        _finite(r[:, outer, :].ravel()),
        bins=bins,
        histtype="step",
        lw=1.2,
        label=f"outer planes (col {outer[0]}, {outer[1]})",
    )
    for R, lbl in ((RESID_CHI2_4, "chi2=4 @ width 1"), (RESID_ALIGNDIST, "ALIGNDIST")):
        ax.axvline(R, color="k", ls="--", lw=0.9)
        ax.text(R, ax.get_ylim()[1], f" {lbl}", va="top", fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel("|per-plane fit residual| (mm)")
    ax.set_ylabel("plane-axis entries")
    ax.set_title("Telescope line-fit residuals")
    ax.legend(fontsize=8)
    return _save(fig, path)


def footprint(pose, path: Path, probe_size_mm: float, title: str = "") -> Path:
    """Probe-frame landing points of the fitted tracks, with the footprint box.

    Inliers and Mahalanobis outliers are drawn separately: a healthy
    configuration puts the inliers inside the box and leaves only a thin flat
    pedestal outside it.  A configuration that bought yield with junk shows a
    pedestal comparable to the in-box population -- the same quantity
    ``on_probe_purity`` estimates numerically.
    """
    import scan_geometric_cuts as scan

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for group, label in (
        (pose.inliers, "inliers"),
        (pose.outliers, "Mahalanobis outliers"),
    ):
        if group:
            u, v = scan.probe_frame_coords(group, pose)
            ax.scatter(u, v, s=6, alpha=0.5, label=label)
    L = probe_size_mm
    ax.add_patch(
        plt.Rectangle((0, 0), L, L, fill=False, ec="k", lw=1.5, label="probe footprint")
    )
    ax.set_xlim(-0.75 * L, 1.75 * L)
    ax.set_ylim(-0.75 * L, 1.75 * L)
    ax.set_aspect("equal")
    ax.set_xlabel("probe u (mm)")
    ax.set_ylabel("probe v (mm)")
    ax.set_title(title or "Track landing points in the probe frame")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, path)


# ── Grid figures ──────────────────────────────────────────────────────────


def load_grid(path: Path) -> list[dict]:
    """Read ``scan_grid_<label>.csv`` back into typed rows."""
    rows: list[dict] = []
    with Path(path).open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict = {}
            for k, v in raw.items():
                if v == "":
                    row[k] = None
                elif v in ("True", "False"):
                    row[k] = v == "True"
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def _stage(rows, name):
    return [r for r in rows if r.get("stage") == name]


def funnel_bars(rows, path: Path, gate_order) -> Path:
    """Stacked gate funnel, one bar per B1 grid point."""
    rows = _stage(rows, "B1")
    fig, ax = plt.subplots(figsize=(max(7.0, 0.32 * len(rows)), 4.6))
    if rows:
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for gate in (*gate_order, "accepted"):
            key = "n_accepted" if gate == "accepted" else f"gate_{gate}"
            vals = np.array([r.get(key) or 0.0 for r in rows])
            ax.bar(x, vals, bottom=bottom, label=gate)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['chi2_track']:g}/{int(r['min_anchor_planes'])}" for r in rows],
            rotation=90,
            fontsize=6,
        )
    ax.set_xlabel("chi2_track / min_anchor_planes")
    ax.set_ylabel("clusters")
    ax.set_title("Gate funnel per configuration")
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, path)


def yield_vs_chi2(rows, path: Path) -> Path:
    """Acceptance and inlier count vs chi2_track, one curve per anchor setting."""
    rows = _stage(rows, "B1")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax.twinx()
    for anchor in sorted({r["min_anchor_planes"] for r in rows}):
        sel = sorted(
            (r for r in rows if r["min_anchor_planes"] == anchor),
            key=lambda r: r["chi2_track"],
        )
        c = [r["chi2_track"] for r in sel]
        ax.plot(c, [r["n_inliers"] for r in sel], "o-", label=f"anchor >= {anchor:g}")
        ax2.plot(c, [r["sigma_zp"] for r in sel], "s--", alpha=0.45)
    ax.axvline(4.0, color="k", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("chi2_track")
    ax.set_ylabel("stage-5 inliers")
    ax2.set_ylabel("sigma_zp (mm, dashed)")
    ax.set_title("Yield vs chi2_track (dotted = shipped default 4.0)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def sigma_vs_n(rows, path: Path) -> Path:
    """**The decision plot**: sigma_zp against inlier count, with 1/sqrt(N).

    Genuine events added by a looser cut fall on the 1/sqrt(N) reference --
    down and to the right.  Junk drags the fit and moves the point *up* and to
    the right even as N grows.  Equivalently: ``sigma_zp * sqrt(N)`` stays flat
    for real gain, and rises for contamination.
    """
    pts = [
        r
        for r in rows
        if r.get("n_inliers")
        and r.get("sigma_zp") is not None
        and math.isfinite(r["sigma_zp"])
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if pts:
        n = np.array([r["n_inliers"] for r in pts], dtype=float)
        s = np.array([r["sigma_zp"] for r in pts], dtype=float)
        stages = [r.get("stage", "?") for r in pts]
        for st in sorted(set(stages)):
            m = np.array([x == st for x in stages])
            ax.scatter(n[m], s[m], s=18, alpha=0.7, label=st)
        # 1/sqrt(N) reference anchored on the best-resolution point.
        i = int(np.argmin(s))
        ref_n = np.linspace(n.min(), n.max(), 100)
        ax.plot(
            ref_n,
            s[i] * np.sqrt(n[i] / ref_n),
            "k--",
            lw=1.0,
            label="1/sqrt(N) reference",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("stage-5 inliers")
    ax.set_ylabel("sigma_zp (mm)")
    ax.set_title("Resolution vs yield: below the line = real gain, above = junk")
    ax.legend(fontsize=8)
    return _save(fig, path)


def pose_stability(rows, path: Path) -> Path:
    """Fitted t_x, t_y, theta, z_p (with 1-sigma bars) against chi2_track."""
    rows = sorted(_stage(rows, "B1"), key=lambda r: r["chi2_track"])
    specs = [
        ("t_x", "sigma_tx", "t_x (mm)"),
        ("t_y", "sigma_ty", "t_y (mm)"),
        ("theta_deg", None, "theta (deg)"),
        ("z_p", "sigma_zp", "z_p (mm)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.5), sharex=True)
    for ax, (key, skey, label) in zip(axes, specs):
        for anchor in sorted({r["min_anchor_planes"] for r in rows}):
            sel = [r for r in rows if r["min_anchor_planes"] == anchor]
            x = [r["chi2_track"] for r in sel]
            y = [r[key] for r in sel]
            err = [r[skey] for r in sel] if skey else None
            ax.errorbar(
                x, y, yerr=err, fmt="o-", capsize=2, ms=3, label=f"anchor >= {anchor:g}"
            )
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.axvline(4.0, color="k", ls=":", lw=0.9)
    axes[-1].set_xlabel("chi2_track")
    axes[0].set_title("Pose stability across the cut scan")
    axes[0].legend(fontsize=8)
    return _save(fig, path)


def purity_vs_chi2(rows, path: Path) -> Path:
    """Estimated on-probe purity and signal count against chi2_track."""
    rows = sorted(_stage(rows, "B1"), key=lambda r: r["chi2_track"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax.twinx()
    for anchor in sorted({r["min_anchor_planes"] for r in rows}):
        sel = [r for r in rows if r["min_anchor_planes"] == anchor]
        x = [r["chi2_track"] for r in sel]
        ax.plot(x, [r["purity"] for r in sel], "o-", label=f"anchor >= {anchor:g}")
        ax2.plot(x, [r["signal_count"] for r in sel], "s--", alpha=0.45)
    ax.set_xscale("log")
    ax.set_xlabel("chi2_track")
    ax.set_ylabel("estimated on-probe purity")
    ax2.set_ylabel("estimated signal count (dashed)")
    ax.set_title("Purity and signal count vs chi2_track")
    ax.legend(fontsize=8)
    return _save(fig, path)


def heatmap_chi2_anchor(rows, path: Path) -> Path:
    """n_inliers / purity / sigma_zp over the (chi2_track x anchor) grid."""
    rows = _stage(rows, "B1")
    chi2s = sorted({r["chi2_track"] for r in rows})
    anchors = sorted({r["min_anchor_planes"] for r in rows})
    lookup = {(r["chi2_track"], r["min_anchor_planes"]): r for r in rows}
    metrics = [("n_inliers", "inliers"), ("purity", "purity"), ("sigma_zp", "sigma_zp")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    for ax, (key, label) in zip(axes, metrics):
        grid = np.array(
            [
                [
                    # `or np.nan` would swallow a legitimate zero (e.g. a
                    # configuration that yielded no inliers at all).
                    float(v)
                    if (v := lookup.get((c, a), {}).get(key)) is not None
                    else np.nan
                    for c in chi2s
                ]
                for a in anchors
            ],
            dtype=float,
        )
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(chi2s)))
        ax.set_xticklabels([f"{c:g}" for c in chi2s], rotation=90, fontsize=7)
        ax.set_yticks(range(len(anchors)))
        ax.set_yticklabels([f"{a:g}" for a in anchors], fontsize=7)
        ax.set_xlabel("chi2_track")
        ax.set_ylabel("min_anchor_planes")
        ax.set_title(label, fontsize=9)
        fig.colorbar(im, ax=ax)
    return _save(fig, path)


def anomaly_bins(series: dict[str, list[dict]], path: Path) -> Path:
    """Per-time-bin z_p, resid_rms and inlier fraction, one line per config.

    The ANOMALY veto: on ``testLab_20210723`` the 17:15 and 18:10 UTC bins must
    remain visible outliers.  A configuration that smooths them into the rest of
    the window is disqualified no matter what it did for yield.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)
    for label, rows in series.items():
        if not rows:
            continue
        t0 = min(r["bin_start_ns"] for r in rows)
        x = [(r["bin_start_ns"] - t0) / 60e9 for r in rows]  # minutes
        for ax, key in zip(axes, ("z_p", "resid_rms", "inlier_frac")):
            ax.plot(x, [r.get(key) for r in rows], "o-", ms=3, label=label)
    for ax, label in zip(axes, ("z_p (mm)", "resid RMS (mm)", "inlier fraction")):
        ax.set_ylabel(label)
    axes[-1].set_xlabel("minutes from window start")
    axes[0].set_title("Per-bin behaviour across the window")
    axes[0].legend(fontsize=7)
    return _save(fig, path)


# ── Driver ────────────────────────────────────────────────────────────────


def render_all(caches: dict, grid_csv: Path, out_dir: Path) -> list[Path]:
    """Render every figure the cache(s) and grid CSV alone can support.

    The footprint and anomaly-bin figures need a fitted pose / a per-bin replay,
    so the caller drives those directly (see :func:`footprint` and
    :func:`anomaly_bins`).
    """
    from monrad.pose import GATE_ORDER

    written: list[Path] = []
    for tag, cache in caches.items():
        written.append(chi2_hist(cache, out_dir / f"chi2_hist_{tag}.png"))
        written.append(chi2_vs_width(cache, out_dir / f"chi2_vs_width_{tag}.png"))
        written.append(residual_hist(cache, out_dir / f"residual_hist_{tag}.png"))
    if Path(grid_csv).exists():
        rows = load_grid(grid_csv)
        written.append(funnel_bars(rows, out_dir / "funnel_bars.png", GATE_ORDER))
        written.append(yield_vs_chi2(rows, out_dir / "yield_vs_chi2.png"))
        written.append(sigma_vs_n(rows, out_dir / "sigma_vs_n.png"))
        written.append(pose_stability(rows, out_dir / "pose_stability.png"))
        written.append(purity_vs_chi2(rows, out_dir / "purity_vs_chi2.png"))
        written.append(heatmap_chi2_anchor(rows, out_dir / "heatmap_chi2_anchor.png"))
    return written
