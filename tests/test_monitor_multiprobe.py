"""Tests for monrad.monitor.multiprobe (monitoring Step 3).

Two synthetic probes, generated with the same seed/n_tracks (hence
byte-identical telescope tracks — see generate()'s module docstring for why
golden, non-folded encoding draws no further rng state) but distinct poses
and channel counts, sharing one telescope acquisition.  Each probe's
recovered timeseries is checked against its own truth independently.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from monrad.alignment import AlignmentCorrection, PlaneCorrection, save_alignment
from monrad.monitor.multiprobe import _parse_args, monitor_probes
from monrad.pose import PoseFitter
from monrad.synthetic.generate import Z_TEL, generate

# (t_x, t_y, theta, z_p, n_probe_ch) truth per probe.
_PROBE1_ZP = 300.0
_PROBE2_ZP = 500.0


@pytest.fixture(scope="module")
def multiprobe_run(tmp_path_factory):
    """Run monitor_probes on two synthetic probes sharing one telescope acquisition."""
    src1 = tmp_path_factory.mktemp("mp_src1")
    src2 = tmp_path_factory.mktemp("mp_src2")
    info1 = generate(
        src1,
        n_tracks=5000,
        seed=42,
        t_x=300.0,
        t_y=350.0,
        theta=0.29671,
        z_p=_PROBE1_ZP,
        n_probe_ch=30,
    )
    info2 = generate(
        src2,
        n_tracks=5000,
        seed=42,
        t_x=500.0,
        t_y=150.0,
        theta=-0.29671,
        z_p=_PROBE2_ZP,
        n_probe_ch=40,
    )

    out = tmp_path_factory.mktemp("mp_out")
    all_results = monitor_probes(
        info1["tel_dir"],
        [info1["probe_dir"], info2["probe_dir"]],
        window_s=150.0,
        z_tel=np.array(Z_TEL, dtype=float),
        n_probe_ch=[30, 40],
        out_dir=out,
        make_plots=True,
    )
    return all_results, out, info1, info2


def test_multiprobe_returns_one_list_per_probe(multiprobe_run):
    all_results, _, _, _ = multiprobe_run
    assert len(all_results) == 2


@pytest.mark.parametrize("idx,z_p_truth", [(0, _PROBE1_ZP), (1, _PROBE2_ZP)])
def test_multiprobe_recovers_each_zp_within_6sigma(multiprobe_run, idx, z_p_truth):
    """z_p is the best-constrained parameter; each probe's windows should be
    within a few sigma of that probe's own truth, independent of the other."""
    all_results, _, _, _ = multiprobe_run
    results = all_results[idx]
    assert results, f"expected at least one window for probe {idx + 1}"
    for r in results:
        assert abs(r.z_p - z_p_truth) < 6 * r.sigma_zp, (
            f"probe {idx + 1}: z_p={r.z_p:.2f} deviates >6σ from "
            f"{z_p_truth} (σ_zp={r.sigma_zp:.3f})"
        )


def test_multiprobe_windows_internally_ordered(multiprobe_run):
    all_results, _, _, _ = multiprobe_run
    for results in all_results:
        for i in range(1, len(results)):
            assert results[i].utc_start >= results[i - 1].utc_end


def test_multiprobe_sigmas_finite_positive(multiprobe_run):
    all_results, _, _, _ = multiprobe_run
    for results in all_results:
        for r in results:
            for s in (r.sigma_tx, r.sigma_ty, r.sigma_zp, r.sigma_theta):
                assert math.isfinite(s) and s > 0


def test_multiprobe_csv_written_per_probe(multiprobe_run):
    all_results, out, _, _ = multiprobe_run
    for k, results in enumerate(all_results, start=1):
        csv_path = out / f"pose_timeseries_probe{k}.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert len(lines) == len(results) + 1


def test_multiprobe_plot_written_per_probe(multiprobe_run):
    _, out, _, _ = multiprobe_run
    assert (out / "pose_timeseries_probe1.png").exists()
    assert (out / "pose_timeseries_probe2.png").exists()


# ── n_probe_ch broadcast / per-probe / mismatch (library-level) ─────────────


def test_monitor_probes_rejects_empty_probe_list(multiprobe_run):
    _, _, info1, _ = multiprobe_run
    with pytest.raises(ValueError):
        monitor_probes(
            info1["tel_dir"], [], z_tel=np.array(Z_TEL, dtype=float), make_plots=False
        )


def test_monitor_probes_rejects_n_probe_ch_length_mismatch(multiprobe_run):
    _, _, info1, info2 = multiprobe_run
    with pytest.raises(ValueError):
        monitor_probes(
            info1["tel_dir"],
            [info1["probe_dir"], info2["probe_dir"]],
            z_tel=np.array(Z_TEL, dtype=float),
            n_probe_ch=[30, 40, 50],
            make_plots=False,
        )


def test_monitor_probes_broadcasts_single_n_probe_ch(tmp_path_factory):
    """A single n_probe_ch value broadcasts to every probe rather than erroring."""
    src1 = tmp_path_factory.mktemp("mp_bcast_src1")
    src2 = tmp_path_factory.mktemp("mp_bcast_src2")
    info1 = generate(src1, n_tracks=200, seed=7, n_probe_ch=30)
    info2 = generate(src2, n_tracks=200, seed=7, n_probe_ch=30)
    all_results = monitor_probes(
        info1["tel_dir"],
        [info1["probe_dir"], info2["probe_dir"]],
        z_tel=np.array(Z_TEL, dtype=float),
        n_probe_ch=[30],
        min_fit=30,
        make_plots=False,
    )
    assert len(all_results) == 2


# ── fibers_per_ribbon (N) — per-probe, differing values (regression) ────────


def test_monitor_probes_rejects_fibers_per_ribbon_length_mismatch(multiprobe_run):
    _, _, info1, info2 = multiprobe_run
    with pytest.raises(ValueError):
        monitor_probes(
            info1["tel_dir"],
            [info1["probe_dir"], info2["probe_dir"]],
            z_tel=np.array(Z_TEL, dtype=float),
            fibers_per_ribbon=[10, 5, 3],
            make_plots=False,
        )


def test_monitor_probes_rejects_n_probe_ch_exceeding_fibers_per_ribbon_range(
    multiprobe_run,
):
    """n_probe_ch and fibers_per_ribbon must be mutually consistent per probe
    (probe 2: n_probe_ch=40 needs channels up to 39, but N=3 only covers
    0..29) — catches the class of misconfiguration where the two flags
    silently alias channels instead of erroring (see
    docs/handoffs/2026-07-10-fibers-per-ribbon-pr-review-findings.md #2)."""
    _, _, info1, info2 = multiprobe_run
    with pytest.raises(ValueError):
        monitor_probes(
            info1["tel_dir"],
            [info1["probe_dir"], info2["probe_dir"]],
            z_tel=np.array(Z_TEL, dtype=float),
            n_probe_ch=[30, 40],
            fibers_per_ribbon=[10, 3],
            make_plots=False,
        )


def test_monitor_probes_recovers_each_probe_with_distinct_fibers_per_ribbon(
    tmp_path_factory,
):
    """Two probes sharing one telescope acquisition, wired with different
    fiber×ribbon combine factors (N=10 and N=5), each recover their own z_p
    truth when monitor_probes is given the matching per-probe N — the
    motivating end-to-end scenario for n_fibers_per_ribbon."""
    z_p1, z_p2 = 300.0, 500.0
    src1 = tmp_path_factory.mktemp("mp_nfib_src1")
    src2 = tmp_path_factory.mktemp("mp_nfib_src2")
    info1 = generate(
        src1,
        n_tracks=5000,
        seed=42,
        t_x=300.0,
        t_y=350.0,
        theta=0.29671,
        z_p=z_p1,
        n_probe_ch=30,
        n_probe_fibers_per_ribbon=10,
    )
    info2 = generate(
        src2,
        n_tracks=5000,
        seed=42,
        t_x=500.0,
        t_y=150.0,
        theta=-0.29671,
        z_p=z_p2,
        n_probe_ch=30,
        n_probe_fibers_per_ribbon=5,
    )

    all_results = monitor_probes(
        info1["tel_dir"],
        [info1["probe_dir"], info2["probe_dir"]],
        window_s=150.0,
        z_tel=np.array(Z_TEL, dtype=float),
        n_probe_ch=[30, 30],
        fibers_per_ribbon=[10, 5],
        make_plots=False,
    )
    assert len(all_results) == 2
    for idx, z_p_truth in ((0, z_p1), (1, z_p2)):
        results = all_results[idx]
        assert results, f"expected at least one window for probe {idx + 1}"
        for r in results:
            assert abs(r.z_p - z_p_truth) < 6 * r.sigma_zp, (
                f"probe {idx + 1}: z_p={r.z_p:.2f} deviates >6sigma from "
                f"{z_p_truth} (sigma_zp={r.sigma_zp:.3f})"
            )


# ── CLI: repeatable --probe, --n-probe-ch broadcast / per-probe / mismatch ──


def test_cli_probe_flag_is_repeatable():
    args, _ = _parse_args(
        [
            "--telescope",
            "unused",
            "--probe",
            "unused1",
            "--probe",
            "unused2",
            "--z-tel",
            "0",
            "400",
            "800",
        ]
    )
    assert args.probe == [Path("unused1"), Path("unused2")]


def test_cli_single_n_probe_ch_value_accepted_for_two_probes():
    args, _ = _parse_args(
        [
            "--telescope",
            "unused",
            "--probe",
            "unused1",
            "--probe",
            "unused2",
            "--z-tel",
            "0",
            "400",
            "800",
            "--n-probe-ch",
            "30",
        ]
    )
    assert args.n_probe_ch == [30]


def test_cli_per_probe_n_probe_ch_values_accepted():
    args, _ = _parse_args(
        [
            "--telescope",
            "unused",
            "--probe",
            "unused1",
            "--probe",
            "unused2",
            "--z-tel",
            "0",
            "400",
            "800",
            "--n-probe-ch",
            "30",
            "40",
        ]
    )
    assert args.n_probe_ch == [30, 40]


def test_cli_n_probe_ch_count_matching_neither_rejected():
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--telescope",
                "unused",
                "--probe",
                "unused1",
                "--probe",
                "unused2",
                "--z-tel",
                "0",
                "--n-probe-ch",
                "30",
                "40",
                "50",
            ]
        )


def test_cli_single_fibers_per_ribbon_value_accepted_for_two_probes():
    args, _ = _parse_args(
        [
            "--telescope",
            "unused",
            "--probe",
            "unused1",
            "--probe",
            "unused2",
            "--z-tel",
            "0",
            "400",
            "800",
            "--fibers-per-ribbon",
            "5",
        ]
    )
    assert args.fibers_per_ribbon == [5]


def test_cli_per_probe_fibers_per_ribbon_values_accepted():
    args, _ = _parse_args(
        [
            "--telescope",
            "unused",
            "--probe",
            "unused1",
            "--probe",
            "unused2",
            "--z-tel",
            "0",
            "400",
            "800",
            "--fibers-per-ribbon",
            "10",
            "5",
        ]
    )
    assert args.fibers_per_ribbon == [10, 5]


def test_cli_fibers_per_ribbon_count_matching_neither_rejected():
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--telescope",
                "unused",
                "--probe",
                "unused1",
                "--probe",
                "unused2",
                "--z-tel",
                "0",
                "--fibers-per-ribbon",
                "10",
                "5",
                "3",
            ]
        )


def test_cli_min_fit_below_floor_rejected():
    from monrad.pose import _MIN_COINCS

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


@pytest.mark.parametrize("bad_n", [0, 11])
def test_cli_fibers_per_ribbon_out_of_range_rejected(bad_n):
    """--fibers-per-ribbon outside 1..10 errors at parse time, not with an
    uncaught ZeroDivisionError deep in split_channel (see
    docs/handoffs/2026-07-10-fibers-per-ribbon-pr-review-findings.md #3)."""
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--telescope",
                "unused",
                "--probe",
                "unused",
                "--z-tel",
                "0",
                "--fibers-per-ribbon",
                str(bad_n),
            ]
        )


# ── Time-varying alignment (shared-search identity across a mid-stream switch) ─

_Z_MP = np.array(Z_TEL, dtype=float)


def _mp_corr(delta_x: float) -> AlignmentCorrection:
    planes = [
        PlaneCorrection(delta_x, 0.0, 0.0, 0.0, 0.0, 0.0),
        PlaneCorrection(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        PlaneCorrection(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    return AlignmentCorrection(planes, False)


def _mp_write_window(dir_: Path, label: str, corr: AlignmentCorrection) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    save_alignment(
        corr,
        dir_ / f"alignment_{label}.json",
        date=label,
        z_tel=_Z_MP,
        files=[f"{label}.bin"],
        n_events=100,
    )


def test_multiprobe_directory_switch_preserves_identity(
    multiprobe_run, tmp_path, monkeypatch
):
    """A mid-stream alignment switch applies the *same* correction object to
    every probe's fitter, so the shared-telescope-search identity invariant
    (all fitters share one AlignmentCorrection) still holds, and the run still
    produces per-probe windows."""
    all_results, _, info1, info2 = multiprobe_run
    base = all_results[0]
    first, last = base[0].utc_start, base[-1].utc_end
    mid = first + (last - first) / 2
    label0 = first.strftime("%Y%m%d_%H%M%S")
    label1 = mid.strftime("%Y%m%d_%H%M%S")
    assert label0 != label1
    adir = tmp_path / "mp_sched"
    _mp_write_window(adir, label0, _mp_corr(0.0))
    _mp_write_window(adir, label1, _mp_corr(5.0))  # distinct second window

    switches: list[tuple[int, int]] = []  # (id(fitter), id(correction))
    orig = PoseFitter.update_alignment

    def spy(self, correction):
        switches.append((id(self), id(correction)))
        return orig(self, correction)

    monkeypatch.setattr(PoseFitter, "update_alignment", spy)
    results = monitor_probes(
        info1["tel_dir"],
        [info1["probe_dir"], info2["probe_dir"]],
        window_s=150.0,
        z_tel=_Z_MP,
        n_probe_ch=[30, 40],
        alignment_path=adir,
        make_plots=False,
    )
    assert len(results) == 2
    assert all(r for r in results)  # each probe produced windows

    # A switch fired, and at each switched-to correction both probes' fitters
    # received the *same* object (identity invariant preserved).
    assert switches, "expected at least one mid-stream alignment switch"
    by_corr: dict[int, set[int]] = {}
    for fid, cid in switches:
        by_corr.setdefault(cid, set()).add(fid)
    # the boundary switch touches both fitters with one shared correction id.
    assert any(len(fitters) == 2 for fitters in by_corr.values())


def test_multiprobe_alignment_label_reflects_switch(multiprobe_run, tmp_path):
    """Each probe's per-window alignment_label names the schedule window(s)
    its coincidences were decoded under, mirroring the single-probe driver."""
    all_results, _, info1, info2 = multiprobe_run
    base = all_results[0]
    first, last = base[0].utc_start, base[-1].utc_end
    mid = first + (last - first) / 2
    label0 = first.strftime("%Y%m%d_%H%M%S")
    label1 = mid.strftime("%Y%m%d_%H%M%S")
    adir = tmp_path / "mp_sched_label"
    _mp_write_window(adir, label0, _mp_corr(0.0))
    _mp_write_window(adir, label1, _mp_corr(5.0))

    results = monitor_probes(
        info1["tel_dir"],
        [info1["probe_dir"], info2["probe_dir"]],
        window_s=150.0,
        z_tel=_Z_MP,
        n_probe_ch=[30, 40],
        alignment_path=adir,
        make_plots=False,
    )
    assert len(results) == 2
    for probe_results in results:
        assert probe_results
        for r in probe_results:
            assert set(r.alignment_label.split(",")) <= {label0, label1}
        assert any(label1 in r.alignment_label.split(",") for r in probe_results)
