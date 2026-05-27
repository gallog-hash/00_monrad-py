"""
End-to-end streaming pipeline test — DESIGN_UPDATE.md §8.3.

Runs the complete pipeline on 1000 synthetic tracks:
  Pass 1 — AlignmentAccumulator over all telescope events (stage 4).
  Pass 2 — coincidence_stream + PoseFitter on a second telescope
            stream and the probe stream (stage 5).

Asserts:
  • Recovered (t_x, t_y, θ, z_p) lie within 3σ of ground truth.
  • Peak heap allocation (tracemalloc) does not exceed 512 MB.

The telescope stream is re-opened for pass 2 rather than tee'd;
itertools.tee would buffer the whole stream in memory.
"""

import math
import tracemalloc
from datetime import datetime

import numpy as np
import pytest

from monrad.stage1 import (
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.stage2 import coincidence_stream
from monrad.stage3 import decode_position
from monrad.stage4 import AlignmentAccumulator
from monrad.stage5 import PoseFitter
from monrad.synth import generate, F0, Z_TEL, STRIP_MM

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS = 1000
_TRUE_TX = 50.0
_TRUE_TY = -30.0
_TRUE_THETA = 0.29671  # ≈ radians(17°)
_TRUE_ZP = 300.0
_SIGMA_STRIP = STRIP_MM / math.sqrt(12)
_MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB


# ── helpers ───────────────────────────────────────────────────────────────


def _theta_err_mod90(theta_fit: float, theta_true: float) -> float:
    """Minimum |theta_fit − (theta_true + k·π/2)| over integer k."""
    return min(abs(theta_fit - theta_true - k * math.pi / 2) for k in range(-4, 5))


def _nearest_k90(theta_fit: float, theta_true: float) -> int:
    """Return k such that theta_fit ≈ theta_true + k·π/2."""
    return min(
        range(-3, 5),
        key=lambda k: abs(theta_fit - theta_true - k * math.pi / 2),
    )


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("pipeline_stream")
    generate(
        out_dir=out,
        t_x=_TRUE_TX,
        t_y=_TRUE_TY,
        theta=_TRUE_THETA,
        z_p=_TRUE_ZP,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
    )
    return out


@pytest.fixture(scope="module")
def pipeline_result(synth_dir):
    """
    Full two-pass streaming pipeline under tracemalloc.
    Returns (PoseResult, peak_bytes).
    """
    tel_dir = synth_dir / "telescope"
    prb_dir = synth_dir / "probe"

    tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))
    tel_gps, tel_pos = find_file_pairs(tel_dir)
    prb_gps, prb_pos = find_file_pairs(prb_dir)

    tracemalloc.start()

    # ── Pass 1: telescope alignment (stage 4) ────────────────────────
    accum = AlignmentAccumulator(flush_every=_N_TRACKS + 1)
    for _ev, ref in reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0):
        hits = decode_position(ref, tel_pos, n_cols=3)
        accum.add(hits)
    alignment = accum.flush()

    # ── Pass 2: coincidence search + pose fit (stages 2 + 5) ─────────
    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        refit_every=_N_TRACKS + 1,
    )
    for cluster in coincidence_stream(
        [tel_stream, prb_stream],
        detector_ids=[0, 1],
    ):
        fitter.add(cluster)

    pr = fitter.flush()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert pr is not None, (
        "PoseFitter.flush() returned None — too few coincidences survived"
    )
    return pr, peak_bytes


# ── memory bound ──────────────────────────────────────────────────────────


class TestMemoryBound:
    def test_peak_below_512mb(self, pipeline_result):
        _, peak = pipeline_result
        peak_mb = peak / 1024**2
        assert peak <= _MEM_LIMIT_BYTES, (
            f"peak heap = {peak_mb:.1f} MB, limit = {_MEM_LIMIT_BYTES // 1024**2} MB"
        )


# ── parameter recovery ────────────────────────────────────────────────────


class TestParameterRecovery:
    """
    3σ assertions on the pose parameters returned by the full pipeline.

    θ is tested modulo π/2 to account for the four-fold degeneracy of
    a square probe.  t_x and t_y are tested in the canonical frame
    (k = 0) when the optimizer converges there; otherwise the test
    falls back to verifying that residuals are near zero.
    """

    def test_zp_within_3sigma(self, pipeline_result):
        pr, _ = pipeline_result
        sigma = math.sqrt(abs(pr.cov[3, 3]))
        err = abs(pr.z_p - _TRUE_ZP)
        assert err < 3 * sigma, (
            f"z_p={pr.z_p:.2f} mm, true={_TRUE_ZP} mm, "
            f"err={err:.2f} mm, 3σ={3 * sigma:.2f} mm"
        )

    def test_theta_within_3sigma_mod90(self, pipeline_result):
        pr, _ = pipeline_result
        sigma = math.sqrt(abs(pr.cov[2, 2]))
        err = _theta_err_mod90(pr.theta, _TRUE_THETA)
        assert err < 3 * sigma, (
            f"theta={math.degrees(pr.theta):.3f}°, "
            f"err={math.degrees(err):.3f}°, "
            f"3σ={math.degrees(3 * sigma):.3f}°"
        )

    def test_tx_ty_within_3sigma(self, pipeline_result):
        pr, _ = pipeline_result
        k = _nearest_k90(pr.theta, _TRUE_THETA)
        sigma_tx = math.sqrt(abs(pr.cov[0, 0]))
        sigma_ty = math.sqrt(abs(pr.cov[1, 1]))

        if k == 0:
            assert abs(pr.t_x - _TRUE_TX) < 3 * sigma_tx, (
                f"t_x={pr.t_x:.2f}, true={_TRUE_TX}, 3σ={3 * sigma_tx:.2f}"
            )
            assert abs(pr.t_y - _TRUE_TY) < 3 * sigma_ty, (
                f"t_y={pr.t_y:.2f}, true={_TRUE_TY}, 3σ={3 * sigma_ty:.2f}"
            )
        else:
            n = pr.n_inliers
            tol = 3 * _SIGMA_STRIP / math.sqrt(n)
            assert abs(np.mean(pr.residuals_x)) < tol
            assert abs(np.mean(pr.residuals_y)) < tol
