"""
Corner-probe edge-case regression test.

Laboratory-testing scenario: a single 30 cm probe sits on a *corner* of
the telescope's uppermost plane (z_p ≈ 0, mounted nearly flat).  At real
sea-level muon flux several messy things happen inside the 200 ns
coincidence window beyond the textbook "one muon through probe and
telescope".  This module builds a synthetic dataset that injects each
plausible edge case as a *labelled* event, runs the real stage 1-5
pipeline over it, and asserts how each one is handled.

The dataset reuses the byte-level writers in ``monrad.synth`` (so the
files are format-identical to ``generate()``) but hand-crafts the event
content per scenario.  Edge cases:

  E1  genuine golden coincidence (the common case)
  E2  pile-up in one window, non-adjacent  -> telescope 'unresolved'
  E3  pile-up in one window, adjacent      -> merges to a 'cluster'
  E4  two telescope events <200 ns apart   -> 2 tel + 1 prb in cluster
  E5  accidental (probe muon + unrelated telescope muon)
  E6  probe-only (muon misses telescope)
  E7  telescope-only (muon misses probe)
  E8  genuine track, one plane charge-shares -> 'cluster'
  E9  genuine track, stray probe bit       -> probe 'unresolved'
  E10 genuine track, probe word forced INVALID
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pytest

from monrad.stage1 import (
    _utc_to_ns,
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.stage2 import coincidence_stream
from monrad.stage3 import decode_position
from monrad.stage4 import AlignmentCorrection
from monrad.stage5 import (
    PoseFitter,
    fit_probe_pose,
    _tel_line_fit,
    _sigma_tel_at_z,
    _CHI2_TRACK,
)
from monrad.synth import (
    F0,
    Z_TEL,
    STRIP_MM,
    N_TEL,
    _ch_to_u64,
    _build_gps_stream,
    _write_header,
    _write_gps_bin,
    _write_pos_bin,
    _quantize,
)

# ── geometry of the corner-probe setup ─────────────────────────────
TRUE_TX = 20.0  # mm — probe origin near the (0,0) corner
TRUE_TY = 20.0  # mm
TRUE_THETA = 0.05  # rad — small mounting rotation
TRUE_ZP = 15.0  # mm — probe resting on top plane (z≈0) + thickness
N_PROBE = 30  # 30 cm probe → 30 channels per axis
START_UTC = datetime(2023, 4, 18, 19, 21, 0)

SLOT = 200_000  # ticks between scenario slots (2 ms @ 100 MHz)
WINDOW_NS = 200  # stage-2 coincidence window
N_GOOD = 200  # genuine coincidences (enough to dilute one accidental)

TEL_ID = 0
PRB_ID = 1
_SIGMA_STRIP = STRIP_MM / math.sqrt(12)


# ── hand-crafted event model ───────────────────────────────────────


@dataclass
class TelEvent:
    tick: int
    plane_chans: list[list[tuple[int, int]]]  # 3 planes, each a list of (cx,cy)
    label: str
    raw_or: dict[int, int] = field(default_factory=dict)  # plane -> extra OR bits


@dataclass
class PrbEvent:
    tick: int
    chans: list[tuple[int, int]]  # one (cu,cv); >1 = noise / pile-up
    label: str
    raw_or: int = 0  # extra OR bits (force invalid)


def _word(chans: list[tuple[int, int]], gen: int, n_max: int) -> int:
    """OR a list of (cx,cy) channel hits into one u64 with GEN baked in."""
    w = 0
    for cx, cy in chans:
        cx = max(0, min(n_max - 1, cx))
        cy = max(0, min(n_max - 1, cy))
        w |= _ch_to_u64(cx, cy, 0)
    return w | (gen << 52)


# ── synthetic track helpers ────────────────────────────────────────


def _planes_of(ax, bx, ay, by):
    """Return [(cx,cy)] per telescope plane, or None if out of range."""
    out = []
    for z in Z_TEL:
        cx = _quantize(ax + bx * z, N_TEL)
        cy = _quantize(ay + by * z, N_TEL)
        if cx is None or cy is None:
            return None
        out.append((cx, cy))
    return out


def _probe_of(ax, bx, ay, by):
    """Return probe (cu,cv) for a track, or None if it misses the probe."""
    xp = ax + bx * TRUE_ZP
    yp = ay + by * TRUE_ZP
    c, s = math.cos(TRUE_THETA), math.sin(TRUE_THETA)
    u = (xp - TRUE_TX) * c + (yp - TRUE_TY) * s
    v = -(xp - TRUE_TX) * s + (yp - TRUE_TY) * c
    cu = _quantize(u, N_PROBE)
    cv = _quantize(v, N_PROBE)
    if cu is None or cv is None:
        return None
    return cu, cv


def _line_chi2(planes) -> float:
    """χ² of the quantized telescope line — same cut stage 5 applies."""
    x = np.array([(cx + 0.5) * STRIP_MM for cx, _ in planes])
    y = np.array([(cy + 0.5) * STRIP_MM for _, cy in planes])
    *_, chi2 = _tel_line_fit(x, y, np.asarray(Z_TEL), _SIGMA_STRIP)
    return chi2


def _sample_through_probe(rng, clean=False):
    """Sample a track crossing all 3 planes AND the probe.

    clean=True additionally requires the quantized line-fit χ² to clear
    the stage-5 track cut, so the caller gets a track that survives.
    """
    while True:
        ax = rng.uniform(40.0, 290.0)
        ay = rng.uniform(40.0, 290.0)
        u = rng.uniform(0, 1)
        zen = math.acos(np.cbrt(1.0 - u))
        phi = rng.uniform(0, 2 * math.pi)
        bx = max(-0.22, min(0.22, math.tan(zen) * math.cos(phi)))
        by = max(-0.22, min(0.22, math.tan(zen) * math.sin(phi)))
        planes = _planes_of(ax, bx, ay, by)
        probe = _probe_of(ax, bx, ay, by)
        if planes is None or probe is None:
            continue
        if clean and _line_chi2(planes) >= _CHI2_TRACK:
            continue
        return (ax, bx, ay, by), planes, probe


def _sample_telescope_only(rng, clean=False):
    """Sample a track crossing the telescope, anywhere (may miss probe)."""
    while True:
        ax = rng.uniform(0.0, 990.0)
        ay = rng.uniform(0.0, 990.0)
        u = rng.uniform(0, 1)
        zen = math.acos(np.cbrt(1.0 - u))
        phi = rng.uniform(0, 2 * math.pi)
        bx = max(-0.22, min(0.22, math.tan(zen) * math.cos(phi)))
        by = max(-0.22, min(0.22, math.tan(zen) * math.sin(phi)))
        planes = _planes_of(ax, bx, ay, by)
        if planes is None:
            continue
        if clean and _line_chi2(planes) >= _CHI2_TRACK:
            continue
        return (ax, bx, ay, by), planes


def _probe_cell_of_track(planes):
    """Probe cell a telescope track points at (clamped into the probe)."""
    z = Z_TEL
    xc = np.array([(cx + 0.5) * STRIP_MM for cx, _ in planes])
    yc = np.array([(cy + 0.5) * STRIP_MM for _, cy in planes])
    bx = (xc[-1] - xc[0]) / (z[-1] - z[0])
    by = (yc[-1] - yc[0]) / (z[-1] - z[0])
    ax = xc[0] - bx * z[0]
    ay = yc[0] - by * z[0]
    pr = _probe_of(ax, bx, ay, by)
    return pr if pr is not None else (N_PROBE // 2, N_PROBE // 2)


# ── dataset construction ───────────────────────────────────────────


def _build_events(rng):
    tel_events: list[TelEvent] = []
    prb_events: list[PrbEvent] = []
    slot = 0

    def next_slot():
        nonlocal slot
        slot += 1
        return slot * SLOT

    # E1: genuine golden coincidences
    for _ in range(N_GOOD):
        t = next_slot()
        _, planes, probe = _sample_through_probe(rng)
        tel_events.append(TelEvent(t, [[p] for p in planes], "E1-genuine"))
        prb_events.append(PrbEvent(t, [probe], "E1-genuine"))

    # E2: pile-up in one window, non-adjacent → unresolved telescope hit
    t = next_slot()
    _, planesA, probeA = _sample_through_probe(rng)
    _, planesB = _sample_telescope_only(rng)
    merged = [[planesA[k], planesB[k]] for k in range(3)]
    tel_events.append(TelEvent(t, merged, "E2-pileup-unresolved"))
    prb_events.append(PrbEvent(t, [probeA], "E2-pileup-unresolved"))

    # E3: pile-up in one window, adjacent strips → merges to one cluster
    t = next_slot()
    _, planesA, probeA = _sample_through_probe(rng)

    def _adj(c):  # stay inside the same ribbon decade → contiguous merge
        return c + 1 if c % 10 < 9 else c - 1

    planesB = [(_adj(cx), _adj(cy)) for cx, cy in planesA]
    merged = [[planesA[k], planesB[k]] for k in range(3)]
    tel_events.append(TelEvent(t, merged, "E3-pileup-adjacent-cluster"))
    prb_events.append(PrbEvent(t, [probeA], "E3-pileup-adjacent-cluster"))

    # E4: two telescope events in distinct windows, <200 ns apart
    t = next_slot()
    _, planesA, probeA = _sample_through_probe(rng)
    _, planesB = _sample_telescope_only(rng)
    tel_events.append(TelEvent(t, [[p] for p in planesA], "E4-two-window-A"))
    tel_events.append(TelEvent(t + 5, [[p] for p in planesB], "E4-two-window-B"))
    prb_events.append(PrbEvent(t, [probeA], "E4-two-window"))

    # E5: accidental — probe muon misses telescope, unrelated muon Q hits it
    t = next_slot()
    _, planesQ = _sample_telescope_only(rng, clean=True)
    cu_q, cv_q = _probe_cell_of_track(planesQ)
    cu = (cu_q + 15) % N_PROBE  # gross spatial mismatch → outlier
    cv = (cv_q + 15) % N_PROBE
    tel_events.append(TelEvent(t + 3, [[p] for p in planesQ], "E5-accidental"))
    prb_events.append(PrbEvent(t, [(cu, cv)], "E5-accidental"))

    # E6: probe-only (muon through probe, misses the telescope)
    t = next_slot()
    prb_events.append(PrbEvent(t, [(10, 10)], "E6-probe-only"))

    # E7: telescope-only (muon through telescope, misses the probe)
    t = next_slot()
    _, planes = _sample_telescope_only(rng)
    tel_events.append(TelEvent(t, [[p] for p in planes], "E7-telescope-only"))

    # E8: genuine, but charge sharing makes one plane a cluster hit
    t = next_slot()
    _, planes, probe = _sample_through_probe(rng, clean=True)
    chans = [[p] for p in planes]
    cx, cy = planes[1]
    cy2 = cy + 1 if cy % 10 < 9 else cy - 1
    chans[1] = [(cx, cy), (cx, cy2)]  # adjacent fiber bit on plane 2 (Y)
    tel_events.append(TelEvent(t, chans, "E8-charge-share-cluster"))
    prb_events.append(PrbEvent(t, [probe], "E8-charge-share-cluster"))

    # E9: genuine track, but a noise bit makes the probe unresolved
    t = next_slot()
    _, planes, probe = _sample_through_probe(rng)
    cu, cv = probe
    noise = ((cu + 4) % N_PROBE, (cv + 4) % N_PROBE)  # far → 2 clusters
    tel_events.append(TelEvent(t, [[p] for p in planes], "E9-probe-noise-unresolved"))
    prb_events.append(PrbEvent(t, [probe, noise], "E9-probe-noise-unresolved"))

    # E10: genuine track, but the probe word is INVALID (ribbon all 1s)
    t = next_slot()
    _, planes, probe = _sample_through_probe(rng)
    tel_events.append(TelEvent(t, [[p] for p in planes], "E10-probe-invalid"))
    prb_events.append(PrbEvent(t, [probe], "E10-probe-invalid", raw_or=(0x3FF << 32)))

    return tel_events, prb_events


def _write_files(out_dir, tel_events, prb_events):
    tel_dir = out_dir / "telescope"
    prb_dir = out_dir / "probe"
    tel_dir.mkdir(parents=True, exist_ok=True)
    prb_dir.mkdir(parents=True, exist_ok=True)

    tel_events = sorted(tel_events, key=lambda e: e.tick)
    prb_events = sorted(prb_events, key=lambda e: e.tick)

    label_by_slot: dict[int, str] = {}
    for e in tel_events + prb_events:
        base = round(e.tick / SLOT) * SLOT
        label_by_slot.setdefault(base, e.label.split("-A")[0].split("-B")[0])

    tel_ticks = [e.tick for e in tel_events]
    tel_blocks = []
    for gen, e in enumerate(tel_events):
        words = []
        for k in range(3):
            w = _word(e.plane_chans[k], gen % 2048, N_TEL)
            w |= e.raw_or.get(k, 0)
            words.append(w)
        tel_blocks.append(words)

    prb_ticks = [e.tick for e in prb_events]
    prb_blocks = []
    for gen, e in enumerate(prb_events):
        w = _word(e.chans, gen % 2048, N_PROBE)
        w |= e.raw_or
        prb_blocks.append([w])

    duration = max(tel_ticks[-1], prb_ticks[-1]) + SLOT
    pps = list(range(0, duration + F0, F0))

    ts = START_UTC.strftime("%Y%m%d_%H%M%S")
    _write_header(tel_dir / f"{ts}_header.txt", START_UTC, F0)
    _write_gps_bin(tel_dir / f"{ts}_GPS.bin", _build_gps_stream(tel_ticks, pps))
    _write_pos_bin(tel_dir / f"{ts}.bin", tel_blocks, n_cols=3)

    _write_header(prb_dir / f"{ts}_header.txt", START_UTC, F0)
    _write_gps_bin(prb_dir / f"{ts}_GPS.bin", _build_gps_stream(prb_ticks, pps))
    _write_pos_bin(prb_dir / f"{ts}.bin", prb_blocks, n_cols=1)

    return label_by_slot


# ── verdict bundle ─────────────────────────────────────────────────


@dataclass
class Verdict:
    n_tel: int
    n_prb: int
    accepted: bool
    tel_quals: list[str] | None = None
    prb_qual: str | None = None
    chi2: float | None = None


@dataclass
class Results:
    verdicts: dict[str, Verdict]  # one per non-E1 scenario
    e1_accept: int
    e1_reject: int
    no_cluster: set[str]
    pose_clean: object
    pose_mixed: object
    genuine_maha_max: float
    accidental_maha: float


def _run_pipeline(out_dir, label_by_slot) -> Results:
    tel_dir = out_dir / "telescope"
    prb_dir = out_dir / "probe"
    tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))
    tel_gps, tel_pos = find_file_pairs(tel_dir)
    prb_gps, prb_pos = find_file_pairs(prb_dir)
    utc0_ns = _utc_to_ns(START_UTC)

    def tns_to_slot(t_ns):
        tick = round((t_ns - utc0_ns) * tel_f0 / 1_000_000_000)
        return round(tick / SLOT) * SLOT

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    alignment = AlignmentCorrection.identity()
    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=alignment,
        tel_id=TEL_ID,
        prb_id=PRB_ID,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
    )

    verdicts: dict[str, Verdict] = {}
    genuine, accidental = [], []
    e1_accept = e1_reject = 0
    seen_slots = set()

    for cluster in coincidence_stream(
        [tel_stream, prb_stream], detector_ids=[TEL_ID, PRB_ID], window_ns=WINDOW_NS
    ):
        n_tel = sum(1 for d, _, _ in cluster if d == TEL_ID)
        n_prb = sum(1 for d, _, _ in cluster if d == PRB_ID)
        t_ns = min(ev.t_ns for _, ev, _ in cluster)
        slot = tns_to_slot(t_ns)
        seen_slots.add(slot)
        label = label_by_slot.get(slot, f"slot@{slot}")

        co = fitter._decode_cluster(cluster)
        accepted = co is not None

        tel_quals = prb_qual = None
        chi2 = None
        if n_tel == 1 and n_prb == 1:
            tref = next(r for d, _, r in cluster if d == TEL_ID)
            pref = next(r for d, _, r in cluster if d == PRB_ID)
            th = [h for h in decode_position(tref, tel_pos, n_cols=3) if h]
            ph = [h for h in decode_position(pref, prb_pos, n_cols=1) if h]
            tel_quals = [str(h.quality) for h in th]
            prb_qual = str(ph[0].quality)
            if all(q in ("golden", "cluster") for q in tel_quals):
                x = np.array([h.x_mm for h in th])
                y = np.array([h.y_mm for h in th])
                *_, chi2 = _tel_line_fit(x, y, np.asarray(Z_TEL), th[0].sigma_x)

        if label == "E1-genuine":
            if accepted:
                e1_accept += 1
            else:
                e1_reject += 1
        else:
            verdicts[label] = Verdict(n_tel, n_prb, accepted, tel_quals, prb_qual, chi2)

        if accepted:
            (accidental if label.startswith("E5") else genuine).append(co)

    no_cluster = {
        label
        for slot, label in label_by_slot.items()
        if slot not in seen_slots and not label.startswith("E1")
    }

    z = alignment.corrected_z_tel(Z_TEL)
    pose_clean = fit_probe_pose(genuine, z, alignment)
    pose_mixed = fit_probe_pose(genuine + accidental, z, alignment)

    def _maha(co, pr):
        c, s = math.cos(pr.theta), math.sin(pr.theta)
        xm = pr.t_x + co.u * c - co.v * s
        ym = pr.t_y + co.u * s + co.v * c
        xp = co.a_x + co.b_x * pr.z_p
        yp = co.a_y + co.b_y * pr.z_p
        var = max(co.sigma_prb**2 + _sigma_tel_at_z(co.cov_ab, pr.z_p), 1e-12)
        return math.sqrt((xm - xp) ** 2 / var + (ym - yp) ** 2 / var)

    return Results(
        verdicts=verdicts,
        e1_accept=e1_accept,
        e1_reject=e1_reject,
        no_cluster=no_cluster,
        pose_clean=pose_clean,
        pose_mixed=pose_mixed,
        genuine_maha_max=max(_maha(co, pose_clean) for co in genuine),
        accidental_maha=_maha(accidental[0], pose_clean),
    )


# ── fixture ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> Results:
    out = tmp_path_factory.mktemp("corner_probe")
    rng = np.random.default_rng(7)
    tel_events, prb_events = _build_events(rng)
    label_by_slot = _write_files(out, tel_events, prb_events)
    return _run_pipeline(out, label_by_slot)


# ── helpers for assertions ─────────────────────────────────────────


def _theta_err_mod90(theta_fit, theta_true):
    return min(abs(theta_fit - theta_true - k * math.pi / 2) for k in range(-4, 5))


# ── tests: per-scenario handling ───────────────────────────────────


class TestPerScenarioHandling:
    def test_genuine_majority_accepted(self, results):
        n = results.e1_accept + results.e1_reject
        assert n == N_GOOD
        # The χ²<4 line cut rejects a tail of near-vertical corner tracks
        # (quantization noise); the bulk must still survive.
        assert results.e1_accept / n > 0.70

    def test_E2_pileup_same_window_unresolved_rejected(self, results):
        v = results.verdicts["E2-pileup-unresolved"]
        assert (v.n_tel, v.n_prb) == (1, 1)
        assert not v.accepted
        assert "unresolved" in v.tel_quals

    def test_E3_pileup_adjacent_merges_to_cluster(self, results):
        v = results.verdicts["E3-pileup-adjacent-cluster"]
        # Two adjacent muons OR into one contiguous cluster per plane —
        # eligible (not 'unresolved'), and collinear so it survives.
        assert all(q in ("golden", "cluster") for q in v.tel_quals)
        assert "cluster" in v.tel_quals
        assert v.accepted

    def test_E4_two_window_double_track_rejected(self, results):
        v = results.verdicts["E4-two-window"]
        # Two telescope events land in the same coincidence cluster.
        assert v.n_tel == 2 and v.n_prb == 1
        assert not v.accepted  # ambiguous → dropped wholesale

    def test_E5_accidental_accepted_then_flagged_outlier(self, results):
        v = results.verdicts["E5-accidental"]
        # It passes the per-cluster gates (1 tel + 1 prb, clean track)…
        assert (v.n_tel, v.n_prb) == (1, 1)
        assert v.accepted
        # …but is a gross spatial outlier the pose fit must reject.
        assert results.accidental_maha > _CHI2_TRACK
        assert results.genuine_maha_max < _CHI2_TRACK

    def test_E6_probe_only_no_cluster(self, results):
        assert "E6-probe-only" in results.no_cluster

    def test_E7_telescope_only_no_cluster(self, results):
        assert "E7-telescope-only" in results.no_cluster

    def test_E8_charge_share_cluster_eligible(self, results):
        v = results.verdicts["E8-charge-share-cluster"]
        # A charge-shared plane decodes as 'cluster', not 'unresolved',
        # so it clears the quality gate.
        assert all(q in ("golden", "cluster") for q in v.tel_quals)
        assert "cluster" in v.tel_quals

    def test_E9_probe_noise_unresolved_rejected(self, results):
        v = results.verdicts["E9-probe-noise-unresolved"]
        assert not v.accepted
        assert v.prb_qual == "unresolved"

    def test_E10_probe_invalid_rejected(self, results):
        v = results.verdicts["E10-probe-invalid"]
        assert not v.accepted
        assert v.prb_qual == "invalid"


# ── tests: pose recovery and accidental robustness ─────────────────


class TestPoseRecovery:
    def test_clean_pose_recovered(self, results):
        pr = results.pose_clean
        assert abs(pr.t_x - TRUE_TX) < 4.0
        assert abs(pr.t_y - TRUE_TY) < 4.0
        assert _theta_err_mod90(pr.theta, TRUE_THETA) < math.radians(1.5)
        assert abs(pr.z_p - TRUE_ZP) < 8.0

    def test_zp_is_the_soft_direction_at_the_corner(self, results):
        # At z_p≈0 with near-vertical tracks, z_p is the least-constrained
        # parameter (small slope leverage — DESIGN §8.6).  This is *why* a
        # high-leverage accidental can run away along z_p and defeat the
        # one-pass Mahalanobis cut unless enough genuine statistics dilute
        # it; see handoff.md.  The covariance must expose that softness.
        sigma = np.sqrt(np.abs(np.diag(results.pose_clean.cov)))
        sigma_tx, sigma_ty, _, sigma_zp = sigma
        assert sigma_zp > sigma_tx
        assert sigma_zp > sigma_ty
