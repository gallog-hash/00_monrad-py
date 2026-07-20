"""
Stage 5 (part) — the streaming PoseFitter accumulator.

PoseFitter accumulates telescope-probe coincidences, decodes each cluster to a
Coincidence (combinatorial telescope track finder + probe hit), and refits the
probe pose every ``refit_every`` coincidences (DESIGN.md §8.8).
"""

import itertools
import math
from pathlib import Path
from typing import Callable

import numpy as np

from ..alignment import AlignmentCorrection
from ..decoders.position import POS_HALF_BITS
from ..reconstruction import (
    GOOD_QUALITIES,
    decode_position,
    reconstruct_plane_candidates,
)
from ..timing import PosRef
from .optimize import _fit_triple, fit_probe_pose, tel_align_arrays
from .types import Coincidence, DecodeReport, PoseResult, TelescopeTrackResult

_CHI2_TRACK = 4.0  # telescope line-fit χ² threshold — DESIGN.md §8.2
N_TEL_PLANES = 3  # telescope planes the combinatorial search fits a line through


class PoseFitter:
    """
    Accumulates telescope-probe coincidences and refits the probe pose
    every refit_every new coincidences.  Implements DESIGN.md §8.8.
    """

    MIN_FIT = 30
    REFIT_EVERY = 500

    def __init__(
        self,
        tel_z: np.ndarray,
        alignment: AlignmentCorrection,
        tel_id: int,
        prb_id: int,
        tel_pos_paths: list[Path],
        prb_pos_paths: list[Path],
        refit_every: int = REFIT_EVERY,
        tot_thresh: int = 1,
        tot_weights: bool = False,
        min_anchor_planes: int = 1,
        on_decode: Callable[[DecodeReport], None] | None = None,
        prb_fibers_per_ribbon: int = POS_HALF_BITS,
        chi2_track: float | None = None,
        max_cluster_width: int | None = None,
    ) -> None:
        if not 0 <= min_anchor_planes <= N_TEL_PLANES:
            raise ValueError(
                f"min_anchor_planes must be in [0, {N_TEL_PLANES}], "
                f"got {min_anchor_planes}"
            )
        self.tel_z = tel_z
        self.alignment = alignment
        self.tel_id = tel_id
        self.prb_id = prb_id
        self.tel_pos_paths = tel_pos_paths
        self.prb_pos_paths = prb_pos_paths
        self.refit_every = refit_every
        self.tot_thresh = tot_thresh
        self.tot_weights = tot_weights
        # Minimum number of telescope planes that must decode to a single
        # resolved candidate (an "anchor") before the combinatorial χ² search
        # is allowed to run.  1 (default) reproduces the original gate; 0
        # removes it entirely (search every all-ambiguous cluster too — more
        # tracks, far heavier compute, and pile-up can fabricate a low-χ²
        # track); N_TEL_PLANES requires every plane already resolved.
        self.min_anchor_planes = min_anchor_planes
        self.on_decode = on_decode
        self.prb_fibers_per_ribbon = prb_fibers_per_ribbon
        self.chi2_track = _CHI2_TRACK if chi2_track is None else chi2_track
        self.max_cluster_width = max_cluster_width
        self._coincs: list[Coincidence] = []
        self._since_last = 0
        self.result: PoseResult | None = None

    def update_alignment(self, correction: AlignmentCorrection) -> None:
        self.alignment = correction

    def add(
        self,
        cluster: list[tuple[int, object, object]],
    ) -> "PoseResult | None":
        """
        Decode positions for the cluster and accumulate the coincidence.
        Returns a new PoseResult when a refit is triggered; otherwise None.
        """
        co = self.decode_cluster(cluster)
        if co is None:
            return None
        self._coincs.append(co)
        self._since_last += 1
        if len(self._coincs) >= self.MIN_FIT and self._since_last >= self.refit_every:
            return self._refit()
        return None

    def flush(self) -> "PoseResult | None":
        """Force a fit on whatever is buffered."""
        if len(self._coincs) < self.MIN_FIT:
            return None
        return self._refit()

    def decode_cluster(self, cluster: list) -> "Coincidence | None":
        """
        Decode one coincidence cluster to a :class:`Coincidence` (or None).

        Public wrapper over :meth:`_decode_cluster` so monitoring drivers
        (``monrad.monitor``) can reuse the exact decode path the streaming
        accumulator applies — combinatorial telescope track finder, alignment
        correction, track/probe quality cuts — without poking at a private
        method.
        """
        return self._decode_cluster(cluster)

    def decode_telescope_track(self, cluster: list) -> TelescopeTrackResult:
        """
        Run the combinatorial telescope-track search for one cluster and
        return its outcome, without touching this fitter's probe side.

        Public wrapper over :meth:`_decode_telescope_track` so multi-probe
        drivers (``monrad.monitor.multiprobe``) can compute this once per
        cluster and share it across every :class:`PoseFitter` that watches
        the same cluster and agrees on tel_id/tel_z/alignment/tot_thresh/
        tot_weights/min_anchor_planes/tel_pos_paths — pass the result to each
        fitter's :meth:`decode_from_telescope_track` (finding 9).
        """
        return self._decode_telescope_track(cluster)

    def decode_from_telescope_track(
        self, cluster: list, tel_result: TelescopeTrackResult
    ) -> "Coincidence | None":
        """
        Finish decoding a cluster's probe side given a telescope-track
        result already computed by :meth:`decode_telescope_track` (by this
        fitter or shared from another — see that method's docstring).
        """
        prb_refs = [ref for det_id, _ev, ref in cluster if det_id == self.prb_id]
        if len(prb_refs) != 1:
            self._fire_report("ambiguous_cluster")
            return None
        return self._finish_decode(prb_refs[0], tel_result)

    def _fire_report(
        self,
        reason: str,
        cand_counts: tuple[int, int, int] | None = None,
        chi2: float | None = None,
        prb_quality: str | None = None,
        tel_quality: tuple[str, str, str] | None = None,
    ) -> None:
        if self.on_decode is not None:
            self.on_decode(
                DecodeReport(
                    accepted=(reason == "accepted"),
                    reason=reason,
                    cand_counts=cand_counts,
                    chi2=chi2,
                    prb_quality=prb_quality,
                    tel_quality=tel_quality,
                )
            )

    def _decode_cluster(
        self,
        cluster: list,
    ) -> "Coincidence | None":
        """
        Extract telescope and probe hits from a coincidence cluster,
        apply alignment correction, fit a telescope line, apply the
        track quality cut, and return a Coincidence or None.

        A genuine coincidence pairs exactly one telescope track with exactly
        one hit in *this* probe.  A cluster carrying two or more events from
        either of those two detectors is ambiguous (two particles inside the
        window, or a random coincidence) — reject it rather than silently
        picking one and fabricating a pairing.  Events belonging to *other*
        probe detectors are ignored here: a single telescope event may
        legitimately be in coincidence with several distinct probes, each
        handled by its own PoseFitter.

        The probe-side ambiguity check runs *before* the (expensive)
        telescope search, mirroring the combined gate this method used to
        have, so a probe-ambiguous cluster never pays for a search whose
        result it can't use.
        """
        prb_refs = [ref for det_id, _ev, ref in cluster if det_id == self.prb_id]
        if len(prb_refs) != 1:
            self._fire_report("ambiguous_cluster")
            return None
        tel_result = self._decode_telescope_track(cluster)
        return self._finish_decode(prb_refs[0], tel_result)

    def _decode_telescope_track(self, cluster: list) -> TelescopeTrackResult:
        """
        Run the combinatorial telescope-track search for one cluster.

        Depends only on the cluster's telescope entry (never on which probe,
        if any, is asking), so its result is safe to share across every
        PoseFitter watching the same cluster with an identical telescope-side
        configuration (finding 9).
        """
        tel_entries = [
            (ev, ref) for det_id, ev, ref in cluster if det_id == self.tel_id
        ]
        if len(tel_entries) != 1:
            return TelescopeTrackResult(accepted=False, reason="ambiguous_cluster")
        tel_ev, tel_ref = tel_entries[0]

        # Enumerate per-plane candidate positions (golden/cluster axes give
        # one candidate; mirror-fold-ambiguous axes give their full
        # ribbon×fiber cross-product) and search every one-candidate-per-
        # plane triple for the lowest-χ² straight line.  This resolves the
        # per-plane ambiguity globally from which combination actually lies on
        # a track, instead of recovering one plane from two already-clean ones
        # (see DESIGN.md §8.2).
        cands = reconstruct_plane_candidates(
            tel_ref,
            self.tel_pos_paths,
            n_cols=3,
            max_per_plane=16,
            tot_thresh=self.tot_thresh,
            tot_weights=self.tot_weights,
            max_cluster_width=self.max_cluster_width,
        )
        cand_counts = (len(cands[0]), len(cands[1]), len(cands[2]))
        if any(len(c) == 0 for c in cands):
            # A triple needs all 3 planes; single-half dropouts are out of
            # scope for this phase-1 combinatorial search.
            return TelescopeTrackResult(
                accepted=False, reason="zero_candidate_plane", cand_counts=cand_counts
            )
        # Anchor-plane gate (tunable via min_anchor_planes).  An "anchor" is a
        # plane that decoded to a single resolved candidate; zero-candidate
        # planes were already rejected above, so an anchor is exactly a plane
        # with len(cands) == 1.  Require at least min_anchor_planes of them
        # before running the combinatorial χ² search.
        #
        # The mirror-fold/pile-up ambiguity is identical at the bit level for
        # both causes, so a search over fully-ambiguous planes cannot tell "one
        # particle, fold-mirrored" from "two particles overlapping in the same
        # window" — it only finds whichever combination minimises χ², which a
        # genuine pile-up can do by coincidence (see
        # TestPerScenarioHandling::test_E2_pileup_same_window_unresolved_rejected).
        # min_anchor_planes=1 (default) keeps that guard, requiring one
        # already-resolved combinatorial reference; 0 disables it (search every
        # cluster — more tracks, far heavier compute); N_TEL_PLANES demands
        # all planes resolved.
        n_anchor = sum(1 for c in cands if len(c) == 1)
        if n_anchor < self.min_anchor_planes:
            return TelescopeTrackResult(
                accepted=False, reason="no_anchor_plane", cand_counts=cand_counts
            )

        # The alignment-corrected plane z and the per-plane delta/tilt arrays
        # are constant across every triple of this event, so compute them once
        # instead of per-triple in _fit_triple (up to 16³ triples per cluster).
        z_corr = self.alignment.corrected_z_tel(self.tel_z)
        tel_align = tel_align_arrays(self.alignment)

        best_chi2 = math.inf
        best_fit = None
        best_cands: tuple[object, object, object] | None = None
        for c0, c1, c2 in itertools.product(*cands):
            x_raw = np.array([c0.x_mm, c1.x_mm, c2.x_mm])
            y_raw = np.array([c0.y_mm, c1.y_mm, c2.y_mm])
            sigma_x_arr = np.array([c0.sigma_x, c1.sigma_x, c2.sigma_x])
            sigma_y_arr = np.array([c0.sigma_y, c1.sigma_y, c2.sigma_y])
            fit = _fit_triple(x_raw, y_raw, sigma_x_arr, sigma_y_arr, tel_align, z_corr)
            if fit[-1] < best_chi2:
                best_chi2 = fit[-1]
                best_fit = fit
                best_cands = (c0, c1, c2)

        if best_fit is None or best_chi2 >= self.chi2_track:
            return TelescopeTrackResult(
                accepted=False,
                reason="chi2_track_cut",
                cand_counts=cand_counts,
                chi2=(best_chi2 if best_fit is not None else None),
            )
        a_x, b_x, a_y, b_y, cov_x, cov_y, _ = best_fit
        assert best_cands is not None  # set whenever best_fit is

        # Per-plane quality straight from the winning triple: each candidate
        # carries its own "golden"/"cluster" label (DESIGN.md §8.2).
        tel_quality = (
            best_cands[0].quality,
            best_cands[1].quality,
            best_cands[2].quality,
        )
        return TelescopeTrackResult(
            accepted=True,
            reason="accepted",
            cand_counts=cand_counts,
            chi2=best_chi2,
            a_x=a_x,
            b_x=b_x,
            a_y=a_y,
            b_y=b_y,
            cov_ab_x=cov_x,
            cov_ab_y=cov_y,
            tel_quality=tel_quality,
            t_ns=tel_ev.t_ns,
        )

    def _finish_decode(
        self, prb_ref: PosRef, tel_result: TelescopeTrackResult
    ) -> "Coincidence | None":
        """Probe-side decode given an already-resolved telescope track."""
        if not tel_result.accepted:
            self._fire_report(
                tel_result.reason,
                cand_counts=tel_result.cand_counts,
                chi2=tel_result.chi2,
            )
            return None
        # tel_quality is always set once tel_result.accepted is True (see
        # _decode_telescope_track's success return).
        assert tel_result.tel_quality is not None
        tel_quality = tel_result.tel_quality

        # Decode probe (1 plane)
        prb_hits = decode_position(
            prb_ref,
            self.prb_pos_paths,
            n_cols=1,
            tot_thresh=self.tot_thresh,
            tot_weights=self.tot_weights,
            n_fibers_per_ribbon=self.prb_fibers_per_ribbon,
            max_cluster_width=self.max_cluster_width,
        )
        prb_hit = prb_hits[0]
        if prb_hit.quality not in GOOD_QUALITIES:
            self._fire_report(
                "probe_quality",
                cand_counts=tel_result.cand_counts,
                chi2=tel_result.chi2,
                prb_quality=prb_hit.quality,
                tel_quality=tel_quality,
            )
            return None

        self._fire_report(
            "accepted",
            cand_counts=tel_result.cand_counts,
            chi2=tel_result.chi2,
            prb_quality=prb_hit.quality,
            tel_quality=tel_quality,
        )
        return Coincidence(
            a_x=tel_result.a_x,
            b_x=tel_result.b_x,
            a_y=tel_result.a_y,
            b_y=tel_result.b_y,
            cov_ab_x=tel_result.cov_ab_x,
            cov_ab_y=tel_result.cov_ab_y,
            u=prb_hit.x_mm,
            v=prb_hit.y_mm,
            sigma_prb_x=prb_hit.sigma_x,
            sigma_prb_y=prb_hit.sigma_y,
            tel_quality=tel_quality,
            t_ns=tel_result.t_ns,
        )

    def _refit(self) -> "PoseResult":
        result = fit_probe_pose(
            self._coincs,
            self.alignment.corrected_z_tel(self.tel_z),
            self.alignment,
        )
        self._since_last = 0
        self.result = result
        return result
