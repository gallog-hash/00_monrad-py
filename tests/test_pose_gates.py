"""
Tests for the stage-5 pre-fit geometric gates: ``filter_rigidity`` and
``filter_off_probe`` (docs/handoffs/2026-07-07-off-probe-track-gate-strategy.md).

Both gates reject coincidences whose telescope track is geometrically
inconsistent with its time-matched probe hit *before* ``fit_probe_pose`` sees
them — the mechanism behind the testLab 20210723 z_p burst anomaly.
"""

import math

import numpy as np

from monrad.pose import Coincidence, PoseResult, filter_off_probe, filter_rigidity

_SIGMA = 1.0  # unused by either gate, kept for a valid Coincidence
_ZERO_COV = (0.0, 0.0, 0.0)


def _flat_coinc(x: float, y: float, u: float, v: float) -> Coincidence:
    """A zero-slope 'track' (a_x=x, b_x=0) whose projection is x,y at every z."""
    return Coincidence(x, 0.0, y, 0.0, _ZERO_COV, _ZERO_COV, u, v, _SIGMA, _SIGMA)


def _make_ref_pose(t_x: float, t_y: float, theta: float, z_p: float) -> PoseResult:
    return PoseResult(
        t_x=t_x,
        t_y=t_y,
        theta=theta,
        z_p=z_p,
        cov=np.eye(4),
        chi2_curve=np.zeros((1, 2)),
        residuals_x=np.array([]),
        residuals_y=np.array([]),
        n_inliers=0,
        half_params=np.zeros((2, 4)),
        inliers=[],
    )


# ── filter_rigidity ──────────────────────────────────────────────────────────


class TestFilterRigidity:
    _THETA = 0.3
    _TX, _TY = 50.0, -30.0

    def _genuine_cluster(self, n: int, seed: int) -> list[Coincidence]:
        """n coincidences whose (u,v) and (X,Y) obey the same rigid transform —
        the pairwise-distance invariant holds exactly (up to float error)."""
        rng = np.random.default_rng(seed)
        c, s = math.cos(self._THETA), math.sin(self._THETA)
        coincs = []
        for _ in range(n):
            u, v = rng.uniform(0, 400, size=2)
            x = self._TX + u * c - v * s
            y = self._TY + u * s + v * c
            coincs.append(_flat_coinc(x, y, u, v))
        return coincs

    def test_below_min_is_noop(self):
        coincs = self._genuine_cluster(2, 0)
        kept, dropped = filter_rigidity(coincs, z_ref=800.0, max_resid_mm=10.0)
        assert kept == coincs
        assert dropped == []

    def test_clean_cluster_is_noop(self):
        coincs = self._genuine_cluster(10, 1)
        kept, dropped = filter_rigidity(coincs, z_ref=800.0, max_resid_mm=1.0)
        assert dropped == []
        assert len(kept) == len(coincs)

    def test_drops_cross_particle_track(self):
        """A track whose (X,Y) is unrelated to its probe hit (u,v) — the
        cross-particle contamination mechanism — violates the invariant
        against every genuine coincidence and gets dropped."""
        core = self._genuine_cluster(10, 2)
        wild_u, wild_v = 100.0, 100.0
        wild = _flat_coinc(x=-2000.0, y=1500.0, u=wild_u, v=wild_v)
        coincs = core + [wild]

        kept, dropped = filter_rigidity(coincs, z_ref=800.0, max_resid_mm=50.0)

        assert dropped == [wild]
        assert set(kept) == set(core)

    def test_drops_everything_in_fully_contaminated_window(self):
        """A window with no genuine coincidences at all (every track is
        cross-particle) must drop everything, down to 0 survivors — the
        caller's min_fit check is what skips the window, not a bypass here.
        This is the real-data failure mode: a premature 'keep >= 3' floor
        guard would make the gate a no-op on exactly the windows it exists
        to catch (see docs/handoffs/2026-07-07-off-probe-track-gate-strategy.md)."""
        wild = [
            _flat_coinc(x=-2000.0 + i, y=1500.0 - i, u=50.0 * i, v=50.0 * i)
            for i in range(5)
        ]

        kept, dropped = filter_rigidity(wild, z_ref=800.0, max_resid_mm=50.0)

        assert kept == []
        assert set(dropped) == set(wild)


# ── filter_off_probe ─────────────────────────────────────────────────────────


class TestFilterOffProbe:
    _THETA = 0.2
    _TX, _TY = 100.0, 100.0
    _ZP = 800.0
    _L = 400.0  # probe footprint side (mm)

    def _ref_pose(self) -> PoseResult:
        return _make_ref_pose(self._TX, self._TY, self._THETA, self._ZP)

    def _track_landing_at(self, u_pred: float, v_pred: float) -> Coincidence:
        """A zero-slope track whose projection lands at (u_pred, v_pred) in the
        reference pose's probe frame, regardless of its own (u, v) hit."""
        c, s = math.cos(self._THETA), math.sin(self._THETA)
        x = self._TX + u_pred * c - v_pred * s
        y = self._TY + u_pred * s + v_pred * c
        return _flat_coinc(x, y, u=u_pred, v=v_pred)

    def test_noop_when_all_within_footprint(self):
        coincs = [
            self._track_landing_at(50.0, 50.0),
            self._track_landing_at(350.0, 350.0),
            self._track_landing_at(0.0, 400.0),
        ]
        kept, dropped = filter_off_probe(
            coincs, self._ref_pose(), probe_size_mm=self._L, max_off_probe_mm=10.0
        )
        assert dropped == []
        assert kept == coincs

    def test_drops_track_landing_off_probe(self):
        core = [
            self._track_landing_at(50.0, 50.0),
            self._track_landing_at(350.0, 350.0),
            self._track_landing_at(200.0, 200.0),
        ]
        wild = self._track_landing_at(-500.0, 200.0)  # 500 mm outside u in [0, L]
        coincs = core + [wild]

        kept, dropped = filter_off_probe(
            coincs, self._ref_pose(), probe_size_mm=self._L, max_off_probe_mm=10.0
        )

        assert dropped == [wild]
        assert kept == core

    def test_drops_everything_when_no_track_lands_on_probe(self):
        """A window where every track lands off-probe drops down to 0
        survivors — the caller's min_fit check skips the window, this
        function does not bypass to preserve a floor (see filter_rigidity's
        docstring for why: a bypass here would no-op on exactly the windows
        the gate exists to catch)."""
        wild = [
            self._track_landing_at(-500.0, 200.0),
            self._track_landing_at(900.0, 200.0),
            self._track_landing_at(200.0, -500.0),
        ]

        kept, dropped = filter_off_probe(
            wild, self._ref_pose(), probe_size_mm=self._L, max_off_probe_mm=10.0
        )

        assert kept == []
        assert set(dropped) == set(wild)
