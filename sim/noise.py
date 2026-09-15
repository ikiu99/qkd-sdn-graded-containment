"""Observation noise model.

Phase 3 reference: section 3.3 of phase3-4-build-spec.

> In a noiseless simulator every detector is perfect.

With a perfect observer K_obs == K_exp holds exactly for every healthy node, the
residual is identically zero, any threshold scores AUC = 1.0 and the whole of
phase 4 is vacuous.  This module is therefore a first class component, not a
refinement.

Four independent sources:

  N1  relative QBER measurement noise            sigma_qber_rel
  N2  key rate jitter                            sigma_skr_rel
  N3  buffer report noise, per edge *side*       sigma_report_rel
  N4  key consumption invisible to the controller  unmanaged_fraction

N1-N3 live here.  N4 is a traffic stream, so it lives in the demand path: see
``runner`` and ``SessionManager`` with ``managed=False``.  N4 is the only source
that produces a non zero x1 residual for a *healthy* node - remove it and F1
scores AUC = 1.0 against profile G, which is exactly the failure the spec warns
about.

A fifth, free source comes from sessions that straddle a window boundary; it
needs no code but should be named as an independent source in the analysis.

All four scale together through ``noise.scale``, which is the noise sweep axis.

Stream isolation: the observation noise draws from its own generator, seeded
from ``seed_noise``.  If it shared the attack stream, changing the attack
profile would move the noise realisation and profile-to-profile comparisons
would be contaminated; if it shared the demand stream, changing lambda would.
"""
from __future__ import annotations

import numpy as np

from .config import NoiseConfig
from .state import NetworkState
from .topology import Topology

QBER_MAX = 0.5


class ObservationModel:
    """Turns the true state into what the SDN controller gets to see."""

    def __init__(self, cfg: NoiseConfig, topo: Topology, rng: np.random.Generator):
        self.cfg = cfg
        self.topo = topo
        self.rng = rng
        self.n_edges = topo.n_edges
        self.sigma_qber = float(cfg.sigma_qber_rel * cfg.scale)
        self.sigma_skr = float(cfg.sigma_skr_rel * cfg.scale)
        self.sigma_report = float(cfg.sigma_report_rel * cfg.scale)

    # ------------------------------------------------------------------ #
    def observe(self, state: NetworkState, t: float) -> None:
        """Refresh every ``*_obs`` field from the true state.

        Called once per detector sample (every ``T_sample``), not every step:
        that is when the controller actually polls, and it keeps the RNG cost
        proportional to the number of samples rather than the number of steps.
        """
        rng = self.rng
        n = self.n_edges

        # N1 - QBER measurement noise
        if self.sigma_qber > 0.0:
            q = state.qber_true * (1.0 + rng.normal(0.0, self.sigma_qber, n))
            np.clip(q, 0.0, QBER_MAX, out=q)
        else:
            q = state.qber_true.copy()
        state.qber_obs = q

        # N2 - key rate jitter
        if self.sigma_skr > 0.0:
            r = state.R_eff * (1.0 + rng.normal(0.0, self.sigma_skr, n))
            np.clip(r, 0.0, None, out=r)
        else:
            r = state.R_eff.copy()
        state.R_obs = r

        # N3 - buffer report noise, drawn independently for each end of the link
        if self.sigma_report > 0.0:
            rep = state.buffers[:, None] * (1.0 + rng.normal(0.0, self.sigma_report,
                                                             (n, 2)))
            np.clip(rep, 0.0, None, out=rep)
        else:
            rep = np.repeat(state.buffers[:, None], 2, axis=1)
        state.buffer_reported = rep

    # ------------------------------------------------------------------ #
    def report_noise(self, size) -> np.ndarray:
        """Multiplicative report noise factor, for an attacker that overwrites a
        report and still has to look like a noisy measurement."""
        if self.sigma_report <= 0.0:
            return np.ones(size)
        return 1.0 + self.rng.normal(0.0, self.sigma_report, size)
