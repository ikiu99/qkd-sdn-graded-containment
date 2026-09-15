"""Compromised relay attack: ground truth plus the four behavioural profiles.

Phase 3 reference: section 3.4 of phase3-4-build-spec.

Each profile switches on exactly one detection feature, and that one to one
correspondence is what makes the ablation table of phase 4 meaningful:

    P  passive       nothing observable changes      -> no feature
    G  greedy        extra key drained off ledger    -> F1  (x1)
    L  liar          over reports adjacent buffers   -> F2  (x2)
    E  eavesdropper  taps one adjacent link          -> F3  (x3)

Profile P is the control condition and is *deliberately* undetectable: in this
implementation it provably changes nothing in the simulated dynamics, so a run
with profile P is bit identical to a run with no attack at all.  An AUC near 0.5
there is the correct result, not a failure.

Determinism contract: the compromised set V_c is drawn **before** any profile
specific random draw, so the same ``seed_attack`` compromises the same nodes in
all four profiles and the profiles stay paired for comparison.
"""
from __future__ import annotations

import math

import numpy as np

from .config import AttackConfig
from .interfaces import Attack
from .noise import ObservationModel
from .state import NetworkState
from .topology import Topology

INF = float("inf")


class CompromiseAttack(Attack):
    """Persistent compromise of a node subset, starting at t_c."""

    def __init__(self, cfg: AttackConfig, n_nodes: int, rng: np.random.Generator,
                 horizon: float, topo: Topology, dt: float = 1.0,
                 key_flow: np.ndarray | None = None,
                 noise: ObservationModel | None = None, **_: object):
        self.cfg = cfg
        self.topo = topo
        self.rng = rng
        self.dt = float(dt)
        self.noise = noise
        self.t_c = float(0.25 * horizon if cfg.t_c is None else cfg.t_c)
        self.profile = cfg.profile

        # --- ground truth: choose V_c FIRST, before any profile specific draw,
        #     so that the same seed gives the same compromised set in P/G/L/E ---
        n_c = int(math.ceil(cfg.f * n_nodes))
        n_c = max(0, min(n_c, n_nodes))
        if n_c == 0:
            victims = np.empty(0, dtype=np.int64)
        elif cfg.selection == "random":
            # degree-1 nodes stay in the draw: excluding them would bias the
            # sample.  They are reported separately in the analysis instead,
            # because their D_raw is structurally zero.
            victims = rng.choice(n_nodes, size=n_c, replace=False)
        elif cfg.selection == "top_keyflow":
            if key_flow is None:
                raise ValueError("selection='top_keyflow' needs the phase 2 key flow "
                                 "baseline (config/calibration_*.yaml)")
            victims = np.argsort(np.asarray(key_flow, dtype=float),
                                 kind="stable")[-n_c:]
        else:
            raise ValueError(f"unknown attack.selection {cfg.selection!r}")

        self.victims = np.sort(np.asarray(victims, dtype=np.int64))
        self._mask = np.zeros(n_nodes, dtype=bool)
        self._mask[self.victims] = True
        self._zero_mask = np.zeros(n_nodes, dtype=bool)

        # edges incident to any compromised node, used by G and L
        adj = sorted({int(e) for i in self.victims for e in topo.node_edges[i]})
        self.adj_edges = np.array(adj, dtype=np.int64)

        # --- profile specific draws, after V_c ---
        self.tap_edges = np.empty(0, dtype=np.int64)
        if cfg.profile == "E":
            taps = []
            for i in self.victims:                       # sorted -> deterministic
                inc = topo.node_edges[i]
                if inc.size:
                    taps.append(int(inc[int(rng.integers(inc.size))]))
            # np.unique, not a sum: two compromised nodes sharing a tapped link
            # must not double the effect
            self.tap_edges = np.unique(np.array(taps, dtype=np.int64))

        # profile L: the (edge, side) pairs each compromised node reports on
        if cfg.profile == "L" and self.adj_edges.size:
            ends = topo.edge_ends[self.adj_edges]                    # (m,2)
            lo_bad = self._mask[ends[:, 0]]
            hi_bad = self._mask[ends[:, 1]]
            pairs = ([(int(e), 0) for e, bad in zip(self.adj_edges, lo_bad) if bad] +
                     [(int(e), 1) for e, bad in zip(self.adj_edges, hi_bad) if bad])
            pairs.sort()
            self.lie_edges = np.array([p[0] for p in pairs], dtype=np.int64)
            self.lie_sides = np.array([p[1] for p in pairs], dtype=np.int64)
        else:
            self.lie_edges = np.empty(0, dtype=np.int64)
            self.lie_sides = np.empty(0, dtype=np.int64)

        # profile G: EMA of the managed consumption rate per edge.  Tracked from
        # t=0 so the attacker already has a baseline when it switches on.
        self._baseline = np.zeros(topo.n_edges, dtype=float)
        self._prev_ledger = np.zeros(topo.n_edges, dtype=float)
        self._rate_buf = np.zeros(topo.n_edges, dtype=float)
        self._drain_buf = np.zeros(topo.n_edges, dtype=float)
        # gamma*dt on the links incident to a compromised node, 0 elsewhere, so
        # the per step drain is whole-array arithmetic with no fancy indexing
        self._gain = np.zeros(topo.n_edges, dtype=float)
        self._gain[self.adj_edges] = cfg.gamma * self.dt
        self._alpha_b = self.dt / 300.0
        self.armed = False
        # key siphoned on the most recent step, so the runner can book it as
        # consumption without taking a second full sum over the buffers
        self.last_drain = 0.0

    # ------------------------------------------------------------------ #
    def compromised_nodes(self, t: float) -> np.ndarray:
        return self._mask if t >= self.t_c else self._zero_mask

    # ------------------------------------------------------------------ #
    def apply(self, state: NetworkState, t: float) -> None:
        if self.profile == "G":
            self._track_baseline(state)

        if t < self.t_c:
            return
        if not self.armed:
            self._arm(state, t)

        if self.profile == "G":
            self._drain(state)

    def _track_baseline(self, state: NetworkState) -> None:
        """EMA of the *managed* consumption rate of each edge.

        Derived from the ledger delta rather than from edge_rate, because in AES
        edge_rate is zero and all consumption arrives as pulses.
        """
        rate = self._rate_buf
        np.subtract(state.ledger_consumed, self._prev_ledger, out=rate)
        rate /= self.dt
        np.copyto(self._prev_ledger, state.ledger_consumed)
        rate -= self._baseline
        rate *= self._alpha_b
        self._baseline += rate

    def _arm(self, state: NetworkState, t: float) -> None:
        self.armed = True
        state.compromised = self._mask.copy()
        state.t_compromise[self._mask] = self.t_c

        if self.profile == "E" and self.tap_edges.size:
            # np.maximum, never +=: overlapping taps must not stack
            np.maximum.at(state.attack_qber_delta, self.tap_edges, self.cfg.q)
            np.maximum.at(state.attack_skr_drop, self.tap_edges, self.cfg.s)
            state.qber_true = self.topo.qber_base + state.attack_qber_delta
            state.R_eff = self.topo.R * (1.0 - state.attack_skr_drop)

    def _drain(self, state: NetworkState) -> None:
        """Profile G: siphon gamma times the edge's normal managed rate.

        Never recorded in ledger_consumed, which is exactly what produces the
        x1 residual.  An edge carrying no traffic is drained by nothing: the
        attacker has nothing to steal there.
        """
        if self.adj_edges.size == 0:
            return
        take = self._drain_buf
        np.multiply(self._baseline, self._gain, out=take)
        np.minimum(take, state.buffers, out=take)   # cannot steal what is absent
        state.buffers -= take
        state.actual_consumed += take
        self.last_drain = float(take.sum())

    # ------------------------------------------------------------------ #
    def observe(self, state: NetworkState, t: float) -> None:
        """Tamper with the reports after the honest observation model ran.

        Profile L only: the compromised end of each adjacent link over reports
        its buffer by (1+delta), while the healthy end keeps reporting honestly,
        which is precisely the discrepancy x2 measures.
        """
        if not self.armed or self.profile != "L" or self.lie_edges.size == 0:
            return
        e, side = self.lie_edges, self.lie_sides
        factor = (self.noise.report_noise(e.size) if self.noise is not None
                  else np.ones(e.size))
        state.buffer_reported[e, side] = (state.buffers[e] * (1.0 + self.cfg.delta)
                                          * factor)
