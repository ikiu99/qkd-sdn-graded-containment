"""Metric accumulators and node level telemetry sampling.

Phase 2 reference: section 4.7 of phase2-build-spec.
Phase 3 reference: section 3.6 of phase3-4-build-spec.

Absolute rule: no event with t < warmup enters a metric, while the simulation
itself runs normally through the warm-up window so that buffers and the session
population reach steady state first.

Key accounting is kept twice: post warm-up (reported) and whole run (used by the
conservation invariant, which must hold over the entire simulation).
"""
from __future__ import annotations

import numpy as np

from .state import NetworkState, Session
from .topology import Topology

TELEMETRY_COLUMNS = (
    "t", "node_id",
    "x1", "x2", "x3", "e1", "e2", "e3",
    "S", "S_bar", "prior", "rho", "isolated",
    "node_key_flow", "adjacent_buffer_mean", "qber_obs_mean",
    "exposed", "compromised", "t_compromise",
)


class MetricsAccumulator:
    def __init__(self, warmup: float, topo: Topology | None = None,
                 keep_session_log: bool = False):
        self.warmup = float(warmup)
        self.topo = topo

        # counters (post warm-up)
        self.n_admitted = 0
        self.n_rejected = 0
        self.n_completed = 0
        self.n_interrupted = 0
        self.n_starvation_events = 0
        self.n_unreachable = 0          # rejected because no path existed at all
        self.n_no_disjoint = 0          # rejected: fewer than m disjoint legs
        self.sum_path_hops = 0
        self.sum_leg_hops = 0
        self.sum_legs = 0
        self.n_background_admitted = 0
        self.n_reclassified = 0         # sessions re-marked as exposed at t_c

        # key accounting
        self.key_generated = 0.0        # post warm-up
        self.key_consumed = 0.0
        self.key_consumed_managed = 0.0
        self.key_generated_all = 0.0    # whole run, for the conservation invariant
        self.key_consumed_all = 0.0

        # damage (phase 3)
        self.n_closed = 0               # managed sessions closed post warm-up
        self.n_closed_exposed = 0
        self.compromised_relays: set[int] = set()

        # per step aggregates (post warm-up)
        self.n_steps = 0
        self.sum_buffer = 0.0
        self.sum_active = 0

        # telemetry
        self._tel: dict[str, list] = {k: [] for k in TELEMETRY_COLUMNS}
        self._last_cum: np.ndarray | None = None
        self._last_tel_t: float | None = None

        # cumulative consumption snapshot at the end of warm-up, for Phi_i
        self._cum_at_warmup: np.ndarray | None = None

        # optional full session log, used by test_damage_accounting
        self.keep_session_log = bool(keep_session_log)
        self.session_log: list[Session] = []

    # ------------------------------------------------------------------ #
    def active(self, t: float) -> bool:
        return t >= self.warmup

    def note_generated(self, bits: float, t: float) -> None:
        self.key_generated_all += bits
        if t >= self.warmup:
            self.key_generated += bits

    def note_consumed(self, bits: float, managed_bits: float, t: float) -> None:
        """``bits`` is everything that left the buffers, ``managed_bits`` only the
        part the controller authorised.  The difference is the N4 background
        stream plus whatever a greedy attacker siphoned off."""
        self.key_consumed_all += bits
        if t >= self.warmup:
            self.key_consumed += bits
            self.key_consumed_managed += managed_bits

    # ------------------------------------------------------------------ #
    def on_admit(self, session: Session, t: float) -> None:
        if t < self.warmup:
            return
        self.n_admitted += 1
        # Three quantities, not one.  mean_path_len (total union hops) is the
        # key-cost axis; mean_leg_len = sum_leg_hops / sum_legs is path STRETCH
        # and is what PSI measures.  Folding the m-fold replication into stretch
        # would correlate the two Pareto axes and make the frontier misleading.
        self.sum_path_hops += session.hops
        self.sum_leg_hops += session.hops
        self.sum_legs += session.n_legs

    def on_background_admit(self, session: Session, t: float) -> None:
        if t < self.warmup:
            return
        self.n_background_admitted += 1

    def on_reject(self, req, t: float, reason: str = "key") -> None:
        if t < self.warmup:
            return
        self.n_rejected += 1
        if reason == "unreachable":
            self.n_unreachable += 1
        elif reason == "no_disjoint":
            # structural: the graph cannot supply m node-disjoint legs.  Kept
            # apart from a key shortage, or B2's rejection rate would read as a
            # capacity problem when it is a topology one.
            self.n_no_disjoint += 1

    def on_complete(self, session: Session, t: float) -> None:
        if t < self.warmup:
            return
        self.n_completed += 1

    def on_interrupt(self, session: Session, t: float) -> None:
        if t < self.warmup:
            return
        self.n_interrupted += 1

    def on_starvation(self, n_edges: int, t: float) -> None:
        if t < self.warmup:
            return
        self.n_starvation_events += n_edges

    def note_relays(self, session: Session, compromised: np.ndarray) -> None:
        """Record which compromised nodes actually acted as a relay.

        A compromised node of degree 1, or one that no path ever crosses, has a
        structurally zero D_raw; counting it as a 'compromised relay' would
        understate the per relay damage.
        """
        if compromised is None or not compromised.any():
            return
        for leg in session.legs:
            for n in leg[1:-1]:
                if compromised[n]:
                    self.compromised_relays.add(int(n))

    def note_closed(self, session: Session, t: float) -> None:
        if self.keep_session_log:
            self.session_log.append(session)
        if t < self.warmup or not session.managed:
            return
        self.n_closed += 1
        if session.exposed:
            self.n_closed_exposed += 1

    def on_step(self, state: NetworkState, t: float) -> None:
        if self._cum_at_warmup is None and t >= self.warmup:
            self._cum_at_warmup = state.actual_consumed.copy()
        if t < self.warmup:
            return
        self.n_steps += 1
        self.sum_buffer += float(state.buffers.sum())
        self.sum_active += len(state.sessions)

    # ------------------------------------------------------------------ #
    def log_telemetry(self, state: NetworkState, t: float, detector=None) -> None:
        """Sample node level telemetry.

        Called once per ``telemetry_period``, which is deliberately coarser than
        the detector's own ``T_sample``: the detector needs 120 samples in its
        base window, the log does not, and 86400 steps x 50 nodes would be 4.3M
        rows per run.
        """
        cum = state.actual_consumed
        if self._last_cum is None:
            delta = np.zeros_like(cum)
            period = 1.0
        else:
            delta = cum - self._last_cum
            period = max(t - float(self._last_tel_t), 1e-9)
        self._last_cum = cum.copy()
        self._last_tel_t = t

        if t < self.warmup or self.topo is None:
            return

        topo = self.topo
        x = getattr(detector, "x", None)
        ev = getattr(detector, "e", None)
        S = getattr(detector, "S", None)
        prior = getattr(detector, "prior", None)
        exposed_nodes = _exposed_node_mask(state, topo.n_nodes)

        for i in range(topo.n_nodes):
            adj = topo.node_edges[i]
            self._tel["t"].append(float(t))
            self._tel["node_id"].append(int(i))
            for k in range(3):
                self._tel[f"x{k + 1}"].append(float(x[i, k]) if x is not None else 0.0)
                self._tel[f"e{k + 1}"].append(float(ev[i, k]) if ev is not None else 0.0)
            self._tel["S"].append(float(S[i]) if S is not None else 0.0)
            self._tel["S_bar"].append(float(state.S_bar[i]))
            self._tel["prior"].append(float(prior[i]) if prior is not None else 0.0)
            self._tel["rho"].append(float(state.rho[i]))
            self._tel["isolated"].append(bool(state.isolated[i]))
            self._tel["node_key_flow"].append(float(delta[adj].sum() / period))
            self._tel["adjacent_buffer_mean"].append(float(state.buffers[adj].mean()))
            self._tel["qber_obs_mean"].append(float(state.qber_obs[adj].mean()))
            self._tel["exposed"].append(bool(exposed_nodes[i]))
            self._tel["compromised"].append(bool(state.compromised[i]))
            self._tel["t_compromise"].append(float(state.t_compromise[i]))

    def telemetry_table(self) -> dict[str, list]:
        return self._tel

    # ------------------------------------------------------------------ #
    def node_key_flow(self, state: NetworkState) -> np.ndarray:
        """Total key relayed through each node after warm-up, bits.

        This is the raw quantity behind Phi_i of phase 1 section 7.
        """
        if self.topo is None:
            raise RuntimeError("node_key_flow needs a topology")
        base = (self._cum_at_warmup if self._cum_at_warmup is not None
                else np.zeros_like(state.actual_consumed))
        delta = state.actual_consumed - base
        return np.array([delta[self.topo.node_edges[i]].sum()
                         for i in range(self.topo.n_nodes)], dtype=float)

    # ------------------------------------------------------------------ #
    def summary(self, state: NetworkState | None = None) -> dict:
        offered = self.n_admitted + self.n_rejected
        n_e = self.topo.n_edges if self.topo is not None else 1
        b_max = self.topo.B_max if self.topo is not None else 1.0
        out = {
            "n_admitted": self.n_admitted,
            "n_rejected": self.n_rejected,
            "n_offered": offered,
            "n_completed": self.n_completed,
            "n_interrupted": self.n_interrupted,
            "n_starvation_events": self.n_starvation_events,
            "n_unreachable": self.n_unreachable,
            "n_no_disjoint": self.n_no_disjoint,
            "n_background_admitted": self.n_background_admitted,
            "RR": (self.n_rejected / offered) if offered else 0.0,
            "KPD": (self.key_consumed_managed / self.n_admitted) if self.n_admitted else 0.0,
            "mean_path_len": (self.sum_path_hops / self.n_admitted) if self.n_admitted else 0.0,
            "mean_leg_len": (self.sum_leg_hops / self.sum_legs) if self.sum_legs else 0.0,
            "mean_legs": (self.sum_legs / self.n_admitted) if self.n_admitted else 0.0,
            "mean_buffer_util": (self.sum_buffer / (self.n_steps * n_e * b_max))
                                if self.n_steps else 0.0,
            "mean_active_sessions": (self.sum_active / self.n_steps) if self.n_steps else 0.0,
            "total_key_generated": self.key_generated,
            "total_key_consumed": self.key_consumed,
            "total_key_consumed_managed": self.key_consumed_managed,
            "total_key_generated_all": self.key_generated_all,
            "total_key_consumed_all": self.key_consumed_all,
            "n_steps_recorded": self.n_steps,
            # --- phase 3 damage ---
            "n_reclassified_at_tc": self.n_reclassified,
            "n_compromised_relays": len(self.compromised_relays),
            "exposed_session_ratio": (self.n_closed_exposed / self.n_closed)
                                     if self.n_closed else 0.0,
        }
        if state is not None:
            out.update({
                "D_eff": state.D_eff,
                "D_raw": state.D_raw,
                "D_raw_hops": state.D_raw_hops,
                "D_eff_relay": state.D_eff_relay,
                "D_raw_relay": state.D_raw_relay,
                "D_eff_endpoint": state.D_eff - state.D_eff_relay,
                "n_compromised": int(state.compromised.sum()),
            })
        return out


def _exposed_node_mask(state: NetworkState, n_nodes: int) -> np.ndarray:
    """Nodes currently carrying at least one exposed session."""
    mask = np.zeros(n_nodes, dtype=bool)
    for sid in state.exposed_sessions:
        s = state.sessions.get(sid)
        if s is None:
            continue
        for leg in s.legs:
            for n in leg:
                mask[n] = True
    return mask
