"""Mutable simulation state.

Phase 2 reference: section 2 of phase2-build-spec.
Phase 3 reference: section 3.1 of phase3-4-build-spec.

Naming rule, mandatory from phase 3 on: every observable exists twice, as
``*_true`` (only the simulator knows it) and ``*_obs`` (what the controller
sees).  The phase 4 detector may read **only** the ``_obs`` family plus
``buffer_reported`` and ``ledger_consumed``; touching ``*_true``,
``buffers`` or ``compromised`` is a bug, and ``test_detector_isolation``
enforces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import Topology

INF = float("inf")


@dataclass
class Session:
    sid: int
    src: int
    dst: int
    legs: tuple[tuple[int, ...], ...]   # one node path per XOR share
    edges: tuple[int, ...]       # union of all legs' edge ids, no duplicates
    t_start: float
    t_end: float                 # t_start + T_s
    data_rate: float             # bit/s of protected traffic, both KM modes
    key_rate: float              # bit/s key consumption per edge (0 in AES / reserve_full)
    next_rekey: float            # AES only, time of the next pulse (inf when none)
    pulse_bits: float = 0.0      # AES only, key bits consumed per edge per pulse
    total_key_need: float = 0.0  # bits per edge over the whole session
    exposed_bits: float = 0.0    # accumulated D_eff contributed by this session
    status: str = "active"       # "active" | "completed" | "interrupted"
    managed: bool = True         # False = N4 background traffic, invisible to the ledger
    exposed: bool = False        # any node on the path is compromised
    exposed_relay: bool = False  # a compromised node acts as an intermediate
    t_exposed: float = INF       # when it entered the exposed set
    t_close: float = INF         # when it completed or was interrupted
    n_exposed_edges: int = 0     # hops incident to a compromised node

    @property
    def hops(self) -> int:
        """Total hops over every leg - the key-cost measure.

        Under B2 this is m times a single leg, because each leg carries a
        full-size XOR share.  Path *stretch* is ``mean_leg_len`` in metrics.py,
        deliberately a separate quantity: folding the m-fold replication into
        stretch would correlate the two Pareto axes, and replication already
        shows up in KPD and RR.
        """
        return len(self.edges)

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def path(self) -> tuple[int, ...]:
        """The single path, for the single-leg case only.

        Deliberately raises rather than silently returning leg 0: a caller that
        assumes one path while looking at a multipath session would produce a
        quietly wrong answer, and there is no safe default here.
        """
        if len(self.legs) != 1:
            raise AttributeError(
                f"session {self.sid} has {len(self.legs)} legs; use .legs")
        return self.legs[0]

    def interiors(self) -> tuple[frozenset[int], ...]:
        return tuple(frozenset(leg[1:-1]) for leg in self.legs)

    def all_nodes(self):
        seen = set()
        for leg in self.legs:
            seen.update(leg)
        return seen


@dataclass
class NetworkState:
    t: float
    buffers: np.ndarray              # (|E|,) bit - TRUE buffer, simulator only
    edge_rate: np.ndarray            # (|E|,) aggregate consumption rate, bit/s
    edge_rate_mgd: np.ndarray        # (|E|,) same, managed sessions only (the ledger)
    edge_n_mgd: np.ndarray           # (|E|,) int, count of active managed sessions
    edge_sessions: list[set[int]]    # per edge, set of active sids
    sessions: dict[int, Session]
    S_bar: np.ndarray                # (|V|,) smoothed score
    node_weight: np.ndarray          # (|V|,) routing weight, all one under NullPolicy
    rho: np.ndarray                  # (|V|,) rate quota, all one under NullPolicy
    isolated: np.ndarray             # (|V|,) bool, all False under NullPolicy
    n_paths: int = 1                 # legs per session; >1 only under B2
    # B8 only.  hybrid_tau < 1 turns on the risk trigger: a session whose
    # cheapest single path exceeds this cumulative compromise probability
    # is relayed over hybrid_m XOR legs instead.  It lives on the state
    # rather than being passed down because admission is the only place
    # that knows which path a given session would actually take.
    hybrid_tau: float = 1.0
    hybrid_m: int = 2
    rekey_heap: list = field(default_factory=list)   # AES: (t_rekey, sid) min heap

    # --- ground truth (phase 3) --------------------------------------------
    compromised: np.ndarray = None    # (|V|,) bool
    t_compromise: np.ndarray = None   # (|V|,) float, inf where healthy

    # --- quantum layer observables ----------------------------------------
    qber_true: np.ndarray = None      # (|E|,) real error rate
    qber_obs: np.ndarray = None       # (|E|,) reported error rate (noisy)
    R_eff: np.ndarray = None          # (|E|,) effective generation rate after the attack
    R_obs: np.ndarray = None          # (|E|,) reported generation rate (noisy)

    # --- key layer observables --------------------------------------------
    buffer_reported: np.ndarray = None   # (|E|,2) report of each end of the link
    # column 0 = report of the lower numbered node, column 1 = the higher one

    # --- controller ledger -------------------------------------------------
    ledger_consumed: np.ndarray = None   # (|E|,) key the controller expects to be gone
    actual_consumed: np.ndarray = None   # (|E|,) key that really left the buffer

    # --- attack effect, simulator only -------------------------------------
    attack_qber_delta: np.ndarray = None  # (|E|,) absolute QBER increase
    attack_skr_drop: np.ndarray = None    # (|E|,) relative SKR drop in [0,1)

    # --- damage accounting -------------------------------------------------
    exposed_sessions: set[int] = field(default_factory=set)
    exposed_rate: float = 0.0        # sum of data_rate over exposed sessions
    exposed_key_rate: float = 0.0    # sum of key_rate over exposed sessions
    exposed_hop_key_rate: float = 0.0  # same, weighted by exposed hop count
    D_eff: float = 0.0               # protected data bits exposed
    D_raw: float = 0.0               # key bits exposed, counted once per session
    D_raw_hops: float = 0.0          # key bits summed over compromised-incident hops

    # Relay-only exposure: the same quantities counting ONLY sessions where a
    # compromised node acts as an intermediate.  A session whose own endpoint is
    # compromised is exposed no matter how it is routed - the attacker owns one
    # end of it - so that part of D_eff is a floor no relay policy can touch, and
    # mixing the two would cap DRR at a value set by the traffic matrix rather
    # than by the policy.  D_eff stays the headline metric; D_eff_relay is what
    # measures what a policy can actually do.
    exposed_rate_relay: float = 0.0
    exposed_key_rate_relay: float = 0.0
    D_eff_relay: float = 0.0
    D_raw_relay: float = 0.0


def init_state(topo: Topology, t0: float = 0.0, buffers_init: str = "full") -> NetworkState:
    """Fresh state. ``buffers_init``: "full" starts at B_max, "empty" at zero.

    Buffers start full by default; the warm-up window then lets the offered load
    pull them down to their steady state before any metric is recorded.
    """
    if buffers_init == "full":
        buffers = np.full(topo.n_edges, topo.B_max, dtype=float)
    elif buffers_init == "empty":
        buffers = np.zeros(topo.n_edges, dtype=float)
    else:
        raise ValueError("buffers_init must be 'full' or 'empty'")

    n_e, n_v = topo.n_edges, topo.n_nodes
    return NetworkState(
        t=t0,
        buffers=buffers,
        edge_rate=np.zeros(n_e, dtype=float),
        edge_rate_mgd=np.zeros(n_e, dtype=float),
        edge_n_mgd=np.zeros(n_e, dtype=np.int64),
        edge_sessions=[set() for _ in range(n_e)],
        sessions={},
        S_bar=np.zeros(n_v, dtype=float),
        node_weight=np.ones(n_v, dtype=float),
        rho=np.ones(n_v, dtype=float),
        isolated=np.zeros(n_v, dtype=bool),
        rekey_heap=[],
        compromised=np.zeros(n_v, dtype=bool),
        t_compromise=np.full(n_v, INF, dtype=float),
        qber_true=topo.qber_base.copy(),
        qber_obs=topo.qber_base.copy(),
        R_eff=topo.R.copy(),
        R_obs=topo.R.copy(),
        buffer_reported=np.repeat(buffers[:, None], 2, axis=1),
        ledger_consumed=np.zeros(n_e, dtype=float),
        actual_consumed=np.zeros(n_e, dtype=float),
        attack_qber_delta=np.zeros(n_e, dtype=float),
        attack_skr_drop=np.zeros(n_e, dtype=float),
    )


def recompute_edge_rate_from_sessions(state: NetworkState,
                                      managed_only: bool = False) -> np.ndarray:
    """Rebuild ``edge_rate`` by walking the active sessions.

    Reference implementation for the incremental bookkeeping - expensive, used
    only by the invariant check and the tests.
    """
    out = np.zeros_like(state.edge_rate)
    for s in state.sessions.values():
        if s.status != "active" or (managed_only and not s.managed):
            continue
        for e in s.edges:
            out[e] += s.key_rate
    return out


def recompute_exposure(state: NetworkState) -> tuple[float, float, float]:
    """Reference recomputation of the incremental exposure aggregates."""
    rate = key_rate = hop_key_rate = 0.0
    for sid in state.exposed_sessions:
        s = state.sessions.get(sid)
        if s is None or s.status != "active":
            continue
        rate += s.data_rate
        key_rate += s.key_rate
        hop_key_rate += s.key_rate * s.n_exposed_edges
    return rate, key_rate, hop_key_rate


def recompute_exposure_relay(state: NetworkState) -> tuple[float, float]:
    rate = key_rate = 0.0
    for sid in state.exposed_sessions:
        s = state.sessions.get(sid)
        if s is None or s.status != "active" or not s.exposed_relay:
            continue
        rate += s.data_rate
        key_rate += s.key_rate
    return rate, key_rate


def assert_invariants(state: NetworkState, topo: Topology, deep: bool = False) -> None:
    """Runtime invariants, section 6 of phase2-build-spec."""
    assert np.all(state.buffers >= -1e-6), "negative key buffer"
    assert np.all(state.buffers <= topo.B_max + 1e-6), "buffer above B_max"
    assert np.all(state.edge_rate >= -1e-6), "negative aggregate consumption rate"
    if deep:
        ref = recompute_edge_rate_from_sessions(state)
        assert np.allclose(state.edge_rate, ref, atol=1e-6), (
            "edge_rate drifted from the active sessions"
        )
        ref_mgd = recompute_edge_rate_from_sessions(state, managed_only=True)
        assert np.allclose(state.edge_rate_mgd, ref_mgd, atol=1e-6), (
            "edge_rate_mgd drifted from the active managed sessions"
        )
        rate, key_rate, hop_key_rate = recompute_exposure(state)
        assert abs(state.exposed_rate - rate) < 1e-6, "exposed_rate drifted"
        assert abs(state.exposed_key_rate - key_rate) < 1e-6, "exposed_key_rate drifted"
        assert abs(state.exposed_hop_key_rate - hop_key_rate) < 1e-6, (
            "exposed_hop_key_rate drifted")
        rate_r, key_rate_r = recompute_exposure_relay(state)
        assert abs(state.exposed_rate_relay - rate_r) < 1e-6, "exposed_rate_relay drifted"
        assert abs(state.exposed_key_rate_relay - key_rate_r) < 1e-6, (
            "exposed_key_rate_relay drifted")
