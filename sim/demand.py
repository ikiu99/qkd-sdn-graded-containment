"""Demand generation, admission control, key consumption and damage accounting.

Phase 2 reference: sections 3 and 4.5 of phase2-build-spec.
Phase 3 reference: sections 3.3 (N4) and 3.5 of phase3-4-build-spec.

Performance contract (phase 2 section 3, not optional): per edge aggregate
consumption rate, updated only when a session starts or ends, so a step costs
O(|E|) instead of O(sessions x hops).  AES consumption is not a rate but a train
of 256 bit pulses, driven by a min heap, because those pulses have a real effect
on the buffer dynamics and must not be smoothed into an average.

Damage accounting obeys the same contract: walking the active sessions every
step would destroy it, so exposure is kept as three running aggregates that
change only on three events - session open, session close, and the single
reclassification at t_c.  The per step cost is then O(1).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .config import DemandConfig
from .metrics import MetricsAccumulator
from .routing import RouteTable
from .state import NetworkState, Session
from .topology import Topology, edges_of_path

INF = float("inf")



def path_risk(S_bar: np.ndarray, interior) -> float:
    """Probability that a path carries at least one compromised relay.

        P = 1 - prod over interior nodes of (1 - S_bar_i)

    Cumulative rather than the maximum along the path, because exposure needs
    only ONE compromised relay: a long path through moderately suspect nodes is
    genuinely more dangerous than a short path through one slightly worse node,
    and the maximum cannot tell those apart.

    It reads S_bar as a probability, which it is not exactly - it is a
    calibrated posterior, not a frequency.  What a trigger needs is monotonicity
    in the right direction, and that holds; the threshold it is compared against
    is swept rather than derived, so nothing rests on the calibration being
    exact.
    """
    if not len(interior):
        return 0.0
    clean = 1.0
    for j in interior:
        clean *= 1.0 - min(max(float(S_bar[j]), 0.0), 1.0)
    return 1.0 - clean


def session_is_exposed(s: Session, comp: np.ndarray) -> tuple[bool, bool]:
    """(exposed, exposed_via_relay) under XOR key sharing.

    The key is reconstructable only from ALL m shares, so a multipath session
    leaks only when every leg carries a compromised relay - ALL over the legs,
    ANY within a leg.  Getting this backwards is the single most expensive
    mistake available here: an OR over the union would report B2 as leaking
    whenever *one* leg is touched, which is the opposite of what XOR does, and
    would overstate the paper's headline metric with no error anywhere.

    Endpoints are separate.  A compromised source or destination owns its own
    plaintext however the traffic is routed, so it exposes the session
    regardless of m - and no relay policy can do anything about it, which is why
    that share is tracked apart as the endpoint floor.
    """
    if comp[s.src] or comp[s.dst]:
        return True, False
    relay = all(any(comp[n] for n in leg[1:-1]) for leg in s.legs)
    return relay, relay


@dataclass
class DemandRequest:
    rid: int
    src: int
    dst: int
    t: float
    T_s: float
    data_rate: float          # bit/s of protected traffic, drawn in both KM modes


class DemandGenerator:
    """Poisson arrivals, uniform src/dst, uniform duration and data rate."""

    def __init__(self, cfg: DemandConfig, n_nodes: int, rng: np.random.Generator,
                 lam: float | None = None, rid_offset: int = 0):
        self.cfg = cfg
        self.n_nodes = int(n_nodes)
        self.rng = rng
        self.lam = float(cfg.lam if lam is None else lam)
        self._next_rid = int(rid_offset)

    def arrivals(self, t: float, dt: float) -> list[DemandRequest]:
        if self.lam <= 0.0:
            return []
        cfg = self.cfg
        n = int(self.rng.poisson(self.lam * dt))
        if n == 0:
            return []
        out = []
        for _ in range(n):
            src = int(self.rng.integers(self.n_nodes))
            dst = int(self.rng.integers(self.n_nodes - 1))
            if dst >= src:            # uniform over the n-1 nodes different from src
                dst += 1
            T_s = float(self.rng.uniform(cfg.T_s_min, cfg.T_s_max))
            # data_rate is drawn in BOTH KM modes.  In AES it does not affect key
            # consumption at all, but D_eff is defined in units of protected data
            # (phase 3, section 3.5), so the damage metric needs it either way.
            rate = float(self.rng.uniform(cfg.rate_min, cfg.rate_max))
            out.append(DemandRequest(self._next_rid, src, dst, float(t), T_s, rate))
            self._next_rid += 1
        return out


def admission_rate(cfg: DemandConfig, req: DemandRequest) -> float:
    """Mean key consumption rate per edge used by the admission test, bit/s."""
    if cfg.km_mode == "OTP":
        return req.data_rate
    return cfg.key_size / cfg.T_rk


def total_key_need(cfg: DemandConfig, req: DemandRequest) -> float:
    """Total key consumed per edge over the whole session, bit."""
    if cfg.km_mode == "OTP":
        return req.data_rate * req.T_s
    return math.ceil(req.T_s / cfg.T_rk) * cfg.key_size


class SessionManager:
    def __init__(self, cfg: DemandConfig, topo: Topology,
                 metrics: MetricsAccumulator, rng_policy: np.random.Generator):
        self.cfg = cfg
        self.topo = topo
        self.metrics = metrics
        self.rng_policy = rng_policy
        self._next_sid = 0
        self._end_heap: list[tuple[float, int]] = []
        self._reclassified = False
        # B8: how often the risk trigger fired, and how often the graph
        # could actually supply the legs it asked for.  The gap between
        # them is the sparse-topology failure mode, and it is reported
        # rather than hidden behind the damage number.
        self.n_hybrid_triggered = 0
        self.n_hybrid_served = 0
        self._last_reject = "key"

    # ------------------------------------------------------------------ #
    # admission
    # ------------------------------------------------------------------ #
    def try_admit(self, req: DemandRequest, state: NetworkState,
                  route_table: RouteTable, topo: Topology,
                  cfg: DemandConfig, managed: bool = True) -> Session | None:
        self._last_reject = "key"
        draws: dict[int, bool] = {}      # one quota coin per node per demand
        adm_rate = admission_rate(cfg, req)
        need_total = total_key_need(cfg, req)
        need = (adm_rate * cfg.T_adm if cfg.admission == "optimistic" else need_total)

        if state.n_paths > 1:
            return self._try_admit_multipath(req, state, route_table, topo, cfg,
                                             adm_rate, need_total, need, managed,
                                             draws)

        # Lazy in K: admission usually stops at the cheapest candidate, and
        # materialising all K costs 17-32x more than one Dijkstra.
        for i in range(route_table.K):
            path = route_table.candidate(req.src, req.dst, i)
            if path is None:
                break
            inner = path[1:-1]
            if any(state.isolated[j] for j in inner):
                continue
            if not self._quota_ok(inner, state, draws):
                continue

            # Policy B8: this path is the one the session would actually take,
            # so this is the only point at which its risk can be evaluated.  If
            # it is too likely to carry a compromised relay, spend key on XOR
            # redundancy instead; if the graph cannot supply the legs, fall
            # through and take this path anyway rather than refuse service.
            if state.hybrid_tau < 1.0 and path_risk(state.S_bar, inner) > state.hybrid_tau:
                self.n_hybrid_triggered += 1
                legs = route_table.disjoint_paths(req.src, req.dst, state.hybrid_m)
                if legs and not any(state.isolated[j]
                                    for leg in legs for j in leg[1:-1]):
                    m_edges: list[int] = []
                    for leg in legs:
                        m_edges.extend(edges_of_path(topo, leg))
                    m_eidx = np.fromiter(m_edges, dtype=np.int64,
                                         count=len(m_edges))
                    if np.all(state.buffers[m_eidx] >= need):
                        self.n_hybrid_served += 1
                        return self._open(req, tuple(legs), tuple(m_edges),
                                          m_eidx, state, cfg, adm_rate,
                                          need_total, managed)

            edges = edges_of_path(topo, path)
            eidx = np.fromiter(edges, dtype=np.int64, count=len(edges))
            if not np.all(state.buffers[eidx] >= need):
                continue
            return self._open(req, (path,), edges, eidx, state, cfg,
                              adm_rate, need_total, managed)
        return None

    def _try_admit_multipath(self, req, state, route_table, topo, cfg,
                             adm_rate, need_total, need, managed, draws):
        """Policy B2: one XOR key share per node-disjoint leg, all or nothing.

        Strict by design - fewer than m node-disjoint legs means the demand is
        rejected rather than served with weaker protection, because with XOR
        sharing a single leg offers none at all.  The caller distinguishes that
        rejection as ``no_disjoint`` so a structural shortfall is never counted
        as a key shortage.

        Every leg carries a full-size share, so the per-edge key requirement is
        the same as on a single path and the admission test is one ``np.all``
        over the union.  Node-disjoint implies edge-disjoint, so the union has
        no repeats - asserted at ``_open``, because a near-disjoint bug would
        silently double count edge_rate and the deep invariant would repeat the
        same mistake.
        """
        legs = route_table.disjoint_paths(req.src, req.dst, state.n_paths)
        if not legs:
            self._last_reject = "no_disjoint"
            return None
        for leg in legs:
            if any(state.isolated[j] for j in leg[1:-1]):
                return None
        interior = sorted({j for leg in legs for j in leg[1:-1]})
        if not self._quota_ok(tuple(interior), state, draws):
            return None

        edges: list[int] = []
        for leg in legs:
            edges.extend(edges_of_path(topo, leg))
        eidx = np.fromiter(edges, dtype=np.int64, count=len(edges))
        if not np.all(state.buffers[eidx] >= need):
            return None
        return self._open(req, tuple(legs), tuple(edges), eidx, state, cfg,
                          adm_rate, need_total, managed)

    @property
    def last_reject(self) -> str:
        """Why the most recent try_admit failed: key | no_disjoint | unreachable."""
        return self._last_reject

    def _quota_ok(self, inner, state: NetworkState, draws: dict) -> bool:
        """Probabilistic rate quota over the intermediate relays (phase 5 lever 2).

        ``rho_i`` is the maximum share of relay traffic allowed through node i
        (phase 1, section 8.3), so the acceptance probability of a path is the
        PRODUCT over its intermediate relays.  That compounds sharply - three
        nodes at rho=0.5 pass 12.5% - and the paper should say so.

        ``draws`` memoises the coin per node for the lifetime of one demand.
        Without it the draw is repeated for every candidate path, so a node
        appearing on 3 of the 4 candidates gets three independent trials and its
        effective pass-through becomes 1-(1-rho)^3, which depends on K rather
        than on rho.

        With NullPolicy every rho is 1 and no random number is drawn at all, so
        the policy stream stays untouched in phases 2 to 4 and under B0/B2.
        """
        for i in inner:
            r = state.rho[i]
            if r >= 1.0:
                continue
            passed = draws.get(i)
            if passed is None:
                passed = r > 0.0 and self.rng_policy.random() < r
                draws[i] = passed
            if not passed:
                return False
        return True

    def _open(self, req: DemandRequest, legs: tuple[tuple[int, ...], ...],
              edges: tuple[int, ...], eidx: np.ndarray, state: NetworkState,
              cfg: DemandConfig, adm_rate: float, need_total: float,
              managed: bool) -> Session:
        assert len(set(edges)) == len(edges), (
            "legs are not disjoint; edge_rate would be double counted")
        sid = self._next_sid
        self._next_sid += 1
        reserve = cfg.admission == "reserve_full"

        if reserve:
            key_rate = 0.0
            next_rekey = INF
            pulse_bits = 0.0
        elif cfg.km_mode == "OTP":
            key_rate = req.data_rate       # one key bit per data bit per hop
            next_rekey = INF
            pulse_bits = 0.0
        else:                              # AES: rate is zero, pulses do the work
            key_rate = 0.0
            next_rekey = req.t
            pulse_bits = float(cfg.key_size)

        s = Session(
            sid=sid, src=req.src, dst=req.dst, legs=legs, edges=edges,
            t_start=req.t, t_end=req.t + req.T_s, data_rate=req.data_rate,
            key_rate=key_rate, next_rekey=next_rekey, pulse_bits=pulse_bits,
            total_key_need=need_total, managed=managed,
        )
        state.sessions[sid] = s

        for e in edges:
            state.edge_rate[e] += key_rate
            state.edge_sessions[e].add(sid)
        if managed:
            for e in edges:
                state.edge_rate_mgd[e] += key_rate
                state.edge_n_mgd[e] += 1

        self._mark_exposed(state, s, req.t)

        if reserve:
            # the whole session demand is deducted immediately.  The key material
            # is relayed up front, so an exposed session is charged up front too.
            state.buffers[eidx] -= need_total
            state.actual_consumed[eidx] += need_total
            if managed:
                state.ledger_consumed[eidx] += need_total
            self.metrics.note_consumed(need_total * len(edges),
                                       need_total * len(edges) if managed else 0.0,
                                       req.t)
            if s.exposed:
                state.D_raw += need_total
                state.D_raw_hops += need_total * s.n_exposed_edges
                state.D_eff += req.data_rate * req.T_s
                s.exposed_bits = req.data_rate * req.T_s
                if s.exposed_relay:
                    state.D_raw_relay += need_total
                    state.D_eff_relay += req.data_rate * req.T_s
        elif cfg.km_mode == "AES":
            heapq.heappush(state.rekey_heap, (s.next_rekey, sid))

        heapq.heappush(self._end_heap, (s.t_end, sid))
        if managed:
            self.metrics.note_relays(s, state.compromised)
        return s

    # ------------------------------------------------------------------ #
    # damage accounting (phase 3, section 3.5)
    # ------------------------------------------------------------------ #
    def _mark_exposed(self, state: NetworkState, s: Session, t: float) -> None:
        """Move a session into the exposed set, once.

        Counted once per session even when the path crosses several compromised
        nodes: it is the same key material, and it is not disclosed twice.  The
        N4 background stream is excluded from the damage metrics entirely - it
        exists to create ledger noise, not to be measured.
        """
        if s.exposed or not s.managed or s.status != "active":
            return
        comp = state.compromised
        if not comp.any():
            return
        exposed, relay = session_is_exposed(s, comp)
        if not exposed:
            return
        ends = self.topo.edge_ends
        s.exposed = True
        s.exposed_relay = relay
        s.t_exposed = t
        # D_raw_hops counts key over hops incident to a compromised node, but
        # only on legs that actually leak: under XOR a partially compromised leg
        # discloses nothing.
        leaking = ([leg for leg in s.legs if any(comp[n] for n in leg[1:-1])]
                   if relay else list(s.legs))
        leak_edges = {e for leg in leaking for e in edges_of_path(self.topo, leg)}
        s.n_exposed_edges = int(sum(1 for e in leak_edges
                                    if comp[ends[e, 0]] or comp[ends[e, 1]]))
        state.exposed_sessions.add(s.sid)
        state.exposed_rate += s.data_rate
        state.exposed_key_rate += s.key_rate
        state.exposed_hop_key_rate += s.key_rate * s.n_exposed_edges
        if s.exposed_relay:
            state.exposed_rate_relay += s.data_rate
            state.exposed_key_rate_relay += s.key_rate

    def _unmark_exposed(self, state: NetworkState, s: Session, t: float) -> None:
        if not s.exposed:
            return
        state.exposed_sessions.discard(s.sid)
        state.exposed_rate -= s.data_rate
        state.exposed_key_rate -= s.key_rate
        state.exposed_hop_key_rate -= s.key_rate * s.n_exposed_edges
        if s.exposed_relay:
            state.exposed_rate_relay -= s.data_rate
            state.exposed_key_rate_relay -= s.key_rate
        if self.cfg.km_mode == "OTP" and self.cfg.admission != "reserve_full":
            # exposure never switches off again, so the exposed span is exact
            s.exposed_bits = s.data_rate * max(0.0, t - s.t_exposed)

    def reclassify(self, state: NetworkState, t: float) -> int:
        """One off pass over the active sessions at t_c.

        Runs exactly once, the first step on which any node is compromised.
        """
        n = 0
        for s in list(state.sessions.values()):
            before = s.exposed
            self._mark_exposed(state, s, t)
            n += int(s.exposed and not before)
        return n

    # ------------------------------------------------------------------ #
    # per step
    # ------------------------------------------------------------------ #
    def step(self, state: NetworkState, t: float, dt: float) -> None:
        """Consume key, fire AES pulses, detect starvation, retire sessions.

        Order matters: a session admitted at t0 with duration T_s consumes on the
        steps t0+dt .. t0+T_s inclusive, i.e. exactly T_s/dt steps.
        """
        consumed, managed = self._consume_continuous(state, dt)

        # Reclassify *after* this step's accrual, never before: a session that
        # becomes exposed at t_c must start accruing damage at t_c+dt, exactly
        # like one that is admitted already exposed.  Doing it first over counts
        # every reclassified session by one dt.
        if not self._reclassified and state.compromised.any():
            self._reclassified = True
            self.metrics.n_reclassified = self.reclassify(state, t)

        pulsed = self._apply_rekey_pulses(state, t)
        consumed += pulsed
        managed += pulsed
        consumed -= self._handle_starvation(state, t)
        self.metrics.note_consumed(consumed, managed, t)
        self._retire_expired(state, t)

    def _consume_continuous(self, state: NetworkState, dt: float) -> tuple[float, float]:
        if not state.sessions:
            return 0.0, 0.0
        draw = state.edge_rate * dt
        state.buffers -= draw
        state.actual_consumed += draw
        state.ledger_consumed += state.edge_rate_mgd * dt

        if self.cfg.km_mode == "OTP":
            # O(1) damage accrual - the whole point of the aggregate design
            state.D_eff += state.exposed_rate * dt
            state.D_raw += state.exposed_key_rate * dt
            state.D_raw_hops += state.exposed_hop_key_rate * dt
            state.D_eff_relay += state.exposed_rate_relay * dt
            state.D_raw_relay += state.exposed_key_rate_relay * dt
        return float(draw.sum()), float(state.edge_rate_mgd.sum() * dt)

    def _apply_rekey_pulses(self, state: NetworkState, t: float) -> float:
        """Fire every due AES pulse.

        A pulse is all or nothing: if any hop cannot pay the 256 bits, that
        session cannot re-key and is interrupted on the spot.  Handling the AES
        shortfall here, per session, is more faithful than routing it through the
        aggregate starvation rule below, which is defined on a *rate* and would
        see edge_rate == 0 for AES sessions.

        Damage is charged at the pulse, never continuously: a 256 bit key relayed
        at t_pulse protects the traffic of [t_pulse, t_pulse + T_rk).  Keys
        relayed *before* t_c are therefore not exposed even if the traffic they
        protect continues past t_c - a real OTP/AES difference worth stating in
        the paper.
        """
        if self.cfg.km_mode != "AES":
            return 0.0
        heap = state.rekey_heap
        T_rk = self.cfg.T_rk
        total = 0.0
        while heap and heap[0][0] <= t:
            t_rk, sid = heapq.heappop(heap)
            s = state.sessions.get(sid)
            if s is None or s.status != "active" or t_rk != s.next_rekey:
                continue                                  # stale heap entry
            bits = s.pulse_bits
            if any(state.buffers[e] < bits for e in s.edges):
                self.metrics.on_starvation(1, t)
                self._close(state, sid, t, "interrupted")
                continue
            for e in s.edges:
                state.buffers[e] -= bits
                state.actual_consumed[e] += bits
            if s.managed:
                for e in s.edges:
                    state.ledger_consumed[e] += bits
            total += bits * len(s.edges)

            if s.exposed:
                span = min(T_rk, max(0.0, s.t_end - t_rk))
                state.D_raw += bits                       # once per session
                state.D_raw_hops += bits * s.n_exposed_edges
                state.D_eff += s.data_rate * span
                s.exposed_bits += s.data_rate * span
                # The relay/endpoint split is charged on the same pulse and by
                # the same span.  Omitting it here does not show up as an error
                # anywhere: D_eff stays correct, D_eff_endpoint silently absorbs
                # the whole of it, and DRR_relay - the metric every policy is
                # judged on - becomes 0/0 for the entire AES half of the matrix.
                if s.exposed_relay:
                    state.D_raw_relay += bits
                    state.D_eff_relay += s.data_rate * span

            nxt = t_rk + T_rk
            if nxt < s.t_end:
                s.next_rekey = nxt
                heapq.heappush(heap, (nxt, sid))
            else:
                s.next_rekey = INF
        return total

    def _handle_starvation(self, state: NetworkState, t: float) -> float:
        """Clamp negative buffers and tear down sessions until the link can
        sustain its load again.  Returns the key that was *not* actually
        available (so the caller can subtract it from the consumption total).

        Cut condition, straight from the spec: keep interrupting while
        edge_rate[e] > R[e], i.e. while the link drains faster than it refills.
        If the aggregate rate is already sustainable the buffer recovers on the
        next step by itself and nothing is cut - so an edge can never stay
        starved forever.

        The controller ledger is deliberately *not* clawed back here: the
        controller expects the key it authorised to have been spent, and the
        difference is itself a legitimate source of x1 residual.
        """
        starved = np.flatnonzero(state.buffers < 0.0)
        if starved.size == 0:
            return 0.0

        deficit = float(-state.buffers[starved].sum())
        state.actual_consumed[starved] += state.buffers[starved]   # undo the overdraw
        state.buffers[starved] = 0.0
        self.metrics.on_starvation(int(starved.size), t)

        R = state.R_eff
        for e in starved:
            e = int(e)
            # LIFO: the newest sessions are torn down first, so that older
            # sessions - the ones with more invested in them - survive.
            victims = sorted(state.edge_sessions[e], reverse=True)
            while victims and state.edge_rate[e] > R[e]:
                self._close(state, victims.pop(0), t, "interrupted")
        return deficit

    def tear_down_through(self, state: NetworkState, nodes, t: float) -> int:
        """Cut every active session that relays through one of ``nodes``.

        Isolation normally closes the door to *new* admissions and leaves traffic
        already in flight alone, which is what an operator would do - live
        sessions are not killed on suspicion.  The cost is a tail: a session
        admitted just before the node was isolated keeps feeding the attacker for
        up to T_s_max afterwards, and with a dwell timer of the same order that
        tail is not negligible.  This method is the other end of that choice, so
        both can be measured rather than assumed.
        """
        if not len(nodes):
            return 0
        bad = set(int(i) for i in nodes)
        victims = [sid for sid, s in state.sessions.items()
                   if any(n in bad for leg in s.legs for n in leg[1:-1])]
        for sid in victims:
            self._close(state, sid, t, "interrupted")
        return len(victims)

    def _retire_expired(self, state: NetworkState, t: float) -> None:
        heap = self._end_heap
        while heap and heap[0][0] <= t:
            t_end, sid = heapq.heappop(heap)
            s = state.sessions.get(sid)
            if s is None or s.status != "active" or s.t_end != t_end:
                continue
            self._close(state, sid, t, "completed")

    def _close(self, state: NetworkState, sid: int, t: float, status: str) -> None:
        s = state.sessions.pop(sid, None)
        if s is None:
            return
        s.status = status
        s.t_close = t
        self._unmark_exposed(state, s, t)
        for e in s.edges:
            state.edge_sessions[e].discard(sid)
            if state.edge_sessions[e]:
                state.edge_rate[e] -= s.key_rate
            else:
                state.edge_rate[e] = 0.0     # keeps the incremental sum exact
            if s.managed:
                state.edge_n_mgd[e] -= 1
                if state.edge_n_mgd[e]:
                    state.edge_rate_mgd[e] -= s.key_rate
                else:
                    state.edge_rate_mgd[e] = 0.0   # keeps the incremental sum exact
        if s.managed:
            if status == "completed":
                self.metrics.on_complete(s, t)
            else:
                self.metrics.on_interrupt(s, t)
        self.metrics.note_closed(s, t)
