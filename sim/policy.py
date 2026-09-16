"""Response policies B0-B4.

    B0  no defence - the reference every comparison is measured against
    B1  binary: isolate at a fixed threshold on the same S_bar. This is the
        state of the art and the industrial patents, so it is given the same
        hysteresis and dwell timer as B3 - a baseline that flaps is a strawman.
    B2  static multipath: XOR key shares over m node-disjoint paths, with no
        detection at all. Security by redundancy rather than by detection.
    B3  graded continuous response (the proposal): three levers driven by one
        score, with rho reaching zero exactly at S_iso so throttling and
        isolation meet continuously.
    B4  oracle: perfect knowledge of V_c. The achievable upper bound - it is
        still subject to the partition guard, because isolating a cut vertex is
        not something any policy can do.

Information boundary. ``apply`` receives the score and the time, and nothing
else. Phase 4 established that the detector may not read privileged state, and
the policy sits on the same side of that line; B4's ground truth arrives through
an explicitly injected ``oracle`` callable so the one place that cheats is
visible. ``test_policy_isolation`` enforces it.

No policy draws a random number. ``rng_policy`` is already consumed by the
admission quota test, and a second consumer would shift those draws and
contaminate the comparison; every tie here is broken on node id instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .config import PolicyConfig
from .interfaces import Policy
from .topology import Topology

INF = float("inf")


@dataclass
class PolicyDecision:
    node_weight: np.ndarray
    rho: np.ndarray
    isolated: np.ndarray
    n_paths: int = 1
    n_vetoed: int = 0        # wanted isolation this tick but would have partitioned
    n_flips: int = 0         # isolation state changes this tick
    counters: dict = field(default_factory=dict)
    edge_cost: object = None  # per-edge routing multiplier, or None
    hybrid: tuple = (1.0, 1, "product")   # (trigger, legs, path-risk aggregation)


# shared mechanics, as pure functions so they can be tested without a simulator
def hysteresis_step(S_bar: np.ndarray, prev: np.ndarray, since: np.ndarray,
                    t: float, hi: float, lo: float, dwell: float):
    """One step of the isolate/release state machine.

    Rise at ``hi``, fall at ``lo = hi - hysteresis``, and no transition until the
    node has held its current state for ``dwell`` seconds. Without this a score
    hovering at the threshold flaps every tick, which costs a route table
    rebuild each time and makes the binary baseline look worse than it is.
    """
    ready = (t - since) >= dwell
    rise = (~prev) & (S_bar >= hi) & ready
    fall = prev & (S_bar < lo) & ready
    new = prev.copy()
    new[rise] = True
    new[fall] = False
    return new, (rise | fall)


def partition_guard(G: nx.Graph, want: np.ndarray, S_bar: np.ndarray):
    """Largest prefix of the risk ordering that can be isolated safely.

    Phase 1, section 8.3: if isolating node i would disconnect the network,
    clamp rho_i to rho_min instead and record an "impossible isolation".

    Candidates are taken in DESCENDING risk with an explicit node-id tiebreak -
    ties are common, since during detector warm-up every node sits at exactly
    its prior. Descending order gives the highest-risk node first refusal;
    isolating a low-risk node first could turn a high-risk one into a cut
    vertex. This is a monotone-in-risk heuristic, not an optimum: the sets
    whose removal keeps G connected do not form a matroid.

    Two conditions, not one. The graph minus the isolated set must stay
    connected, AND every isolated node must keep at least one non-isolated
    neighbour - isolation means "no longer a relay", not "off the network", so a
    node cut off entirely would have its own demands rejected as unreachable and
    that structural artifact would land in RR indistinguishable from a key
    shortage.
    """
    order = sorted((int(i) for i in np.flatnonzero(want)),
                   key=lambda i: (-float(S_bar[i]), i))
    accepted: set[int] = set()
    vetoed: list[int] = []
    for i in order:
        trial = accepted | {i}
        rest = [n for n in G.nodes() if n not in trial]
        if not rest or not nx.is_connected(G.subgraph(rest)):
            vetoed.append(i)
            continue
        if any(all(nb in trial for nb in G.neighbors(j)) for j in trial):
            vetoed.append(i)
            continue
        accepted = trial
    return accepted, vetoed


class BasePolicy(Policy):
    """Hysteresis, dwell, partition guard and the counters every policy reports."""

    # Does this policy act on S_bar? If it does, it must not act before the
    # detector is warm: until then SuspicionDetector returns the prior alone,
    # and under selection='top_keyflow' the prior correlates with the
    # compromised set by construction - a policy driven by it would look
    # effective for exactly the reason the phase 4 analysis warns against.
    needs_detector = False

    def __init__(self, cfg: PolicyConfig, n_nodes: int = 0,
                 topo: Topology | None = None, **_: object):
        self.cfg = cfg
        self.n = int(n_nodes)
        self.topo = topo
        self.G = topo.G if topo is not None else None

        self._isolated = np.zeros(self.n, dtype=bool)
        self._since = np.full(self.n, -INF, dtype=float)
        self._vetoed = np.zeros(self.n, dtype=bool)

        self.n_flaps = 0                 # transitions after the first
        self.n_isolation_events = 0      # healthy -> isolated transitions
        self.n_veto_ticks = 0
        self.ever_vetoed: set[int] = set()
        self.ever_isolated: set[int] = set()
        self.sum_isolated = 0
        self.n_ticks = 0

    # -- subclass hooks ------------------------------------------------- #
    def _threshold(self) -> float | None:
        """Score at which this policy isolates, or None if it never does."""
        return None

    def _want_isolated(self, S_bar: np.ndarray, t: float) -> np.ndarray:
        thr = self._threshold()
        if thr is None:
            return np.zeros(self.n, dtype=bool)
        new, changed = hysteresis_step(
            S_bar, self._isolated, self._since, t,
            hi=thr, lo=thr - self.cfg.hysteresis, dwell=self.cfg.dwell)
        if changed.any():
            self._since[changed] = t
            self.n_isolation_events += int((new & changed).sum())
            self.n_flaps += int(changed.sum())
        return new

    def _levers(self, S_bar: np.ndarray, isolated: np.ndarray):
        """(node_weight, rho) before the veto floor is applied."""
        return np.ones(self.n, dtype=float), np.ones(self.n, dtype=float)

    def n_paths(self) -> int:
        return 1

    def hybrid(self) -> tuple[float, int, str]:
        """(trigger, legs, aggregation) for a policy deciding redundancy per session.

        A trigger of 1.0 never fires, which is every policy but B8.
        """
        return 1.0, 1, "product"

    def edge_cost(self, state=None):
        """Per-edge routing cost multiplier, or None to leave it at one.

        B0-B4 steer by NODE weight, which is all the graded policy's kappa lever
        needs. The literature baselines weight LINKS - by their own remaining
        key, or by their key-generation rate - and that is not expressible as a
        node weight, so it gets its own hook.
        """
        return None

    # -- the ABC entry point -------------------------------------------- #
    def apply(self, S_bar: np.ndarray, t: float = 0.0,
              state=None) -> PolicyDecision:
        want = self._want_isolated(S_bar, t)

        vetoed: list[int] = []
        if want.any() and self.G is not None:
            accepted, vetoed = partition_guard(self.G, want, S_bar)
            isolated = np.zeros(self.n, dtype=bool)
            for i in accepted:
                isolated[i] = True
        else:
            isolated = want.copy()

        # a vetoed node never became isolated, so it must not hold the latch
        self._isolated = isolated
        self._vetoed[:] = False
        for i in vetoed:
            self._vetoed[i] = True
            self.ever_vetoed.add(int(i))
        if vetoed:
            self.n_veto_ticks += 1

        weight, rho = self._levers(S_bar, isolated)
        # phase 1 section 8.3: what cannot be isolated is throttled to rho_min
        rho[self._vetoed] = self.cfg.rho_min
        rho[isolated] = 0.0

        self.ever_isolated.update(int(i) for i in np.flatnonzero(isolated))
        self.sum_isolated += int(isolated.sum())
        self.n_ticks += 1

        return PolicyDecision(
            node_weight=weight, rho=rho, isolated=isolated,
            n_paths=self.n_paths(), n_vetoed=len(vetoed),
            n_flips=0, counters=self.counters(),
            edge_cost=self.edge_cost(state),
            hybrid=self.hybrid())

    def counters(self) -> dict:
        return {
            "policy_n_flaps": self.n_flaps,
            "policy_n_isolation_events": self.n_isolation_events,
            "policy_n_veto_ticks": self.n_veto_ticks,
            "policy_n_nodes_ever_vetoed": len(self.ever_vetoed),
            "policy_n_nodes_ever_isolated": len(self.ever_isolated),
            "policy_mean_isolated": (self.sum_isolated / self.n_ticks)
                                    if self.n_ticks else 0.0,
        }


class NoDefencePolicy(BasePolicy):
    """B0. Plain shortest path, no throttling, no isolation."""


class BinaryPolicy(BasePolicy):
    """B1. Hard isolation at a fixed threshold on the same S_bar as B3.

    Equivalent to the existing literature and to the industrial patents. Given
    the same hysteresis and dwell timer as B3 on purpose: the interesting claim
    is that a *graded* response beats a well-implemented binary one, not that it
    beats a flapping one.
    """

    needs_detector = True

    def _threshold(self) -> float:
        return self.cfg.tau


class MultipathPolicy(BasePolicy):
    """B2. XOR key shares over m node-disjoint paths, with no detection at all.

    Security by redundancy: the attacker learns nothing unless it holds a relay
    on every leg. The cost is m-fold key consumption and a rejection whenever
    the graph cannot supply m node-disjoint paths.
    """

    def n_paths(self) -> int:
        return int(self.cfg.m_paths)


class BlindThrottlePolicy(BasePolicy):
    """BT - the volume control. Uniform rho everywhere, no detection at all.

    Not one of the five policies of phase 1; it exists to separate two effects
    that any throttling policy mixes together. Admitting fewer sessions reduces
    exposure *mechanically*, whether or not the throttle is aimed at anything:
    a policy that rejects half the traffic cuts damage roughly in half while
    targeting nothing. So a DRR of 0.76 bought with a 54 point rise in
    rejection is not 0.76 of detection value.

    BT throttles every node identically at ``rho_blind``, which makes it the
    matched-cost null hypothesis: the targeting benefit of B1/B3 is their DRR
    minus BT's DRR at the same rejection rate, not their DRR.
    """

    def __init__(self, cfg: PolicyConfig, n_nodes: int = 0,
                 topo: Topology | None = None, **_: object):
        super().__init__(cfg, n_nodes=n_nodes, topo=topo)
        self._rho_blind = float(cfg.rho_blind)

    def _levers(self, S_bar: np.ndarray, isolated: np.ndarray):
        return (np.ones(self.n, dtype=float),
                np.full(self.n, self._rho_blind, dtype=float))


class GradedPolicy(BasePolicy):
    """B3, the proposal. Three levers driven by one continuous score.

        w_route,i = w_base,i * exp(kappa * S_bar_i)
        rho_i     = clip((S_iso - S_bar_i) / (S_iso - rho_start), 0, 1)
        isolate   when S_bar_i >= S_iso

    rho still reaches zero exactly at S_iso, so throttling and isolation join
    continuously and share a single parameter - the property phase 1 section 8.3
    is built around. What changes is where throttling *starts*.

    The spec writes rho = max(0, 1 - S_bar/S_iso), which anchors rho = 1 at
    S_bar = 0. With a configuration prior in the score no node is ever at zero:
    measured on T1, every healthy node sat at S_bar ~ 0.22, so every node in the
    network got rho ~ 0.56 and B3 became a blanket 44% throttle that raised the
    rejection rate by 26 points while separating compromised from healthy nodes
    by 0.006 - it throttled everything and protected nothing.

    ``rho_start`` re-anchors the ramp at the quiet operating level that
    ``calibrate.calibrate_score`` already measures, so a node at ordinary risk is
    untouched and the lever acts on the tail, which is what it was for.
    Defaults to 0.0, i.e. the literal spec formula, so the change is explicit and
    sweepable rather than baked in.
    """

    needs_detector = True

    def _threshold(self) -> float:
        return self.cfg.S_iso

    def _route_weight(self, S_bar: np.ndarray) -> np.ndarray:
        """Turn the risk score into a routing weight.

        ``exp`` is the phase 1 formula and was measured inert: it makes a
        suspect node a constant factor dearer, which the quota lever already
        does on the same node and more strongly, so there is nothing left for it
        to steer away from.

        ``logrisk`` is the objective the damage model implies. Routing cost
        sums 0.5*(w_u + w_v) over the edges of a path, so every interior node
        contributes w_i exactly once and the path cost becomes

            hops + kappa * sum over interior of -log(1 - S_bar)

        Minimising that maximises the probability the whole path is clean, which
        is what exposure depends on. The practical difference is the shape: a
        node at S_bar = 0.9 costs 2.3*kappa here and diverges as S_bar -> 1,
        where exp(kappa*S) merely doubles. A single very suspect relay can
        therefore be routed around, which is the case that matters.
        """
        s = np.clip(S_bar, 0.0, 1.0)
        if self.cfg.route_mode == "logrisk":
            return 1.0 + self.cfg.kappa * (-np.log1p(-np.minimum(s, 0.999)))
        return np.exp(self.cfg.kappa * s)

    def _levers(self, S_bar: np.ndarray, isolated: np.ndarray):
        weight = self._route_weight(S_bar)
        span = max(self.cfg.S_iso - self.cfg.rho_start, 1e-9)
        rho = np.clip((self.cfg.S_iso - S_bar) / span, 0.0, 1.0)
        return weight, rho


class HybridPolicy(GradedPolicy):
    """B8. Graded throttling, plus redundancy for the sessions that need it.

    B2 pays for XOR redundancy on every session whether or not anything is
    wrong; B3 pays nothing but can only act on evidence. The paper's own
    conclusion is that the two are complements, and this is the policy that
    makes them one. A session is relayed over ``m_paths`` node-disjoint legs
    only when the cheapest single path it would otherwise take is itself
    suspect:

        P(path carries a compromised relay) = 1 - prod(1 - S_bar_i)
                                              over the interior nodes

    and that exceeds ``hybrid_tau``. Everything else takes one path.

    The trigger is cumulative rather than the maximum along the path because
    exposure needs only *one* compromised relay, so a long path through
    moderately suspect nodes is genuinely more dangerous than a short path
    through one - and the maximum cannot tell them apart. The cost is an
    assumption: it reads ``S_bar`` as a probability, which it is not exactly,
    being a calibrated posterior rather than a frequency. It is monotone in the
    right direction, which is what a trigger needs.

    When the trigger fires but the graph has no ``m_paths`` node-disjoint set,
    the session is admitted on one path rather than rejected. That is the
    weaker security choice and the deliberate one: refusing would turn a
    detection policy into an availability outage exactly on the sparse
    topologies where disjoint paths are scarce, which is the failure mode that
    makes B2 at m=3 reject 87% of traffic.
    """

    needs_detector = True

    def n_paths(self) -> int:
        return 1                      # the trigger decides, per session

    def hybrid(self) -> tuple[float, int, str]:
        return (float(self.cfg.hybrid_tau), int(self.cfg.m_paths),
                str(self.cfg.hybrid_agg))


class OraclePolicy(BasePolicy):
    """B4. Perfect knowledge of V_c - the achievable upper bound.

    Still subject to the partition guard: isolating a cut vertex is impossible
    for any policy, so an oracle that ignored the guard would be an unachievable
    bound rather than a useful one. No hysteresis is needed, since ground truth
    does not flap, but the dwell machinery is inherited and harmless.
    """

    def __init__(self, cfg: PolicyConfig, n_nodes: int = 0,
                 topo: Topology | None = None, oracle=None, **_: object):
        super().__init__(cfg, n_nodes=n_nodes, topo=topo)
        if oracle is None:
            raise ValueError("policy B4 needs an oracle; the runner injects "
                             "attack.compromised_nodes")
        self._oracle = oracle

    def _want_isolated(self, S_bar: np.ndarray, t: float) -> np.ndarray:
        want = np.asarray(self._oracle(t), dtype=bool).copy()
        changed = want != self._isolated
        if changed.any():
            self._since[changed] = t
            self.n_isolation_events += int((want & changed).sum())
            self.n_flaps += int(changed.sum())
        return want


# Literature baselines.
#
# B0-B4 and BT are all constructions of this paper, which makes any comparison
# between them self-referential: showing that our graded policy beats our own
# binary one is an ablation, not a comparison with the state of the art. The
# three policies below are reimplementations of published methods, so that the
# claim has something outside this work to stand against.
#
# Every one of them is an ADAPTATION and the departures are listed in each
# docstring. None of the three was published against this threat model - key
# material stolen by a compromised relay - so a faithful port is impossible in
# principle; what is ported is each method's decision rule.
class TrustDeratePolicy(BasePolicy):
    """B5. Edge-trust derating, after Luo & Li, *Entropy* 27(11):1100, 2025.

    Their trust level for a link is (their Eq. 8, notation theirs)

        C_e = (min(W_e^A, W_e^B) - f) / (n - f)
              * exp( -beta * ( (G_e^A + S_e^A) - (G_e^B + S_e^B) )^2 )

    and it multiplies the link's usable key-pool capacity in the flow
    constraint, so a suspect link is *derated* rather than removed. Two things
    are worth saying plainly about it, because they are what makes this a
    baseline rather than a rival:

    * The second factor is exactly our ``x2``: the two endpoints of a link
      report its key state separately, and an honest-honest pair agrees while an
      honest-malicious pair does not. Their detector is that one observable.
      They have no analogue of ``x1`` (key-accounting residual) or ``x3``
      (quantum-layer deviation), which is the gap this paper is about.
    * It is memoryless - recomputed from scratch every round, with no EWMA and
      no per-node history - so it is reimplemented here on the raw feature
      rather than on the smoothed score ``S_bar``.

    ADAPTED, and the paper must say so: the first factor counts Byzantine
    witness acknowledgements from their causal-consistency layer. This
    simulator has no consensus protocol and no equivocating control plane, so
    that factor cannot be reproduced and is dropped. What remains is their
    trust *mechanism* measured against our threat model, which is the honest
    comparison: it is the published method that comes closest to detecting a
    compromised relay from telemetry, and the question is how far one observable
    gets you.
    """

    needs_detector = True

    def __init__(self, cfg: PolicyConfig, n_nodes: int = 0,
                 topo: Topology | None = None, detector=None, **_: object):
        super().__init__(cfg, n_nodes=n_nodes, topo=topo)
        self._det = detector

    def _threshold(self) -> float | None:
        return None                      # derating only; it never isolates

    def _levers(self, S_bar: np.ndarray, isolated: np.ndarray):
        x = getattr(self._det, "x", None)
        if x is None or x.shape[0] != self.n:
            return np.ones(self.n), np.ones(self.n)
        d = np.abs(np.asarray(x[:, 1], dtype=float))      # the x2 discrepancy
        rho = np.exp(-self.cfg.beta_trust * np.square(d))
        return np.ones(self.n, dtype=float), np.clip(rho, 0.0, 1.0)


class NonOverlappingMultipathPolicy(MultipathPolicy):
    """B6. XOR over non-overlapping paths, after Kiktenko, Tayduganov & Fedorov,
    *Entropy* 26(12):1102, 2024.

    Their scheme is the same one B2 implements - a key split into XOR shares,
    one per internally node-disjoint path, so that an adversary must hold a
    relay on *every* path - and B2 should be credited to them rather than
    presented as ours. What is theirs and not ours is the **path selection**:
    B2 takes the minimum-cost disjoint set (fewest hops), while their Algorithm
    1 picks the set minimising

        D(P^M) = max over paths, max over links, of ( T_ij - R_eff_ij )

    i.e. the set whose worst link has the smallest key-rate deficiency, and then
    debits every link it used so the next demand avoids it. It is water-filling
    on the key supply, not shortest path.

    ADAPTED, twice, and both need stating:

    * Their algorithm enumerates **every simple path** between a pair and then
      every M-subset of non-overlapping ones. That is exponential, and their
      own evaluation runs at N = 6 and N = 10 nodes; at N = 50 it does not
      terminate. Selection here is over the same min-cost-flow machinery B2
      uses, with the link cost replaced by a deficiency surrogate.
    * Their deficiency is defined against a *target rate matrix* solved offline
      for a whole period. Online, the standing analogue of "this link is short
      of key" is its own generation rate, so the surrogate cost is R_max / R_e:
      a link that generates little key is expensive, which is the same ordering
      their D(P^M) induces, without the global optimisation.

    The consequence is worth measuring rather than assuming: it should route
    around key-poor links, which helps when key is the binding constraint and
    costs hops when it is not.
    """

    def edge_cost(self, state=None) -> np.ndarray | None:
        R = np.asarray(self.topo.R, dtype=float)
        return float(R.max()) / np.maximum(R, 1e-12)


class KeyAwareRoutingPolicy(BasePolicy):
    """B7. Key-availability routing, after Bi, Miao & Di, *Appl. Sci.* 13:8690,
    2023.

    A routing baseline with no detection at all, included to answer the obvious
    objection that a good key-aware routing algorithm might make a detector
    unnecessary. Their link weight (their Eq. 12) is

        W_ij = alpha * K_ij^t + beta * P_ij,      alpha + beta = 1,

    with ``K_ij^t`` the key remaining in the link's pool and ``P_ij`` its
    blocking probability, itself (their Eqs. 9-10) the chance of at least one
    blocking event in unit time under a Poisson model,

        P = 1 - exp(-lambda).

    The controller then prefers links that score high.

    ADAPTED: their Eq. 12 as printed adds the blocking probability to the key
    term and then *maximises* the sum, which would prefer the more congested
    link; read as a benefit score it must be the availability ``1 - P`` that
    enters, and that is what is implemented. The paper does not give values for
    alpha and beta beyond ``alpha + beta = 1``, so ``policy.alpha_key`` is a
    swept parameter here and defaults to 0.5. ``lambda`` is the link's key
    demand relative to its key supply, which is the quantity their "blocking
    intensity" measures in a key-limited network.

    Their recomputation is per request; here it is gated by ``routing.T_route``,
    which is the same statement at controller granularity.
    """

    def edge_cost(self, state=None) -> np.ndarray | None:
        if state is None:
            return None
        B = float(self.topo.B_max)
        k_hat = np.clip(np.asarray(state.buffers, dtype=float) / max(B, 1e-12),
                        0.0, 1.0)
        R = np.maximum(np.asarray(self.topo.R, dtype=float), 1e-12)
        lam = np.asarray(state.edge_rate, dtype=float) / R
        avail = np.exp(-np.clip(lam, 0.0, 50.0))          # 1 - P, Eqs. 9-10
        a = float(self.cfg.alpha_key)
        w = a * k_hat + (1.0 - a) * avail
        return 1.0 / np.maximum(w, 1e-6)
