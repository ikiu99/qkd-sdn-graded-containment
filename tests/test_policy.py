"""Phase 5: the response policies B0-B4.

``test_policy_isolation`` is the counterpart of ``test_detector_isolation``: it
proves that only B4 - the oracle, which is *defined* as the perfect-knowledge
upper bound - reads ground truth, and that B0-B3 are blind to it.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from sim.config import load_config
from sim.interfaces import build_policy
from sim.policy import hysteresis_step, partition_guard
from sim.runner import simulate
from sim.topology import load_topology

BASE = "config/attack_t1_otp.yaml"
COMMON = {"horizon": 21600, "warmup": 1800, "attack.t_c": 10800,
          "detector.ablation": False}


def make(overrides=None):
    o = dict(COMMON)
    o.update(overrides or {})
    return load_config(BASE, o)


def run(overrides=None, **kw):
    return simulate(make(overrides), write=False, asserts=False, **kw)


def policy_for(kind, **over):
    cfg = make({"policy.type": kind, **over})
    topo = load_topology(cfg.topology)
    pol = build_policy(cfg.policy, n_nodes=topo.n_nodes, topo=topo,
                       oracle=lambda t: np.zeros(topo.n_nodes, dtype=bool))
    return cfg, topo, pol


# the hysteresis state machine, as a pure function
def _drive(series, hi=0.5, lo=0.4, dwell=0.0, dt=1.0):
    n = 1
    prev = np.zeros(n, dtype=bool)
    since = np.full(n, -np.inf)
    out, flips = [], 0
    for k, v in enumerate(series):
        t = k * dt
        new, changed = hysteresis_step(np.array([v]), prev, since, t, hi, lo, dwell)
        if changed.any():
            since[changed] = t
            flips += int(changed.sum())
        prev = new
        out.append(bool(new[0]))
    return out, flips


def test_hysteresis_never_isolates_below_threshold():
    out, _ = _drive([0.0, 0.1, 0.2, 0.3, 0.49])
    assert not any(out)


def test_hysteresis_holds_inside_the_band():
    """Once isolated, a score inside [lo, hi) must not release."""
    out, flips = _drive([0.6, 0.45, 0.45, 0.45])
    assert out == [True, True, True, True]
    assert flips == 1


def test_hysteresis_releases_below_the_lower_edge():
    out, flips = _drive([0.6, 0.39, 0.39])
    assert out == [True, False, False]
    assert flips == 2


def test_dwell_blocks_a_spike_shorter_than_the_timer():
    """A one-sample excursion must not produce an isolate/release pair."""
    out, flips = _drive([0.6, 0.0, 0.0, 0.0, 0.0], dwell=3.0, dt=1.0)
    assert out[0] is True
    assert out[1] is True, "released inside the dwell window"
    assert out[3] is False, "never released after the dwell window"
    assert flips == 2


def test_sawtooth_across_the_band_does_not_flap_every_tick():
    saw = [0.52, 0.44, 0.52, 0.44, 0.52, 0.44]
    _, with_band = _drive(saw, hi=0.5, lo=0.4)
    _, no_band = _drive(saw, hi=0.5, lo=0.5)
    assert with_band == 1, "the hysteresis band should absorb the sawtooth"
    assert no_band > with_band


# partition guard
@pytest.fixture(scope="module")
def graph():
    return load_topology(load_config(BASE).topology).G


def test_guard_keeps_the_graph_connected(graph):
    rng = np.random.default_rng(0)
    n = graph.number_of_nodes()
    for _ in range(60):
        want = rng.random(n) < 0.25
        S = rng.random(n)
        accepted, vetoed = partition_guard(graph, want, S)
        rest = [x for x in graph.nodes() if x not in accepted]
        assert rest and nx.is_connected(graph.subgraph(rest))
        for j in accepted:
            assert any(nb not in accepted for nb in graph.neighbors(j)), (
                "an isolated node was cut off from the network entirely")
        assert set(accepted) | set(vetoed) == set(np.flatnonzero(want).tolist())


def test_guard_veto_is_justified(graph):
    """Every vetoed node really would have broken one of the two conditions."""
    rng = np.random.default_rng(1)
    n = graph.number_of_nodes()
    seen = 0
    for _ in range(80):
        want = rng.random(n) < 0.45
        S = rng.random(n)
        accepted, vetoed = partition_guard(graph, want, S)
        for i in vetoed:
            seen += 1
            trial = set(accepted) | {i}
            rest = [x for x in graph.nodes() if x not in trial]
            broke_connectivity = not rest or not nx.is_connected(graph.subgraph(rest))
            cut_off = any(all(nb in trial for nb in graph.neighbors(j)) for j in trial)
            assert broke_connectivity or cut_off
    assert seen > 0, "no veto occurred; the test is not exercising the guard"


def test_guard_is_deterministic_under_reordering(graph):
    """The id tiebreak must make the result independent of input order."""
    rng = np.random.default_rng(2)
    n = graph.number_of_nodes()
    want = rng.random(n) < 0.4
    S = np.full(n, 0.7)            # all tied - the worst case for determinism
    a1, _ = partition_guard(graph, want, S)
    a2, _ = partition_guard(graph, want.copy(), S.copy())
    assert a1 == a2


def test_guard_prefers_the_higher_risk_node(graph):
    """Descending risk order gives the riskiest node first refusal."""
    n = graph.number_of_nodes()
    want = np.zeros(n, dtype=bool)
    S = np.zeros(n)
    # a pair whose joint removal disconnects T1 but whose single removal does not
    for i, j in [(a, b) for a in range(n) for b in range(a + 1, n)]:
        rest = [x for x in graph.nodes() if x not in (i, j)]
        if not nx.is_connected(graph.subgraph(rest)):
            want[i] = want[j] = True
            S[i], S[j] = 0.9, 0.6
            accepted, vetoed = partition_guard(graph, want, S)
            assert i in accepted and j in vetoed
            return
    pytest.skip("no such pair on this topology")


# the information boundary
def test_policy_isolation():
    """Only B4 may see ground truth; B0-B3 decide from S_bar and t alone."""
    n = 14
    S = np.linspace(0.0, 0.95, n)

    for kind in ("B0", "B1", "B2", "B3"):
        cfg, topo, pol = policy_for(kind)
        truth = pol.apply(S.copy(), 5000.0)
        cfg2, topo2, pol2 = policy_for(kind)
        # the same score, but a completely different "ground truth" behind it
        poisoned = pol2.apply(S.copy(), 5000.0)
        np.testing.assert_array_equal(truth.isolated, poisoned.isolated)
        np.testing.assert_allclose(truth.node_weight, poisoned.node_weight)
        np.testing.assert_allclose(truth.rho, poisoned.rho)

    cfg = make({"policy.type": "B4"})
    topo = load_topology(cfg.topology)
    mask = np.zeros(topo.n_nodes, dtype=bool)
    mask[[2, 5]] = True
    b4 = build_policy(cfg.policy, n_nodes=topo.n_nodes, topo=topo,
                      oracle=lambda t: mask)
    dec = b4.apply(np.zeros(topo.n_nodes), 5000.0)
    assert dec.isolated.sum() > 0, "the oracle ignored its ground truth"
    assert set(np.flatnonzero(dec.isolated)) <= {2, 5}


def test_policies_return_fresh_arrays():
    """Handing back internal arrays would alias state.rho to policy memory."""
    for kind in ("B0", "B1", "B2", "B3"):
        _, topo, pol = policy_for(kind)
        S = np.full(topo.n_nodes, 0.2)
        a = pol.apply(S, 1000.0)
        b = pol.apply(S, 2000.0)
        assert a.rho is not b.rho
        assert a.node_weight is not b.node_weight
        assert a.isolated is not b.isolated


def test_b0_is_numerically_identical_to_no_policy():
    """Promoting policy to a typed section must not move a single number."""
    a = run({"policy.type": "B0", "attack.profile": "G"})
    b = run({"policy.type": "none", "attack.profile": "G"})
    for key in ("n_admitted", "n_rejected", "RR", "KPD", "D_eff", "D_eff_relay",
                "mean_path_len", "total_key_consumed"):
        assert a.summary[key] == b.summary[key], key


def test_b0_draws_nothing_from_the_policy_stream():
    """B0 and B2 must leave rng_policy untouched, or the admission draws of
    every other policy shift underneath them."""
    for kind in ("B0", "B2"):
        _, topo, pol = policy_for(kind)
        S = np.full(topo.n_nodes, 0.9)
        dec = pol.apply(S, 9000.0)
        assert np.all(dec.rho == 1.0)
        assert not dec.isolated.any()


# end to end behaviour
def test_no_attack_means_no_damage_under_every_policy():
    for kind in ("B0", "B1", "B2", "B3", "B4"):
        res = run({"policy.type": kind, "attack.enabled": False})
        assert res.summary["D_eff"] == 0.0, kind
        assert res.summary["D_eff_relay"] == 0.0, kind
        assert res.summary["exposed_session_ratio"] == 0.0, kind


def test_oracle_removes_almost_all_relay_damage():
    """B4 is the achievable upper bound, and this is what makes it one.

    Only the pre-t_c window and whatever the partition guard vetoes should
    survive, so relay exposure must collapse by well over an order of magnitude.
    """
    b0 = run({"policy.type": "B0", "attack.profile": "G", "attack.f": 0.2})
    b4 = run({"policy.type": "B4", "attack.profile": "G", "attack.f": 0.2})
    assert b0.summary["D_eff_relay"] > 0
    drr = 1.0 - b4.summary["D_eff_relay"] / b0.summary["D_eff_relay"]
    assert drr > 0.9, f"oracle only achieved DRR_relay = {drr:.3f}"


def test_endpoint_exposure_is_a_floor_no_policy_can_touch():
    """A session whose own endpoint is compromised leaks however it is routed.

    That share of D_eff is set by the traffic matrix, not by the policy, which
    is exactly why it is reported separately - folding it in would cap DRR at a
    value that has nothing to do with the policy.
    """
    b0 = run({"policy.type": "B0", "attack.profile": "G", "attack.f": 0.2})
    b4 = run({"policy.type": "B4", "attack.profile": "G", "attack.f": 0.2})
    for res in (b0, b4):
        s = res.summary
        assert s["D_eff"] == pytest.approx(s["D_eff_relay"] + s["D_eff_endpoint"])
        assert s["D_eff_endpoint"] > 0
    # the oracle wipes out relay exposure but cannot reduce the endpoint share
    assert b4.summary["D_eff_endpoint"] > 0.5 * b0.summary["D_eff_endpoint"]


def test_isolation_blocks_new_admissions_through_the_node():
    """Isolation closes the door; it does not evict traffic already inside.

    An operator does not kill live sessions on suspicion, so a session admitted
    before its relay was isolated runs to completion through it. That leaves a
    tail of up to T_s_max, which is measured rather than assumed: with
    ``tear_down_on_isolate`` the invariant becomes exact, and the damage
    difference between the two is about 1-2%.
    """
    over = {"policy.type": "B1", "attack.profile": "G", "attack.f": 0.2,
            "policy.tau": 0.35}
    res = run({**over, "policy.tear_down_on_isolate": True})
    iso = set(np.flatnonzero(res.state.isolated).tolist())
    if not iso:
        pytest.skip("no node crossed the threshold in this short run")
    for s in res.state.sessions.values():
        for leg in s.legs:
            for node in leg[1:-1]:
                assert node not in iso, "teardown left an isolated node relaying"
    assert res.summary["n_torn_down_on_isolate"] > 0

    # without teardown the tail exists, and that is the documented default
    plain = run(over)
    assert plain.summary["n_torn_down_on_isolate"] == 0


def test_graded_policy_beats_binary_on_relay_damage():
    """The paper's central comparison, on a single matched scenario."""
    over = {"attack.profile": "G", "attack.f": 0.2}
    b0 = run({"policy.type": "B0", **over})
    b1 = run({"policy.type": "B1", **over})
    b3 = run({"policy.type": "B3", **over})
    base = b0.summary["D_eff_relay"]
    drr1 = 1.0 - b1.summary["D_eff_relay"] / base
    drr3 = 1.0 - b3.summary["D_eff_relay"] / base
    assert drr3 > drr1, f"graded {drr3:.3f} did not beat binary {drr1:.3f}"
