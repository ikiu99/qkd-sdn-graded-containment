"""Phase 7: analytical model validation.

Cross-validation against QKDNetSim is deferred to future work, so the simulator
is instead checked against closed forms it must reproduce.  These are deliberately
*interaction* tests - each one exercises several components at once against an
answer derived outside the simulator - rather than more unit tests of parts that
already have them.

Each test states the closed form and the conditions under which it holds; a test
that only holds in a narrow regime says so, because a validation check that
quietly stops applying is worse than none.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from sim.config import load_config
from sim.routing import RouteTable
from sim.runner import simulate
from sim.state import (recompute_edge_rate_from_sessions, recompute_exposure,
                       recompute_exposure_relay)
from sim.topology import load_topology

T1 = "config/base_t1_otp.yaml"
T3 = "config/base_t3_otp.yaml"


def run(base, overrides, **kw):
    kw.setdefault("asserts", False)
    return simulate(load_config(base, overrides), write=False, **kw)


# --------------------------------------------------------------------------- #
# 1. queueing: the session population is M/G/inf when nothing blocks
# --------------------------------------------------------------------------- #
def test_session_population_matches_m_g_infinity():
    """E[active] = (1 + unmanaged_fraction) * lambda * (E[T_s] + dt).

    Both corrections matter and neither is a fudge.  ``mean_active_sessions``
    counts the N4 background stream as well as the managed one, and a session
    admitted at t0 is held in the population for the steps t0 .. t0+T_s
    inclusive, which is T_s + dt of occupancy.  Requires a regime with no
    rejection and no interruption, so lambda is tiny and B_max is huge.
    """
    lam = 0.02
    res = run(T1, {"horizon": 43200, "warmup": 3600, "demand.lam": lam,
                   "demand.rate_min": 1.0, "demand.rate_max": 2.0,
                   "topology.B_max": 1e9})
    s = res.summary
    assert s["n_rejected"] == 0, "the closed form needs a loss-free regime"
    assert s["n_interrupted"] == 0

    cfg = load_config(T1, {})
    e_ts = 0.5 * (cfg.demand.T_s_min + cfg.demand.T_s_max)
    expected = (1.0 + cfg.noise.unmanaged_fraction) * lam * (e_ts + cfg.dt)
    assert s["mean_active_sessions"] == pytest.approx(expected, rel=0.06), (
        f"got {s['mean_active_sessions']:.2f}, M/G/inf predicts {expected:.2f}")


def test_offered_load_matches_poisson_rate():
    """Arrivals are Poisson(lambda): the offered count over the measured window
    must match lambda * (horizon - warmup) within sampling error."""
    lam, horizon, warmup = 0.4, 43200, 1800
    res = run(T1, {"horizon": horizon, "warmup": warmup, "demand.lam": lam})
    expected = lam * (horizon - warmup)
    got = res.summary["n_offered"]
    assert got == pytest.approx(expected, rel=4 / np.sqrt(expected)), (
        f"offered {got}, Poisson predicts {expected:.0f}")


# --------------------------------------------------------------------------- #
# 2. fluid limit: throughput is capped by generation, not by anything else
# --------------------------------------------------------------------------- #
def test_consumption_cannot_exceed_generation_in_the_long_run():
    """Over a long overloaded run the network cannot consume more key than it
    makes, and with the buffers driven to the floor it must consume nearly all
    of it.  This is the fluid limit, and it is the one statement the whole
    buffer model has to satisfy."""
    res = run(T3, {"horizon": 43200, "warmup": 1800, "demand.lam": 5.0})
    s = res.summary
    ratio = s["total_key_consumed"] / s["total_key_generated"]
    assert ratio <= 1.0 + 1e-9, "consumed more key than was generated"
    assert ratio > 0.85, f"an overloaded network left {1 - ratio:.1%} of its key unused"
    assert s["RR"] > 0.4, "the network is not actually saturated at this lambda"

    # The binding constraint is a minority of core links, not the network mean.
    # Under uniform src/dst the buffer distribution is sharply bimodal - measured
    # here as 36% of links below a tenth full while the median link sits at
    # 99.9% - so mean_buffer_util is a misleading summary and is deliberately not
    # what is asserted on.  Worth stating in the paper: raising lambda does not
    # raise the mean, it deepens the core.
    frac_drained = float((res.state.buffers < 0.1 * res.topo.B_max).mean())
    assert frac_drained > 0.2, (
        f"only {frac_drained:.0%} of links drained; the fluid limit is not binding")


@pytest.mark.parametrize("km", ["OTP", "AES"])
def test_key_conservation_closes_exactly(km):
    """generation - consumption == change in total buffer, over the whole run,
    with attack, policy and multipath all active at once."""
    cfg = "config/attack_t3_aes.yaml" if km == "AES" else "config/attack_t3_otp.yaml"
    res = run(cfg, {"horizon": 21600, "warmup": 1800, "attack.t_c": 10800,
                    "attack.profile": "G", "policy.type": "B2",
                    "policy.m_paths": 2, "detector.ablation": False})
    delta = float(res.state.buffers.sum() - res.buffers_initial.sum())
    net = res.metrics.key_generated_all - res.metrics.key_consumed_all
    assert net == pytest.approx(delta, rel=1e-9, abs=1e-3)


# --------------------------------------------------------------------------- #
# 3. routing: the path table reproduces networkx, and admission biases it
# --------------------------------------------------------------------------- #
def test_unloaded_path_length_matches_networkx():
    """With unit weights the cached first path is a shortest path, so the mean
    over all pairs must equal nx.average_shortest_path_length exactly."""
    for name in ("nsfnet", "usnet", "net50"):
        topo = load_topology(load_config(T1, {}).topology.__class__(
            name=name, path=f"data/topologies/{name}.json"))
        rt = RouteTable(topo, K=4)
        rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, bool))
        pairs = list(itertools.combinations(sorted(topo.G), 2))
        mine = np.mean([len(rt.candidate(s, d, 0)) - 1 for s, d in pairs])
        ref = nx.average_shortest_path_length(topo.G)
        assert mine == pytest.approx(ref, rel=1e-12), name


def test_admission_is_biased_towards_short_paths():
    """A diagnostic, not a bug: a long path needs key on more hops, so it is
    rejected more often and the ADMITTED mean sits below the topological mean.
    Quantifying that bias is what stops it being mistaken for a routing error."""
    cfg = load_config(T3, {"horizon": 43200, "warmup": 1800})
    topo = load_topology(cfg.topology)
    topological = nx.average_shortest_path_length(topo.G)
    res = run(T3, {"horizon": 43200, "warmup": 1800})
    admitted = res.summary["mean_path_len"]
    assert admitted < topological, "admission should favour short paths"
    assert admitted > 0.5 * topological, (
        f"admitted mean {admitted:.2f} is implausibly far below the "
        f"topological mean {topological:.2f}")


# --------------------------------------------------------------------------- #
# 4. damage accounting against a closed form
# --------------------------------------------------------------------------- #
def test_total_compromise_exposes_every_admitted_session():
    """With f = 1.0 every node is compromised, so every managed session is
    exposed from the moment it is classified and D_eff must equal the closed
    form over the session log.  This exercises admission, the damage aggregates,
    the t_c reclassification and the close path together."""
    horizon, t_c = 21600, 10800
    res = run("config/attack_t3_otp.yaml",
              {"horizon": horizon, "warmup": 1800, "attack.t_c": t_c,
               "attack.enabled": True, "attack.profile": "P", "attack.f": 1.0,
               "detector.enabled": False},
              keep_session_log=True)

    t_end = float(horizon) - 1.0          # the last step runs at horizon - dt
    expected = 0.0
    seen = [s for s in res.metrics.session_log if s.managed and s.exposed]
    seen += [s for s in res.state.sessions.values() if s.managed and s.exposed]
    for s in seen:
        end = s.t_close if np.isfinite(s.t_close) else t_end
        expected += s.data_rate * max(0.0, end - s.t_exposed)
    assert res.state.D_eff == pytest.approx(expected, rel=1e-9, abs=1e-6)

    # and with everyone compromised, every session that ran past t_c must be in
    for s in res.state.sessions.values():
        if s.managed:
            assert s.exposed


def test_no_attack_leaves_no_damage_under_every_policy():
    for kind in ("B0", "B1", "B2", "B3", "B4", "BT"):
        res = run("config/attack_t3_otp.yaml",
                  {"horizon": 21600, "warmup": 1800, "attack.enabled": False,
                   "policy.type": kind, "detector.ablation": False})
        assert res.summary["D_eff"] == 0.0, kind
        assert res.summary["exposed_session_ratio"] == 0.0, kind


# --------------------------------------------------------------------------- #
# 5. slow-reference mode: every incremental aggregate against its O(n) twin
# --------------------------------------------------------------------------- #
def test_incremental_aggregates_match_full_recomputation():
    """The strongest interaction test available here.

    Every O(1) aggregate the simulator maintains - edge_rate, the managed
    variant, and the three exposure sums - is rebuilt from scratch by walking the
    session set, with attack, detector, policy and multipath all running.  A
    drift anywhere in the open/close/starve/reclassify paths shows up here and
    nowhere else.
    """
    res = run("config/attack_t3_otp.yaml",
              {"horizon": 21600, "warmup": 1800, "attack.t_c": 10800,
               "attack.profile": "G", "policy.type": "B3",
               "detector.ablation": False}, asserts=True)
    st = res.state
    np.testing.assert_allclose(st.edge_rate,
                               recompute_edge_rate_from_sessions(st), atol=1e-6)
    np.testing.assert_allclose(st.edge_rate_mgd,
                               recompute_edge_rate_from_sessions(st, managed_only=True),
                               atol=1e-6)
    rate, key_rate, hop = recompute_exposure(st)
    assert st.exposed_rate == pytest.approx(rate, abs=1e-6)
    assert st.exposed_key_rate == pytest.approx(key_rate, abs=1e-6)
    assert st.exposed_hop_key_rate == pytest.approx(hop, abs=1e-6)
    rate_r, key_r = recompute_exposure_relay(st)
    assert st.exposed_rate_relay == pytest.approx(rate_r, abs=1e-6)
    assert st.exposed_key_rate_relay == pytest.approx(key_r, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. the key rate model against its own definition
# --------------------------------------------------------------------------- #
def test_key_rate_model_and_fill_time():
    """R_e = R_max exp(-L_e/L_0), and an empty buffer fills in B_max/R_e seconds.

    The abstract rate model of phase 1 is the thing phase 7 was meant to
    cross-validate against QKDNetSim; in its absence it is at least checked
    against its own closed form and against the buffer dynamics it drives.
    """
    cfg = load_config(T1, {})
    topo = load_topology(cfg.topology)
    expected = cfg.topology.R_max * np.exp(-topo.edge_len / cfg.topology.L_0)
    np.testing.assert_allclose(topo.R, expected, rtol=1e-12)

    from sim import keygen
    e = int(np.argmax(topo.R))
    buf = np.zeros(topo.n_edges)
    need = topo.B_max / topo.R[e]
    for _ in range(int(np.ceil(need))):
        keygen.step(buf, topo.R, 1.0, topo.B_max)
    assert buf[e] == pytest.approx(topo.B_max, rel=1e-9)
    assert buf.max() <= topo.B_max + 1e-9
