"""Section 4.5 acceptance criteria - admission, accounting, starvation, AES pulses."""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import T1
from sim.config import DemandConfig, TopologyConfig
from sim.demand import DemandRequest, SessionManager
from sim.metrics import MetricsAccumulator
from sim.routing import RouteTable
from sim.state import init_state, recompute_edge_rate_from_sessions
from sim.topology import load_topology


def build(km_mode="OTP", B_max=10e6, **demand_kw):
    topo = load_topology(TopologyConfig(name="nsfnet", path=T1, B_max=B_max))
    dcfg = DemandConfig(km_mode=km_mode, **demand_kw)
    state = init_state(topo)
    metrics = MetricsAccumulator(0.0, topo)
    mgr = SessionManager(dcfg, topo, metrics, np.random.default_rng(0))
    routes = RouteTable(topo, K=4)
    routes.rebuild(state.node_weight, state.isolated)
    return topo, dcfg, state, metrics, mgr, routes


def test_session_accounting():
    """Sum of what a session consumed equals the drop in every buffer it used."""
    topo, dcfg, state, metrics, mgr, routes = build()
    T_s, rate = 100.0, 500.0
    req = DemandRequest(rid=0, src=0, dst=9, t=0.0, T_s=T_s, data_rate=rate)
    s = mgr.try_admit(req, state, routes, topo, dcfg)
    assert s is not None and s.hops >= 2

    before = state.buffers.copy()
    for step in range(1, int(T_s) + 1):
        mgr.step(state, float(step), 1.0)

    drop = before - state.buffers
    expected = np.zeros(topo.n_edges)
    for e in s.edges:
        expected[e] = rate * T_s
    assert np.allclose(drop, expected, atol=1e-6)
    assert metrics.key_consumed_all == pytest.approx(rate * T_s * s.hops)
    assert s.status == "completed"
    assert metrics.n_completed == 1
    assert np.all(state.edge_rate == 0.0)
    assert state.sessions == {}


def test_reserve_full_deducts_up_front():
    topo, dcfg, state, metrics, mgr, routes = build(admission="reserve_full")
    T_s, rate = 100.0, 500.0
    req = DemandRequest(0, 0, 9, 0.0, T_s, rate)
    before = state.buffers.copy()
    s = mgr.try_admit(req, state, routes, topo, dcfg)
    assert s is not None
    drop = before - state.buffers
    for e in s.edges:
        assert drop[e] == pytest.approx(rate * T_s)
    assert s.key_rate == 0.0                       # already paid for
    for step in range(1, int(T_s) + 1):
        mgr.step(state, float(step), 1.0)
    assert metrics.key_consumed_all == pytest.approx(rate * T_s * s.hops)


def test_interrupt_on_starvation():
    """With a deliberately small buffer the session is cut, and edge_rate and the
    per edge session sets are updated correctly."""
    B_max = 1e6
    rate = 20000.0                       # above R_e of every T1 link (max 13406)
    topo, dcfg, state, metrics, mgr, routes = build(B_max=B_max, T_adm=5.0)
    req = DemandRequest(0, 0, 9, 0.0, 3600.0, rate)
    s = mgr.try_admit(req, state, routes, topo, dcfg)
    assert s is not None

    # no key generation on purpose: the buffer drains in exactly B_max/rate steps
    drain = int(B_max // rate)
    for step in range(1, drain + 1):
        mgr.step(state, float(step), 1.0)
    assert s.status == "active"
    assert state.buffers[list(s.edges)].max() == pytest.approx(0.0, abs=1e-9)

    mgr.step(state, float(drain + 1), 1.0)
    assert s.status == "interrupted"
    assert metrics.n_interrupted == 1
    assert state.sessions == {}
    assert np.all(state.buffers >= -1e-9)
    assert np.all(state.edge_rate == 0.0)
    assert np.allclose(state.edge_rate, recompute_edge_rate_from_sessions(state))
    assert all(len(x) == 0 for x in state.edge_sessions)
    # the deficit was clawed back, so consumption never exceeds what existed
    assert metrics.key_consumed_all == pytest.approx(B_max * s.hops)


def test_starvation_cuts_newest_first():
    """LIFO: older sessions, which have more invested in them, survive."""
    # on the shortest 0->9 path the weakest link generates 3234 bit/s: one
    # session at 3000 bit/s is sustainable, two are not, so the cut rule must
    # remove exactly the newer one
    B_max = 2e6
    topo, dcfg, state, metrics, mgr, routes = build(B_max=B_max, T_adm=5.0)
    s_old = mgr.try_admit(DemandRequest(0, 0, 9, 0.0, 3600.0, 3000.0),
                          state, routes, topo, dcfg)
    s_new = mgr.try_admit(DemandRequest(1, 0, 9, 0.0, 3600.0, 3000.0),
                          state, routes, topo, dcfg)
    assert s_old is not None and s_new is not None and s_old.edges == s_new.edges

    for step in range(1, 500):
        mgr.step(state, float(step), 1.0)
        if s_new.status != "active":
            break
    assert s_new.status == "interrupted"
    assert s_old.status == "active"


@pytest.mark.parametrize("T_s", [60.0, 120.0, 250.0, 600.0])
def test_aes_pulse_count(T_s):
    """Exactly ceil(T_s / T_rk) re-key pulses per edge."""
    topo, dcfg, state, metrics, mgr, routes = build(km_mode="AES", T_rk=60.0,
                                                    key_size=256)
    req = DemandRequest(0, 0, 9, 0.0, T_s, 0.0)
    s = mgr.try_admit(req, state, routes, topo, dcfg)
    assert s is not None and s.key_rate == 0.0

    before = state.buffers.copy()
    for step in range(1, int(T_s) + 1):
        mgr.step(state, float(step), 1.0)

    expected_pulses = math.ceil(T_s / dcfg.T_rk)
    for e in s.edges:
        assert (before[e] - state.buffers[e]) == pytest.approx(
            expected_pulses * dcfg.key_size)
    assert metrics.key_consumed_all == pytest.approx(
        expected_pulses * dcfg.key_size * s.hops)
    assert s.status == "completed"
    assert state.rekey_heap == [] or all(
        sid not in state.sessions for _, sid in state.rekey_heap)


def test_admission_rejects_when_buffer_too_low():
    """Optimistic admission needs key_rate * T_adm on every hop."""
    topo, dcfg, state, metrics, mgr, routes = build(T_adm=60.0)
    state.buffers[:] = 100.0                        # far below any need
    req = DemandRequest(0, 0, 9, 0.0, 100.0, 500.0)
    assert mgr.try_admit(req, state, routes, topo, dcfg) is None
    assert state.sessions == {}


def test_admission_skips_isolated_transit_node():
    topo, dcfg, state, metrics, mgr, routes = build()
    state.isolated[8] = True
    routes.rebuild(state.node_weight, state.isolated)
    for src, dst in [(0, 9), (7, 11), (6, 12)]:
        s = mgr.try_admit(DemandRequest(0, src, dst, 0.0, 100.0, 500.0),
                          state, routes, topo, dcfg)
        if s is not None:
            assert 8 not in s.path[1:-1]
