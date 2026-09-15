"""Phase 5: XOR key sharing over node-disjoint legs (policy B2).

The exposure rule is the dangerous part. ``_mark_exposed`` used to be an OR over
the whole path; under XOR the correct rule is ALL over the legs and ANY within a
leg - the exact opposite - and getting it backwards would overstate the paper's
headline metric with no error and no failing test anywhere. It is therefore
checked exhaustively against a closed form rather than spot-checked.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from sim.config import load_config
from sim.demand import session_is_exposed
from sim.runner import simulate
from sim.state import Session

BASE = "config/attack_t1_otp.yaml"
COMMON = {"horizon": 21600, "warmup": 1800, "attack.t_c": 10800,
          "detector.ablation": False}


def make(overrides=None):
    o = dict(COMMON)
    o.update(overrides or {})
    return load_config(BASE, o)


def run(overrides=None, **kw):
    kw.setdefault("asserts", False)
    return simulate(make(overrides), write=False, **kw)


def session_with(legs):
    return Session(sid=0, src=legs[0][0], dst=legs[0][-1], legs=tuple(legs),
                   edges=(), t_start=0.0, t_end=1.0, data_rate=1.0,
                   key_rate=1.0, next_rekey=float("inf"))


def test_xor_exposure_truth_table():
    """Every compromise mask on a 6-node two-leg session, against the closed form.

        0 -- 1 -- 2 -- 5      leg A, relays {1, 2}
        0 -- 3 -- 4 -- 5      leg B, relays {3, 4}
    """
    legs = [(0, 1, 2, 5), (0, 3, 4, 5)]
    s = session_with(legs)
    n = 6
    for bits in itertools.product([False, True], repeat=n):
        comp = np.array(bits, dtype=bool)
        exposed, relay = session_is_exposed(s, comp)

        endpoint = comp[0] or comp[5]
        both_legs = (comp[1] or comp[2]) and (comp[3] or comp[4])
        assert exposed == (endpoint or both_legs), f"mask {bits}"
        # a compromised endpoint owns the plaintext outright, so the relay share
        # is zero and the damage is attributed to the endpoint floor
        assert relay == (both_legs and not endpoint), f"mask {bits}"


def test_one_compromised_leg_leaks_nothing():
    """The whole point of XOR: m-1 shares reveal nothing."""
    s = session_with([(0, 1, 2, 5), (0, 3, 4, 5)])
    comp = np.zeros(6, dtype=bool)
    comp[1] = comp[2] = True                # every relay of leg A
    assert session_is_exposed(s, comp) == (False, False)


def test_single_leg_reduces_to_any_over_the_path():
    """With m=1 the rule must collapse to the phase 3 behaviour."""
    s = session_with([(0, 1, 2, 3)])
    for i in range(4):
        comp = np.zeros(4, dtype=bool)
        comp[i] = True
        exposed, relay = session_is_exposed(s, comp)
        assert exposed is True
        assert relay == (i in (1, 2))


def test_legs_are_node_disjoint_in_a_full_run():
    res = run({"policy.type": "B2", "policy.m_paths": 2,
               "attack.profile": "G", "attack.f": 0.2})
    assert res.summary["n_admitted"] > 0
    checked = 0
    for s in res.state.sessions.values():
        assert s.n_legs == 2
        a, b = set(s.legs[0][1:-1]), set(s.legs[1][1:-1])
        assert not (a & b), f"session {s.sid} legs share relay {a & b}"
        assert len(set(s.edges)) == len(s.edges), "the union has a repeated edge"
        checked += 1
    assert checked > 0


def test_strict_multipath_admits_exactly_m_legs():
    """B2 is strict: an admitted session always has m legs, never fewer."""
    for m in (2, 3):
        res = run({"policy.type": "B2", "policy.m_paths": m,
                   "attack.profile": "G"})
        assert res.summary["mean_legs"] == pytest.approx(float(m))
        for s in res.state.sessions.values():
            assert s.n_legs == m


def test_structural_rejections_are_labelled_separately():
    """A demand refused for want of m disjoint legs is not a key shortage."""
    res = run({"policy.type": "B2", "policy.m_paths": 3, "attack.profile": "G"})
    assert res.summary["n_no_disjoint"] > 0, "m=3 on T1 should be infeasible somewhere"
    b0 = run({"policy.type": "B0", "attack.profile": "G"})
    assert b0.summary["n_no_disjoint"] == 0


def test_metric_split_is_consistent():
    """mean_path_len = mean_leg_len * mean_legs, by construction."""
    for kind, m in (("B0", 1), ("B2", 2), ("B2", 3)):
        res = run({"policy.type": kind, "policy.m_paths": m,
                   "attack.profile": "G"})
        s = res.summary
        assert s["mean_path_len"] == pytest.approx(s["mean_leg_len"] * s["mean_legs"])
        assert s["mean_legs"] == pytest.approx(float(m if kind == "B2" else 1))


def test_multipath_costs_more_key_and_rejects_more():
    """The trade-off B2 exists to demonstrate, asserted rather than assumed."""
    over = {"attack.profile": "G", "attack.f": 0.2}
    b0 = run({"policy.type": "B0", **over})
    m2 = run({"policy.type": "B2", "policy.m_paths": 2, **over})
    m3 = run({"policy.type": "B2", "policy.m_paths": 3, **over})

    assert b0.summary["RR"] < m2.summary["RR"] < m3.summary["RR"]
    assert b0.summary["KPD"] < m2.summary["KPD"] < m3.summary["KPD"]
    assert (b0.summary["D_eff_relay"] > m2.summary["D_eff_relay"]
            > m3.summary["D_eff_relay"]), "more legs must not leak more"


def test_multipath_approaches_the_oracle_on_relay_damage():
    over = {"attack.profile": "G", "attack.f": 0.2}
    b0 = run({"policy.type": "B0", **over})
    m2 = run({"policy.type": "B2", "policy.m_paths": 2, **over})
    drr = 1.0 - m2.summary["D_eff_relay"] / b0.summary["D_eff_relay"]
    assert drr > 0.8, f"XOR over two disjoint legs only achieved DRR {drr:.3f}"


def test_aes_pulse_count_under_multipath():
    """Each leg pays its own 256-bit pulse on every hop."""
    res = simulate(load_config("config/attack_t3_aes.yaml",
                               {"horizon": 21600, "warmup": 1800,
                                "attack.t_c": 10800, "detector.ablation": False,
                                "policy.type": "B2", "policy.m_paths": 2}),
                   write=False, asserts=True)
    assert res.summary["n_admitted"] > 0
    assert res.summary["mean_legs"] == pytest.approx(2.0)
    # key conservation still holds with m-fold pulses
    delta = float(res.state.buffers.sum() - res.buffers_initial.sum())
    net = res.metrics.key_generated_all - res.metrics.key_consumed_all
    assert net == pytest.approx(delta, rel=1e-9, abs=1e-3)


def test_deep_invariants_hold_under_multipath():
    """edge_rate, edge_rate_mgd and the exposure aggregates must stay exact."""
    res = run({"policy.type": "B2", "policy.m_paths": 2,
               "attack.profile": "G", "attack.f": 0.2}, asserts=True)
    from sim.state import (recompute_edge_rate_from_sessions, recompute_exposure,
                           recompute_exposure_relay)
    np.testing.assert_allclose(res.state.edge_rate,
                               recompute_edge_rate_from_sessions(res.state), atol=1e-6)
    rate, key_rate, hop = recompute_exposure(res.state)
    assert res.state.exposed_rate == pytest.approx(rate, abs=1e-6)
    assert res.state.exposed_key_rate == pytest.approx(key_rate, abs=1e-6)
    rate_r, key_r = recompute_exposure_relay(res.state)
    assert res.state.exposed_rate_relay == pytest.approx(rate_r, abs=1e-6)
    assert res.state.exposed_key_rate_relay == pytest.approx(key_r, abs=1e-6)
