"""The reimplemented published methods, and the physics they all run on.

Two things are pinned here.

First, the three literature baselines (B5 Luo & Li 2025, B6 Kiktenko et al.
2024, B7 Bi et al. 2023) must actually differ from our own policies in the way
their papers say they do - otherwise the comparison in the paper is against a
relabelled copy of B0 and means nothing.

Second, the key-rate model's decay constant is not a free parameter, and the
rescaling that put it on a physical footing has to be exactly rate-preserving or
every number in the results section moved when it was applied.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.config import load_config
from sim.runner import simulate
from sim.topology import key_rate

BASE = "config/attack_t3_otp.yaml"
COMMON = {"horizon": 7200, "warmup": 1800, "attack.t_c": 3600,
          "attack.enabled": True}


def run(overrides=None):
    o = dict(COMMON)
    o.update(overrides or {})
    return simulate(load_config(BASE, o), write=False, asserts=False).summary


# --------------------------------------------------------------------------- #
# the physical footing
# --------------------------------------------------------------------------- #
def test_L0_is_the_value_the_fibre_attenuation_implies():
    """L_0 is fixed by physics, not chosen.

    Secret key rate scales with channel transmittance eta = 10^(-alpha L / 10),
    so exp(-L / L_0) = 10^(-alpha L / 10) forces L_0 = 10 / (alpha ln 10).  At
    the standard 1550 nm fibre attenuation of 0.2 dB/km that is 21.71 km.  The
    phase 1 spec's 50 km would need alpha = 0.087 dB/km, below the Rayleigh
    scattering limit of silica - a fibre that does not exist.
    """
    cfg = load_config(BASE, {})
    alpha = 10.0 / (cfg.topology.L_0 * math.log(10.0))
    assert alpha == pytest.approx(0.2, abs=5e-4), (
        f"L_0 = {cfg.topology.L_0} km implies alpha = {alpha:.4f} dB/km; "
        "the model is only physical at alpha = 0.2")


def test_length_rescaling_is_exactly_rate_preserving():
    """Scaling every link length and L_0 together cannot move a single rate.

    This is what made the correction free: link length enters the model ONLY
    through key_rate() - qber_base is an independent per-edge draw - so the
    relabelling that moved the links from [20, 120] km to [8.7, 52] km left the
    simulation identical, and every number measured before it stayed valid.
    """
    L = np.array([8.69, 19.4, 33.0, 52.11])
    for k in (0.434294, 0.5, 2.0, 7.3):
        a = key_rate(L, 20000.0, 21.7147)
        b = key_rate(L * k, 20000.0, 21.7147 * k)
        assert np.allclose(a, b, rtol=1e-12, atol=0.0)


def test_only_the_ratio_of_key_supply_to_demand_matters():
    """Scale the key supply and the key demand together: nothing dimensionless moves.

    This is the answer to "your key rates are far below a current commercial
    QKD system".  They are, and it cannot matter: every metric this paper
    reports is either dimensionless (rejection rate, AUC, hop count) or a ratio
    of two damages (DRR), and all of them are invariant.  Only the extensive
    quantities scale, and they scale exactly.
    """
    o = {"attack.profile": "E", "policy.type": "B3", "policy.S_iso": 0.5}
    base = run(o)
    s = 25.0
    scaled = dict(o)
    scaled.update({"topology.R_max": 20000.0 * s, "topology.B_max": 10.0e6 * s,
                   "demand.rate_min": 10.0 * s, "demand.rate_max": 1000.0 * s})
    got = run(scaled)
    for m in ("RR", "n_admitted", "mean_leg_len", "mean_buffer_util", "auc"):
        assert got[m] == pytest.approx(base[m], rel=1e-9), f"{m} is not invariant"
    for m in ("D_eff_relay", "KPD"):
        assert got[m] == pytest.approx(base[m] * s, rel=1e-6), f"{m} did not scale"


# --------------------------------------------------------------------------- #
# the literature baselines
# --------------------------------------------------------------------------- #
def test_b5_detects_the_liar_and_nothing_else():
    """Luo & Li's trust is built on ONE observable, and it shows.

    Their C_e derates a link by the disagreement between its two endpoints'
    reports - which is exactly our x2, the feature that responds to profile L
    and to nothing else.  So a faithful port must move L and leave P, G and E
    alone.  That is the whole argument for x1 and x3 stated as a test: the
    published method is not weak, it is narrow.
    """
    drr = {}
    for p in "PGLE":
        b0 = run({"attack.profile": p, "policy.type": "B0"})
        b5 = run({"attack.profile": p, "policy.type": "B5"})
        drr[p] = 1.0 - b5["D_eff_relay"] / b0["D_eff_relay"]
    assert drr["L"] > 0.25, f"B5 should catch the liar; got {drr['L']:.3f}"
    for p in "PGE":
        assert drr[p] < 0.15, (
            f"B5 moved profile {p} by {drr[p]:.3f}; x2 carries no evidence "
            "against it, so anything large here is blind throttling")


def test_b6_selects_different_paths_from_b2():
    """B6 differs from B2 in path selection, which is the only thing it changes.

    Both relay XOR shares over internally node-disjoint paths - that scheme is
    Kiktenko et al.'s and B2 should be credited to them.  Theirs picks the set
    whose worst link is shortest of key; ours picks the cheapest by hop count.
    If the two came out identical the baseline would be a relabelled B2.
    """
    a = run({"attack.profile": "E", "policy.type": "B2", "policy.m_paths": 2})
    b = run({"attack.profile": "E", "policy.type": "B6", "policy.m_paths": 2})
    assert a["mean_legs"] == pytest.approx(2.0)
    assert b["mean_legs"] == pytest.approx(2.0)
    assert b["D_eff_relay"] != a["D_eff_relay"], "B6 chose exactly B2's paths"


def test_b7_moves_routing_without_reading_the_detector():
    """Bi et al. weight links by remaining key; no detector is involved.

    The point of including it is to answer "could good key-aware routing make a
    detector unnecessary?", so the test pins the two halves of that: it must
    change the routing, and it must not need the score to do it.
    """
    from sim.policy import KeyAwareRoutingPolicy
    assert KeyAwareRoutingPolicy.needs_detector is False
    b0 = run({"attack.profile": "E", "policy.type": "B0"})
    b7 = run({"attack.profile": "E", "policy.type": "B7"})
    assert b7["mean_leg_len"] != pytest.approx(b0["mean_leg_len"], rel=1e-6), (
        "B7 produced B0's paths; the edge cost never reached the route table")


def test_edge_cost_defaults_leave_the_node_weighted_policies_untouched():
    """The edge-cost hook must be invisible to B0-B4.

    It was added for the literature baselines, and a hook that quietly changed
    the policies the paper's own results rest on would invalidate them.
    """
    from sim.policy import GradedPolicy, NoDefencePolicy
    for cls in (NoDefencePolicy, GradedPolicy):
        assert cls.edge_cost(cls.__new__(cls), None) is None
