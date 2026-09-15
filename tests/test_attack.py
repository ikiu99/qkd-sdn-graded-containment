"""Phase 3: attack injection, observation noise and damage accounting.

Section 3.7 of phase3-4-build-spec.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.config import load_config
from sim.runner import simulate

BASE = "config/base_t1_otp.yaml"
HORIZON = 14400          # 4 h: warm-up 1800, t_c 3600, 3 h of attack
COMMON = {"horizon": HORIZON, "warmup": 1800, "attack.t_c": 3600}


def make(overrides=None, **kw):
    o = dict(COMMON)
    o.update(overrides or {})
    return load_config(BASE, o)


def run(overrides=None, **kw):
    return simulate(make(overrides), write=False, asserts=True, **kw)


# --------------------------------------------------------------------------- #
def test_no_attack_zero_damage():
    res = run({"attack.enabled": False})
    assert res.summary["D_eff"] == 0.0
    assert res.summary["D_raw"] == 0.0
    assert res.summary["n_compromised"] == 0
    assert not res.state.exposed_sessions


def test_passive_profile_changes_nothing():
    """Profile P is the control condition and must be undetectable.

    Stronger than the KS test the spec asks for: with no policy feedback, a
    passive compromise cannot touch the dynamics at all, so the two runs are
    identical observable for observable.  If this ever fails, profile P has
    started leaving a trace and the 'undetectable by construction' claim is void.
    """
    off = run({"attack.enabled": False})
    passive = run({"attack.enabled": True, "attack.profile": "P"})

    for key in ("n_admitted", "n_rejected", "n_interrupted", "RR", "KPD",
                "total_key_generated", "total_key_consumed", "mean_buffer_util"):
        assert off.summary[key] == passive.summary[key], key

    for field in ("buffers", "qber_obs", "R_obs", "ledger_consumed",
                  "actual_consumed", "buffer_reported"):
        np.testing.assert_array_equal(getattr(off.state, field),
                                      getattr(passive.state, field), err_msg=field)

    # ...while the ground truth is recorded all the same
    assert passive.summary["n_compromised"] > 0
    assert passive.summary["D_eff"] > 0.0


def test_passive_observables_ks():
    """The spec's formulation: the _obs distributions must be indistinguishable."""
    off = run({"attack.enabled": False})
    passive = run({"attack.enabled": True, "attack.profile": "P"})
    a = np.sort(off.state.qber_obs)
    b = np.sort(passive.state.qber_obs)
    grid = np.union1d(a, b)
    d = np.abs(np.searchsorted(a, grid, "right") / a.size
               - np.searchsorted(b, grid, "right") / b.size)
    assert float(d.max()) == 0.0          # p = 1, they are the same sample


def test_damage_accounting():
    """Incremental D_raw/D_eff == full recomputation from the session log."""
    res = run({"attack.enabled": True, "attack.profile": "G"},
              keep_session_log=True)
    # the last step runs at horizon - dt, so that is where accrual stops for a
    # session still open at the end
    t_end = float(HORIZON) - 1.0

    # independent recomputation straight from the session log: exposure starts
    # at t_exposed and never switches off, so the exposed span is exact
    d_raw = d_eff = 0.0
    seen = [s for s in res.metrics.session_log if s.managed and s.exposed]
    seen += [s for s in res.state.sessions.values() if s.managed and s.exposed]
    assert seen, "no exposed session to check"
    for s in seen:
        end = s.t_close if np.isfinite(s.t_close) else t_end
        span = max(0.0, end - s.t_exposed)
        d_raw += s.key_rate * span
        d_eff += s.data_rate * span

    assert res.state.D_raw == pytest.approx(d_raw, rel=1e-9, abs=1e-6)
    assert res.state.D_eff == pytest.approx(d_eff, rel=1e-9, abs=1e-6)
    assert res.state.D_eff > 0.0


@pytest.mark.parametrize("km_mode", ["OTP", "AES"])
def test_relay_endpoint_split_is_charged_in_both_key_modes(km_mode):
    """D_eff_relay is accrued on whichever path charges D_eff, not just OTP's.

    This is the regression for a bug that produced no error anywhere: the relay
    split was added with the phase 5 policies, after the AES re-key path had
    already been written, and only the OTP continuous-accrual branch was
    updated.  D_eff stayed exact, the split identity D_eff = relay + endpoint
    still held because endpoint is *defined* as the remainder, and every AES run
    silently reported D_eff_relay = 0 - which turns DRR_relay, the metric every
    policy is judged on, into 0/0 for half the experiment matrix.

    So the assertion that matters is not the identity; it is that the relay
    share is non-zero and matches an independent recomputation from the session
    log.  Damage is charged over the exposed span in OTP and at each re-key
    pulse in AES, so the two modes are compared against their own definitions.
    """
    o = {"attack.enabled": True, "attack.profile": "G", "attack.f": 0.2,
         "demand.km_mode": km_mode}
    if km_mode == "AES":
        o["demand.lam"] = 4.0          # hold roughly the offered key load
    res = run(o, keep_session_log=True)
    st = res.state

    assert st.D_eff > 0.0, "no damage at all; the test cell is wrong"
    assert st.D_eff_relay > 0.0, (
        f"{km_mode}: relay damage is zero while D_eff = {st.D_eff:.3g}; the "
        "accrual path for this key mode does not update the split")
    assert st.D_eff_relay <= st.D_eff + 1e-6
    assert res.summary["D_eff_endpoint"] == pytest.approx(
        st.D_eff - st.D_eff_relay, rel=1e-9, abs=1e-6)

    # Independent recomputation: every exposed session contributes its own
    # data_rate x span to exactly one of the two buckets.  OTP charges the
    # exposed span continuously, so the span is reconstructed from the log the
    # way test_damage_accounting does it - ``exposed_bits`` is only stamped when
    # a session closes and would undercount the ones still open at the horizon.
    # AES charges per pulse, where ``exposed_bits`` is the running total and the
    # span is not reconstructible from t_exposed alone.
    seen = [s for s in res.metrics.session_log if s.managed and s.exposed]
    seen += [s for s in st.sessions.values() if s.managed and s.exposed]
    if km_mode == "AES":
        bits = {s.sid: s.exposed_bits for s in seen}
    else:
        t_end = float(HORIZON) - 1.0
        bits = {s.sid: s.data_rate * max(
            0.0, (s.t_close if np.isfinite(s.t_close) else t_end) - s.t_exposed)
            for s in seen}
    relay = sum(bits[s.sid] for s in seen if s.exposed_relay)
    total = sum(bits.values())
    assert total > 0.0
    assert st.D_eff_relay == pytest.approx(relay, rel=1e-6, abs=1e-3)
    assert st.D_eff == pytest.approx(total, rel=1e-6, abs=1e-3)


def test_exposed_set_consistency():
    """The three running aggregates match a full walk of the exposed set.

    Checked continuously by the deep invariant every 1000 steps; this pins the
    end state explicitly.
    """
    from sim.state import recompute_exposure
    res = run({"attack.enabled": True, "attack.profile": "G"})
    rate, key_rate, hop = recompute_exposure(res.state)
    assert res.state.exposed_rate == pytest.approx(rate, abs=1e-6)
    assert res.state.exposed_key_rate == pytest.approx(key_rate, abs=1e-6)
    assert res.state.exposed_hop_key_rate == pytest.approx(hop, abs=1e-6)


def test_reclassification_at_tc():
    """Right after t_c every active session crossing a compromised node is exposed."""
    cfg = make({"attack.enabled": True, "attack.profile": "P"})
    res = simulate(cfg, write=False, asserts=False)
    assert res.summary["n_reclassified_at_tc"] > 0

    # nothing crossing a compromised node may be left outside the exposed set
    for s in res.state.sessions.values():
        if not s.managed:
            continue
        crosses = any(res.state.compromised[n] for n in s.path)
        assert s.exposed == crosses, f"session {s.sid} exposed={s.exposed}"


def test_damage_counted_once_per_session():
    """A path crossing two compromised nodes is charged once, not twice."""
    res = run({"attack.enabled": True, "attack.profile": "P", "attack.f": 0.5},
              keep_session_log=True)
    multi = [s for s in res.metrics.session_log
             if s.managed and s.exposed
             and sum(res.state.compromised[n] for n in s.path) >= 2]
    assert multi, "test needs at least one path crossing two compromised nodes"
    for s in multi:
        span = (s.t_close if np.isfinite(s.t_close)
                else float(HORIZON) - 1.0) - s.t_exposed
        assert s.exposed_bits == pytest.approx(s.data_rate * span, rel=1e-9), (
            "damage scaled with the number of compromised nodes on the path")


def test_e_profile_no_double_count():
    """Two compromised nodes sharing a tapped link must not stack the effect."""
    res = run({"attack.enabled": True, "attack.profile": "E", "attack.f": 0.6})
    st = res.state
    tapped = st.attack_qber_delta > 0
    assert tapped.any()
    assert np.allclose(st.attack_qber_delta[tapped], 0.03), \
        "overlapping taps stacked; use np.maximum, not +="
    assert np.allclose(st.attack_skr_drop[tapped], 0.25)
    assert st.attack_qber_delta.max() <= 0.03 + 1e-12


def test_greedy_drains_off_ledger():
    """Profile G raises actual consumption without touching the ledger."""
    clean = run({"attack.enabled": False})
    greedy = run({"attack.enabled": True, "attack.profile": "G"})

    resid_clean = float((clean.state.actual_consumed
                         - clean.state.ledger_consumed).sum())
    resid_greedy = float((greedy.state.actual_consumed
                          - greedy.state.ledger_consumed).sum())
    assert resid_greedy > resid_clean, "the greedy drain never reached the buffers"

    topo, st = greedy.topo, greedy.state
    victim_edges = np.unique(np.concatenate(
        [topo.node_edges[i] for i in np.flatnonzero(st.compromised)]))
    other = np.setdiff1d(np.arange(topo.n_edges), victim_edges)
    resid = st.actual_consumed - st.ledger_consumed
    assert resid[victim_edges].mean() > resid[other].mean(), \
        "the residual is not concentrated on the compromised node's links"


def test_liar_only_touches_reports():
    """Profile L changes what is reported, never the real buffer."""
    clean = run({"attack.enabled": False})
    liar = run({"attack.enabled": True, "attack.profile": "L"})
    np.testing.assert_allclose(clean.state.buffers, liar.state.buffers)

    st = liar.state
    over = st.buffer_reported.max(axis=1) > st.buffers * 1.2
    assert over.any(), "no link is being over reported"


def test_background_traffic_is_off_ledger_and_unmetered():
    """N4 consumes real key, is invisible to the ledger and to RR/KPD."""
    with_bg = run({"attack.enabled": False})
    without = run({"attack.enabled": False, "noise.unmanaged_fraction": 0.0})

    assert with_bg.summary["n_background_admitted"] > 0
    assert without.summary["n_background_admitted"] == 0
    # background sessions never enter the offered/admitted counters
    assert with_bg.summary["n_offered"] == with_bg.summary["n_admitted"] \
        + with_bg.summary["n_rejected"]

    # The residual is signed: the ledger is deliberately not clawed back when a
    # starved link cannot deliver the key it authorised, so it can run negative.
    # What N4 has to do is push it up.
    resid_bg = float((with_bg.state.actual_consumed
                      - with_bg.state.ledger_consumed).sum())
    resid_no = float((without.state.actual_consumed
                      - without.state.ledger_consumed).sum())
    assert resid_bg > resid_no


def test_noise_present():
    """Healthy nodes must carry a non zero raw x1.

    This is the guard against quietly dropping N4: without it K_obs == K_exp for
    every healthy node, the residual is identically zero and F1 scores a
    meaningless AUC of 1.0 against profile G.
    """
    res = run({"attack.enabled": False, "detector.enabled": True})
    hist = res.detector._hist[:, :, 0]
    assert np.isfinite(hist).all()
    assert hist.var() > 0.0

    quiet = simulate(make({"attack.enabled": False, "detector.enabled": True,
                           "noise.unmanaged_fraction": 0.0,
                           "noise.sigma_report_rel": 0.0}),
                     write=False, asserts=False)
    assert res.detector._hist[:, :, 0].mean() > quiet.detector._hist[:, :, 0].mean()


def test_attack_seed_isolation():
    """Changing the demand seed must not move the compromised set."""
    a = run({"attack.enabled": True, "attack.profile": "P", "seed_demand": 2})
    b = run({"attack.enabled": True, "attack.profile": "P", "seed_demand": 99})
    np.testing.assert_array_equal(a.state.compromised, b.state.compromised)


def test_same_victims_across_profiles():
    """The four profiles stay paired: same seed, same V_c."""
    sets = [run({"attack.enabled": True, "attack.profile": p}).state.compromised
            for p in ("P", "G", "L", "E")]
    for other in sets[1:]:
        np.testing.assert_array_equal(sets[0], other)


def test_key_conservation_with_attack():
    """Generation minus consumption still equals the buffer change, even when a
    greedy attacker is siphoning key off the books."""
    res = run({"attack.enabled": True, "attack.profile": "G"})
    delta = float(res.state.buffers.sum() - res.buffers_initial.sum())
    net = res.metrics.key_generated_all - res.metrics.key_consumed_all
    assert net == pytest.approx(delta, rel=1e-9, abs=1e-3)
