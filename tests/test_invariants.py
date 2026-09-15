"""The four runtime invariants."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import make_cfg
from sim import keygen
from sim.runner import simulate
from sim.state import recompute_edge_rate_from_sessions


@pytest.mark.parametrize("km_mode", ["OTP", "AES"])
def test_key_conservation(tmp_path, km_mode):
    """generated - consumed == change in total buffer, over the whole run."""
    cfg = make_cfg(tmp_path, demand={"km_mode": km_mode,
                                     "lam": 0.2 if km_mode == "OTP" else 3.0})
    res = simulate(cfg, write=False, asserts=True)
    m = res.metrics

    delta_buffer = float(res.state.buffers.sum() - res.buffers_initial.sum())
    balance = m.key_generated_all - m.key_consumed_all
    scale = max(abs(balance), abs(delta_buffer), 1.0)
    assert abs(balance - delta_buffer) / scale < 1e-9
    assert m.key_consumed_all > 0.0


def test_buffer_bounds(tmp_path):
    cfg = make_cfg(tmp_path, demand={"lam": 1.0})
    res = simulate(cfg, write=False, asserts=True)
    assert np.all(res.state.buffers >= -1e-6)
    assert np.all(res.state.buffers <= res.topo.B_max + 1e-6)


def test_edge_rate_consistency(tmp_path):
    """The incremental edge_rate must equal a full walk of the active sessions.

    This is the invariant that the O(|E|) architecture is most likely to break.
    """
    cfg = make_cfg(tmp_path, demand={"lam": 1.0})
    res = simulate(cfg, write=False, asserts=True)
    ref = recompute_edge_rate_from_sessions(res.state)
    assert np.allclose(res.state.edge_rate, ref, atol=1e-6)
    assert np.all(res.state.edge_rate >= -1e-6)
    # a non trivial state, otherwise the assertion above is vacuous
    assert len(res.state.sessions) > 0


def test_buffer_fills_in_expected_time(topo):
    """An empty buffer is exactly full after B_max / R_e seconds and never
    exceeds B_max (acceptance criterion of section 4.3)."""
    e = int(np.argmax(topo.R))
    R_e = float(topo.R[e])
    n = int(np.ceil(topo.B_max / R_e))

    buffers = np.zeros(topo.n_edges)
    for i in range(n):
        keygen.step(buffers, topo.R, 1.0, topo.B_max)
        assert buffers[e] <= topo.B_max + 1e-9
        if i == n - 2:
            assert buffers[e] < topo.B_max          # not yet full one step earlier
    assert buffers[e] == pytest.approx(topo.B_max, rel=1e-12)

    keygen.step(buffers, topo.R, 1.0, topo.B_max)
    assert np.all(buffers <= topo.B_max + 1e-9)
