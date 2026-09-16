"""Coordinated liars, and the alternative path-risk aggregations."""
from __future__ import annotations

import numpy as np
import pytest

from sim.demand import path_risk


def _line_topology(n: int = 3):
    """0 - 1 - ... - n-1, the smallest graph with an interior node."""
    import networkx as nx

    from sim.topology import Topology

    G = nx.path_graph(n)
    edges = [(u, v) for u, v in G.edges()]
    idx = {(u, v): k for k, (u, v) in enumerate(edges)}
    node_edges = [np.array([k for k, (u, v) in enumerate(edges) if i in (u, v)],
                           dtype=np.int64) for i in range(n)]
    return Topology(
        name="line", G=G, n_nodes=n, n_edges=len(edges), edge_index=idx,
        edge_list=edges, edge_len=np.full(len(edges), 20.0),
        R=np.full(len(edges), 1.0e4), B_max=1.0e7, node_edges=node_edges,
        node_deg=np.array([e.size for e in node_edges]),
        edge_ends=np.array(edges, dtype=np.int64),
        exposure=np.zeros(n), qber_base=np.full(len(edges), 0.02), meta={})


def _pair_reports(collude: bool, seed: int = 11):
    """Run the liar's observe() on a link whose two ends are both compromised."""
    from sim.attack import CompromiseAttack
    from sim.config import AttackConfig
    from sim.noise import ObservationModel
    from sim.state import init_state

    topo = _line_topology(3)                 # 0 - 1 - 2, two edges
    rng = np.random.default_rng(seed)
    noise = ObservationModel.__new__(ObservationModel)
    noise.report_noise = lambda n: 1.0 + 0.02 * np.arange(1, n + 1)

    cfg = AttackConfig(enabled=True, profile="L", f=1.0, delta=0.5, t_c=0.0,
                       collude=collude)
    atk = CompromiseAttack(cfg, 3, rng, horizon=100.0, topo=topo, noise=noise)
    atk.armed = True

    state = init_state(topo)
    atk.observe(state, 1.0)
    return state.buffer_reported


def test_colluding_ends_report_the_same_number():
    """Both ends of a doubly compromised link agree, so x2 sees nothing."""
    rep = _pair_reports(collude=True)
    for e in range(rep.shape[0]):
        assert rep[e, 0] == pytest.approx(rep[e, 1]), (
            "a coordinating pair must report one agreed value")


def test_independent_liars_still_disagree():
    """Without coordination the two draws differ, which is the signal x2 uses."""
    rep = _pair_reports(collude=False)
    assert any(rep[e, 0] != rep[e, 1] for e in range(rep.shape[0])), (
        "independent report noise should leave a discrepancy")


def test_collusion_is_off_by_default():
    from sim.config import AttackConfig
    assert AttackConfig().collude is False


# --------------------------------------------------------------------------- #
def test_path_risk_product_counts_length():
    """Two moderate relays are worse than one, under the product form."""
    S = np.array([0.0, 0.4, 0.4, 0.0])
    assert path_risk(S, [1], "product") == pytest.approx(0.4)
    assert path_risk(S, [1, 2], "product") == pytest.approx(0.64)


def test_path_risk_max_ignores_length():
    S = np.array([0.0, 0.4, 0.4, 0.0])
    assert path_risk(S, [1], "max") == pytest.approx(0.4)
    assert path_risk(S, [1, 2], "max") == pytest.approx(0.4)


def test_path_risk_mean_is_length_normalised():
    """The mean form returns the per-hop risk, so a longer path of equals ties."""
    S = np.array([0.0, 0.4, 0.4, 0.0])
    assert path_risk(S, [1, 2], "mean") == pytest.approx(0.4)


def test_path_risk_orders_the_same_way_within_a_length():
    S = np.array([0.1, 0.9])
    for agg in ("product", "max", "mean"):
        assert path_risk(S, [0], agg) < path_risk(S, [1], agg)


def test_path_risk_empty_interior_is_zero():
    for agg in ("product", "max", "mean"):
        assert path_risk(np.array([0.9, 0.9]), [], agg) == 0.0
