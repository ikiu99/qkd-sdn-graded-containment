"""Section 4.4 acceptance criteria."""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from sim.routing import RouteTable
from sim.topology import edges_of_path


def _weighted_reference(topo, node_weight):
    G = nx.Graph()
    G.add_nodes_from(topo.G.nodes())
    for u, v in topo.edge_list:
        G.add_edge(u, v, w=0.5 * (node_weight[u] + node_weight[v]))
    return G


def test_first_path_is_dijkstra(topo):
    """With NullPolicy weights the first cached path is the plain Dijkstra path;
    the same must hold for a non uniform weight vector."""
    for weights in (np.ones(topo.n_nodes),
                    np.linspace(1.0, 3.0, topo.n_nodes)):
        rt = RouteTable(topo, K=4)
        rt.rebuild(weights, np.zeros(topo.n_nodes, dtype=bool))
        ref_G = _weighted_reference(topo, weights)
        for s in range(topo.n_nodes):
            for d in range(topo.n_nodes):
                if s == d:
                    continue
                first = rt.paths(s, d)[0]
                ref = nx.dijkstra_path(ref_G, s, d, weight="w")
                assert rt.path_cost(first) == pytest.approx(
                    nx.dijkstra_path_length(ref_G, s, d, weight="w"))
                assert len(first) == len(ref)


def test_paths_are_cost_ordered_and_capped(topo):
    rt = RouteTable(topo, K=3)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, dtype=bool))
    paths = rt.paths(0, 9)
    assert 1 <= len(paths) <= 3
    costs = [rt.path_cost(p) for p in paths]
    assert costs == sorted(costs)
    assert len(set(paths)) == len(paths)


def test_isolated_excluded(topo):
    iso = np.zeros(topo.n_nodes, dtype=bool)
    victim = 8                                   # a degree 4 transit node in T1
    iso[victim] = True
    rt = RouteTable(topo, K=4)
    rt.rebuild(np.ones(topo.n_nodes), iso)
    for s in range(topo.n_nodes):
        for d in range(topo.n_nodes):
            if s == d or victim in (s, d):
                continue
            for p in rt.paths(s, d):
                assert victim not in p


def test_rebuild_only_when_stale(topo):
    w = np.ones(topo.n_nodes)
    iso = np.zeros(topo.n_nodes, dtype=bool)
    rt = RouteTable(topo, K=4)

    assert rt.is_stale(w, iso) is True            # never built yet
    rt.rebuild(w, iso)
    v = rt.version

    for _ in range(10):                           # unchanged weights -> no rebuild
        assert rt.is_stale(w.copy(), iso.copy()) is False
    assert rt.version == v

    w2 = w.copy()
    w2[3] = 2.0
    assert rt.is_stale(w2, iso) is True
    rt.rebuild(w2, iso)
    assert rt.version == v + 1


def test_reverse_pair_is_mirrored(topo):
    rt = RouteTable(topo, K=4)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, dtype=bool))
    fwd = rt.paths(2, 11)
    rev = rt.paths(11, 2)
    assert [tuple(reversed(p)) for p in fwd] == rev


def test_disjoint_paths_share_no_edge(topo):
    from sim.topology import edges_of_path
    rt = RouteTable(topo, K=4, disjoint_K=3)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, dtype=bool))
    seen: set[int] = set()
    for p in rt.disjoint_paths(0, 13):
        eids = set(edges_of_path(topo, p))
        assert not (eids & seen)
        seen |= eids


# --------------------------------------------------------------------------- #
# phase 5: node-disjoint path sets for policy B2
# --------------------------------------------------------------------------- #
def test_disjoint_paths_share_no_node(topo):
    """The security claim of B2 rests on this.

    B2 relays one XOR key share per leg, so a relay sitting on two legs sees two
    shares and reconstructs the key.  Edge-disjointness is not enough: two
    edge-disjoint paths may share an intermediate node.
    """
    rt = RouteTable(topo, K=4, disjoint_K=3)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, bool))
    checked = 0
    for s, d in itertools.combinations(sorted(topo.G), 2):
        legs = rt.disjoint_paths(s, d, 2)
        if not legs:
            continue
        checked += 1
        assert len(legs) == 2
        a, b = set(legs[0][1:-1]), set(legs[1][1:-1])
        assert not (a & b), f"{s}->{d} legs share relay {a & b}"
        ea = set(edges_of_path(topo, legs[0]))
        eb = set(edges_of_path(topo, legs[1]))
        assert not (ea & eb), "node-disjoint must imply edge-disjoint"
        for leg in legs:
            assert leg[0] == s and leg[-1] == d
    assert checked == len(list(itertools.combinations(sorted(topo.G), 2)))


def test_disjoint_paths_are_minimum_cost(topo):
    """Exact minimum cost, not greedy over the K Yen paths.

    Brute forced against every node-disjoint pair of simple paths, so the test
    states the property rather than a number that happened to come out.  Greedy
    filtering of Yen's list can miss the optimum because Yen's paths are minor
    perturbations of each other and tend to share interior nodes.
    """
    rt = RouteTable(topo, K=4, disjoint_K=2)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, bool))
    G = rt._G

    for s_, d_ in list(itertools.combinations(sorted(topo.G), 2))[:12]:
        legs = rt.disjoint_paths(s_, d_, 2)
        assert len(legs) == 2
        got = sum(rt.path_cost(p) for p in legs)

        simple = [tuple(p) for p in nx.all_simple_paths(G, s_, d_, cutoff=8)]
        best = min(
            (rt.path_cost(a) + rt.path_cost(b)
             for a, b in itertools.combinations(simple, 2)
             if not (set(a[1:-1]) & set(b[1:-1]))),
            default=None)
        assert best is not None
        assert got == pytest.approx(best), f"{s_}->{d_}: got {got}, optimum {best}"


def test_disjoint_paths_strict_when_infeasible(topo):
    """Fewer than m disjoint paths must give [], never a short list."""
    rt = RouteTable(topo, K=4, disjoint_K=3)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, bool))
    infeasible = 0
    for s, d in itertools.combinations(sorted(topo.G), 2):
        legs = rt.disjoint_paths(s, d, 3)
        assert legs == [] or len(legs) == 3
        infeasible += not legs
    assert infeasible > 0, "T1 should have pairs without three disjoint paths"


def test_lazy_in_k_matches_full_materialisation(topo):
    """candidate(i) must return exactly what paths() would have."""
    rt = RouteTable(topo, K=4)
    rt.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, bool))
    ref = RouteTable(topo, K=4)
    ref.rebuild(np.ones(topo.n_nodes), np.zeros(topo.n_nodes, bool))
    for s, d in itertools.combinations(sorted(topo.G), 2):
        full = ref.paths(s, d)
        lazy = [rt.candidate(s, d, i) for i in range(len(full))]
        assert lazy == full
        assert rt.candidate(s, d, len(full)) is None


def test_t_route_gates_rebuild(topo):
    """With T_route set, a weight change does not rebuild until the period is up."""
    rt = RouteTable(topo, K=4, T_route=300.0)
    w = np.ones(topo.n_nodes)
    iso = np.zeros(topo.n_nodes, bool)
    rt.rebuild(w, iso, t=0.0)
    assert rt.n_rebuilds == 1

    w2 = w.copy()
    w2[0] = 5.0
    assert not rt.is_stale(w2, iso, t=100.0), "rebuilt before T_route elapsed"
    assert rt.is_stale(w2, iso, t=300.0)
    rt.rebuild(w2, iso, t=300.0)
    assert rt.n_rebuilds == 2
    assert not rt.is_stale(w2, iso, t=900.0), "unchanged weights must not rebuild"
