"""Structural guarantees the topology files must provide.

Phase 5 policy B2 relays XOR key shares over m node-disjoint paths and rejects a
demand outright when fewer than m exist, and the partition guard refuses to
isolate a node whose removal would disconnect the graph. Both are therefore
bounded by the *graph*, not by the policy: on a topology with articulation
points those bounds would silently become the result. Real QKD backbones are
laid out 2-connected precisely so no single trusted relay is a single point of
failure, so that is asserted here rather than assumed.
"""
from __future__ import annotations

import itertools

import networkx as nx
import pytest

from sim.config import TopologyConfig
from sim.topology import load_topology

TOPOLOGIES = ("nsfnet", "usnet", "net50")


def load(name):
    return load_topology(TopologyConfig(name=name, path=f"data/topologies/{name}.json"))


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_two_connected(name):
    topo = load(name)
    G = topo.G
    assert nx.is_connected(G)
    assert not list(nx.articulation_points(G)), "a cut vertex would cap B2 and the guard"
    assert min(dict(G.degree()).values()) >= 2


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_every_pair_has_two_node_disjoint_paths(name):
    topo = load(name)
    G = topo.G
    for s, d in itertools.combinations(sorted(G), 2):
        assert len(list(nx.node_disjoint_paths(G, s, d))) >= 2, f"{name}: {s}->{d}"


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_connectivity_stats_are_frozen_and_honest(name):
    """The B2 ceiling recorded in the file must match the graph it describes."""
    topo = load(name)
    meta = topo.meta
    assert meta["node_connectivity_min"] >= 2
    assert meta["articulation_points"] == 0
    assert meta["frac_ge_2_disjoint"] == 1.0
    # m=3 is NOT universally available - that is the point of sweeping m
    assert 0.0 < meta["frac_ge_3_disjoint"] <= 1.0

    G = topo.G
    pairs = list(itertools.combinations(sorted(G), 2))
    k = [len(list(nx.node_disjoint_paths(G, s, d))) for s, d in pairs]
    assert meta["frac_ge_3_disjoint"] == pytest.approx(
        sum(x >= 3 for x in k) / len(pairs), abs=5e-5)
