"""Graph loading, stable edge indexing and per-link secure key rate.

Phase 2 reference: section 4.2 of phase2-build-spec.
Key rate model (phase1 section 1.3):  R_e = R_max * exp(-L_e / L_0)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np

from .config import TopologyConfig


@dataclass(frozen=True)
class Topology:
    name: str
    G: nx.Graph
    n_nodes: int
    n_edges: int
    edge_index: dict[tuple[int, int], int]    # (min,max) -> edge_id
    edge_list: list[tuple[int, int]]
    edge_len: np.ndarray                      # km, shape (|E|,)
    R: np.ndarray                             # bit/s, shape (|E|,)
    B_max: float
    node_edges: list[np.ndarray]              # node_id -> ids of incident edges
    node_deg: np.ndarray                      # shape (|V|,) int, incident edge count
    edge_ends: np.ndarray                     # shape (|E|,2) int, (lower, higher) node
    exposure: np.ndarray                      # shape (|V|,), phase 4 prior input
    qber_base: np.ndarray                     # shape (|E|,), phase 3 quantum layer
    meta: dict                                # frozen graph facts, incl. the B2 ceiling

    def edge_id(self, u: int, v: int) -> int:
        return self.edge_index[(u, v) if u < v else (v, u)]


def key_rate(edge_len_km: np.ndarray, R_max: float, L_0: float) -> np.ndarray:
    """R_e = R_max * exp(-L_e / L_0). Abstract model, calibrated in phase 7."""
    return R_max * np.exp(-np.asarray(edge_len_km, dtype=float) / L_0)


def load_topology(cfg: TopologyConfig, qber_min: float = 0.015,
                  qber_max: float = 0.030) -> Topology:
    """Load the graph and derive every frozen per-link quantity.

    ``qber_base`` (phase 3, section 3.2) is built from the per edge unit draw
    stored in the topology JSON, which came from rng_topology: the ordering is
    frozen with the topology while the range stays a sweepable config value.
    """
    with open(cfg.path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    nodes = sorted(int(n) for n in doc["nodes"])
    if nodes != list(range(len(nodes))):
        raise ValueError("topology node ids must be 0..n-1")

    # canonical (min,max) keys, lexicographic order -> deterministic edge ids
    pairs: dict[tuple[int, int], float] = {}
    for entry in doc["edges"]:
        u, v, length = int(entry[0]), int(entry[1]), float(entry[2])
        if u == v:
            raise ValueError(f"self loop at node {u}")
        key = (min(u, v), max(u, v))
        if key in pairs:
            raise ValueError(f"duplicate edge {key}")
        if length <= 0:
            raise ValueError(f"edge {key} has non positive length")
        pairs[key] = length

    edge_list = sorted(pairs)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    edge_len = np.array([pairs[e] for e in edge_list], dtype=float)
    R = key_rate(edge_len, cfg.R_max, cfg.L_0)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i, (u, v) in enumerate(edge_list):
        G.add_edge(u, v, eid=i, length=edge_len[i], R=float(R[i]), w=1.0)

    if not nx.is_connected(G):
        raise ValueError(f"topology {cfg.path} is not connected")

    node_edges = [
        np.array(sorted(edge_index[(min(u, v), max(u, v))] for v in G.neighbors(u)),
                 dtype=np.int64)
        for u in nodes
    ]

    raw_qber = doc.get("qber_unit")
    if raw_qber is None:
        qber_unit = np.full(len(edge_list), 0.5, dtype=float)
    else:
        qber_unit = np.array(raw_qber, dtype=float)
    if qber_unit.shape != (len(edge_list),):
        raise ValueError("qber_unit must have one entry per edge")
    qber_base = qber_min + qber_unit * (qber_max - qber_min)

    raw_exposure = doc.get("exposure")
    exposure = (np.array(raw_exposure, dtype=float) if raw_exposure is not None
                else np.zeros(len(nodes), dtype=float))
    if exposure.shape != (len(nodes),):
        raise ValueError("exposure must have one entry per node")

    return Topology(
        name=str(doc.get("name", cfg.name)),
        G=G,
        n_nodes=len(nodes),
        n_edges=len(edge_list),
        edge_index=edge_index,
        edge_list=edge_list,
        edge_len=edge_len,
        R=R,
        B_max=float(cfg.B_max),
        node_edges=node_edges,
        node_deg=np.array([len(a) for a in node_edges], dtype=np.int64),
        edge_ends=np.array(edge_list, dtype=np.int64).reshape(len(edge_list), 2),
        exposure=exposure,
        qber_base=qber_base,
        meta=dict(doc.get("meta", {})),
    )


def edges_of_path(topo: Topology, path: Sequence[int]) -> tuple[int, ...]:
    """Edge ids along a node path; length is len(path) - 1."""
    idx = topo.edge_index
    out = []
    for u, v in zip(path[:-1], path[1:]):
        key = (u, v) if u < v else (v, u)
        try:
            out.append(idx[key])
        except KeyError as exc:
            raise KeyError(f"path uses non existent edge {key}") from exc
    return tuple(out)
