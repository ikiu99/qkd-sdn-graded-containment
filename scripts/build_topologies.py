"""Generate the three topology JSON files under data/topologies/.

Run:  python scripts/build_topologies.py

Design note (stated in the paper's method section)
--------------------------------------------------
The secure key rate model is

    R_e = R_max * exp(-L_e / L_0)

and L_0 is NOT a free parameter.  Secret key rate scales with channel
transmittance eta = 10^(-alpha L / 10), so

    L_0 = 10 / (alpha * ln 10) = 21.71 km   at alpha = 0.2 dB/km,

the standard 1550 nm single-mode fibre attenuation assumed throughout the QKD
networking literature.  (The phase 1 spec's 50 km would imply alpha = 0.087
dB/km, below the Rayleigh scattering limit of silica: no such fibre exists.)

With the *geographic* span of NSFNET / USNET (600-2800 km) the model yields no
usable key on any link, which is the physical reason trusted-relay QKD networks
exist at all: individual QKD links are short.  The reference topologies are used
here for their *connectivity* - literature comparability - while link lengths
are linearly rescaled into [8.7, 52.1] km, preserving the
relative ordering.  That range is where deployed metropolitan QKD networks
actually sit (Tokyo QKD Network 1-45 km; Hefei 46-node metro; MadQCI Madrid),
and it gives R_e in [1.8, 13.4] kbit/s - a dynamic range in which buffers,
admission and starvation are all meaningful.

The adjacency lists below follow commonly used variants of NSFNET-14 and a
24-node US mesh.  They live in plain JSON files, so swapping in the exact edge
list of whichever citation the paper uses is a one-file edit and requires no
code change.
"""
from __future__ import annotations

import itertools
import json
import math
import os

import networkx as nx
import numpy as np

L_MIN_KM = 8.685890
L_MAX_KM = 52.115338
SEED_TOPOLOGY = 1
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "topologies")

# --------------------------------------------------------------------------- #
# T1 - NSFNET, 14 nodes / 21 links
# --------------------------------------------------------------------------- #
NSFNET_NODES = [
    ("Seattle", 47.61, -122.33),
    ("Palo Alto", 37.44, -122.14),
    ("San Diego", 32.72, -117.16),
    ("Salt Lake City", 40.76, -111.89),
    ("Boulder", 40.01, -105.27),
    ("Houston", 29.76, -95.37),
    ("Lincoln", 40.81, -96.70),
    ("Champaign", 40.11, -88.24),
    ("Pittsburgh", 40.44, -79.99),
    ("Atlanta", 33.75, -84.39),
    ("Ann Arbor", 42.28, -83.74),
    ("Ithaca", 42.44, -76.50),
    ("Princeton", 40.35, -74.66),
    ("College Park", 38.99, -76.94),
]
NSFNET_EDGES = [
    (0, 1), (0, 2), (0, 7), (1, 2), (1, 3), (2, 5), (3, 4), (3, 10), (4, 5),
    (4, 6), (5, 9), (5, 12), (6, 7), (7, 8), (8, 9), (8, 11), (8, 12), (9, 13),
    (10, 11), (10, 13), (11, 12),
]

# --------------------------------------------------------------------------- #
# T2 - USNET, 24 nodes / 43 links
# --------------------------------------------------------------------------- #
USNET_NODES = [
    ("Seattle", 47.61, -122.33),
    ("Portland", 45.52, -122.68),
    ("San Francisco", 37.77, -122.42),
    ("Los Angeles", 34.05, -118.24),
    ("San Diego", 32.72, -117.16),
    ("Las Vegas", 36.17, -115.14),
    ("Salt Lake City", 40.76, -111.89),
    ("Boise", 43.62, -116.20),
    ("Denver", 39.74, -104.99),
    ("Albuquerque", 35.08, -106.65),
    ("Phoenix", 33.45, -112.07),
    ("Dallas", 32.78, -96.80),
    ("Houston", 29.76, -95.37),
    ("Kansas City", 39.10, -94.58),
    ("Minneapolis", 44.98, -93.27),
    ("Chicago", 41.88, -87.63),
    ("St Louis", 38.63, -90.20),
    ("Atlanta", 33.75, -84.39),
    ("Nashville", 36.16, -86.78),
    ("Detroit", 42.33, -83.05),
    ("Cleveland", 41.50, -81.69),
    ("New York", 40.71, -74.01),
    ("Washington DC", 38.91, -77.04),
    ("Boston", 42.36, -71.06),
]
USNET_EDGES = [
    (0, 1), (0, 7), (1, 2), (1, 7), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5),
    (4, 10), (5, 6), (5, 10), (6, 7), (6, 8), (7, 8), (8, 9), (8, 13), (8, 14),
    (9, 10), (9, 11), (10, 11), (11, 12), (11, 13), (12, 13), (12, 17),
    (13, 14), (13, 16), (14, 15), (14, 19), (15, 16), (15, 19), (15, 20),
    (16, 18), (16, 21), (17, 18), (17, 22), (18, 19), (19, 20), (20, 21),
    (20, 22), (21, 22), (21, 23), (22, 23),
]


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def rescale(raw: list[float]) -> list[float]:
    """Linear map of raw distances into [L_MIN_KM, L_MAX_KM], order preserving."""
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-12:
        return [0.5 * (L_MIN_KM + L_MAX_KM) for _ in raw]
    span = L_MAX_KM - L_MIN_KM
    return [round(L_MIN_KM + span * (d - lo) / (hi - lo), 3) for d in raw]


def _qber_unit(n_edges: int) -> list[float]:
    """Per edge uniform draw in [0,1), frozen into the topology file.

    Phase 3 section 3.2 wants qber_base_e ~ U(0.015, 0.030) from rng_topology.
    Storing the *unit* draw rather than the rate keeps the range a sweepable
    config parameter (noise.qber_base_min / max) while the per edge ordering
    stays frozen with the topology.  Its own substream, so adding it does not
    move the exposure values drawn above.
    """
    rng = np.random.default_rng([SEED_TOPOLOGY, 99])
    return [round(float(x), 6) for x in rng.random(n_edges)]


def _exposure(n: int, rng: np.random.Generator) -> list[float]:
    """Physical protection level per node, 0 = hardened datacentre, 1 = remote site.

    phase1-model-spec section 13 leaves this open for phase 2; it is drawn once
    per topology from rng_topology and frozen into the JSON so every run of every
    phase sees exactly the same node exposure.  Unused in phase 2.
    """
    return [round(float(x), 4) for x in rng.random(n)]


def connectivity_stats(edge_pairs, n_nodes: int) -> dict:
    """All-pairs node connectivity, frozen into the topology file.

    ``frac_ge_m`` is the ceiling on policy B2: it relays XOR key shares over m
    node-disjoint paths and rejects a demand outright when fewer than m exist,
    so this fraction bounds B2's acceptance rate before key supply is even
    considered.  Having it in the data file means a reader can tell a structural
    rejection from a key shortage without re-deriving the graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((u, v) for u, v in edge_pairs)
    pairs = list(itertools.combinations(range(n_nodes), 2))
    k = [len(list(nx.node_disjoint_paths(G, s, d))) for s, d in pairs]
    return {
        "node_connectivity_min": int(min(k)),
        "articulation_points": len(set(nx.articulation_points(G))),
        "frac_ge_2_disjoint": round(sum(x >= 2 for x in k) / len(pairs), 4),
        "frac_ge_3_disjoint": round(sum(x >= 3 for x in k) / len(pairs), 4),
        "mean_degree": round(2 * G.number_of_edges() / n_nodes, 3),
    }


def build_geo(name: str, nodes, edge_pairs, rng) -> dict:
    coords = [(lat, lon) for _, lat, lon in nodes]
    raw = [haversine_km(coords[u], coords[v]) for u, v in edge_pairs]
    lens = rescale(raw)
    return {
        "name": name,
        "nodes": list(range(len(nodes))),
        "edges": [[u, v, L] for (u, v), L in zip(edge_pairs, lens)],
        "node_names": [n[0] for n in nodes],
        "coords": [[lat, lon] for _, lat, lon in nodes],
        "exposure": _exposure(len(nodes), rng),
        "qber_unit": _qber_unit(len(edge_pairs)),
        "meta": {
            "geographic_km": [round(d, 1) for d in raw],
            "length_model": f"linear rescale of geographic km into [{L_MIN_KM}, {L_MAX_KM}] km",
            **connectivity_stats(edge_pairs, len(nodes)),
        },
    }


def build_net50(n: int, target_edges: int, rng) -> dict:
    """Deterministic 50-node geometric network: Euclidean MST plus the shortest
    remaining candidate edges until target_edges is reached (guarantees a
    connected graph and a realistic, roughly planar mesh)."""
    pos = rng.random((n, 2))
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))

    # Prim MST
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = d[0].copy()
    parent = np.zeros(n, dtype=int)
    mst: list[tuple[int, int]] = []
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        j = int(np.argmin(cand))
        mst.append((min(j, int(parent[j])), max(j, int(parent[j]))))
        in_tree[j] = True
        closer = d[j] < best
        parent[closer] = j
        best = np.minimum(best, d[j])

    chosen = set(mst)
    candidates = sorted(
        ((float(d[u, v]), u, v) for u in range(n) for v in range(u + 1, n)
         if (u, v) not in chosen)
    )
    for _, u, v in candidates:
        if len(chosen) >= target_edges:
            break
        chosen.add((u, v))

    chosen, n_added, n_trimmed = biconnect(chosen, d, n, target_edges)

    edge_pairs = sorted(chosen)
    raw = [float(d[u, v]) for u, v in edge_pairs]
    lens = rescale(raw)
    return {
        "name": "net50",
        "nodes": list(range(n)),
        "edges": [[u, v, L] for (u, v), L in zip(edge_pairs, lens)],
        "coords": [[round(float(x), 5), round(float(y), 5)] for x, y in pos],
        "exposure": _exposure(n, rng),
        "qber_unit": _qber_unit(len(edge_pairs)),
        "meta": {
            "generator": f"euclidean MST + shortest fill to {target_edges} edges, "
                         f"then biconnectivity augmentation "
                         f"(+{n_added} edges, -{n_trimmed} redundant), "
                         f"seed_topology={SEED_TOPOLOGY}",
            "length_model": f"linear rescale of unit-square distance into "
                            f"[{L_MIN_KM}, {L_MAX_KM}] km",
            **connectivity_stats(edge_pairs, n),
        },
    }


def biconnect(chosen: set, d, n: int, target_edges: int):
    """Make the graph 2-node-connected, then trim back to ~target_edges.

    Why this is not optional.  MST + shortest-fill leaves articulation points:
    the 50 node graph came out with 4 degree-1 nodes and 15 cut vertices, and
    only 17.4% of node pairs had two node-disjoint paths (both reference
    topologies have 100%).  Policy B2 relays XOR key shares over node-disjoint
    paths, so on such a graph it would reject 83% of demands for a reason that
    is a property of this generator rather than of the policy - and the phase 5
    partition guard would veto roughly a third of B1's isolations for the same
    reason.  Real QKD backbones are laid out 2-connected precisely so that no
    single trusted relay is a single point of failure.

    Augmentation walks the block-cut tree: each added edge joins two *leaf*
    blocks, which strictly reduces their number, so the loop terminates.  Among
    all such pairs the shortest is taken, which keeps the mesh geometric.
    Redundant long edges are then dropped so the edge count stays comparable to
    the pre-augmentation graph, keeping the average degree - and therefore the
    key supply per node - unchanged.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(sorted(chosen))

    n_added = 0
    while not nx.is_biconnected(G):
        cuts = set(nx.articulation_points(G))
        leaves = [b for b in nx.biconnected_components(G) if len(b & cuts) <= 1]
        best = None
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                for u in sorted(leaves[i] - cuts) or sorted(leaves[i]):
                    for v in sorted(leaves[j] - cuts) or sorted(leaves[j]):
                        if u == v or G.has_edge(u, v):
                            continue
                        w = float(d[u, v])
                        if best is None or w < best[0]:
                            best = (w, min(u, v), max(u, v))
        if best is None:
            break
        G.add_edge(best[1], best[2])
        n_added += 1

    # trim the longest edges whose removal keeps the graph 2-connected
    n_trimmed = 0
    for _, u, v in sorted(((float(d[u, v]), u, v) for u, v in G.edges()), reverse=True):
        if G.number_of_edges() <= target_edges:
            break
        G.remove_edge(u, v)
        if nx.is_biconnected(G):
            n_trimmed += 1
        else:
            G.add_edge(u, v)

    return {(min(u, v), max(u, v)) for u, v in G.edges()}, n_added, n_trimmed


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED_TOPOLOGY)
    specs = [
        ("nsfnet.json", build_geo("nsfnet", NSFNET_NODES, NSFNET_EDGES, rng)),
        ("usnet.json", build_geo("usnet", USNET_NODES, USNET_EDGES, rng)),
        ("net50.json", build_net50(50, 90, rng)),
    ]
    for fname, doc in specs:
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=False)
            fh.write("\n")
        lens = [e[2] for e in doc["edges"]]
        rates = [20000.0 * math.exp(-L / 21.714724) for L in lens]
        m = doc["meta"]
        print(f"{fname:12s} nodes={len(doc['nodes']):3d} edges={len(doc['edges']):3d} "
              f"R=[{min(rates):.0f},{max(rates):.0f}] bit/s  "
              f"kappa_min={m['node_connectivity_min']}  "
              f"cuts={m['articulation_points']}  "
              f">=2 disjoint {m['frac_ge_2_disjoint']:.1%}  "
              f">=3 {m['frac_ge_3_disjoint']:.1%}")


if __name__ == "__main__":
    main()
