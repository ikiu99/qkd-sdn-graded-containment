"""Path table with periodic rebuild.

Phase 2 reference: section 4.4 of phase2-build-spec.
Phase 5: node-disjoint path sets for policy B2.

Mirrors a real SDN controller: flow rules are pushed periodically, not per
packet.  Three optimisations matter, and all three are exact - they change how
fast a path is produced, never which path:

1. ``rebuild`` only runs when the node weights or the isolation mask actually
   changed *and* ``T_route`` has elapsed.  Under NullPolicy that is exactly once,
   at t=0; under the graded policy B3 the weights drift continuously, so without
   the ``T_route`` gate the cache would be dropped at every policy tick.
2. The table is lazy in the pair: the paths of a pair are computed on first
   request, never all 1225 pairs up front.
3. The table is also lazy in K.  Admission walks the candidates in cost order
   and usually stops at the first, but ``shortest_simple_paths`` is a generator
   and materialising all K costs 17x (T1) to 32x (T3) more than one Dijkstra.
   The generator is kept alive and advanced only when a caller asks for the
   next candidate.

Determinism note: ``shortest_simple_paths`` breaks equal-cost ties by graph
insertion order, so the pruned graph is always derived from ``topo.G`` in the
same order.  Changing how it is constructed can silently change which of two
equal-cost paths is returned.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .topology import Topology, edges_of_path

INF = float("inf")


class _LazyPaths:
    """Cost-ordered candidate paths for one pair, materialised on demand."""

    __slots__ = ("_found", "_gen", "_limit")

    def __init__(self, gen, limit: int):
        self._found: list[tuple[int, ...]] = []
        self._gen = gen
        self._limit = int(limit)

    def _advance(self) -> bool:
        if self._gen is None or len(self._found) >= self._limit:
            return False
        try:
            self._found.append(tuple(next(self._gen)))
            return True
        except (StopIteration, nx.NetworkXNoPath, nx.NodeNotFound):
            self._gen = None
            return False

    def get(self, i: int):
        while len(self._found) <= i and self._advance():
            pass
        return self._found[i] if i < len(self._found) else None

    def all(self) -> list[tuple[int, ...]]:
        while self._advance():
            pass
        return self._found

    def known(self) -> list[tuple[int, ...]]:
        return self._found

    def reversed_copy(self) -> "_LazyPaths":
        out = _LazyPaths(None, self._limit)
        out._found = [tuple(reversed(p)) for p in self._found]
        return out


class RouteTable:
    def __init__(self, topo: Topology, K: int, disjoint_K: int = 3,
                 T_route: float = 0.0):
        self.topo = topo
        self.K = int(K)
        self.disjoint_K = int(disjoint_K)
        self.T_route = float(T_route)
        self.version = 0
        self.n_rebuilds = 0
        self._cache: dict[tuple[int, int], _LazyPaths] = {}
        self._disjoint_cache: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
        self._G: nx.Graph | None = None
        self._G_pruned: nx.Graph | None = None
        self._node_weight = np.ones(topo.n_nodes, dtype=float)
        self._edge_cost = np.ones(topo.n_edges, dtype=float)
        self._isolated = np.zeros(topo.n_nodes, dtype=bool)
        self._built = False
        self._last_rebuild_t = -INF

    # ------------------------------------------------------------------ #
    def is_stale(self, node_weight: np.ndarray, isolated: np.ndarray,
                 t: float | None = None,
                 edge_cost: np.ndarray | None = None) -> bool:
        """Would a rebuild change anything, and is it due?

        The ``T_route`` gate is not just a speed knob: it is the model statement
        that the controller recomputes and pushes flow rules on a period rather
        than reacting continuously.
        """
        if not self._built:
            return True
        changed = not (
            np.array_equal(self._node_weight, node_weight)
            and np.array_equal(self._isolated, isolated)
            and (edge_cost is None
                 or np.array_equal(self._edge_cost, edge_cost))
        )
        if not changed:
            return False
        if t is None or self.T_route <= 0.0:
            return True
        return (t - self._last_rebuild_t) >= self.T_route

    def rebuild(self, node_weight: np.ndarray, isolated: np.ndarray,
                t: float | None = None,
                edge_cost: np.ndarray | None = None) -> None:
        """Re-weight the routing graph and drop the path cache.

        Edge cost of (u,v) is 0.5*(node_weight[u] + node_weight[v]).  The
        isolation-pruned graph is built once here rather than copied per
        uncached query.  Bumps ``version``.
        """
        self._node_weight = np.array(node_weight, dtype=float, copy=True)
        self._isolated = np.array(isolated, dtype=bool, copy=True)
        if edge_cost is not None:
            self._edge_cost = np.array(edge_cost, dtype=float, copy=True)

        G = nx.Graph()
        G.add_nodes_from(self.topo.G.nodes())
        for u, v, data in self.topo.G.edges(data=True):
            G.add_edge(u, v, eid=data["eid"],
                       w=(0.5 * (self._node_weight[u] + self._node_weight[v])
                          * self._edge_cost[data["eid"]]))
        self._G = G

        drop = np.flatnonzero(self._isolated)
        if drop.size:
            H = G.copy()
            H.remove_nodes_from(int(i) for i in drop)
            self._G_pruned = H
        else:
            self._G_pruned = G

        self._cache.clear()
        self._disjoint_cache.clear()
        self.version += 1
        self.n_rebuilds += 1
        self._built = True
        if t is not None:
            self._last_rebuild_t = float(t)

    # ------------------------------------------------------------------ #
    def _graph_for(self, src: int, dst: int) -> nx.Graph:
        """Routing graph with isolated nodes removed.

        The endpoints of the requested pair are always kept, so a demand that
        originates or terminates at an isolated node can still be evaluated -
        isolation means "no longer a relay", not "off the network".  The
        admission test separately rejects paths whose *intermediate* nodes are
        isolated.
        """
        assert self._G_pruned is not None, "call rebuild() before paths()"
        keep = [i for i in (src, dst) if self._isolated[i]]
        if not keep:
            return self._G_pruned
        H = self._G_pruned.copy()
        for i in keep:
            H.add_node(i)
            for j in self.topo.G.neighbors(i):
                if not self._isolated[j] or j in keep:
                    eid = self.topo.edge_id(i, j)
                    H.add_edge(i, j, eid=eid,
                               w=(0.5 * (self._node_weight[i]
                                         + self._node_weight[j])
                                  * self._edge_cost[eid]))
        return H

    def _lazy(self, src: int, dst: int) -> _LazyPaths:
        key = (src, dst)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        G = self._graph_for(src, dst)
        try:
            gen = nx.shortest_simple_paths(G, src, dst, weight="w")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            gen = iter(())
        lazy = _LazyPaths(gen, self.K)
        self._cache[key] = lazy
        return lazy

    def candidate(self, src: int, dst: int, i: int):
        """The i-th cheapest path, or None. Only computes as far as asked."""
        if src == dst:
            return None
        return self._lazy(src, dst).get(i)

    def paths(self, src: int, dst: int) -> list[tuple[int, ...]]:
        """All (up to K) node paths, ordered by cost. Empty if unreachable.

        Forces full materialisation - prefer :meth:`candidate` on the hot path.
        """
        if src == dst:
            return []
        lazy = self._lazy(src, dst)
        found = lazy.all()
        # undirected and the edge cost is symmetric, so the reverse pair is free
        if (dst, src) not in self._cache:
            self._cache[(dst, src)] = lazy.reversed_copy()
        return found

    def reachable(self, src: int, dst: int) -> bool:
        """Is there any path at all? Costs one Dijkstra, not a full Yen."""
        return self.candidate(src, dst, 0) is not None

    # ------------------------------------------------------------------ #
    def disjoint_paths(self, src: int, dst: int,
                       m: int | None = None) -> list[tuple[int, ...]]:
        """Minimum-cost set of m internally **node-disjoint** paths, or [].

        Node-disjoint, not edge-disjoint, and the difference is the whole
        security claim of policy B2: it relays XOR key shares, one per leg, so a
        relay that sits on two legs sees two shares and reconstructs the key.
        Two edge-disjoint paths may share an intermediate node, which would make
        B2 look protective while offering nothing.

        Solved exactly as a min-cost flow on the node-split graph (each node
        becomes in->out with capacity 1, so no node can carry two legs), rather
        than greedily filtering the K Yen paths: Yen's paths are minor
        perturbations of each other and share most of their nodes, so a greedy
        filter would report the disjointness of *K* rather than of the network.

        Returns [] when fewer than m disjoint paths exist - B2 is strict, and
        the caller turns that into a distinct ``no_disjoint`` rejection so a
        structural shortfall is never confused with a key shortage.
        """
        m = self.disjoint_K if m is None else int(m)
        if src == dst or m < 1:
            return []
        key = (src, dst, m)
        cached = self._disjoint_cache.get(key)
        if cached is not None:
            return cached

        out = self._min_cost_disjoint(src, dst, m)
        self._disjoint_cache[key] = out
        self._disjoint_cache[(dst, src, m)] = [tuple(reversed(p)) for p in out]
        return out

    def _min_cost_disjoint(self, src: int, dst: int, m: int) -> list[tuple[int, ...]]:
        G = self._graph_for(src, dst)
        if src not in G or dst not in G:
            return []

        D = nx.DiGraph()
        for node in G.nodes():
            cap = m if node in (src, dst) else 1
            D.add_edge(("i", node), ("o", node), capacity=cap, weight=0)
        for u, v, data in G.edges(data=True):
            # integer weights: network_simplex needs them, and the scale keeps
            # the cost ordering intact
            w = int(round(data["w"] * 1000.0))
            D.add_edge(("o", u), ("i", v), capacity=1, weight=w)
            D.add_edge(("o", v), ("i", u), capacity=1, weight=w)

        D.nodes[("i", src)]["demand"] = -m
        D.nodes[("o", dst)]["demand"] = m
        try:
            _, flow = nx.network_simplex(D)
        except (nx.NetworkXUnfeasible, nx.NetworkXUnbounded):
            return []

        # decompose the flow into m src->dst walks
        nxt: dict[int, list[int]] = {}
        for (ta, a), targets in flow.items():
            if ta != "o":
                continue
            for (tb, b), f in targets.items():
                if tb == "i" and f > 0:
                    nxt.setdefault(a, []).append(b)
        for succ in nxt.values():
            succ.sort()                      # deterministic decomposition

        out: list[tuple[int, ...]] = []
        for _ in range(m):
            path, node = [src], src
            while node != dst:
                succ = nxt.get(node)
                if not succ:
                    return []
                node = succ.pop(0)
                if node in path:             # should not happen, capacity 1 forbids it
                    return []
                path.append(node)
            out.append(tuple(path))
        out.sort(key=self.path_cost)
        return out

    # ------------------------------------------------------------------ #
    def path_cost(self, path: tuple[int, ...]) -> float:
        return sum(0.5 * (self._node_weight[u] + self._node_weight[v])
                   for u, v in zip(path[:-1], path[1:]))

    def edges_of(self, path) -> tuple[int, ...]:
        return edges_of_path(self.topo, path)
