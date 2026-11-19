"""Utilities for deterministic oriented-edge indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import networkx as nx


Vertex = Hashable
OrientedEdge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class OrientedEdgeIndex:
    """Deterministic indexing data for the oriented edges of a graph."""

    edges: tuple[OrientedEdge, ...]
    index: dict[OrientedEdge, int]

    def reverse(self, edge: OrientedEdge) -> OrientedEdge:
        """Return the reversed oriented edge."""
        u, v = edge
        return (v, u)

    def origin(self, edge: OrientedEdge) -> Vertex:
        """Return the origin vertex of an oriented edge."""
        return edge[0]

    def terminus(self, edge: OrientedEdge) -> Vertex:
        """Return the terminus vertex of an oriented edge."""
        return edge[1]


def _vertex_sort_key(vertex: Vertex) -> tuple[str, str]:
    """
    Provide a deterministic ordering even when vertex labels have
    different Python types.
    """
    return (type(vertex).__name__, repr(vertex))


def _canonical_undirected_edge(
    u: Vertex,
    v: Vertex,
) -> tuple[Vertex, Vertex]:
    """Return a consistently ordered version of an undirected edge."""
    if _vertex_sort_key(u) <= _vertex_sort_key(v):
        return (u, v)
    return (v, u)


def build_oriented_edge_index(graph: nx.Graph) -> OrientedEdgeIndex:
    """
    Construct a deterministic index of all oriented edges of a finite,
    undirected, loopless simple graph.

    Each undirected edge {u, v} produces (u, v) and (v, u).
    """
    if graph.is_directed():
        raise TypeError("Expected an undirected NetworkX graph.")

    if nx.number_of_selfloops(graph) != 0:
        raise ValueError("Self-loops are not supported in Version 1.")

    if graph.is_multigraph():
        raise TypeError("MultiGraphs are not supported in Version 1.")

    undirected_edges = sorted(
        (
            _canonical_undirected_edge(u, v)
            for u, v in graph.edges()
        ),
        key=lambda edge: (
            _vertex_sort_key(edge[0]),
            _vertex_sort_key(edge[1]),
        ),
    )

    oriented_edges: list[OrientedEdge] = []

    for u, v in undirected_edges:
        oriented_edges.append((u, v))
        oriented_edges.append((v, u))

    edges = tuple(oriented_edges)
    index = {edge: i for i, edge in enumerate(edges)}

    return OrientedEdgeIndex(edges=edges, index=index)
