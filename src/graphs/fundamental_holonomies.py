"""Spanning-tree gauge fixing and fundamental SU(2) graph holonomies."""

from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from collections import deque
from typing import Hashable

import networkx as nx
import numpy as np

from src.ihara.oriented_edges import build_oriented_edge_index


Vertex = Hashable
CanonicalEdge = tuple[Vertex, Vertex]
Matrix2 = np.ndarray


def vertex_sort_key(vertex: Vertex) -> tuple[str, str]:
    """Return a deterministic sort key for arbitrary hashable vertices."""
    return (type(vertex).__name__, repr(vertex))


def canonical_undirected_edge(
    first: Vertex,
    second: Vertex,
) -> CanonicalEdge:
    """
    Return a deterministic orientation for an undirected edge.

    The result is used only as a dictionary key and is not a graph orientation.
    """
    if vertex_sort_key(first) <= vertex_sort_key(second):
        return (first, second)

    return (second, first)


def deterministic_spanning_tree(graph: nx.Graph) -> nx.Graph:
    r"""
    Construct a deterministic breadth-first spanning tree.

    The graph must be connected, loopless, undirected, and simple.
    """
    if graph.is_directed():
        raise TypeError("Spanning-tree construction requires an undirected graph.")

    if graph.is_multigraph():
        raise TypeError("Spanning-tree construction requires a simple graph.")

    if not nx.is_connected(graph):
        raise ValueError("Spanning-tree construction requires a connected graph.")

    root = min(graph.nodes(), key=vertex_sort_key)

    tree = nx.Graph()
    tree.add_nodes_from(graph.nodes(data=True))

    visited: set[Vertex] = {root}
    queue: deque[Vertex] = deque([root])

    while queue:
        current = queue.popleft()

        neighbors = sorted(
            graph.neighbors(current),
            key=vertex_sort_key,
        )

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                tree.add_edge(current, neighbor)

    return tree


def fundamental_chords(
    graph: nx.Graph,
    tree: nx.Graph | None = None,
) -> list[CanonicalEdge]:
    r"""
    Return the non-tree edges of a connected graph in deterministic order.

    Each chord determines one fundamental cycle relative to the spanning tree.
    """
    if tree is None:
        tree = deterministic_spanning_tree(graph)

    graph_edges = {
        canonical_undirected_edge(first, second)
        for first, second in graph.edges()
    }

    tree_edges = {
        canonical_undirected_edge(first, second)
        for first, second in tree.edges()
    }

    chords = graph_edges - tree_edges

    return sorted(
        chords,
        key=lambda edge: (
            vertex_sort_key(edge[0]),
            vertex_sort_key(edge[1]),
        ),
    )


def fundamental_cycle_rank(graph: nx.Graph) -> int:
    r"""
    Return the number of independent fundamental cycles of a connected graph.

    $$
    |E|-|V|+1.
    $$
    """
    if not nx.is_connected(graph):
        raise ValueError(
            "Fundamental cycle rank is currently defined for connected graphs."
        )

    return (
        graph.number_of_edges()
        - graph.number_of_nodes()
        + 1
    )


def build_gauge_fixed_su2_transport(
    graph: nx.Graph,
    chord_holonomies: dict[CanonicalEdge, Matrix2],
    tree: nx.Graph | None = None,
) -> tuple[object, dict[tuple[Vertex, Vertex], Matrix2]]:
    r"""
    Build a gauge-fixed SU(2) transport from assigned chord holonomies.

    Tree edges receive $I_2$. For each canonical chord $(x,y)$ with assigned
    holonomy $H$, set:

    $$
    T_{(x,y)}=H,
    \qquad
    T_{(y,x)}=H^\dagger.
    $$

    The keys of ``chord_holonomies`` must be exactly the graph's chord set.
    """
    if tree is None:
        tree = deterministic_spanning_tree(graph)

    chords = fundamental_chords(graph, tree)
    expected_chords = set(chords)
    supplied_chords = set(chord_holonomies)

    if supplied_chords != expected_chords:
        missing = expected_chords - supplied_chords
        extra = supplied_chords - expected_chords

        raise ValueError(
            "Chord-holonomy keys must equal the fundamental chord set. "
            f"Missing: {missing}; extra: {extra}."
        )

    oriented = build_oriented_edge_index(graph)

    transport: dict[tuple[Vertex, Vertex], Matrix2] = {
        edge: np.eye(2, dtype=complex)
        for edge in oriented.edges
    }

    for chord in chords:
        first, second = chord
        holonomy = np.asarray(chord_holonomies[chord], dtype=complex)

        if holonomy.shape != (2, 2):
            raise ValueError(
                f"Chord holonomy for {chord} must have shape (2, 2)."
            )

        transport[(first, second)] = holonomy
        transport[(second, first)] = holonomy.conjugate().T

    return oriented, transport
