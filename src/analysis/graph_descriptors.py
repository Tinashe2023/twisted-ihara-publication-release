"""Baseline graph descriptors for Tanner-graph experiments."""

from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from collections import deque
from typing import Hashable

import networkx as nx
import numpy as np


Vertex = Hashable


def deterministic_vertices(graph: nx.Graph) -> list[Vertex]:
    """Return graph vertices in a stable deterministic order."""
    return sorted(
        graph.nodes(),
        key=lambda vertex: (type(vertex).__name__, repr(vertex)),
    )


def cycle_rank(graph: nx.Graph) -> int:
    r"""
    Return the graph cycle rank:

    $$
    \beta_1(X)=|E|-|V|+c,
    $$

    where $c$ is the number of connected components.
    """
    return (
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def sorted_degree_sequence(graph: nx.Graph) -> tuple[int, ...]:
    """Return the nondecreasing vertex-degree sequence."""
    return tuple(sorted(degree for _, degree in graph.degree()))


def adjacency_spectrum(graph: nx.Graph) -> np.ndarray:
    """
    Return the numerically computed adjacency eigenvalues in sorted order.

    Because an undirected graph has a real symmetric adjacency matrix,
    the eigenvalues are real up to floating-point roundoff.
    """
    vertices = deterministic_vertices(graph)

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=vertices,
        dtype=float,
    )

    eigenvalues = np.linalg.eigvalsh(adjacency)

    return np.sort(eigenvalues)


def graph_girth(graph: nx.Graph) -> int | None:
    r"""
    Return the girth of a simple undirected graph.

    The girth is the length of its shortest cycle. Return None if the graph
    is acyclic.

    The method runs breadth-first search from every vertex.
    """
    if graph.is_directed():
        raise TypeError("Girth is implemented for undirected graphs only.")

    if graph.is_multigraph():
        raise TypeError("Girth is implemented for simple graphs only.")

    shortest_cycle = float("inf")

    for root in graph.nodes():
        distances: dict[Vertex, int] = {root: 0}
        parents: dict[Vertex, Vertex | None] = {root: None}
        queue: deque[Vertex] = deque([root])

        while queue:
            current = queue.popleft()

            for neighbor in graph.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    parents[neighbor] = current
                    queue.append(neighbor)

                elif parents[current] != neighbor:
                    candidate_length = (
                        distances[current]
                        + distances[neighbor]
                        + 1
                    )

                    shortest_cycle = min(
                        shortest_cycle,
                        candidate_length,
                    )

    if shortest_cycle == float("inf"):
        return None

    return int(shortest_cycle)


def graph_descriptor_record(
    graph: nx.Graph,
) -> dict[str, object]:
    """
    Return a JSON-friendly collection of baseline graph descriptors.
    """
    spectrum = adjacency_spectrum(graph)
    girth = graph_girth(graph)

    return {
        "vertex_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "component_count": nx.number_connected_components(graph),
        "connected": nx.is_connected(graph),
        "bipartite": nx.is_bipartite(graph),
        "cycle_rank": cycle_rank(graph),
        "girth": girth,
        "degree_sequence": list(sorted_degree_sequence(graph)),
        "adjacency_spectrum": [
            float(np.round(value, 12))
            for value in spectrum
        ],
    }
