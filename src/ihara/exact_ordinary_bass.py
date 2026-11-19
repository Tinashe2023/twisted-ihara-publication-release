"""Exact ordinary Bass determinant polynomials for finite simple graphs."""

from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import sympy as sp

from src.analysis.graph_descriptors import deterministic_vertices


def exact_ordinary_bass_polynomial(
    graph: nx.Graph,
    u: sp.Symbol | None = None,
) -> sp.Expr:
    r"""
    Compute the ordinary non-backtracking determinant polynomial exactly:

    $$
    P_G(u)=\det(I-uB).
    $$

    Bass's determinant formula gives:

    $$
    P_G(u)
    =
    (1-u^2)^{m-n}
    \det\left(I-uA+u^2(D-I)\right),
    $$

    where $n=|V|$, $m=|E|$, $A$ is the adjacency matrix, and $D$ is the
    diagonal degree matrix.
    """
    if graph.is_directed():
        raise TypeError("Bass polynomial requires an undirected graph.")

    if graph.is_multigraph():
        raise TypeError("Bass polynomial currently requires a simple graph.")

    if u is None:
        u = sp.symbols("u")

    vertices = deterministic_vertices(graph)
    vertex_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    adjacency = sp.Matrix(
        [
            [
                int(graph.has_edge(first, second))
                for second in vertices
            ]
            for first in vertices
        ]
    )

    degree_matrix = sp.diag(
        *[
            graph.degree(vertex)
            for vertex in vertices
        ]
    )

    identity = sp.eye(vertex_count)

    vertex_determinant = (
        identity
        - u * adjacency
        + u**2 * (degree_matrix - identity)
    ).det(method="domain-ge")

    polynomial = (
        (1 - u**2) ** (edge_count - vertex_count)
        * vertex_determinant
    )

    return sp.expand(sp.cancel(polynomial))
