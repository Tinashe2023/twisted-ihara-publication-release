"""Exact symbolic U(1)-twisted Bass determinant computations."""

from __future__ import annotations

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import sympy as sp

from src.ihara.bass_formula import degree_matrix_sympy, sorted_vertices
from src.ihara.oriented_edges import (
    OrientedEdge,
    OrientedEdgeIndex,
    build_oriented_edge_index,
)
from src.ihara.symbolic_twists import (
    build_symbolic_u1_twisted_nonbacktracking_matrix,
    characteristic_determinant,
)


def symbolic_twisted_adjacency_matrix(
    graph: nx.Graph,
    transport: dict[OrientedEdge, sp.Expr],
) -> sp.Matrix:
    """
    Construct the exact U(1)-twisted adjacency matrix A_T.

    Its entries are

    $$
    (A_T)_{x,y}
    =
    \begin{cases}
    T_{(x,y)}, & \{x,y\}\in E,\\
    0, & \text{otherwise}.
    \end{cases}
    $$
    """
    vertices = sorted_vertices(graph)
    vertex_index = {
        vertex: index
        for index, vertex in enumerate(vertices)
    }

    matrix = sp.zeros(len(vertices), len(vertices))

    for origin, terminus in graph.edges():
        origin_index = vertex_index[origin]
        terminus_index = vertex_index[terminus]

        matrix[origin_index, terminus_index] = transport[(origin, terminus)]
        matrix[terminus_index, origin_index] = transport[(terminus, origin)]

    return matrix


def lhs_twisted_bass_polynomial(
    graph: nx.Graph,
    transport: dict[OrientedEdge, sp.Expr],
    u: sp.Symbol,
    oriented: OrientedEdgeIndex | None = None,
) -> sp.Expr:
    """Compute det(I - u B_T) exactly."""
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    twisted_nonbacktracking = (
        build_symbolic_u1_twisted_nonbacktracking_matrix(
            graph,
            transport,
            oriented,
        )
    )

    return characteristic_determinant(twisted_nonbacktracking, u)


def rhs_twisted_bass_polynomial(
    graph: nx.Graph,
    transport: dict[OrientedEdge, sp.Expr],
    u: sp.Symbol,
) -> sp.Expr:
    """
    Compute the vertex-side expression in the twisted Bass formula.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    adjacency = symbolic_twisted_adjacency_matrix(graph, transport)
    degree = degree_matrix_sympy(graph)
    identity = sp.eye(n)

    vertex_term = (
        identity
        - u * adjacency
        + u**2 * (degree - identity)
    )

    return sp.cancel(
        (1 - u**2) ** (m - n)
        * vertex_term.det()
    )


def twisted_bass_formula_holds(
    graph: nx.Graph,
    transport: dict[OrientedEdge, sp.Expr],
) -> bool:
    """Return True precisely when the exact twisted Bass identity holds."""
    u = sp.symbols("u")

    lhs = lhs_twisted_bass_polynomial(graph, transport, u)
    rhs = rhs_twisted_bass_polynomial(graph, transport, u)

    difference = sp.cancel(lhs - rhs)

    return sp.simplify(difference) == 0
