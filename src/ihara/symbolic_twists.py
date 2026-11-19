"""Exact symbolic U(1)-twisted non-backtracking computations."""

from __future__ import annotations

import networkx as nx
import sympy as sp

from src.ihara.oriented_edges import (
    OrientedEdge,
    OrientedEdgeIndex,
    build_oriented_edge_index,
)


def trivial_symbolic_transport(
    oriented: OrientedEdgeIndex,
) -> dict[OrientedEdge, sp.Expr]:
    """Assign exact transport 1 to every oriented edge."""
    return {edge: sp.Integer(1) for edge in oriented.edges}


def build_symbolic_u1_twisted_nonbacktracking_matrix(
    graph: nx.Graph,
    transport: dict[OrientedEdge, sp.Expr],
    oriented: OrientedEdgeIndex | None = None,
) -> sp.Matrix:
    """
    Construct an exact symbolic U(1)-twisted non-backtracking matrix.

    The entry indexed by oriented edges e and f is:

    $$
    (B_T)_{e,f}
    =
    \begin{cases}
    T_f, & t(e)=o(f)\text{ and }f\ne\bar e,\\
    0, & \text{otherwise}.
    \end{cases}
    $$
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    size = len(oriented.edges)
    matrix = sp.zeros(size, size)

    for edge in oriented.edges:
        row = oriented.index[edge]
        terminus = oriented.terminus(edge)
        reverse_edge = oriented.reverse(edge)

        for successor in oriented.edges:
            column = oriented.index[successor]

            if (
                terminus == oriented.origin(successor)
                and successor != reverse_edge
            ):
                matrix[row, column] = transport[successor]

    return matrix


def characteristic_determinant(
    matrix: sp.Matrix,
    u: sp.Symbol,
) -> sp.Expr:
    """Compute and simplify det(I - uM)."""
    identity = sp.eye(matrix.rows)
    return sp.factor((identity - u * matrix).det())
