"""Exact symbolic SU(2)-twisted non-backtracking matrices."""

from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import sympy as sp

from src.ihara.oriented_edges import (
    OrientedEdge,
    OrientedEdgeIndex,
    build_oriented_edge_index,
)
from src.representations.symbolic_su2 import (
    SymbolicMatrix2,
    symbolic_adjoint,
    symbolic_identity_su2,
)


def symbolic_identity_su2_transport(
    oriented: OrientedEdgeIndex,
) -> dict[OrientedEdge, SymbolicMatrix2]:
    """Assign the exact 2 by 2 identity matrix to every oriented edge."""
    return {
        edge: symbolic_identity_su2()
        for edge in oriented.edges
    }


def build_symbolic_su2_twisted_nonbacktracking_matrix(
    graph: nx.Graph,
    transport: dict[OrientedEdge, SymbolicMatrix2],
    oriented: OrientedEdgeIndex | None = None,
) -> sp.Matrix:
    r"""
    Construct the exact symbolic block SU(2)-twisted non-backtracking matrix.

    For an allowed transition $e\to f$, the corresponding 2 by 2 block is:

    $$
    (B_T)_{e,f}=T_f.
    $$
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    edge_count = len(oriented.edges)
    matrix = sp.zeros(2 * edge_count, 2 * edge_count)

    for edge in oriented.edges:
        row_edge_index = oriented.index[edge]
        terminus = oriented.terminus(edge)
        reverse_edge = oriented.reverse(edge)

        for successor in oriented.edges:
            if (
                terminus == oriented.origin(successor)
                and successor != reverse_edge
            ):
                column_edge_index = oriented.index[successor]

                row_slice = slice(
                    2 * row_edge_index,
                    2 * row_edge_index + 2,
                )

                column_slice = slice(
                    2 * column_edge_index,
                    2 * column_edge_index + 2,
                )

                matrix[row_slice, column_slice] = transport[successor]

    return matrix


def symbolic_su2_theta_transport(
    graph: nx.Graph,
    direct_holonomy: SymbolicMatrix2,
    lower_holonomy: SymbolicMatrix2,
) -> tuple[
    OrientedEdgeIndex,
    dict[OrientedEdge, SymbolicMatrix2],
]:
    r"""
    Create a gauge-fixed exact SU(2) transport for the standard theta graph.

    Nontrivial transports are placed on:

    $$
    (0,1)\mapsto A,
    \qquad
    (0,3)\mapsto B.
    $$

    Reverse edges receive the exact adjoint matrices.
    """
    oriented = build_oriented_edge_index(graph)
    transport = symbolic_identity_su2_transport(oriented)

    transport[(0, 1)] = direct_holonomy
    transport[(1, 0)] = symbolic_adjoint(direct_holonomy)

    transport[(0, 3)] = lower_holonomy
    transport[(3, 0)] = symbolic_adjoint(lower_holonomy)

    return oriented, transport


def exact_twisted_determinant_polynomial(
    matrix: sp.Matrix,
    u: sp.Symbol,
) -> sp.Expr:
    r"""
    Compute the exact determinant polynomial

    $$
    P_T(u)=\det(I-uB_T).
    $$
    """
    identity = sp.eye(matrix.rows)

    return sp.expand(
        (identity - u * matrix).det(method="domain-ge")
    )
