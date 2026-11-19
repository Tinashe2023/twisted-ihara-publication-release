"""U(1)-twisted non-backtracking matrices."""

from __future__ import annotations

import numpy as np
import networkx as nx

from src.ihara.oriented_edges import OrientedEdge, OrientedEdgeIndex, build_oriented_edge_index
from src.representations.u1_representations import validate_u1_transport


def build_u1_twisted_nonbacktracking_matrix(
    graph: nx.Graph,
    transport: dict[OrientedEdge, complex],
    oriented: OrientedEdgeIndex | None = None,
) -> np.ndarray:
    """
    Construct the U(1)-twisted non-backtracking matrix B_T.

    Its entries are

    $$
    (B_T)_{e,f}
    =
    \begin{cases}
    T_f, & t(e)=o(f) and f != \\bar e, \\\\
    0, & otherwise.
    \end{cases}
    $$

    The matrix is indexed by oriented edges.
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    validate_u1_transport(oriented, transport)

    size = len(oriented.edges)
    matrix = np.zeros((size, size), dtype=complex)

    for edge in oriented.edges:
        row = oriented.index[edge]
        terminus = oriented.terminus(edge)
        reverse_edge = oriented.reverse(edge)

        for successor in oriented.edges:
            column = oriented.index[successor]

            is_continuous = terminus == oriented.origin(successor)
            is_not_backtracking = successor != reverse_edge

            if is_continuous and is_not_backtracking:
                matrix[row, column] = transport[successor]

    return matrix
def u1_gauge_similarity_matrix(
    oriented: OrientedEdgeIndex,
    gauge: dict[object, complex],
) -> np.ndarray:
    """
    Return the diagonal matrix G indexed by oriented edges, where

        G[e,e] = g_{t(e)}.

    For gauge-related transports T and T', our convention gives

        B_(T') = G^{-1} B_T G.
    """
    size = len(oriented.edges)
    matrix = np.zeros((size, size), dtype=complex)

    for edge in oriented.edges:
        index = oriented.index[edge]
        terminus = oriented.terminus(edge)
        matrix[index, index] = gauge[terminus]

    return matrix
