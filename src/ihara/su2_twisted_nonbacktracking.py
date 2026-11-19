"""SU(2)-twisted non-backtracking matrices."""

from __future__ import annotations

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import numpy as np

from src.ihara.oriented_edges import (
    OrientedEdge,
    OrientedEdgeIndex,
    build_oriented_edge_index,
)
from src.representations.su2_representations import (
    Matrix2,
    validate_su2_transport,
)

from src.representations.su2_representations import (
    Matrix2,
    validate_su2_transport,
)


def su2_gauge_similarity_matrix(
    oriented: OrientedEdgeIndex,
    gauge: dict[object, Matrix2],
) -> np.ndarray:
    r"""
    Return the block-diagonal matrix G on oriented-edge fiber space:

    $$
    G_{e,e} = g_{t(e)}.
    $$

    Each diagonal entry is a 2 by 2 block. For gauge-related transports,

    $$
    B_{T'} = G^{-1} B_T G.
    $$
    """
    edge_count = len(oriented.edges)

    matrix = np.zeros(
        (2 * edge_count, 2 * edge_count),
        dtype=complex,
    )

    for edge in oriented.edges:
        edge_index = oriented.index[edge]
        terminus = oriented.terminus(edge)

        block_slice = slice(
            2 * edge_index,
            2 * edge_index + 2,
        )

        matrix[block_slice, block_slice] = gauge[terminus]

    return matrix


def build_su2_twisted_nonbacktracking_matrix(
    graph: nx.Graph,
    transport: dict[OrientedEdge, Matrix2],
    oriented: OrientedEdgeIndex | None = None,
) -> np.ndarray:
    """
    Construct the block SU(2)-twisted non-backtracking matrix.

    For an admissible transition e -> f, the (e, f) block is T_f.
    The matrix has shape (2|E_oriented|, 2|E_oriented|).
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    validate_su2_transport(oriented, transport)

    edge_count = len(oriented.edges)
    matrix = np.zeros(
        (2 * edge_count, 2 * edge_count),
        dtype=complex,
    )

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
