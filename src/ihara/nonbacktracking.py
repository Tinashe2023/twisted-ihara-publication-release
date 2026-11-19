"""Ordinary non-backtracking operators on finite undirected graphs."""

from __future__ import annotations

import numpy as np
import networkx as nx

from src.ihara.oriented_edges import OrientedEdgeIndex, build_oriented_edge_index


def build_nonbacktracking_matrix(
    graph: nx.Graph,
    oriented: OrientedEdgeIndex | None = None,
) -> np.ndarray:
    """
    Construct the ordinary non-backtracking matrix B of a finite graph.

    Rows and columns are indexed by oriented edges. Its entries are

        B[e, f] = 1

    exactly when the path may move from oriented edge e to oriented edge f
    without immediately reversing direction:

        t(e) = o(f) and f != reverse(e).

    Parameters
    ----------
    graph:
        A finite undirected loopless simple NetworkX graph.

    oriented:
        Optional precomputed oriented-edge index. If omitted, one is built.

    Returns
    -------
    numpy.ndarray
        The 2|E| by 2|E| ordinary non-backtracking matrix.
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    size = len(oriented.edges)
    matrix = np.zeros((size, size), dtype=int)

    for edge in oriented.edges:
        edge_index = oriented.index[edge]
        terminus = oriented.terminus(edge)
        reverse_edge = oriented.reverse(edge)

        for successor in oriented.edges:
            successor_index = oriented.index[successor]

            is_continuous = terminus == oriented.origin(successor)
            is_not_immediate_reversal = successor != reverse_edge

            if is_continuous and is_not_immediate_reversal:
                matrix[edge_index, successor_index] = 1

    return matrix
