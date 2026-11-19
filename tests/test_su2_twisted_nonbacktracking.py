import networkx as nx
import numpy as np

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)
from src.representations.su2_representations import (
    identity_su2_transport,
    su2_z_rotation,
)

from src.ihara.nonbacktracking import build_nonbacktracking_matrix


def test_identity_su2_twist_is_kronecker_product() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    ordinary = build_nonbacktracking_matrix(graph, oriented)

    su2_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        identity_su2_transport(oriented),
        oriented,
    )

    expected = np.kron(
        ordinary.astype(complex),
        np.eye(2, dtype=complex),
    )

    assert np.array_equal(su2_matrix, expected)


def test_identity_su2_twist_has_expected_shape() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        identity_su2_transport(oriented),
        oriented,
    )

    # C_4 has 8 oriented edges, each with a 2-dimensional fiber.
    assert matrix.shape == (16, 16)


def test_allowed_transition_has_successor_su2_block() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    successor_matrix = su2_z_rotation(0.5)
    transport[(1, 2)] = successor_matrix
    transport[(2, 1)] = successor_matrix.conjugate().T

    matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    row = oriented.index[(0, 1)]
    column = oriented.index[(1, 2)]

    actual_block = matrix[
        2 * row : 2 * row + 2,
        2 * column : 2 * column + 2,
    ]

    assert np.allclose(actual_block, successor_matrix)


def test_immediate_reversal_has_zero_block() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        identity_su2_transport(oriented),
        oriented,
    )

    row = oriented.index[(0, 1)]
    reverse_column = oriented.index[(1, 0)]

    reversal_block = matrix[
        2 * row : 2 * row + 2,
        2 * reverse_column : 2 * reverse_column + 2,
    ]

    assert np.array_equal(reversal_block, np.zeros((2, 2), dtype=complex))
