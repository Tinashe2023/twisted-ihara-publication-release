import cmath

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.twisted_nonbacktracking import (
    build_u1_twisted_nonbacktracking_matrix,
    u1_gauge_similarity_matrix,
)
from src.representations.gauge_transforms import (
    gauge_transform_u1_transport,
)
from src.representations.u1_representations import trivial_u1_transport


def test_gauge_related_twisted_matrices_are_similar() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    transport = trivial_u1_transport(oriented)

    # Introduce one nontrivial edge phase.
    transport[(0, 1)] = cmath.exp(1j * 0.4)
    transport[(1, 0)] = cmath.exp(-1j * 0.4)

    gauge = {
        0: cmath.exp(1j * 0.1),
        1: cmath.exp(1j * 0.6),
        2: cmath.exp(1j * 1.0),
        3: cmath.exp(1j * 1.4),
    }

    transformed_transport = gauge_transform_u1_transport(
        oriented,
        transport,
        gauge,
    )

    original_matrix = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    transformed_matrix = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transformed_transport,
        oriented,
    )

    similarity_matrix = u1_gauge_similarity_matrix(oriented, gauge)

    expected_matrix = (
        np.linalg.inv(similarity_matrix)
        @ original_matrix
        @ similarity_matrix
    )

    assert np.allclose(transformed_matrix, expected_matrix)


def test_gauge_related_matrices_have_equal_eigenvalues() -> None:
    graph = nx.cycle_graph(5)
    oriented = build_oriented_edge_index(graph)

    transport = trivial_u1_transport(oriented)

    transport[(0, 1)] = cmath.exp(1j * 0.9)
    transport[(1, 0)] = cmath.exp(-1j * 0.9)

    gauge = {
        vertex: cmath.exp(1j * (vertex + 1) * 0.25)
        for vertex in graph.nodes()
    }

    transformed_transport = gauge_transform_u1_transport(
        oriented,
        transport,
        gauge,
    )

    original_matrix = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    transformed_matrix = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transformed_transport,
        oriented,
    )

    original_eigenvalues = np.linalg.eigvals(original_matrix)
    transformed_eigenvalues = np.linalg.eigvals(transformed_matrix)

    # Eigenvalues form an unordered multiset. Match each eigenvalue from the
    # original matrix to the closest eigenvalue of the gauge-transformed matrix.
    distance_matrix = np.abs(
        original_eigenvalues[:, np.newaxis]
        - transformed_eigenvalues[np.newaxis, :]
    )

    row_indices, column_indices = linear_sum_assignment(distance_matrix)

    assert np.allclose(
        original_eigenvalues[row_indices],
        transformed_eigenvalues[column_indices],
        atol=1e-10,
        rtol=1e-10,
    )
