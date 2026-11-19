import networkx as nx
import numpy as np
import pytest

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
    su2_gauge_similarity_matrix,
)
from src.representations.su2_gauge_transforms import (
    gauge_transform_su2_transport,
    validate_su2_vertex_gauge,
)
from src.representations.su2_representations import (
    identity_su2_transport,
    su2_from_parameters,
    su2_z_rotation,
)


def test_su2_vertex_gauge_is_valid() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    gauge = {
        0: su2_z_rotation(0.1),
        1: su2_z_rotation(0.3),
        2: su2_z_rotation(0.7),
        3: su2_z_rotation(1.1),
    }

    validate_su2_vertex_gauge(oriented, gauge)


def test_missing_su2_vertex_gauge_is_rejected() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)

    gauge = {
        0: su2_z_rotation(0.1),
        1: su2_z_rotation(0.2),
    }

    with pytest.raises(ValueError):
        validate_su2_vertex_gauge(oriented, gauge)


def test_gauge_transform_preserves_reverse_edge_condition() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    gauge = {
        0: su2_z_rotation(0.2),
        1: su2_z_rotation(0.5),
        2: su2_z_rotation(0.9),
    }

    transformed = gauge_transform_su2_transport(
        oriented,
        transport,
        gauge,
    )

    assert np.allclose(
        transformed[(1, 0)],
        transformed[(0, 1)].conjugate().T,
    )


def test_gauge_related_su2_twisted_matrices_are_similar() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    edge_transport = su2_from_parameters(
        alpha=0.6 + 0.0j,
        beta=0.0 + 0.8j,
    )

    transport[(0, 1)] = edge_transport
    transport[(1, 0)] = edge_transport.conjugate().T

    gauge = {
        0: su2_z_rotation(0.1),
        1: su2_z_rotation(0.4),
        2: su2_z_rotation(0.8),
        3: su2_z_rotation(1.2),
    }

    transformed_transport = gauge_transform_su2_transport(
        oriented,
        transport,
        gauge,
    )

    original_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    transformed_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transformed_transport,
        oriented,
    )

    similarity_matrix = su2_gauge_similarity_matrix(
        oriented,
        gauge,
    )

    expected_matrix = (
        similarity_matrix.conjugate().T
        @ original_matrix
        @ similarity_matrix
    )

    assert np.allclose(transformed_matrix, expected_matrix)


def test_gauge_related_su2_matrices_have_equal_determinants() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    transport[(0, 1)] = su2_z_rotation(0.6)
    transport[(1, 0)] = su2_z_rotation(-0.6)

    gauge = {
        0: su2_z_rotation(0.2),
        1: su2_z_rotation(0.5),
        2: su2_z_rotation(0.9),
        3: su2_z_rotation(1.4),
    }

    transformed_transport = gauge_transform_su2_transport(
        oriented,
        transport,
        gauge,
    )

    original_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    transformed_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transformed_transport,
        oriented,
    )

    assert np.allclose(
        np.linalg.det(np.eye(original_matrix.shape[0]) - 0.3 * original_matrix),
        np.linalg.det(
            np.eye(transformed_matrix.shape[0])
            - 0.3 * transformed_matrix
        ),
    )
