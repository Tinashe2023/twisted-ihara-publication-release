import numpy as np
import networkx as nx
import pytest

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.ihara.oriented_edges import build_oriented_edge_index
from src.representations.su2_representations import (
    identity_su2_transport,
    su2_from_parameters,
    su2_z_rotation,
    validate_su2_matrix,
    validate_su2_transport,
)

from src.representations.su2_representations import su2_x_rotation


def test_x_rotation_is_su2() -> None:
    matrix = su2_x_rotation(0.7)

    validate_su2_matrix(matrix)


def test_z_rotation_is_su2() -> None:
    matrix = su2_z_rotation(0.7)

    validate_su2_matrix(matrix)


def test_general_parameterized_su2_matrix_is_valid() -> None:
    alpha = 0.6 + 0.0j
    beta = 0.0 + 0.8j

    matrix = su2_from_parameters(alpha, beta)

    validate_su2_matrix(matrix)


def test_nonunitary_matrix_is_rejected() -> None:
    matrix = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.5],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError):
        validate_su2_matrix(matrix)


def test_identity_transport_is_valid() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    validate_su2_transport(
        oriented,
        identity_su2_transport(oriented),
    )


def test_invalid_reverse_transport_is_rejected() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    transport[(0, 1)] = su2_z_rotation(0.5)
    transport[(1, 0)] = su2_z_rotation(0.5)

    with pytest.raises(ValueError):
        validate_su2_transport(oriented, transport)
