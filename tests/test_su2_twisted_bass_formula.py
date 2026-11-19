import networkx as nx
import numpy as np

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.su2_twisted_bass_formula import (
    build_su2_twisted_adjacency_matrix,
    lhs_su2_twisted_bass_determinant,
    rhs_su2_twisted_bass_determinant,
    su2_twisted_bass_formula_holds,
)
from src.representations.su2_representations import (
    identity_su2_transport,
    su2_from_parameters,
    su2_z_rotation,
)


def single_edge_su2_transport(
    graph: nx.Graph,
):
    """Use a non-diagonal SU(2) transport on edge {0,1}."""
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    matrix = su2_from_parameters(
        alpha=0.6 + 0.0j,
        beta=0.0 + 0.8j,
    )

    transport[(0, 1)] = matrix
    transport[(1, 0)] = matrix.conjugate().T

    return oriented, transport


def test_su2_twisted_adjacency_has_expected_shape() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    adjacency = build_su2_twisted_adjacency_matrix(
        graph,
        identity_su2_transport(oriented),
        oriented,
    )

    assert adjacency.shape == (8, 8)


def test_su2_twisted_adjacency_is_hermitian() -> None:
    graph = nx.cycle_graph(4)
    oriented, transport = single_edge_su2_transport(graph)

    adjacency = build_su2_twisted_adjacency_matrix(
        graph,
        transport,
        oriented,
    )

    assert np.allclose(
        adjacency,
        adjacency.conjugate().T,
    )


def test_su2_bass_formula_for_identity_transport_on_triangle() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)

    transport = identity_su2_transport(oriented)

    assert su2_twisted_bass_formula_holds(
        graph,
        transport,
        u=0.2,
    )


def test_su2_bass_formula_for_diagonal_cycle_transport() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    transport = identity_su2_transport(oriented)
    phase_matrix = su2_z_rotation(0.7)

    transport[(0, 1)] = phase_matrix
    transport[(1, 0)] = phase_matrix.conjugate().T

    assert su2_twisted_bass_formula_holds(
        graph,
        transport,
        u=0.25,
    )


def test_su2_bass_formula_for_nondiagonal_cycle_transport() -> None:
    graph = nx.cycle_graph(4)
    _, transport = single_edge_su2_transport(graph)

    assert su2_twisted_bass_formula_holds(
        graph,
        transport,
        u=0.2,
    )


def test_su2_bass_formula_for_theta_graph() -> None:
    graph = nx.Graph()

    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (2, 1),
            (0, 3),
            (3, 1),
        ]
    )

    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    first_matrix = su2_z_rotation(0.4)
    second_matrix = su2_from_parameters(
        alpha=0.8 + 0.0j,
        beta=0.0 + 0.6j,
    )

    transport[(0, 1)] = first_matrix
    transport[(1, 0)] = first_matrix.conjugate().T

    transport[(0, 2)] = second_matrix
    transport[(2, 0)] = second_matrix.conjugate().T

    assert su2_twisted_bass_formula_holds(
        graph,
        transport,
        u=0.15,
    )


def test_su2_bass_edge_and_vertex_sides_agree_at_multiple_u_values() -> None:
    graph = nx.cycle_graph(4)
    oriented, transport = single_edge_su2_transport(graph)

    for u in [0.1, 0.2, 0.35]:
        lhs = lhs_su2_twisted_bass_determinant(
            graph,
            transport,
            u,
            oriented,
        )

        rhs = rhs_su2_twisted_bass_determinant(
            graph,
            transport,
            u,
            oriented,
        )

        assert np.allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
