import numpy as np

from src.graphs.fundamental_holonomies import (
    build_gauge_fixed_su2_transport,
    deterministic_spanning_tree,
    fundamental_chords,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.ihara.numerical_characteristic import (
    determinant_polynomial_coefficients,
    determinant_polynomials_are_distinct,
)
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)
from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)


def small_tanner_graph():
    parity_check = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
        ],
        dtype=int,
    )

    return build_tanner_graph(parity_check)


def transport_polynomial(
    graph,
    chord_holonomies,
    tree,
):
    """Return det(I-uB_T) coefficients in ascending powers of u."""
    oriented, transport = build_gauge_fixed_su2_transport(
        graph,
        chord_holonomies,
        tree,
    )

    twisted_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    return determinant_polynomial_coefficients(twisted_matrix)


def test_tanner_transport_experiment_separates_twist_classes() -> None:
    """
    The same Tanner graph admits distinct twisted determinant polynomials for
    trivial, commuting, and noncommuting SU(2) holonomy assignments.
    """
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    trivial_holonomies = {
        chord: np.eye(2, dtype=complex)
        for chord in chords
    }

    commuting_holonomies = {
        chords[0]: su2_z_rotation(0.2),
        chords[1]: su2_z_rotation(0.4),
        chords[2]: su2_z_rotation(0.7),
    }

    noncommuting_holonomies = {
        chords[0]: su2_z_rotation(0.2),
        chords[1]: su2_x_rotation(0.4),
        chords[2]: su2_z_rotation(0.7),
    }

    trivial_polynomial = transport_polynomial(
        graph,
        trivial_holonomies,
        tree,
    )

    commuting_polynomial = transport_polynomial(
        graph,
        commuting_holonomies,
        tree,
    )

    noncommuting_polynomial = transport_polynomial(
        graph,
        noncommuting_holonomies,
        tree,
    )

    assert determinant_polynomials_are_distinct(
        trivial_polynomial,
        commuting_polynomial,
    )

    assert determinant_polynomials_are_distinct(
        commuting_polynomial,
        noncommuting_polynomial,
    )

    assert determinant_polynomials_are_distinct(
        trivial_polynomial,
        noncommuting_polynomial,
    )
