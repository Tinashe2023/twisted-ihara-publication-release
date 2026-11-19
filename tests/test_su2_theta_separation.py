import networkx as nx
import numpy as np

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.examples.theta_graph import (
    theta_graph,
    theta_transport,
)


from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.su2_twisted_bass_formula import (
    lhs_su2_twisted_bass_determinant,
)
from src.representations.su2_representations import (
    identity_su2_transport,
    su2_x_rotation,
    su2_z_rotation,
)

from src.ihara.numerical_characteristic import (
    determinant_polynomial_coefficients,
    determinant_polynomials_are_distinct,
)
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)

def test_theta_graph_full_determinant_polynomials_are_distinct() -> None:
    """
    Compare the complete polynomials det(I-uB_T), not merely one value.

    The commuting and noncommuting pairs have the same individual generator
    traces, but their full twisted determinant polynomials are distinct.
    """
    graph = theta_graph()

    alpha = 0.4
    beta = 0.7

    A = su2_z_rotation(alpha)

    B_commuting = su2_z_rotation(beta)
    B_noncommuting = su2_x_rotation(beta)

    commuting_oriented, commuting_transport = theta_transport(
        graph,
        A,
        B_commuting,
    )

    noncommuting_oriented, noncommuting_transport = theta_transport(
        graph,
        A,
        B_noncommuting,
    )

    commuting_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        commuting_transport,
        commuting_oriented,
    )

    noncommuting_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        noncommuting_transport,
        noncommuting_oriented,
    )

    commuting_coefficients = determinant_polynomial_coefficients(
        commuting_matrix,
    )

    noncommuting_coefficients = determinant_polynomial_coefficients(
        noncommuting_matrix,
    )

    assert determinant_polynomials_are_distinct(
        commuting_coefficients,
        noncommuting_coefficients,
    )

from src.ihara.numerical_characteristic import (
    evaluate_determinant_polynomial,
)

def test_determinant_polynomial_matches_direct_evaluation() -> None:
    graph = theta_graph()

    A = su2_z_rotation(0.4)
    B = su2_x_rotation(0.7)

    oriented, transport = theta_transport(graph, A, B)

    matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    coefficients = determinant_polynomial_coefficients(matrix)

    for u in [0.1, 0.2, 0.35]:
        from_polynomial = evaluate_determinant_polynomial(
            coefficients,
            u,
        )

        direct = np.linalg.det(
            np.eye(matrix.shape[0], dtype=complex) - u * matrix
        )

        assert np.allclose(
            from_polynomial,
            direct,
            atol=1e-8,
            rtol=1e-8,
        )



def test_theta_graph_has_cycle_rank_two() -> None:
    graph = theta_graph()

    cycle_rank = (
        graph.number_of_edges()
        - graph.number_of_nodes()
        + 1
    )

    assert cycle_rank == 2


def test_commuting_and_noncommuting_holonomies_have_same_individual_traces() -> None:
    alpha = 0.4
    beta = 0.7

    first_holonomy = su2_z_rotation(alpha)

    commuting_second_holonomy = su2_z_rotation(beta)
    noncommuting_second_holonomy = su2_x_rotation(beta)

    assert np.allclose(
        np.trace(commuting_second_holonomy),
        np.trace(noncommuting_second_holonomy),
    )

    assert np.allclose(
        np.trace(first_holonomy),
        np.trace(first_holonomy),
    )


def test_commuting_and_noncommuting_pairs_have_distinct_mixed_trace() -> None:
    alpha = 0.4
    beta = 0.7

    first_holonomy = su2_z_rotation(alpha)

    commuting_second_holonomy = su2_z_rotation(beta)
    noncommuting_second_holonomy = su2_x_rotation(beta)

    commuting_mixed_trace = np.trace(
        first_holonomy @ commuting_second_holonomy
    )

    noncommuting_mixed_trace = np.trace(
        first_holonomy @ noncommuting_second_holonomy
    )

    assert not np.isclose(
        commuting_mixed_trace,
        noncommuting_mixed_trace,
    )


def test_theta_graph_separates_commuting_and_noncommuting_su2_holonomies() -> None:
    r"""
    The two representations have matching traces on each generator:

    $$
    \operatorname{tr}(A),\quad \operatorname{tr}(B).
    $$

    However, their mixed-word trace $\operatorname{tr}(AB)$ differs, and
    the twisted non-backtracking determinants are separated.
    """
    graph = theta_graph()
    alpha = 0.4
    beta = 0.7
    u = 0.2

    first_holonomy = su2_z_rotation(alpha)

    commuting_second_holonomy = su2_z_rotation(beta)
    noncommuting_second_holonomy = su2_x_rotation(beta)

    commuting_oriented, commuting_transport = theta_transport(
        graph,
        first_holonomy,
        commuting_second_holonomy,
    )

    noncommuting_oriented, noncommuting_transport = theta_transport(
        graph,
        first_holonomy,
        noncommuting_second_holonomy,
    )

    commuting_determinant = lhs_su2_twisted_bass_determinant(
        graph,
        commuting_transport,
        u,
        commuting_oriented,
    )

    noncommuting_determinant = lhs_su2_twisted_bass_determinant(
        graph,
        noncommuting_transport,
        u,
        noncommuting_oriented,
    )

    assert not np.isclose(
        commuting_determinant,
        noncommuting_determinant,
        atol=1e-9,
        rtol=1e-9,
    )
