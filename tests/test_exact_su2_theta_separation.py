import sympy as sp

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.examples.theta_graph import theta_graph
from src.ihara.symbolic_su2_nonbacktracking import (
    build_symbolic_su2_twisted_nonbacktracking_matrix,
    exact_twisted_determinant_polynomial,
    symbolic_su2_theta_transport,
)
from src.representations.symbolic_su2 import (
    is_symbolic_su2,
    symbolic_su2_x_quarter_turn,
    symbolic_su2_z_quarter_turn,
)


def test_exact_special_matrices_are_in_su2() -> None:
    z_matrix = symbolic_su2_z_quarter_turn()
    x_matrix = symbolic_su2_x_quarter_turn()

    assert is_symbolic_su2(z_matrix)
    assert is_symbolic_su2(x_matrix)


def test_exact_commuting_and_noncommuting_pairs_have_same_generator_traces() -> None:
    A = symbolic_su2_z_quarter_turn()
    B_commuting = symbolic_su2_z_quarter_turn()
    B_noncommuting = symbolic_su2_x_quarter_turn()

    assert sp.simplify(sp.trace(A)) == 0
    assert sp.simplify(
        sp.trace(B_commuting) - sp.trace(B_noncommuting)
    ) == 0


def test_exact_commuting_and_noncommuting_pairs_have_distinct_mixed_traces() -> None:
    A = symbolic_su2_z_quarter_turn()
    B_commuting = symbolic_su2_z_quarter_turn()
    B_noncommuting = symbolic_su2_x_quarter_turn()

    commuting_mixed_trace = sp.trace(A * B_commuting)
    noncommuting_mixed_trace = sp.trace(A * B_noncommuting)

    assert sp.simplify(
        commuting_mixed_trace - noncommuting_mixed_trace
    ) != 0


def test_exact_su2_theta_determinant_polynomials_are_distinct() -> None:
    """
    Compare the full exact determinant polynomials

    $$
    P_T(u)=\det(I-uB_T)
    $$

    for commuting and noncommuting SU(2) theta-graph transports.
    """
    graph = theta_graph()
    u = sp.symbols("u")

    A = symbolic_su2_z_quarter_turn()
    B_commuting = symbolic_su2_z_quarter_turn()
    B_noncommuting = symbolic_su2_x_quarter_turn()

    commuting_oriented, commuting_transport = symbolic_su2_theta_transport(
        graph,
        A,
        B_commuting,
    )

    noncommuting_oriented, noncommuting_transport = (
        symbolic_su2_theta_transport(
            graph,
            A,
            B_noncommuting,
        )
    )

    commuting_matrix = build_symbolic_su2_twisted_nonbacktracking_matrix(
        graph,
        commuting_transport,
        commuting_oriented,
    )

    noncommuting_matrix = (
        build_symbolic_su2_twisted_nonbacktracking_matrix(
            graph,
            noncommuting_transport,
            noncommuting_oriented,
        )
    )

    commuting_polynomial = exact_twisted_determinant_polynomial(
        commuting_matrix,
        u,
    )

    noncommuting_polynomial = exact_twisted_determinant_polynomial(
        noncommuting_matrix,
        u,
    )

    assert sp.simplify(
        commuting_polynomial - noncommuting_polynomial
    ) != 0
