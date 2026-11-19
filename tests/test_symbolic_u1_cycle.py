import networkx as nx
import sympy as sp

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.symbolic_twists import (
    build_symbolic_u1_twisted_nonbacktracking_matrix,
    characteristic_determinant,
    trivial_symbolic_transport,
)


def build_single_phase_cycle_transport(
    graph: nx.Graph,
    theta: sp.Expr,
):
    """Place phase exp(i theta) on edge (0, 1) and its inverse on (1, 0)."""
    oriented = build_oriented_edge_index(graph)
    transport = trivial_symbolic_transport(oriented)

    forward_phase = sp.cos(theta) + sp.I * sp.sin(theta)
    reverse_phase = sp.cos(theta) - sp.I * sp.sin(theta)

    transport[(0, 1)] = forward_phase
    transport[(1, 0)] = reverse_phase

    return oriented, transport



def test_exact_u1_twisted_triangle_identity() -> None:
    graph = nx.cycle_graph(3)
    theta = sp.pi / 3
    u = sp.symbols("u")

    oriented, transport = build_single_phase_cycle_transport(graph, theta)

    matrix = build_symbolic_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    actual = characteristic_determinant(matrix, u)
    expected = sp.expand(
        1 - 2 * sp.cos(theta) * u**3 + u**6
    )

    assert sp.simplify(actual - expected) == 0


def test_exact_u1_twisted_square_identity() -> None:
    graph = nx.cycle_graph(4)
    theta = sp.pi / 3
    u = sp.symbols("u")

    oriented, transport = build_single_phase_cycle_transport(graph, theta)

    matrix = build_symbolic_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    actual = characteristic_determinant(matrix, u)
    expected = sp.expand(
        1 - 2 * sp.cos(theta) * u**4 + u**8
    )

    assert sp.simplify(actual - expected) == 0


def test_exact_u1_twisted_pentagon_identity() -> None:
    graph = nx.cycle_graph(5)
    theta = sp.pi / 2
    u = sp.symbols("u")

    oriented, transport = build_single_phase_cycle_transport(graph, theta)

    matrix = build_symbolic_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    actual = characteristic_determinant(matrix, u)
    expected = sp.expand(
        1 - 2 * sp.cos(theta) * u**5 + u**10
    )

    assert sp.simplify(actual - expected) == 0


def test_trivial_phase_recovers_ordinary_cycle_result() -> None:
    graph = nx.cycle_graph(4)
    theta = sp.Integer(0)
    u = sp.symbols("u")

    oriented, transport = build_single_phase_cycle_transport(graph, theta)

    matrix = build_symbolic_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    actual = characteristic_determinant(matrix, u)
    expected = sp.expand((1 - u**4) ** 2)

    assert sp.simplify(actual - expected) == 0
