import networkx as nx
import sympy as sp

import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.symbolic_twists import trivial_symbolic_transport
from src.ihara.twisted_bass_formula import (
    lhs_twisted_bass_polynomial,
    rhs_twisted_bass_polynomial,
    symbolic_twisted_adjacency_matrix,
    twisted_bass_formula_holds,
)


def exact_phase(theta: sp.Expr) -> sp.Expr:
    """Return exp(i theta) in a simplification-friendly exact form."""
    return sp.cos(theta) + sp.I * sp.sin(theta)


def single_phase_cycle_transport(
    graph: nx.Graph,
    theta: sp.Expr,
):
    """Put phase exp(i theta) on (0, 1), and its inverse on (1, 0)."""
    oriented = build_oriented_edge_index(graph)
    transport = trivial_symbolic_transport(oriented)

    transport[(0, 1)] = exact_phase(theta)
    transport[(1, 0)] = exact_phase(-theta)

    return transport


def theta_graph() -> nx.Graph:
    """Return the four-vertex theta graph used in the separation example."""
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

    return graph


def signed_theta_transport(
    graph: nx.Graph,
    negative_edges: set[tuple[int, int]],
):
    """Return a transport with values in the exact subgroup {+1, -1}."""
    oriented = build_oriented_edge_index(graph)
    transport = trivial_symbolic_transport(oriented)

    for origin, terminus in negative_edges:
        transport[(origin, terminus)] = sp.Integer(-1)
        transport[(terminus, origin)] = sp.Integer(-1)

    return transport


def test_twisted_adjacency_is_hermitian_for_u1_transport() -> None:
    graph = nx.cycle_graph(4)
    transport = single_phase_cycle_transport(graph, sp.pi / 3)

    adjacency = symbolic_twisted_adjacency_matrix(graph, transport)

    assert sp.simplify(adjacency - adjacency.conjugate().T) == sp.zeros(
        graph.number_of_nodes()
    )


def test_twisted_bass_formula_for_trivial_cycle_transport() -> None:
    graph = nx.cycle_graph(4)
    transport = single_phase_cycle_transport(graph, sp.Integer(0))

    assert twisted_bass_formula_holds(graph, transport)


def test_twisted_bass_formula_for_nontrivial_triangle_phase() -> None:
    graph = nx.cycle_graph(3)
    transport = single_phase_cycle_transport(graph, sp.pi / 3)

    assert twisted_bass_formula_holds(graph, transport)


def test_twisted_bass_formula_for_nontrivial_square_phase() -> None:
    graph = nx.cycle_graph(4)
    transport = single_phase_cycle_transport(graph, sp.pi / 3)

    assert twisted_bass_formula_holds(graph, transport)


def test_twisted_bass_formula_for_pentagon_phase() -> None:
    graph = nx.cycle_graph(5)
    transport = single_phase_cycle_transport(graph, sp.pi / 2)

    assert twisted_bass_formula_holds(graph, transport)


def test_twisted_bass_formula_for_signed_theta_transport() -> None:
    graph = theta_graph()
    transport = signed_theta_transport(graph, {(0, 1)})

    assert twisted_bass_formula_holds(graph, transport)


def test_twisted_bass_formula_matches_both_exact_expressions() -> None:
    graph = nx.cycle_graph(4)
    transport = single_phase_cycle_transport(graph, sp.pi / 3)
    u = sp.symbols("u")

    lhs = lhs_twisted_bass_polynomial(graph, transport, u)
    rhs = rhs_twisted_bass_polynomial(graph, transport, u)

    assert sp.simplify(sp.cancel(lhs - rhs)) == 0
