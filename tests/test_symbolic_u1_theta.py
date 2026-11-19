import networkx as nx
import sympy as sp

import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.symbolic_twists import (
    build_symbolic_u1_twisted_nonbacktracking_matrix,
    characteristic_determinant,
    trivial_symbolic_transport,
)


def theta_graph() -> nx.Graph:
    """
    Return a small theta graph with three internally disjoint paths
    between vertices 0 and 1.
    """
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


def signed_transport(
    graph: nx.Graph,
    negative_edges: set[tuple[int, int]],
):
    """
    Construct an exact U(1) transport with values in {+1, -1}.

    A negative edge receives transport -1 in both directions because

    $$
    (-1)^{-1} = -1.
    $$
    """
    oriented = build_oriented_edge_index(graph)
    transport = trivial_symbolic_transport(oriented)

    for edge in negative_edges:
        u, v = edge
        transport[(u, v)] = sp.Integer(-1)
        transport[(v, u)] = sp.Integer(-1)

    return oriented, transport


def theta_determinant(
    graph: nx.Graph,
    negative_edges: set[tuple[int, int]],
    u: sp.Symbol,
) -> sp.Expr:
    """Return the exact twisted determinant for a signed theta transport."""
    oriented, transport = signed_transport(graph, negative_edges)

    matrix = build_symbolic_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    return sp.expand(characteristic_determinant(matrix, u))


def test_theta_graph_has_fundamental_group_rank_two() -> None:
    graph = theta_graph()

    rank = (
        graph.number_of_edges()
        - graph.number_of_nodes()
        + 1
    )

    assert rank == 2


def test_theta_graph_trivial_and_twisted_transports_are_separated() -> None:
    """
    The same underlying graph carries two U(1) transports:

    1. trivial transport;
    2. transport with -1 on the direct edge {0,1}.

    Their twisted determinant polynomials are distinct.
    """
    graph = theta_graph()
    u = sp.symbols("u")

    trivial = theta_determinant(graph, set(), u)
    direct_edge_twist = theta_determinant(graph, {(0, 1)}, u)

    assert sp.simplify(trivial - direct_edge_twist) != 0


def test_distinct_theta_holonomy_patterns_can_be_separated() -> None:
    """
    Compare a direct-edge twist with a twist on one two-edge arm.
    These yield distinct twisted determinant polynomials.
    """
    graph = theta_graph()
    u = sp.symbols("u")

    direct_edge_twist = theta_determinant(graph, {(0, 1)}, u)
    upper_arm_twist = theta_determinant(graph, {(0, 2)}, u)

    assert sp.simplify(direct_edge_twist - upper_arm_twist) != 0


def test_signed_theta_transport_has_constant_term_one() -> None:
    """
    Every determinant det(I-uB_T) has constant coefficient 1.
    """
    graph = theta_graph()
    u = sp.symbols("u")

    polynomial = theta_determinant(graph, {(0, 1)}, u)

    assert polynomial.subs(u, 0) == 1
