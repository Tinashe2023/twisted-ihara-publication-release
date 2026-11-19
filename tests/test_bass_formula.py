import networkx as nx
import sympy as sp

from src.ihara.bass_formula import (
    bass_formula_holds,
    lhs_bass_polynomial,
    rhs_bass_polynomial,
)


def test_bass_formula_for_triangle() -> None:
    graph = nx.cycle_graph(3)

    assert bass_formula_holds(graph)


def test_bass_formula_for_square() -> None:
    graph = nx.cycle_graph(4)

    assert bass_formula_holds(graph)


def test_bass_formula_for_pentagon() -> None:
    graph = nx.cycle_graph(5)

    assert bass_formula_holds(graph)


def test_bass_formula_for_path_graph() -> None:
    graph = nx.path_graph(4)

    assert bass_formula_holds(graph)


def test_bass_formula_for_figure_eight_graph() -> None:
    graph = nx.Graph()

    # Two triangles sharing vertex 0.
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (3, 4),
            (4, 0),
        ]
    )

    assert bass_formula_holds(graph)


def test_cycle_four_polynomial_is_expected() -> None:
    """
    For C_4, B decomposes into two directed 4-cycles, hence

        det(I - uB) = (1-u^4)^2.
    """
    graph = nx.cycle_graph(4)
    u = sp.symbols("u")

    lhs = lhs_bass_polynomial(graph, u)
    rhs = rhs_bass_polynomial(graph, u)
    expected = sp.expand((1 - u**4) ** 2)

    assert sp.expand(lhs - expected) == 0
    assert sp.expand(rhs - expected) == 0
