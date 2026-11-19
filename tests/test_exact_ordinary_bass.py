import networkx as nx
import sympy as sp

from src.ihara.exact_ordinary_bass import (
    exact_ordinary_bass_polynomial,
)


def test_exact_bass_polynomial_for_four_cycle() -> None:
    graph = nx.cycle_graph(4)
    u = sp.symbols("u")

    polynomial = exact_ordinary_bass_polynomial(graph, u)

    expected = (1 - u**4) ** 2

    assert sp.expand(polynomial - expected) == 0


def test_exact_bass_polynomial_has_constant_term_one() -> None:
    graph = nx.complete_graph(4)
    u = sp.symbols("u")

    polynomial = exact_ordinary_bass_polynomial(graph, u)

    assert polynomial.subs(u, 0) == 1


def test_exact_bass_polynomial_is_a_polynomial() -> None:
    graph = nx.cycle_graph(6)
    u = sp.symbols("u")

    polynomial = exact_ordinary_bass_polynomial(graph, u)

    assert sp.Poly(polynomial, u).is_univariate
