"""Symbolic utilities for the ordinary Ihara-Bass determinant formula."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import networkx only for type checking to avoid static analysis errors
if TYPE_CHECKING:
    import networkx as nx

try:
    import sympy as sp  # type: ignore[import-not-found]
except ImportError as exc:
    raise ModuleNotFoundError(
        "sympy is required to use the Ihara-Bass symbolic utilities."
    ) from exc

from src.ihara.nonbacktracking import build_nonbacktracking_matrix
from src.ihara.oriented_edges import build_oriented_edge_index


def sorted_vertices(graph: "nx.Graph") -> list[object]:
    """Return graph vertices in a deterministic order."""
    return sorted(graph.nodes(), key=lambda vertex: (type(vertex).__name__, repr(vertex)))


def adjacency_matrix_sympy(graph: "nx.Graph") -> sp.Matrix:
    """Return the ordinary adjacency matrix as an exact SymPy matrix."""
    import networkx as nx  # local import to ensure runtime availability

    vertices = sorted_vertices(graph)
    return sp.Matrix(
        nx.to_numpy_array(
            graph,
            nodelist=vertices,
            dtype=int,
        )
    )


def degree_matrix_sympy(graph: "nx.Graph") -> sp.Matrix:
    """Return the diagonal degree matrix as an exact SymPy matrix."""
    import networkx as nx  # local import to ensure runtime availability

    vertices = sorted_vertices(graph)
    degrees = [graph.degree(vertex) for vertex in vertices]
    return sp.diag(*degrees)


def nonbacktracking_matrix_sympy(graph: "nx.Graph") -> sp.Matrix:
    """Return the ordinary non-backtracking matrix as an exact SymPy matrix."""
    oriented = build_oriented_edge_index(graph)
    matrix = build_nonbacktracking_matrix(graph, oriented)
    return sp.Matrix(matrix)


def lhs_bass_polynomial(graph: "nx.Graph", u: sp.Symbol) -> sp.Expr:
    """
    Compute the left-hand side of the Bass formula:

        det(I_(2m) - u B).
    """
    B = nonbacktracking_matrix_sympy(graph)
    identity = sp.eye(B.rows)

    return sp.expand((identity - u * B).det())


def rhs_bass_polynomial(graph: "nx.Graph", u: sp.Symbol) -> sp.Expr:
    """
    Compute the right-hand side of the Bass formula:

        (1-u^2)^(m-n) det(I_n - u A + u^2(D-I_n)).
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    A = adjacency_matrix_sympy(graph)
    D = degree_matrix_sympy(graph)
    identity = sp.eye(n)

    vertex_term = identity - u * A + u**2 * (D - identity)

    return sp.cancel((1 - u**2) ** (m - n) * vertex_term.det())


def bass_formula_holds(graph: "nx.Graph") -> bool:
    """Return True exactly when the two Bass expressions agree symbolically."""
    u = sp.symbols("u")

    lhs = lhs_bass_polynomial(graph, u)
    rhs = rhs_bass_polynomial(graph, u)

    return sp.simplify(lhs - rhs) == 0
