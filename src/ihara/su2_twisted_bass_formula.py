"""Numerical SU(2)-twisted Bass determinant computations."""

from __future__ import annotations

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Hashable

import networkx as nx
import numpy as np

from src.ihara.oriented_edges import (
    OrientedEdge,
    OrientedEdgeIndex,
    build_oriented_edge_index,
)
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)
from src.representations.su2_representations import (
    Matrix2,
    validate_su2_transport,
)


Vertex = Hashable


def sorted_vertices(graph: nx.Graph) -> list[Vertex]:
    """Return a deterministic vertex ordering."""
    return sorted(graph.nodes())


def build_su2_twisted_adjacency_matrix(
    graph: nx.Graph,
    transport: dict[OrientedEdge, Matrix2],
    oriented: OrientedEdgeIndex | None = None,
) -> np.ndarray:
    r"""
    Construct the $2n \times 2n$ block SU(2)-twisted adjacency matrix.

    For an undirected graph edge $\{x,y\}$, its blocks are

    $$
    (A_T)_{x,y}=T_{(x,y)},
    \qquad
    (A_T)_{y,x}=T_{(y,x)}.
    $$
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    validate_su2_transport(oriented, transport)

    vertices = sorted_vertices(graph)
    vertex_index = {
        vertex: index
        for index, vertex in enumerate(vertices)
    }

    matrix = np.zeros(
        (2 * len(vertices), 2 * len(vertices)),
        dtype=complex,
    )

    for origin, terminus in graph.edges():
        origin_index = vertex_index[origin]
        terminus_index = vertex_index[terminus]

        origin_slice = slice(2 * origin_index, 2 * origin_index + 2)
        terminus_slice = slice(2 * terminus_index, 2 * terminus_index + 2)

        matrix[origin_slice, terminus_slice] = transport[(origin, terminus)]
        matrix[terminus_slice, origin_slice] = transport[(terminus, origin)]

    return matrix


def su2_degree_correction_matrix(graph: nx.Graph) -> np.ndarray:
    r"""
    Construct the block degree correction

    $$
    (D-I_n)\otimes I_2.
    $$
    """
    vertices = sorted_vertices(graph)

    diagonal_entries = [
        graph.degree(vertex) - 1
        for vertex in vertices
    ]

    scalar_correction = np.diag(diagonal_entries)

    return np.kron(
        scalar_correction,
        np.eye(2, dtype=complex),
    )


def lhs_su2_twisted_bass_determinant(
    graph: nx.Graph,
    transport: dict[OrientedEdge, Matrix2],
    u: complex,
    oriented: OrientedEdgeIndex | None = None,
) -> complex:
    """Compute the edge-side determinant det(I - u B_T)."""
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    nonbacktracking = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    identity = np.eye(nonbacktracking.shape[0], dtype=complex)

    return np.linalg.det(identity - u * nonbacktracking)


def rhs_su2_twisted_bass_determinant(
    graph: nx.Graph,
    transport: dict[OrientedEdge, Matrix2],
    u: complex,
    oriented: OrientedEdgeIndex | None = None,
) -> complex:
    r"""
    Compute the vertex-side SU(2)-twisted Bass expression:

    $$
    (1-u^2)^{2(m-n)}
    \det\left[
    I_{2n}
    -uA_T
    +u^2((D-I_n)\otimes I_2)
    \right].
    $$
    """
    if oriented is None:
        oriented = build_oriented_edge_index(graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    adjacency = build_su2_twisted_adjacency_matrix(
        graph,
        transport,
        oriented,
    )

    correction = su2_degree_correction_matrix(graph)
    identity = np.eye(2 * n, dtype=complex)

    vertex_matrix = (
        identity
        - u * adjacency
        + u**2 * correction
    )

    scalar_factor = (1 - u**2) ** (2 * (m - n))

    return scalar_factor * np.linalg.det(vertex_matrix)


def su2_twisted_bass_formula_holds(
    graph: nx.Graph,
    transport: dict[OrientedEdge, Matrix2],
    u: complex = 0.2,
    tolerance: float = 1e-9,
) -> bool:
    """Numerically test the SU(2)-twisted Bass determinant formula."""
    oriented = build_oriented_edge_index(graph)

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

    return np.allclose(
        lhs,
        rhs,
        atol=tolerance,
        rtol=tolerance,
    )
