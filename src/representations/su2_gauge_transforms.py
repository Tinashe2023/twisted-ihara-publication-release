"""Gauge transformations for SU(2) edge transports."""

from __future__ import annotations

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Hashable

import numpy as np

from src.ihara.oriented_edges import OrientedEdge, OrientedEdgeIndex
from src.representations.su2_representations import (
    Matrix2,
    validate_su2_matrix,
    validate_su2_transport,
)


Vertex = Hashable


def validate_su2_vertex_gauge(
    oriented: OrientedEdgeIndex,
    gauge: dict[Vertex, Matrix2],
    tolerance: float = 1e-10,
) -> None:
    """Validate that every graph vertex is assigned an SU(2) matrix."""
    vertices: set[Vertex] = set()

    for edge in oriented.edges:
        vertices.add(oriented.origin(edge))
        vertices.add(oriented.terminus(edge))

    for vertex in vertices:
        if vertex not in gauge:
            raise ValueError(f"Missing gauge value for vertex {vertex}.")

        validate_su2_matrix(gauge[vertex], tolerance)


def gauge_transform_su2_transport(
    oriented: OrientedEdgeIndex,
    transport: dict[OrientedEdge, Matrix2],
    gauge: dict[Vertex, Matrix2],
    tolerance: float = 1e-10,
) -> dict[OrientedEdge, Matrix2]:
    r"""
    Apply the SU(2) gauge transformation

    $$
    T'_e = g_{o(e)}^\dagger T_e g_{t(e)}.
    $$
    """
    validate_su2_transport(oriented, transport, tolerance)
    validate_su2_vertex_gauge(oriented, gauge, tolerance)

    transformed: dict[OrientedEdge, Matrix2] = {}

    for edge in oriented.edges:
        origin = oriented.origin(edge)
        terminus = oriented.terminus(edge)

        transformed[edge] = (
            gauge[origin].conjugate().T
            @ transport[edge]
            @ gauge[terminus]
        )

    validate_su2_transport(oriented, transformed, tolerance)

    return transformed
