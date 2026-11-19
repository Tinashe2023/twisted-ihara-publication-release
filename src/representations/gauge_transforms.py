"""Gauge transformations for U(1) edge transports."""

from __future__ import annotations

from typing import Hashable

from src.ihara.oriented_edges import OrientedEdge, OrientedEdgeIndex
from src.representations.u1_representations import validate_u1_transport


Vertex = Hashable


def validate_u1_vertex_gauge(
    oriented: OrientedEdgeIndex,
    gauge: dict[Vertex, complex],
    tolerance: float = 1e-10,
) -> None:
    """Validate that every vertex has a U(1) phase."""
    vertices = set()

    for edge in oriented.edges:
        vertices.add(oriented.origin(edge))
        vertices.add(oriented.terminus(edge))

    for vertex in vertices:
        if vertex not in gauge:
            raise ValueError(f"Missing gauge value for vertex {vertex}.")

        value = gauge[vertex]

        if abs(abs(value) - 1.0) > tolerance:
            raise ValueError(
                f"Gauge value at vertex {vertex} does not have modulus one: "
                f"{value}."
            )


def gauge_transform_u1_transport(
    oriented: OrientedEdgeIndex,
    transport: dict[OrientedEdge, complex],
    gauge: dict[Vertex, complex],
) -> dict[OrientedEdge, complex]:
    """
    Apply the U(1) gauge transformation

        T'_e = g_{o(e)}^{-1} T_e g_{t(e)}.

    Since gauge values lie in U(1), inverse(g) equals conjugate(g).
    """
    validate_u1_transport(oriented, transport)
    validate_u1_vertex_gauge(oriented, gauge)

    transformed: dict[OrientedEdge, complex] = {}

    for edge in oriented.edges:
        origin = oriented.origin(edge)
        terminus = oriented.terminus(edge)

        transformed[edge] = (
            gauge[origin].conjugate()
            * transport[edge]
            * gauge[terminus]
        )

    validate_u1_transport(oriented, transformed)

    return transformed
