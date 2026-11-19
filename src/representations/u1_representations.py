"""U(1) edge transports for finite undirected graphs."""

from __future__ import annotations

import cmath
from typing import Hashable

import networkx as nx

from src.ihara.oriented_edges import OrientedEdge, OrientedEdgeIndex


Vertex = Hashable


def phase(theta: float) -> complex:
    """Return exp(i theta)."""
    return cmath.exp(1j * theta)


def validate_u1_transport(
    oriented: OrientedEdgeIndex,
    transport: dict[OrientedEdge, complex],
    tolerance: float = 1e-10,
) -> None:
    """
    Validate a U(1) edge transport.

    Required conditions:
    1. Every oriented edge has a transport value.
    2. Every transport has modulus one.
    3. Reversed edges carry inverse/conjugate transport.
    """
    for edge in oriented.edges:
        if edge not in transport:
            raise ValueError(f"Missing transport for oriented edge {edge}.")

        value = transport[edge]

        if abs(abs(value) - 1.0) > tolerance:
            raise ValueError(
                f"Transport on edge {edge} does not have modulus one: {value}."
            )

        reverse_edge = oriented.reverse(edge)

        if abs(transport[reverse_edge] - value.conjugate()) > tolerance:
            raise ValueError(
                "Transport reversal condition failed for "
                f"{edge} and {reverse_edge}."
            )


def trivial_u1_transport(
    oriented: OrientedEdgeIndex,
) -> dict[OrientedEdge, complex]:
    """Assign transport 1 to every oriented edge."""
    return {edge: 1.0 + 0.0j for edge in oriented.edges}


def transport_from_undirected_phases(
    graph: nx.Graph,
    oriented: OrientedEdgeIndex,
    edge_phases: dict[tuple[Vertex, Vertex], float],
) -> dict[OrientedEdge, complex]:
    """
    Construct a U(1) transport from phase assignments on undirected edges.

    The phase assigned to the canonical orientation (u, v) is theta, giving

        T_(u,v) = exp(i theta),
        T_(v,u) = exp(-i theta).

    The keys in edge_phases must use the same canonical endpoint order as
    the oriented-edge implementation.
    """
    transport: dict[OrientedEdge, complex] = {}

    for u, v in graph.edges():
        canonical = (u, v)
        reverse = (v, u)

        if canonical not in edge_phases and reverse not in edge_phases:
            raise ValueError(f"Missing phase for undirected edge {(u, v)}.")

        theta = edge_phases.get(canonical, edge_phases.get(reverse))

        transport[(u, v)] = phase(theta)
        transport[(v, u)] = phase(-theta)

    validate_u1_transport(oriented, transport)
    return transport
