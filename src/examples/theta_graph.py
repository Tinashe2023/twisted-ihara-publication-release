"""Reusable theta-graph examples and SU(2) transports."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.ihara.oriented_edges import build_oriented_edge_index
from src.representations.su2_representations import (
    Matrix2,
    identity_su2_transport,
)


def theta_graph() -> nx.Graph:
    """
    Return a theta graph with three paths from vertex 0 to vertex 1.

    Paths:
    - direct: 0 -- 1
    - upper:  0 -- 2 -- 1
    - lower:  0 -- 3 -- 1
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


def theta_transport(
    graph: nx.Graph,
    direct_holonomy: Matrix2,
    lower_holonomy: Matrix2,
):
    """
    Construct a gauge-fixed SU(2) transport on a theta graph.

    The upper path remains trivial. The nontrivial matrices are assigned to:

        (0, 1) -> direct_holonomy
        (0, 3) -> lower_holonomy

    Reverse oriented edges receive conjugate-transpose matrices.
    """
    oriented = build_oriented_edge_index(graph)
    transport = identity_su2_transport(oriented)

    transport[(0, 1)] = direct_holonomy
    transport[(1, 0)] = direct_holonomy.conjugate().T

    transport[(0, 3)] = lower_holonomy
    transport[(3, 0)] = lower_holonomy.conjugate().T

    return oriented, transport
