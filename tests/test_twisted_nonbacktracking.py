import cmath

import networkx as nx
import numpy as np

from src.ihara.nonbacktracking import build_nonbacktracking_matrix
from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.twisted_nonbacktracking import (
    build_u1_twisted_nonbacktracking_matrix,
)
from src.representations.u1_representations import trivial_u1_transport


def test_trivial_twist_recovers_ordinary_nonbacktracking_matrix() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    transport = trivial_u1_transport(oriented)

    ordinary = build_nonbacktracking_matrix(graph, oriented)
    twisted = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    assert np.array_equal(twisted, ordinary.astype(complex))


def test_valid_transition_carries_successor_transport() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    transport = trivial_u1_transport(oriented)

    transport[(1, 2)] = cmath.exp(1j * 0.5)
    transport[(2, 1)] = cmath.exp(-1j * 0.5)

    twisted = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    assert np.isclose(
        twisted[oriented.index[(0, 1)], oriented.index[(1, 2)]],
        transport[(1, 2)],
    )


def test_immediate_reversal_remains_zero_under_twist() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    transport = trivial_u1_transport(oriented)

    twisted = build_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    row = oriented.index[(0, 1)]
    reversed_column = oriented.index[(1, 0)]

    assert twisted[row, reversed_column] == 0.0
