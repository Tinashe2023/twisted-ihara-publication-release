import cmath

import networkx as nx
import pytest

from src.ihara.oriented_edges import build_oriented_edge_index
from src.representations.u1_representations import (
    trivial_u1_transport,
    validate_u1_transport,
)


def test_trivial_transport_is_valid() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    transport = trivial_u1_transport(oriented)

    validate_u1_transport(oriented, transport)


def test_reversed_edge_has_conjugate_transport() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)

    transport = trivial_u1_transport(oriented)
    transport[(0, 1)] = cmath.exp(1j * 0.7)
    transport[(1, 0)] = cmath.exp(-1j * 0.7)

    validate_u1_transport(oriented, transport)


def test_nonunit_transport_is_rejected() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)
    transport = trivial_u1_transport(oriented)

    transport[(0, 1)] = 2.0 + 0.0j

    with pytest.raises(ValueError):
        validate_u1_transport(oriented, transport)


def test_inconsistent_reverse_transport_is_rejected() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)
    transport = trivial_u1_transport(oriented)

    transport[(0, 1)] = cmath.exp(1j * 0.4)
    transport[(1, 0)] = cmath.exp(1j * 0.4)

    with pytest.raises(ValueError):
        validate_u1_transport(oriented, transport)
