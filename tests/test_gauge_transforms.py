import cmath

import networkx as nx
import numpy as np
import pytest

from src.ihara.oriented_edges import build_oriented_edge_index
from src.representations.gauge_transforms import (
    gauge_transform_u1_transport,
    validate_u1_vertex_gauge,
)
from src.representations.u1_representations import trivial_u1_transport


def test_unit_modulus_vertex_gauge_is_valid() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    gauge = {
        0: cmath.exp(1j * 0.1),
        1: cmath.exp(1j * 0.2),
        2: cmath.exp(1j * 0.3),
        3: cmath.exp(1j * 0.4),
    }

    validate_u1_vertex_gauge(oriented, gauge)


def test_nonunit_vertex_gauge_is_rejected() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)

    gauge = {
        0: 1.0 + 0.0j,
        1: 2.0 + 0.0j,
        2: 1.0 + 0.0j,
    }

    with pytest.raises(ValueError):
        validate_u1_vertex_gauge(oriented, gauge)


def test_missing_vertex_gauge_is_rejected() -> None:
    graph = nx.cycle_graph(3)
    oriented = build_oriented_edge_index(graph)

    gauge = {
        0: 1.0 + 0.0j,
        1: 1.0 + 0.0j,
    }

    with pytest.raises(ValueError):
        validate_u1_vertex_gauge(oriented, gauge)


def test_gauge_transform_of_trivial_transport() -> None:
    graph = nx.path_graph(3)
    oriented = build_oriented_edge_index(graph)

    transport = trivial_u1_transport(oriented)

    gauge = {
        0: cmath.exp(1j * 0.2),
        1: cmath.exp(1j * 0.7),
        2: cmath.exp(1j * 1.1),
    }

    transformed = gauge_transform_u1_transport(
        oriented,
        transport,
        gauge,
    )

    expected = gauge[0].conjugate() * gauge[1]

    assert np.isclose(transformed[(0, 1)], expected)
    assert np.isclose(transformed[(1, 0)], expected.conjugate())
