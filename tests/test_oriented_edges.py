import networkx as nx
import pytest

from src.ihara.oriented_edges import build_oriented_edge_index


def test_each_undirected_edge_has_two_orientations() -> None:
    graph = nx.cycle_graph(4)

    oriented = build_oriented_edge_index(graph)

    assert len(oriented.edges) == 2 * graph.number_of_edges()


def test_edge_reversal_is_an_involution() -> None:
    graph = nx.cycle_graph(5)

    oriented = build_oriented_edge_index(graph)

    for edge in oriented.edges:
        assert oriented.reverse(oriented.reverse(edge)) == edge


def test_both_orientations_exist() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b")

    oriented = build_oriented_edge_index(graph)

    assert ("a", "b") in oriented.index
    assert ("b", "a") in oriented.index


def test_origin_and_terminus() -> None:
    graph = nx.Graph()
    graph.add_edge(2, 7)

    oriented = build_oriented_edge_index(graph)
    edge = (2, 7)

    assert oriented.origin(edge) == 2
    assert oriented.terminus(edge) == 7


def test_directed_graph_is_rejected() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1)

    with pytest.raises(TypeError):
        build_oriented_edge_index(graph)


def test_self_loop_is_rejected() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 0)

    with pytest.raises(ValueError):
        build_oriented_edge_index(graph)
