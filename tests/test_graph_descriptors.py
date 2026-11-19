import networkx as nx

from src.analysis.graph_descriptors import (
    cycle_rank,
    graph_descriptor_record,
    graph_girth,
    sorted_degree_sequence,
)


def test_cycle_rank_for_tree_is_zero() -> None:
    graph = nx.path_graph(5)

    assert cycle_rank(graph) == 0


def test_cycle_rank_for_single_cycle_is_one() -> None:
    graph = nx.cycle_graph(6)

    assert cycle_rank(graph) == 1


def test_theta_graph_cycle_rank_is_two() -> None:
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

    assert cycle_rank(graph) == 2


def test_girth_for_tree_is_none() -> None:
    graph = nx.path_graph(5)

    assert graph_girth(graph) is None


def test_girth_for_cycle_graph() -> None:
    graph = nx.cycle_graph(6)

    assert graph_girth(graph) == 6


def test_girth_for_theta_graph_is_three() -> None:
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

    assert graph_girth(graph) == 3


def test_descriptor_record_for_cycle_graph() -> None:
    graph = nx.cycle_graph(4)

    record = graph_descriptor_record(graph)

    assert record["vertex_count"] == 4
    assert record["edge_count"] == 4
    assert record["connected"] is True
    assert record["bipartite"] is True
    assert record["cycle_rank"] == 1
    assert record["girth"] == 4
    assert record["degree_sequence"] == [2, 2, 2, 2]


def test_degree_sequence_is_sorted() -> None:
    graph = nx.star_graph(4)

    assert sorted_degree_sequence(graph) == (1, 1, 1, 1, 4)
