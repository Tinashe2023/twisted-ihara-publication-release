import numpy as np
import networkx as nx

from src.ihara.nonbacktracking import build_nonbacktracking_matrix
from src.ihara.oriented_edges import build_oriented_edge_index


def test_nonbacktracking_matrix_has_expected_shape() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)

    matrix = build_nonbacktracking_matrix(graph, oriented)

    expected_size = 2 * graph.number_of_edges()
    assert matrix.shape == (expected_size, expected_size)


def test_immediate_reversal_is_excluded() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    matrix = build_nonbacktracking_matrix(graph, oriented)

    edge = (0, 1)
    reverse_edge = (1, 0)

    row = oriented.index[edge]
    column = oriented.index[reverse_edge]

    assert matrix[row, column] == 0


def test_valid_cycle_successor_is_allowed() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    matrix = build_nonbacktracking_matrix(graph, oriented)

    edge = (0, 1)
    successor = (1, 2)

    row = oriented.index[edge]
    column = oriented.index[successor]

    assert matrix[row, column] == 1


def test_noncontinuous_transition_is_excluded() -> None:
    graph = nx.cycle_graph(4)
    oriented = build_oriented_edge_index(graph)
    matrix = build_nonbacktracking_matrix(graph, oriented)

    edge = (0, 1)
    noncontinuous_edge = (2, 3)

    row = oriented.index[edge]
    column = oriented.index[noncontinuous_edge]

    assert matrix[row, column] == 0


def test_cycle_nonbacktracking_matrix_has_one_successor_per_edge() -> None:
    graph = nx.cycle_graph(5)
    matrix = build_nonbacktracking_matrix(graph)

    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)

    assert np.all(row_sums == 1)
    assert np.all(column_sums == 1)


def test_path_graph_has_terminal_oriented_edges_with_no_successor() -> None:
    graph = nx.path_graph(3)
    oriented = build_oriented_edge_index(graph)
    matrix = build_nonbacktracking_matrix(graph, oriented)

    terminal_edge = (1, 2)
    terminal_row = oriented.index[terminal_edge]

    assert matrix[terminal_row].sum() == 0
