import networkx as nx
import numpy as np
import pytest

from src.graphs.fundamental_holonomies import (
    build_gauge_fixed_su2_transport,
    deterministic_spanning_tree,
    fundamental_chords,
    fundamental_cycle_rank,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)


def small_tanner_graph() -> nx.Graph:
    parity_check = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
        ],
        dtype=int,
    )

    return build_tanner_graph(parity_check)


def test_spanning_tree_has_correct_size() -> None:
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)

    assert nx.is_tree(tree)
    assert tree.number_of_nodes() == graph.number_of_nodes()
    assert tree.number_of_edges() == graph.number_of_nodes() - 1


def test_chord_count_equals_cycle_rank() -> None:
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    assert len(chords) == fundamental_cycle_rank(graph)
    assert len(chords) == 3


def test_chords_are_not_tree_edges() -> None:
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    for first, second in chords:
        assert graph.has_edge(first, second)
        assert not tree.has_edge(first, second)


def test_gauge_fixed_transport_is_identity_on_tree_edges() -> None:
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    holonomies = {
        chords[0]: su2_z_rotation(0.2),
        chords[1]: su2_x_rotation(0.4),
        chords[2]: su2_z_rotation(0.7),
    }

    oriented, transport = build_gauge_fixed_su2_transport(
        graph,
        holonomies,
        tree,
    )

    for first, second in tree.edges():
        assert np.allclose(
            transport[(first, second)],
            np.eye(2),
        )

        assert np.allclose(
            transport[(second, first)],
            np.eye(2),
        )


def test_reverse_chord_transport_is_adjoint() -> None:
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    holonomies = {
        chords[0]: su2_z_rotation(0.2),
        chords[1]: su2_x_rotation(0.4),
        chords[2]: su2_z_rotation(0.7),
    }

    _, transport = build_gauge_fixed_su2_transport(
        graph,
        holonomies,
        tree,
    )

    for first, second in chords:
        assert np.allclose(
            transport[(second, first)],
            transport[(first, second)].conjugate().T,
        )


def test_missing_chord_holonomy_is_rejected() -> None:
    graph = small_tanner_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    incomplete_holonomies = {
        chords[0]: su2_z_rotation(0.2),
    }

    with pytest.raises(ValueError):
        build_gauge_fixed_su2_transport(
            graph,
            incomplete_holonomies,
            tree,
        )
