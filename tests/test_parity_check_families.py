import numpy as np

from src.graphs.parity_check_families import (
    binary_rank_mod_two,
    build_small_4_by_5_family,
    enumerate_binary_matrices_with_row_weights,
    filter_by_column_weights,
)
from src.graphs.tanner_graphs import build_tanner_graph


def test_binary_rank_mod_two() -> None:
    matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=int,
    )

    assert binary_rank_mod_two(matrix) == 2


def test_enumerated_matrices_have_requested_row_weights() -> None:
    matrices = enumerate_binary_matrices_with_row_weights(
        row_count=2,
        column_count=4,
        row_weights=(1, 3),
    )

    assert len(matrices) == 16

    for matrix in matrices:
        assert tuple(np.sum(matrix, axis=1)) == (1, 3)


def test_column_weight_filter() -> None:
    matrices = enumerate_binary_matrices_with_row_weights(
        row_count=2,
        column_count=3,
        row_weights=(2, 2),
    )

    filtered = filter_by_column_weights(
        matrices,
        target_column_weights=(1, 1, 2),
    )

    assert len(filtered) > 0

    for matrix in filtered:
        assert tuple(sorted(np.sum(matrix, axis=0))) == (1, 1, 2)


def test_small_4_by_5_family_has_required_properties() -> None:
    family = build_small_4_by_5_family()

    assert len(family) > 0

    for matrix in family:
        graph = build_tanner_graph(matrix)

        assert matrix.shape == (4, 5)
        assert tuple(np.sum(matrix, axis=1)) == (3, 3, 3, 3)
        assert tuple(sorted(np.sum(matrix, axis=0))) == (2, 2, 2, 3, 3)
        assert graph.number_of_nodes() == 9
        assert graph.number_of_edges() == 12
