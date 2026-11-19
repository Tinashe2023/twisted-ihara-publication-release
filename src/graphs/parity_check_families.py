"""Enumeration and filtering of small binary parity-check matrix families."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from itertools import combinations, product

import networkx as nx
import numpy as np

from src.graphs.tanner_graphs import build_tanner_graph


def enumerate_binary_matrices_with_row_weights(
    row_count: int,
    column_count: int,
    row_weights: tuple[int, ...],
) -> list[np.ndarray]:
    """
    Enumerate binary matrices with prescribed row weights.

    This is intended only for small finite experiments.
    """
    if len(row_weights) != row_count:
        raise ValueError(
            "The number of row weights must equal the row count."
        )

    if any(
        weight < 0 or weight > column_count
        for weight in row_weights
    ):
        raise ValueError("Every row weight must lie between 0 and column_count.")

    row_options: list[list[np.ndarray]] = []

    for weight in row_weights:
        options: list[np.ndarray] = []

        for support in combinations(range(column_count), weight):
            row = np.zeros(column_count, dtype=int)
            row[list(support)] = 1
            options.append(row)

        row_options.append(options)

    matrices = []

    for rows in product(*row_options):
        matrices.append(np.vstack(rows))

    return matrices


def filter_by_column_weights(
    matrices: list[np.ndarray],
    target_column_weights: tuple[int, ...],
) -> list[np.ndarray]:
    """Keep only matrices whose sorted column weights match the target."""
    target = tuple(sorted(target_column_weights))

    return [
        matrix
        for matrix in matrices
        if tuple(sorted(np.sum(matrix, axis=0).tolist())) == target
    ]


def binary_rank_mod_two(matrix: np.ndarray) -> int:
    r"""
    Compute the rank of a binary matrix over $\mathbb{F}_2$.

    Row reduction is performed using XOR operations.
    """
    reduced = np.asarray(matrix, dtype=np.uint8).copy() % 2

    row_count, column_count = reduced.shape
    pivot_row = 0

    for column in range(column_count):
        candidates = np.flatnonzero(reduced[pivot_row:, column])

        if len(candidates) == 0:
            continue

        selected_row = pivot_row + int(candidates[0])

        if selected_row != pivot_row:
            reduced[[pivot_row, selected_row]] = reduced[
                [selected_row, pivot_row]
            ]

        for row in range(row_count):
            if row != pivot_row and reduced[row, column] == 1:
                reduced[row] ^= reduced[pivot_row]

        pivot_row += 1

        if pivot_row == row_count:
            break

    return pivot_row


def bipartition_preserving_isomorphic(
    first: nx.Graph,
    second: nx.Graph,
) -> bool:
    """
    Test Tanner-graph isomorphism while preserving variable/check vertex type.
    """
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        first,
        second,
        node_match=lambda left, right: (
            left.get("bipartite") == right.get("bipartite")
        ),
    )

    return matcher.is_isomorphic()


def nonisomorphic_tanner_representatives(
    matrices: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Remove matrices whose Tanner graphs differ only by check/variable relabeling.

    A Weisfeiler-Lehman hash is used as a fast pre-grouping mechanism.
    Exact
    bipartition-preserving graph isomorphism is then used within each group.
    """
    representatives: list[np.ndarray] = []
    representatives_by_hash: dict[str, list[np.ndarray]] = {}

    for matrix in matrices:
        graph = build_tanner_graph(matrix)

        if not nx.is_connected(graph):
            continue

        graph_hash = nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="bipartite",
        )

        candidates = representatives_by_hash.setdefault(graph_hash, [])

        if any(
            bipartition_preserving_isomorphic(
                graph,
                build_tanner_graph(candidate),
            )
            for candidate in candidates
        ):
            continue

        candidates.append(matrix)
        representatives.append(matrix)

    return representatives


def build_small_4_by_5_family() -> list[np.ndarray]:
    r"""
    Build connected Tanner-inequivalent representatives from the family:

    $$
    H\in\{0,1\}^{4\times5},
    $$

    with row weights $(3,3,3,3)$ and sorted column weights $(2,2,2,3,3)$.
    """
    all_matrices = enumerate_binary_matrices_with_row_weights(
        row_count=4,
        column_count=5,
        row_weights=(3, 3, 3, 3),
    )

    weight_filtered = filter_by_column_weights(
        all_matrices,
        target_column_weights=(2, 2, 2, 3, 3),
    )

    return nonisomorphic_tanner_representatives(weight_filtered)
