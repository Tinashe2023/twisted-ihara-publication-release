import numpy as np

from src.analysis.tanner_fingerprints import (
    coefficient_difference_norm,
    complex_coefficients_to_json,
    fixed_su2_probe_holonomies,
    su2_twisted_fingerprint,
    su2_twisted_fingerprint_from_chord_holonomies,
)
from src.graphs.fundamental_holonomies import (
    deterministic_spanning_tree,
    fundamental_chords,
)
from src.graphs.parity_check_families import build_small_4_by_5_family
from src.graphs.tanner_graphs import build_tanner_graph


def test_fingerprint_has_expected_degree_bound() -> None:
    matrix = build_small_4_by_5_family()[0]
    graph = build_tanner_graph(matrix)

    fingerprint = su2_twisted_fingerprint(graph)

    # The graph has 12 undirected edges, hence 24 oriented edges.
    # An SU(2) block operator therefore has dimension 2 * 24 = 48.
    assert fingerprint.shape == (49,)


def test_fingerprint_from_chord_holonomies_matches_fixed_probe() -> None:
    matrix = build_small_4_by_5_family()[0]
    graph = build_tanner_graph(matrix)
    tree = deterministic_spanning_tree(graph)
    chord_holonomies = fixed_su2_probe_holonomies(
        fundamental_chords(graph, tree)
    )

    assert np.allclose(
        su2_twisted_fingerprint_from_chord_holonomies(
            graph,
            chord_holonomies,
            tree=tree,
        ),
        su2_twisted_fingerprint(graph),
    )


def test_fingerprint_json_conversion() -> None:
    coefficients = np.array(
        [
            1.0 + 0.0j,
            -2.0 + 3.0j,
        ]
    )

    converted = complex_coefficients_to_json(coefficients)

    assert converted == [
        {"real": 1.0, "imaginary": 0.0},
        {"real": -2.0, "imaginary": 3.0},
    ]


def test_fingerprint_distance_is_zero_for_same_polynomial() -> None:
    coefficients = np.array(
        [
            1.0 + 0.0j,
            2.0 - 1.0j,
        ]
    )

    assert coefficient_difference_norm(
        coefficients,
        coefficients,
    ) == 0.0


def test_family_fingerprints_are_computable() -> None:
    family = build_small_4_by_5_family()

    for matrix in family:
        graph = build_tanner_graph(matrix)
        fingerprint = su2_twisted_fingerprint(graph)

        assert np.isclose(fingerprint[0], 1.0)
