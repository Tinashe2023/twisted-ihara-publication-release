"""Compare SU(2)-twisted determinant polynomials on one Tanner graph."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.graphs.fundamental_holonomies import (
    build_gauge_fixed_su2_transport,
    deterministic_spanning_tree,
    fundamental_chords,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.ihara.numerical_characteristic import (
    determinant_polynomial_coefficients,
    trim_small_coefficients,
)
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)
from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)


def polynomial_for_transport(graph, tree, chord_holonomies):
    """Return coefficients of det(I-uB_T) in ascending powers of u."""
    oriented, transport = build_gauge_fixed_su2_transport(
        graph,
        chord_holonomies,
        tree,
    )

    matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    coefficients = determinant_polynomial_coefficients(matrix)

    return trim_small_coefficients(coefficients)


def print_nonzero_coefficients(name, coefficients, tolerance=1e-9):
    """Print non-negligible polynomial coefficients."""
    print(name)

    for power, coefficient in enumerate(coefficients):
        if abs(coefficient) > tolerance:
            print(f"  u^{power:2d}: {coefficient}")

    print()


H = np.array(
    [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
    ],
    dtype=int,
)

graph = build_tanner_graph(H)
tree = deterministic_spanning_tree(graph)
chords = fundamental_chords(graph, tree)

trivial_holonomies = {
    chord: np.eye(2, dtype=complex)
    for chord in chords
}

commuting_holonomies = {
    chords[0]: su2_z_rotation(0.2),
    chords[1]: su2_z_rotation(0.4),
    chords[2]: su2_z_rotation(0.7),
}

noncommuting_holonomies = {
    chords[0]: su2_z_rotation(0.2),
    chords[1]: su2_x_rotation(0.4),
    chords[2]: su2_z_rotation(0.7),
}

trivial_polynomial = polynomial_for_transport(
    graph,
    tree,
    trivial_holonomies,
)

commuting_polynomial = polynomial_for_transport(
    graph,
    tree,
    commuting_holonomies,
)

noncommuting_polynomial = polynomial_for_transport(
    graph,
    tree,
    noncommuting_holonomies,
)

print("SU(2)-twisted Tanner graph experiment")
print()
print(f"Vertices: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Fundamental chords: {len(chords)}")
print()

print("Chord ordering:")
for index, chord in enumerate(chords):
    print(f"  H_{index + 1}: {chord}")

print()

print_nonzero_coefficients(
    "Trivial transport polynomial",
    trivial_polynomial,
)

print_nonzero_coefficients(
    "Commuting transport polynomial",
    commuting_polynomial,
)

print_nonzero_coefficients(
    "Noncommuting transport polynomial",
    noncommuting_polynomial,
)

print_nonzero_coefficients(
    "Commuting minus noncommuting",
    trim_small_coefficients(
        commuting_polynomial - noncommuting_polynomial
    ),
)
