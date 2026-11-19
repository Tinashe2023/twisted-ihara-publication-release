"""Compare ordinary and fixed-SU(2) Tanner graph fingerprints."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis.tanner_fingerprints import (
    coefficient_difference_norm,
    su2_twisted_fingerprint,
)
from src.graphs.parity_check_families import build_small_4_by_5_family
from src.graphs.tanner_graphs import build_tanner_graph
from src.ihara.nonbacktracking import build_nonbacktracking_matrix
from src.ihara.numerical_characteristic import (
    determinant_polynomial_coefficients,
    trim_small_coefficients,
)


def ordinary_fingerprint(graph) -> np.ndarray:
    r"""
    Return coefficients of the ordinary determinant polynomial

    $$
    P_{\mathrm{ord}}(u)=\det(I-uB).
    $$
    """
    nonbacktracking_matrix = build_nonbacktracking_matrix(graph)

    coefficients = determinant_polynomial_coefficients(
        nonbacktracking_matrix,
    )

    return trim_small_coefficients(coefficients)


family = build_small_4_by_5_family()

ordinary_fingerprints = []
su2_fingerprints = []

for matrix in family:
    graph = build_tanner_graph(matrix)

    ordinary_fingerprints.append(
        ordinary_fingerprint(graph)
    )

    su2_fingerprints.append(
        su2_twisted_fingerprint(graph)
    )

print("Ordinary versus SU(2)-twisted Tanner fingerprint comparison")
print()
print(f"Representatives: {len(family)}")
print()

print("Pairwise ordinary determinant-polynomial distances")
print()

for first_index in range(len(ordinary_fingerprints)):
    for second_index in range(first_index + 1, len(ordinary_fingerprints)):
        distance = coefficient_difference_norm(
            ordinary_fingerprints[first_index],
            ordinary_fingerprints[second_index],
        )

        print(
            f"Representative {first_index + 1} vs "
            f"Representative {second_index + 1}: "
            f"{distance:.12g}"
        )

print()
print("Pairwise SU(2)-twisted determinant-polynomial distances")
print()

for first_index in range(len(su2_fingerprints)):
    for second_index in range(first_index + 1, len(su2_fingerprints)):
        distance = coefficient_difference_norm(
            su2_fingerprints[first_index],
            su2_fingerprints[second_index],
        )

        print(
            f"Representative {first_index + 1} vs "
            f"Representative {second_index + 1}: "
            f"{distance:.12g}"
        )

print()
print("Interpretation")
print(
    "An ordinary distance near zero indicates an ordinary "
    "non-backtracking collision."
)
print(
    "A nonzero SU(2) distance for that same pair would demonstrate "
    "separation by the fixed SU(2) probe."
)
