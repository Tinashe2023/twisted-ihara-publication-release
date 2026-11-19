"""Validate SU(2)-twisted separation by direct determinant evaluations."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis.su2_probe_families import (
    holonomies_from_axis_angle_specification,
    seven_chord_probe_specifications,
)
from src.analysis.tanner_fingerprints import (
    su2_twisted_fingerprint_from_chord_holonomies,
)
from src.graphs.fundamental_holonomies import (
    build_gauge_fixed_su2_transport,
    deterministic_spanning_tree,
    fundamental_chords,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)


SEARCH_RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "random_6x6_regular_tanner_search.json"
)

FIRST_INDEX = 1
SECOND_INDEX = 2

# Small values are chosen to keep direct determinant evaluation well-conditioned.
EVALUATION_POINTS = (0.03, 0.07, 0.11)


def representative_matrix(records, index: int) -> np.ndarray:
    """Extract a stored parity-check matrix by representative index."""
    for record in records:
        if record["index"] == index:
            return np.array(
                record["parity_check_matrix"],
                dtype=int,
            )

    raise ValueError(f"Representative {index} was not found.")


def twisted_matrix_from_probe(
    graph,
    tree,
    chords,
    specification,
) -> np.ndarray:
    """Build the SU(2)-twisted non-backtracking matrix for one probe."""
    holonomies = holonomies_from_axis_angle_specification(
        chords,
        specification,
    )

    oriented, transport = build_gauge_fixed_su2_transport(
        graph,
        holonomies,
        tree,
    )

    return build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )


def direct_determinant_value(
    twisted_matrix: np.ndarray,
    u_value: float,
) -> complex:
    r"""
    Evaluate

    $$
    \det(I-uB_T)
    $$

    directly as a matrix determinant.
    """
    identity = np.eye(
        twisted_matrix.shape[0],
        dtype=complex,
    )

    return np.linalg.det(identity - u_value * twisted_matrix)


def relative_difference(
    first: complex,
    second: complex,
) -> float:
    """Return a scale-aware difference between two complex values."""
    scale = max(
        1.0,
        abs(first),
        abs(second),
    )

    return float(abs(first - second) / scale)


with SEARCH_RESULT_PATH.open("r", encoding="utf-8") as input_file:
    search_data = json.load(input_file)

records = search_data["representatives"]

first_graph = build_tanner_graph(
    representative_matrix(records, FIRST_INDEX)
)

second_graph = build_tanner_graph(
    representative_matrix(records, SECOND_INDEX)
)

first_tree = deterministic_spanning_tree(first_graph)
second_tree = deterministic_spanning_tree(second_graph)

first_chords = fundamental_chords(first_graph, first_tree)
second_chords = fundamental_chords(second_graph, second_tree)

if len(first_chords) != 7 or len(second_chords) != 7:
    raise RuntimeError("Both graphs must have exactly seven chords.")

probe_specifications = seven_chord_probe_specifications()

results = []

print("Direct determinant validation of SU(2)-twisted separation")
print()
print(
    f"Ordinary collision pair: "
    f"Representatives {FIRST_INDEX} and {SECOND_INDEX}"
)
print()

for probe_name, specification in probe_specifications.items():
    first_matrix = twisted_matrix_from_probe(
        first_graph,
        first_tree,
        first_chords,
        specification,
    )

    second_matrix = twisted_matrix_from_probe(
        second_graph,
        second_tree,
        second_chords,
        specification,
    )

    first_coefficients = su2_twisted_fingerprint_from_chord_holonomies(
        first_graph,
        holonomies_from_axis_angle_specification(
            first_chords,
            specification,
        ),
        first_tree,
    )

    second_coefficients = su2_twisted_fingerprint_from_chord_holonomies(
        second_graph,
        holonomies_from_axis_angle_specification(
            second_chords,
            specification,
        ),
        second_tree,
    )

    print(f"Probe: {probe_name}")

    evaluation_records = []

    for u_value in EVALUATION_POINTS:
        first_direct = direct_determinant_value(
            first_matrix,
            u_value,
        )

        second_direct = direct_determinant_value(
            second_matrix,
            u_value,
        )

        first_polynomial = np.polynomial.polynomial.polyval(
            u_value,
            first_coefficients,
        )
        second_polynomial = np.polynomial.polynomial.polyval(
            u_value,
            second_coefficients,
        )

        first_reconstruction_error = relative_difference(
            first_direct,
            first_polynomial,
        )

        second_reconstruction_error = relative_difference(
            second_direct,
            second_polynomial,
        )

        separation = relative_difference(
            first_direct,
            second_direct,
        )

        print(f"  u = {u_value}")
        print(
            "    first direct/polynomial relative error: "
            f"{first_reconstruction_error:.3e}"
        )
        print(
            "    second direct/polynomial relative error: "
            f"{second_reconstruction_error:.3e}"
        )
        print(
            "    direct relative separation: "
            f"{separation:.12g}"
        )

        evaluation_records.append(
            {
                "u": u_value,
                "first_direct_real": float(np.real(first_direct)),
                "first_direct_imaginary": float(np.imag(first_direct)),
                "second_direct_real": float(np.real(second_direct)),
                "second_direct_imaginary": float(np.imag(second_direct)),
                "first_reconstruction_relative_error": (
                    first_reconstruction_error
                ),
                "second_reconstruction_relative_error": (
                    second_reconstruction_error
                ),
                "direct_relative_separation": separation,
            }
        )

    print()

    results.append(
        {
            "probe_name": probe_name,
            "evaluations": evaluation_records,
        }
    )

output_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "exact_collision_su2_direct_validation.json"
)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(
        {
            "first_representative": FIRST_INDEX,
            "second_representative": SECOND_INDEX,
            "evaluation_points": list(EVALUATION_POINTS),
            "probe_results": results,
        },
        output_file,
        indent=2,
    )

print(f"Saved validation data to: {output_path}")
