"""Apply controlled seven-chord SU(2) probes to an exact ordinary collision."""

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
    coefficient_difference_norm,
    complex_coefficients_to_json,
    su2_twisted_fingerprint_from_chord_holonomies,
)
from src.graphs.fundamental_holonomies import (
    deterministic_spanning_tree,
    fundamental_chords,
)
from src.graphs.tanner_graphs import build_tanner_graph


SEARCH_RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "random_6x6_regular_tanner_search.json"
)

FIRST_INDEX = 1
SECOND_INDEX = 2


def representative_matrix(records, index: int) -> np.ndarray:
    """Return the parity-check matrix for one stored representative."""
    for record in records:
        if record["index"] == index:
            return np.array(
                record["parity_check_matrix"],
                dtype=int,
            )

    raise ValueError(f"Representative {index} was not found.")


with SEARCH_RESULT_PATH.open("r", encoding="utf-8") as input_file:
    search_data = json.load(input_file)

records = search_data["representatives"]

first_matrix = representative_matrix(records, FIRST_INDEX)
second_matrix = representative_matrix(records, SECOND_INDEX)

first_graph = build_tanner_graph(first_matrix)
second_graph = build_tanner_graph(second_matrix)

first_tree = deterministic_spanning_tree(first_graph)
second_tree = deterministic_spanning_tree(second_graph)

first_chords = fundamental_chords(first_graph, first_tree)
second_chords = fundamental_chords(second_graph, second_tree)

if len(first_chords) != 7 or len(second_chords) != 7:
    raise RuntimeError(
        "Expected exactly seven fundamental chords for each graph."
    )

probe_specifications = seven_chord_probe_specifications()

experiment_records = []

print("Controlled SU(2) separation of exact ordinary collision")
print()
print(
    f"Ordinary collision pair: Representatives "
    f"{FIRST_INDEX} and {SECOND_INDEX}"
)
print(f"First graph chord count:  {len(first_chords)}")
print(f"Second graph chord count: {len(second_chords)}")
print()

for probe_name, specification in probe_specifications.items():
    first_holonomies = holonomies_from_axis_angle_specification(
        first_chords,
        specification,
    )

    second_holonomies = holonomies_from_axis_angle_specification(
        second_chords,
        specification,
    )

    first_fingerprint = su2_twisted_fingerprint_from_chord_holonomies(
        first_graph,
        first_holonomies,
        first_tree,
    )

    second_fingerprint = su2_twisted_fingerprint_from_chord_holonomies(
        second_graph,
        second_holonomies,
        second_tree,
    )

    distance = coefficient_difference_norm(
        first_fingerprint,
        second_fingerprint,
    )

    separated = not np.allclose(
        first_fingerprint,
        second_fingerprint,
        atol=1e-8,
        rtol=1e-8,
    )

    print(f"Probe: {probe_name}")
    print(f"Coefficient distance: {distance:.12g}")
    print(f"Separated numerically: {separated}")
    print()

    experiment_records.append(
        {
            "probe_name": probe_name,
            "specification": [
                {
                    "axis": axis,
                    "angle": angle,
                }
                for axis, angle in specification
            ],
            "coefficient_distance": distance,
            "separated_numerically": bool(separated),
            "first_coefficients": complex_coefficients_to_json(
                first_fingerprint
            ),
            "second_coefficients": complex_coefficients_to_json(
                second_fingerprint
            ),
        }
    )

output_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "exact_collision_su2_probe_experiment.json"
)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(
        {
            "first_representative": FIRST_INDEX,
            "second_representative": SECOND_INDEX,
            "ordinary_collision_verified_exactly": True,
            "probe_experiments": experiment_records,
        },
        output_file,
        indent=2,
    )

print(f"Saved results to: {output_path}")
