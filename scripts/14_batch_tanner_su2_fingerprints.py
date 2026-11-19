"""Compute fixed SU(2)-twisted fingerprints for the 4 by 5 Tanner family."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.graph_descriptors import graph_descriptor_record
from src.analysis.tanner_fingerprints import (
    coefficient_difference_norm,
    complex_coefficients_to_json,
    su2_twisted_fingerprint,
)
from src.graphs.parity_check_families import (
    binary_rank_mod_two,
    build_small_4_by_5_family,
)
from src.graphs.tanner_graphs import build_tanner_graph


family = build_small_4_by_5_family()

fingerprints = []
records = []

for index, matrix in enumerate(family, start=1):
    graph = build_tanner_graph(matrix)
    fingerprint = su2_twisted_fingerprint(graph)

    fingerprints.append(fingerprint)

    records.append(
        {
            "family_index": index,
            "parity_check_matrix": matrix.tolist(),
            "rank_mod_2": binary_rank_mod_two(matrix),
            "baseline_descriptors": graph_descriptor_record(graph),
            "su2_probe_polynomial_coefficients": (
                complex_coefficients_to_json(fingerprint)
            ),
        }
    )

print("Fixed SU(2) Tanner-family fingerprint experiment")
print()
print(f"Representatives: {len(records)}")
print()

for record in records:
    descriptors = record["baseline_descriptors"]

    print(f"Representative {record['family_index']}")
    print(f"rank over F_2: {record['rank_mod_2']}")
    print(f"girth: {descriptors['girth']}")
    print(f"cycle rank: {descriptors['cycle_rank']}")
    print()

print("Pairwise coefficient distances")
print()

for first_index in range(len(fingerprints)):
    for second_index in range(first_index + 1, len(fingerprints)):
        distance = coefficient_difference_norm(
            fingerprints[first_index],
            fingerprints[second_index],
        )

        print(
            f"Representative {first_index + 1} vs "
            f"Representative {second_index + 1}: "
            f"{distance:.12g}"
        )

output_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "tanner_family_4x5_su2_fingerprints.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(records, output_file, indent=2)

print()
print(f"Saved results to: {output_path}")
