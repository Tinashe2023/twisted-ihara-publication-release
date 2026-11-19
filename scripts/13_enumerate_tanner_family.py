"""Enumerate a small Tanner-inequivalent parity-check family."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis.graph_descriptors import graph_descriptor_record
from src.graphs.parity_check_families import (
    binary_rank_mod_two,
    build_small_4_by_5_family,
)
from src.graphs.tanner_graphs import build_tanner_graph


family = build_small_4_by_5_family()

records = []

print("Small 4 by 5 parity-check family")
print()
print(f"Tanner-inequivalent connected representatives: {len(family)}")
print()

for index, matrix in enumerate(family, start=1):
    graph = build_tanner_graph(matrix)
    descriptors = graph_descriptor_record(graph)

    record = {
        "family_index": index,
        "parity_check_matrix": matrix.tolist(),
        "rank_mod_2": binary_rank_mod_two(matrix),
        "classical_dimension_lower_bound": (
            matrix.shape[1] - binary_rank_mod_two(matrix)
        ),
        "descriptors": descriptors,
    }

    records.append(record)

    print(f"Representative {index}")
    print(matrix)
    print(f"rank over F_2: {record['rank_mod_2']}")
    print(
        "dimension n-rank: "
        f"{record['classical_dimension_lower_bound']}"
    )
    print(f"girth: {descriptors['girth']}")
    print(f"degree sequence: {descriptors['degree_sequence']}")
    print()

output_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "tanner_family_4x5.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(records, output_file, indent=2)

print(f"Saved {len(records)} records to:")
print(output_path)
