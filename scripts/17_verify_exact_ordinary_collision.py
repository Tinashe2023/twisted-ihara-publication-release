"""Verify the first candidate ordinary Tanner-graph collision exactly."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sympy as sp

from src.graphs.parity_check_families import (
    bipartition_preserving_isomorphic,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.ihara.exact_ordinary_bass import (
    exact_ordinary_bass_polynomial,
)


RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "random_6x6_regular_tanner_search.json"
)

FIRST_INDEX = 1
SECOND_INDEX = 2


def representative_matrix(records, index: int) -> np.ndarray:
    """Extract one parity-check matrix from the saved search records."""
    for record in records:
        if record["index"] == index:
            return np.array(
                record["parity_check_matrix"],
                dtype=int,
            )

    raise ValueError(f"Representative {index} was not found.")


with RESULT_PATH.open("r", encoding="utf-8") as input_file:
    search_data = json.load(input_file)

records = search_data["representatives"]

first_matrix = representative_matrix(records, FIRST_INDEX)
second_matrix = representative_matrix(records, SECOND_INDEX)

first_graph = build_tanner_graph(first_matrix)
second_graph = build_tanner_graph(second_matrix)

u = sp.symbols("u")

first_polynomial = exact_ordinary_bass_polynomial(
    first_graph,
    u,
)

second_polynomial = exact_ordinary_bass_polynomial(
    second_graph,
    u,
)

difference = sp.expand(first_polynomial - second_polynomial)

are_isomorphic = bipartition_preserving_isomorphic(
    first_graph,
    second_graph,
)

print("Exact ordinary Tanner-graph collision verification")
print()

print(f"Representative {FIRST_INDEX} parity-check matrix:")
print(first_matrix)
print()

print(f"Representative {SECOND_INDEX} parity-check matrix:")
print(second_matrix)
print()

print(
    "Bipartition-preserving Tanner-isomorphic: "
    f"{are_isomorphic}"
)
print()

print("First exact ordinary determinant polynomial:")
print(sp.factor(first_polynomial))
print()

print("Second exact ordinary determinant polynomial:")
print(sp.factor(second_polynomial))
print()

print("Exact polynomial difference:")
print(sp.factor(difference))
print()

print(f"Ordinary collision verified exactly: {difference == 0}")
print(f"Graphs are Tanner-inequivalent: {not are_isomorphic}")

verification_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "exact_ordinary_collision_1_2.json"
)

verification_data = {
    "first_index": FIRST_INDEX,
    "second_index": SECOND_INDEX,
    "tanner_isomorphic": bool(are_isomorphic),
    "ordinary_collision_verified_exactly": bool(difference == 0),
    "first_polynomial": str(sp.factor(first_polynomial)),
    "second_polynomial": str(sp.factor(second_polynomial)),
    "difference": str(sp.factor(difference)),
}

with verification_path.open("w", encoding="utf-8") as output_file:
    json.dump(verification_data, output_file, indent=2)

print()
print(f"Saved exact verification to: {verification_path}")
