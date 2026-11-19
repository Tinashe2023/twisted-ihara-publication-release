"""Compute baseline graph descriptors for a small Tanner graph."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis.graph_descriptors import graph_descriptor_record
from src.graphs.tanner_graphs import build_tanner_graph


H = np.array(
    [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
    ],
    dtype=int,
)

graph = build_tanner_graph(H)
record = graph_descriptor_record(graph)

output_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "small_tanner_graph_baseline.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(record, output_file, indent=2)

print("Small Tanner graph baseline descriptors")
print()

for key, value in record.items():
    print(f"{key}: {value}")

print()
print(f"Saved result to: {output_path}")
