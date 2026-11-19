"""Inspect fundamental-cycle holonomies on a Tanner graph."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.graphs.fundamental_holonomies import (
    build_gauge_fixed_su2_transport,
    deterministic_spanning_tree,
    fundamental_chords,
    fundamental_cycle_rank,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)


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

holonomies = {
    chords[0]: su2_z_rotation(0.2),
    chords[1]: su2_x_rotation(0.4),
    chords[2]: su2_z_rotation(0.7),
}

oriented, transport = build_gauge_fixed_su2_transport(
    graph,
    holonomies,
    tree,
)

print("Tanner graph fundamental holonomies")
print()

print(f"Vertices: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Cycle rank: {fundamental_cycle_rank(graph)}")
print()

print("Spanning-tree edges:")
for edge in sorted(tree.edges(), key=repr):
    print(f"  {edge}")

print()

print("Fundamental chords and assigned SU(2) holonomies:")
for chord in chords:
    print(f"\nChord: {chord}")
    print(transport[chord])

print()
print(f"Oriented-edge count: {len(oriented.edges)}")
