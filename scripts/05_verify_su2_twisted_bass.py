"""Numerically verify the SU(2)-twisted Bass determinant formula."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.su2_twisted_bass_formula import (
    lhs_su2_twisted_bass_determinant,
    rhs_su2_twisted_bass_determinant,
)
from src.representations.su2_representations import (
    identity_su2_transport,
    su2_from_parameters,
)


graph = nx.cycle_graph(4)
oriented = build_oriented_edge_index(graph)
transport = identity_su2_transport(oriented)

edge_matrix = su2_from_parameters(
    alpha=0.6 + 0.0j,
    beta=0.0 + 0.8j,
)

transport[(0, 1)] = edge_matrix
transport[(1, 0)] = edge_matrix.conjugate().T

u = 0.2

lhs = lhs_su2_twisted_bass_determinant(
    graph,
    transport,
    u,
    oriented,
)

rhs = rhs_su2_twisted_bass_determinant(
    graph,
    transport,
    u,
    oriented,
)

print("Graph: C_4")
print(f"u = {u}")
print("Transport on edge (0,1):")
print(edge_matrix)
print()
print(f"Edge-side determinant:   {lhs}")
print(f"Vertex-side determinant: {rhs}")
print(f"Absolute difference:     {abs(lhs - rhs)}")
