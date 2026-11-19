"""Verify the exact U(1)-twisted determinant identity on C_4."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import sympy as sp

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.symbolic_twists import (
    build_symbolic_u1_twisted_nonbacktracking_matrix,
    characteristic_determinant,
    trivial_symbolic_transport,
)

graph = nx.cycle_graph(4)
oriented = build_oriented_edge_index(graph)

theta = sp.pi / 3
u = sp.symbols("u")

transport = trivial_symbolic_transport(oriented)

forward_phase = sp.cos(theta) + sp.I * sp.sin(theta)
reverse_phase = sp.cos(theta) - sp.I * sp.sin(theta)

transport[(0, 1)] = forward_phase
transport[(1, 0)] = reverse_phase

B_t = build_symbolic_u1_twisted_nonbacktracking_matrix(
    graph,
    transport,
    oriented,
)

actual_raw = characteristic_determinant(B_t, u)

expected = sp.expand(
    1 - 2 * sp.cos(theta) * u**4 + u**8
)

actual = sp.collect(
    sp.expand(sp.simplify(actual_raw.rewrite(sp.cos))),
    u,
)

difference = sp.simplify(
    sp.expand(actual.rewrite(sp.cos) - expected)
)

print("Graph: C_4")
print(f"Phase theta: {theta}")
print(f"det(I - u B_T): {actual}")
print(f"Expected:       {expected}")
print(f"Verified:       {difference == 0}")

