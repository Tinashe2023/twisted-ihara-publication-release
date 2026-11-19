"""Verify the exact U(1)-twisted Bass formula on C_4."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import sympy as sp

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.symbolic_twists import trivial_symbolic_transport
from src.ihara.twisted_bass_formula import (
    lhs_twisted_bass_polynomial,
    rhs_twisted_bass_polynomial,
)


graph = nx.cycle_graph(4)
oriented = build_oriented_edge_index(graph)

u = sp.symbols("u")
theta = sp.pi / 3

transport = trivial_symbolic_transport(oriented)
transport[(0, 1)] = sp.cos(theta) + sp.I * sp.sin(theta)
transport[(1, 0)] = sp.cos(theta) - sp.I * sp.sin(theta)

lhs = lhs_twisted_bass_polynomial(graph, transport, u, oriented)
rhs = rhs_twisted_bass_polynomial(graph, transport, u)

difference = sp.simplify(sp.cancel(lhs - rhs))

print("Graph: C_4")
print(f"Phase theta: {theta}")
print(f"det(I - u B_T): {sp.collect(sp.expand(lhs), u)}")
print(f"Vertex-side RHS: {sp.collect(sp.expand(rhs), u)}")
print(f"Verified: {difference == 0}")
