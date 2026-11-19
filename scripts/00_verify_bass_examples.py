"""Print exact Bass-formula polynomials for benchmark graphs."""

import networkx as nx
import sympy as sp
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.ihara.bass_formula import (
    bass_formula_holds,
    lhs_bass_polynomial,
    rhs_bass_polynomial,
)

u = sp.symbols("u")

examples = {
    "C3": nx.cycle_graph(3),
    "C4": nx.cycle_graph(4),
    "C5": nx.cycle_graph(5),
    "P4": nx.path_graph(4),
    "K4": nx.complete_graph(4),
}

for name, graph in examples.items():
    lhs = sp.factor(lhs_bass_polynomial(graph, u))
    rhs = sp.factor(rhs_bass_polynomial(graph, u))

    print(f"\n{name}")
    print(f"  |V| = {graph.number_of_nodes()}, |E| = {graph.number_of_edges()}")
    print(f"  det(I - uB) = {lhs}")
    print(f"  Bass RHS    = {rhs}")
    print(f"  Verified    = {bass_formula_holds(graph)}")
