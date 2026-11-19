"""Print exact twisted determinant polynomials for theta-graph transports."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sympy as sp

from tests.test_symbolic_u1_theta import (
    theta_determinant,
    theta_graph,
)


u = sp.symbols("u")
graph = theta_graph()

examples = {
    "trivial transport": set(),
    "negative direct edge (0,1)": {(0, 1)},
    "negative upper-arm edge (0,2)": {(0, 2)},
    "negative direct and upper edges": {(0, 1), (0, 2)},
}

print("Theta graph")
print(f"|V| = {graph.number_of_nodes()}")
print(f"|E| = {graph.number_of_edges()}")
print(
    "Cycle rank = "
    f"{graph.number_of_edges() - graph.number_of_nodes() + 1}"
)

for name, negative_edges in examples.items():
    polynomial = theta_determinant(graph, negative_edges, u)

    print()
    print(name)
    print(f"  det(I - u B_T) = {sp.factor(polynomial)}")
