"""Print exact SU(2) theta-graph determinant-polynomial separation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sympy as sp

from src.examples.theta_graph import theta_graph
from src.ihara.symbolic_su2_nonbacktracking import (
    build_symbolic_su2_twisted_nonbacktracking_matrix,
    exact_twisted_determinant_polynomial,
    symbolic_su2_theta_transport,
)
from src.representations.symbolic_su2 import (
    symbolic_su2_x_quarter_turn,
    symbolic_su2_z_quarter_turn,
)


graph = theta_graph()
u = sp.symbols("u")

A = symbolic_su2_z_quarter_turn()
B_commuting = symbolic_su2_z_quarter_turn()
B_noncommuting = symbolic_su2_x_quarter_turn()

commuting_oriented, commuting_transport = symbolic_su2_theta_transport(
    graph,
    A,
    B_commuting,
)

noncommuting_oriented, noncommuting_transport = (
    symbolic_su2_theta_transport(
        graph,
        A,
        B_noncommuting,
    )
)

commuting_matrix = build_symbolic_su2_twisted_nonbacktracking_matrix(
    graph,
    commuting_transport,
    commuting_oriented,
)

noncommuting_matrix = build_symbolic_su2_twisted_nonbacktracking_matrix(
    graph,
    noncommuting_transport,
    noncommuting_oriented,
)

commuting_polynomial = exact_twisted_determinant_polynomial(
    commuting_matrix,
    u,
)

noncommuting_polynomial = exact_twisted_determinant_polynomial(
    noncommuting_matrix,
    u,
)

difference = sp.factor(
    commuting_polynomial - noncommuting_polynomial
)

print("Exact SU(2) theta-graph separation")
print()

print("A =")
sp.pprint(A)

print("\nB_commuting =")
sp.pprint(B_commuting)

print("\nB_noncommuting =")
sp.pprint(B_noncommuting)

print("\nTraces")
print(f"tr(A) = {sp.trace(A)}")
print(f"tr(B_commuting) = {sp.trace(B_commuting)}")
print(f"tr(B_noncommuting) = {sp.trace(B_noncommuting)}")
print(f"tr(A B_commuting) = {sp.trace(A * B_commuting)}")
print(f"tr(A B_noncommuting) = {sp.trace(A * B_noncommuting)}")

print("\nCommuting determinant polynomial:")
print(sp.factor(commuting_polynomial))

print("\nNoncommuting determinant polynomial:")
print(sp.factor(noncommuting_polynomial))

print("\nDifference:")
print(difference)

print("\nSeparated exactly:")
print(difference != 0)
