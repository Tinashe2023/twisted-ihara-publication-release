"""Separate commuting and noncommuting SU(2) theta-graph holonomies."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.examples.theta_graph import (
    theta_graph,
    theta_transport,
)

from src.ihara.su2_twisted_bass_formula import (
    lhs_su2_twisted_bass_determinant,
)
from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)


graph = theta_graph()

alpha = 0.4
beta = 0.7
u = 0.2

A = su2_z_rotation(alpha)

B_commuting = su2_z_rotation(beta)
B_noncommuting = su2_x_rotation(beta)

commuting_oriented, commuting_transport = theta_transport(
    graph,
    A,
    B_commuting,
)

noncommuting_oriented, noncommuting_transport = theta_transport(
    graph,
    A,
    B_noncommuting,
)

commuting_determinant = lhs_su2_twisted_bass_determinant(
    graph,
    commuting_transport,
    u,
    commuting_oriented,
)

noncommuting_determinant = lhs_su2_twisted_bass_determinant(
    graph,
    noncommuting_transport,
    u,
    noncommuting_oriented,
)

print("Theta graph SU(2) separation")
print(f"u = {u}")
print()

print("Individual generator traces")
print(f"tr(A)                = {np.trace(A)}")
print(f"tr(B_commuting)      = {np.trace(B_commuting)}")
print(f"tr(B_noncommuting)   = {np.trace(B_noncommuting)}")
print()

print("Mixed-word traces")
print(f"tr(A B_commuting)    = {np.trace(A @ B_commuting)}")
print(f"tr(A B_noncommuting) = {np.trace(A @ B_noncommuting)}")
print()

print("Twisted determinants")
print(f"commuting:            {commuting_determinant}")
print(f"noncommuting:         {noncommuting_determinant}")
print(
    "absolute difference:  "
    f"{abs(commuting_determinant - noncommuting_determinant)}"
)
