"""Compare full SU(2)-twisted determinant polynomials on a theta graph."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.ihara.numerical_characteristic import (
    determinant_polynomial_coefficients,
    trim_small_coefficients,
)
from src.ihara.su2_twisted_nonbacktracking import (
    build_su2_twisted_nonbacktracking_matrix,
)
from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)
from src.examples.theta_graph import (
    theta_graph,
    theta_transport,
)



def print_nonzero_coefficients(
    label: str,
    coefficients: np.ndarray,
    tolerance: float = 1e-9,
) -> None:
    """Print nonzero coefficients of det(I-uB_T)."""
    print(label)

    for power, coefficient in enumerate(coefficients):
        if abs(coefficient) > tolerance:
            print(f"  u^{power:2d}: {coefficient}")

    print()


graph = theta_graph()

alpha = 0.4
beta = 0.7

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

commuting_matrix = build_su2_twisted_nonbacktracking_matrix(
    graph,
    commuting_transport,
    commuting_oriented,
)

noncommuting_matrix = build_su2_twisted_nonbacktracking_matrix(
    graph,
    noncommuting_transport,
    noncommuting_oriented,
)

commuting_coefficients = trim_small_coefficients(
    determinant_polynomial_coefficients(commuting_matrix),
)

noncommuting_coefficients = trim_small_coefficients(
    determinant_polynomial_coefficients(noncommuting_matrix),
)

difference = trim_small_coefficients(
    commuting_coefficients - noncommuting_coefficients,
)

print("SU(2) theta-graph determinant-polynomial comparison")
print()
print(f"tr(A)                    = {np.trace(A)}")
print(f"tr(B_commuting)          = {np.trace(B_commuting)}")
print(f"tr(B_noncommuting)       = {np.trace(B_noncommuting)}")
print(f"tr(A B_commuting)        = {np.trace(A @ B_commuting)}")
print(f"tr(A B_noncommuting)     = {np.trace(A @ B_noncommuting)}")
print()

print_nonzero_coefficients(
    "Commuting transport: det(I-uB_T)",
    commuting_coefficients,
)

print_nonzero_coefficients(
    "Noncommuting transport: det(I-uB_T)",
    noncommuting_coefficients,
)

print_nonzero_coefficients(
    "Coefficientwise difference",
    difference,
)
