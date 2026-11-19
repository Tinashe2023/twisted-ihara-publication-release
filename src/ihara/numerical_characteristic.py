"""Numerical determinant-polynomial utilities."""

from __future__ import annotations

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def determinant_polynomial_coefficients(
    matrix: np.ndarray,
) -> np.ndarray:
    r"""
    Return coefficients of

    $$
    P(u)=\det(I-uM)
    $$

    in ascending powers of $u$:

    $$
    P(u)=c_0+c_1u+\cdots+c_Nu^N.
    $$

    The result is computed from eigenvalues $\lambda_j$ of $M$:

    $$
    \det(I-uM)=\prod_j(1-u\lambda_j).
    $$
    """
    eigenvalues = np.linalg.eigvals(matrix)

    coefficients = np.array([1.0 + 0.0j])

    for eigenvalue in eigenvalues:
        coefficients = np.convolve(
            coefficients,
            np.array([1.0 + 0.0j, -eigenvalue]),
        )

    return coefficients


def evaluate_determinant_polynomial(
    coefficients: np.ndarray,
    u: complex,
) -> complex:
    r"""
    Evaluate an ascending-coefficient polynomial

    $$
    \sum_{k=0}^{N}c_ku^k.
    $$
    """
    return np.polynomial.polynomial.polyval(u, coefficients)


def trim_small_coefficients(
    coefficients: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Replace numerically negligible coefficient components by zero."""
    cleaned = coefficients.copy()

    cleaned.real[np.abs(cleaned.real) < tolerance] = 0.0
    cleaned.imag[np.abs(cleaned.imag) < tolerance] = 0.0

    return cleaned


def determinant_polynomials_are_distinct(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    """Return True when two coefficient vectors differ beyond tolerance."""
    return not np.allclose(
        first,
        second,
        atol=tolerance,
        rtol=tolerance,
    )
