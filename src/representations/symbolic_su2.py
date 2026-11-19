"""Exact symbolic SU(2) matrices and transport validation helpers."""

from __future__ import annotations

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sympy as sp


SymbolicMatrix2 = sp.Matrix


def symbolic_su2_z_quarter_turn() -> SymbolicMatrix2:
    r"""
    Return the exact SU(2) matrix

    $$
    \begin{pmatrix}
    i & 0\\
    0 & -i
    \end{pmatrix}.
    $$
    """
    return sp.Matrix(
        [
            [sp.I, 0],
            [0, -sp.I],
        ]
    )


def symbolic_su2_x_quarter_turn() -> SymbolicMatrix2:
    r"""
    Return the exact SU(2) matrix

    $$
    \begin{pmatrix}
    0 & i\\
    i & 0
    \end{pmatrix}.
    $$
    """
    return sp.Matrix(
        [
            [0, sp.I],
            [sp.I, 0],
        ]
    )


def symbolic_identity_su2() -> SymbolicMatrix2:
    """Return the exact 2 by 2 identity matrix."""
    return sp.eye(2)


def symbolic_adjoint(matrix: SymbolicMatrix2) -> SymbolicMatrix2:
    """Return the conjugate transpose of a symbolic matrix."""
    return matrix.conjugate().T


def is_symbolic_su2(matrix: SymbolicMatrix2) -> bool:
    """
    Return True if the matrix is exactly in SU(2).

    Checks:

    $$
    U^\dagger U = I_2,
    \qquad
    \det(U)=1.
    $$
    """
    if matrix.shape != (2, 2):
        return False

    unitary_check = sp.simplify(
        matrix.conjugate().T * matrix - sp.eye(2)
    )

    determinant_check = sp.simplify(matrix.det() - 1)

    return unitary_check == sp.zeros(2) and determinant_check == 0
