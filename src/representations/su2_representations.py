"""SU(2) edge transports for finite undirected graphs."""

from __future__ import annotations

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Hashable

import numpy as np

from src.ihara.oriented_edges import OrientedEdge, OrientedEdgeIndex


Vertex = Hashable
Matrix2 = np.ndarray


def su2_from_parameters(
    alpha: complex,
    beta: complex,
) -> Matrix2:
    """
    Construct an SU(2) matrix

        [[ alpha,  beta],
         [-beta*, alpha*]]

    provided |alpha|^2 + |beta|^2 = 1.
    """
    return np.array(
        [
            [alpha, beta],
            [-np.conjugate(beta), np.conjugate(alpha)],
        ],
        dtype=complex,
    )

def su2_x_rotation(theta: float) -> Matrix2:
    r"""
    Return the SU(2) rotation

    $$
    \exp(i\theta\sigma_x)
    =
    \begin{pmatrix}
    \cos(\theta) & i\sin(\theta)\\
    i\sin(\theta) & \cos(\theta)
    \end{pmatrix}.
    $$
    """
    return np.array(
        [
            [np.cos(theta), 1j * np.sin(theta)],
            [1j * np.sin(theta), np.cos(theta)],
        ],
        dtype=complex,
    )


def su2_z_rotation(theta: float) -> Matrix2:
    """
    Return the diagonal SU(2) matrix

        diag(exp(i theta), exp(-i theta)).
    """
    return np.array(
        [
            [np.exp(1j * theta), 0.0j],
            [0.0j, np.exp(-1j * theta)],
        ],
        dtype=complex,
    )


def identity_su2_transport(
    oriented: OrientedEdgeIndex,
) -> dict[OrientedEdge, Matrix2]:
    """Assign the 2 by 2 identity matrix to every oriented edge."""
    return {
        edge: np.eye(2, dtype=complex)
        for edge in oriented.edges
    }


def validate_su2_matrix(
    matrix: Matrix2,
    tolerance: float = 1e-10,
) -> None:
    """Validate that a matrix belongs to SU(2), up to numerical tolerance."""
    if matrix.shape != (2, 2):
        raise ValueError(
            f"Expected a 2 by 2 matrix, received shape {matrix.shape}."
        )

    identity = np.eye(2, dtype=complex)

    if not np.allclose(
        matrix.conjugate().T @ matrix,
        identity,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("Matrix is not unitary.")

    if not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("Matrix does not have determinant 1.")


def validate_su2_transport(
    oriented: OrientedEdgeIndex,
    transport: dict[OrientedEdge, Matrix2],
    tolerance: float = 1e-10,
) -> None:
    """
    Validate an SU(2) transport.

    Requirements:
    1. Every oriented edge is assigned a matrix in SU(2).
    2. Reverse edges carry inverse, equivalently adjoint, matrices.
    """
    for edge in oriented.edges:
        if edge not in transport:
            raise ValueError(f"Missing transport for oriented edge {edge}.")

        matrix = transport[edge]
        validate_su2_matrix(matrix, tolerance)

        reverse_edge = oriented.reverse(edge)

        if reverse_edge not in transport:
            raise ValueError(
                f"Missing transport for reverse edge {reverse_edge}."
            )

        if not np.allclose(
            transport[reverse_edge],
            matrix.conjugate().T,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError(
                "Reverse transport condition failed for "
                f"{edge} and {reverse_edge}."
            )
