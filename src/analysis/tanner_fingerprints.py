"""Reproducible SU(2)-twisted fingerprints for Tanner graphs."""

from __future__ import annotations

import numpy as np

from src.graphs.fundamental_holonomies import (
    build_gauge_fixed_su2_transport,
    deterministic_spanning_tree,
    fundamental_chords,
)
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


def fixed_su2_probe_holonomies(
    chords,
) -> dict:
    r"""
    Assign a deterministic noncommuting SU(2) probe to graph chords.

    The assignment cycles through rotations about the $z$ and $x$ axes:

    $$
    R_z(0.2),\quad R_x(0.4),\quad R_z(0.7),\quad R_x(0.9).
    $$

    This is a reproducible chosen representation probe. It is not claimed to
    be a canonical graph-only invariant.
    """
    prototype_holonomies = [
        su2_z_rotation(0.2),
        su2_x_rotation(0.4),
        su2_z_rotation(0.7),
        su2_x_rotation(0.9),
    ]

    if len(chords) > len(prototype_holonomies):
        raise ValueError(
            "The current fixed probe supports at most four chords."
        )

    return {
        chord: prototype_holonomies[index]
        for index, chord in enumerate(chords)
    }


def su2_twisted_fingerprint(
    graph,
    tolerance: float = 1e-9,
) -> np.ndarray:
    r"""
    Return the cleaned coefficient vector of

    $$
    P_T(u)=\det(I-uB_T)
    $$

    under the fixed noncommuting SU(2) probe.
    """
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    chord_holonomies = fixed_su2_probe_holonomies(chords)

    oriented, transport = build_gauge_fixed_su2_transport(
        graph,
        chord_holonomies,
        tree,
    )

    twisted_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    coefficients = determinant_polynomial_coefficients(twisted_matrix)

    return trim_small_coefficients(
        coefficients,
        tolerance=tolerance,
    )


def su2_twisted_fingerprint_from_chord_holonomies(
    graph,
    chord_holonomies: dict,
    tree=None,
    tolerance: float = 1e-9,
) -> np.ndarray:
    r"""
    Compute the coefficient vector of

    $$
    P_T(u)=\det(I-uB_T)
    $$

    for an explicit gauge-fixed fundamental-chord holonomy assignment.
    """
    if tree is None:
        tree = deterministic_spanning_tree(graph)

    oriented, transport = build_gauge_fixed_su2_transport(
        graph,
        chord_holonomies,
        tree,
    )

    twisted_matrix = build_su2_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )

    coefficients = determinant_polynomial_coefficients(
        twisted_matrix,
    )

    return trim_small_coefficients(
        coefficients,
        tolerance=tolerance,
    )


def complex_coefficients_to_json(
    coefficients: np.ndarray,
    tolerance: float = 1e-9,
) -> list[dict[str, float]]:
    """
    Convert complex coefficients into JSON-safe real/imaginary records.
    """
    cleaned = trim_small_coefficients(
        coefficients,
        tolerance=tolerance,
    )

    return [
        {
            "real": float(np.real(coefficient)),
            "imaginary": float(np.imag(coefficient)),
        }
        for coefficient in cleaned
    ]


def coefficient_difference_norm(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    r"""
    Return the Euclidean coefficient distance:

    $$
    \|P_1-P_2\|_2.
    $$
    """
    if first.shape != second.shape:
        raise ValueError(
            "Coefficient vectors must have the same shape."
        )

    return float(np.linalg.norm(first - second))
