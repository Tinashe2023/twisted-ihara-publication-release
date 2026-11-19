"""Random simple regular bipartite graphs represented by parity-check matrices."""

from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np


def random_square_regular_parity_check(
    size: int,
    degree: int,
    generator: np.random.Generator,
    max_attempts: int = 1_000,
) -> np.ndarray:
    r"""
    Generate a simple square $(d,d)$-regular binary parity-check matrix.

    The construction sums ``degree`` random permutation matrices and rejects
    any attempt that creates an entry larger than one.

    The output satisfies:

    $$
    H\in\{0,1\}^{n\times n},
    $$

    with every row and every column having weight ``degree``.
    """
    if size <= 0:
        raise ValueError("size must be positive.")

    if degree <= 0 or degree > size:
        raise ValueError("degree must satisfy 1 <= degree <= size.")

    for _ in range(max_attempts):
        matrix = np.zeros((size, size), dtype=int)

        for _ in range(degree):
            permutation = generator.permutation(size)

            for row_index, column_index in enumerate(permutation):
                matrix[row_index, column_index] += 1

        if np.all(matrix <= 1):
            return matrix

    raise RuntimeError(
        "Could not generate a simple regular parity-check matrix "
        f"after {max_attempts} attempts."
    )
