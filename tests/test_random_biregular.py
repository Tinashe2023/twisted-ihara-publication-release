import numpy as np

from src.graphs.random_biregular import (
    random_square_regular_parity_check,
)


def test_random_square_regular_matrix_has_required_weights() -> None:
    generator = np.random.default_rng(20260924)

    matrix = random_square_regular_parity_check(
        size=6,
        degree=3,
        generator=generator,
    )

    assert matrix.shape == (6, 6)
    assert np.all(np.isin(matrix, [0, 1]))
    assert np.all(np.sum(matrix, axis=0) == 3)
    assert np.all(np.sum(matrix, axis=1) == 3)


def test_generation_is_reproducible_for_fixed_seed() -> None:
    first_generator = np.random.default_rng(12345)
    second_generator = np.random.default_rng(12345)

    first = random_square_regular_parity_check(
        size=6,
        degree=3,
        generator=first_generator,
    )

    second = random_square_regular_parity_check(
        size=6,
        degree=3,
        generator=second_generator,
    )

    assert np.array_equal(first, second)
