"""Tests of the GF(2) rank helper on hand-worked matrices."""

import numpy as np
import pytest
from gf2 import rank


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([[0, 0, 0], [0, 0, 0]], 0),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3),
        # The third row is the sum of the first two.
        ([[1, 1, 0], [0, 1, 1], [1, 0, 1]], 2),
        # Row 3 is row 1 plus row 2; row 4 is independent.
        ([[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 0, 0]], 3),
        # Repeated rows count once.
        ([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], 1),
    ],
)
def test_rank_of_small_matrix(rows: list[list[int]], expected: int) -> None:
    assert rank(np.array(rows, dtype=np.uint8)) == expected


def test_rank_counts_pivots_past_a_64_bit_word() -> None:
    matrix = np.zeros((3, 130), dtype=np.uint8)
    matrix[0, [3, 64]] = 1
    matrix[1, [64, 129]] = 1
    matrix[2, [3, 129]] = 1  # sum of rows 0 and 1
    assert rank(matrix) == 2


def test_rank_does_not_modify_its_input() -> None:
    matrix = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    rank(matrix)
    np.testing.assert_array_equal(matrix, [[1, 1], [1, 0]])
