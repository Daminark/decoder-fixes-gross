"""GF(2) rank for tests: bit-packed forward elimination."""

import numpy as np
from numpy.typing import NDArray


def rank(matrix: NDArray[np.uint8]) -> int:
    """Return the rank over GF(2) of a dense 0/1 matrix of shape [m, n]."""
    m, n = matrix.shape
    packed = np.packbits(matrix, axis=1, bitorder="little")
    padded = np.zeros((m, -(-n // 64) * 8), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    rows = padded.view(np.uint64)
    r = 0
    for i in range(m):
        nonzero = np.flatnonzero(rows[i])
        if nonzero.size == 0:
            continue
        word = int(nonzero[0])
        value = int(rows[i, word])
        bit = np.uint64((value & -value).bit_length() - 1)
        # Clear the pivot bit of row i from every later row; later pivots then differ from it.
        below = rows[i + 1 :]
        has_bit = ((below[:, word] >> bit) & np.uint64(1)).astype(bool)
        below[has_bit] ^= rows[i]
        r += 1
    return r
