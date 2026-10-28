"""Inject an exact fault set into a model and classify the outcome of decoding it.

A fault set e is a set of physical columns of the model; its syndrome is h e. With e_hat
the decoder's correction and r = e xor e_hat, the outcome is "no solution" when h r is
nonzero, "wrong logical" when h r is zero and l r is nonzero, and "correct" otherwise.
"Correct" is the success criterion of Pacenti, Raveendran and Vasić [1]_, Sec. II.B: the
fault set and the correction lie in the same logical coset.

References
----------
.. [1] M. Pacenti, N. Raveendran and B. Vasić, "Trapping Sets of Detector Error Models",
   arXiv:2608.11516 (2026).
"""

import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from decoder_fixes_gross.decoders import Decoder
from decoder_fixes_gross.dem import Dem


class Outcome(IntEnum):
    """Outcome of one decode; the value is the code stored in run outputs."""

    CORRECT = 0
    WRONG_LOGICAL = 1
    NO_SOLUTION = 2


def syndrome(dem: Dem, faults: ArrayLike) -> NDArray[np.uint8]:
    """Return the syndrome h e of a fault set.

    Parameters
    ----------
    dem : Dem
        The model; only its first ``n_base`` columns can be injected.
    faults : array_like of int
        Distinct column indices in [0, n_base), in any order; may be empty.

    Returns
    -------
    numpy.ndarray
        uint8 of shape [n_detectors] with entries 0 and 1.

    Raises
    ------
    ValueError
        If ``faults`` is not a one-dimensional integer array, has an index outside
        [0, n_base), or repeats an index.
    """
    return _mod2(dem.h @ _fault_vector(dem, faults))


def classify(dem: Dem, faults: ArrayLike, correction: ArrayLike) -> Outcome:
    """Return the outcome of a correction for an injected fault set.

    Parameters
    ----------
    dem : Dem
        The model whose full ``h`` and ``l`` score the correction, augmented columns included.
    faults : array_like of int
        The injected set, as for `syndrome`.
    correction : array_like
        0/1 vector over all model columns, as returned by a decoder.

    Returns
    -------
    Outcome
        ``NO_SOLUTION`` if the correction does not reproduce the syndrome, else
        ``WRONG_LOGICAL`` if it differs from the fault set on an observable, else ``CORRECT``.

    Raises
    ------
    ValueError
        If ``faults`` is invalid as for `syndrome`, or ``correction`` is not a 0/1 vector of
        length n_columns.
    """
    n_columns = dem.h.shape[1]
    e_hat = np.asarray(correction)
    if e_hat.shape != (n_columns,) or not np.isin(e_hat, (0, 1)).all():
        raise ValueError(
            f"correction must be a 0/1 vector of length {n_columns}; got shape {e_hat.shape}"
        )
    residual = _fault_vector(dem, faults) ^ e_hat.astype(np.uint8)
    if _mod2(dem.h @ residual).any():
        return Outcome.NO_SOLUTION
    if _mod2(dem.l @ residual).any():
        return Outcome.WRONG_LOGICAL
    return Outcome.CORRECT


@dataclass(frozen=True)
class Trial:
    """One injected fault set, decoded and scored.

    Attributes
    ----------
    outcome : Outcome
        Outcome of the decoder's correction.
    seconds : float
        Wall-clock time of the decode call alone.
    iterations : int or None
        The decoder's iteration count, or None if it has none.
    """

    outcome: Outcome
    seconds: float
    iterations: int | None


def decode_injected(dem: Dem, decoder: Decoder, faults: ArrayLike) -> Trial:
    """Inject a fault set, decode its syndrome and classify the correction.

    Parameters
    ----------
    dem : Dem
        The model the decoder was built from.
    decoder : Decoder
        Decoder whose corrections are over the columns of ``dem``.
    faults : array_like of int
        The injected set, as for `syndrome`.

    Returns
    -------
    Trial
        Outcome, decode time and iteration count.

    Raises
    ------
    ValueError
        If ``faults`` is invalid as for `syndrome`.
    """
    detectors = syndrome(dem, faults)
    start = time.perf_counter()
    decoding = decoder.decode(detectors)
    seconds = time.perf_counter() - start
    return Trial(
        outcome=classify(dem, faults, decoding.correction),
        seconds=seconds,
        iterations=decoding.iterations,
    )


def _fault_vector(dem: Dem, faults: ArrayLike) -> NDArray[np.uint8]:
    """Validate a fault set and return its 0/1 vector over all model columns."""
    columns = np.asarray(faults)
    if columns.size == 0:
        columns = columns.astype(np.int64)
    if columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer):
        raise ValueError(
            f"faults must be a one-dimensional integer array; got shape {columns.shape}, "
            f"dtype {columns.dtype}"
        )
    if ((columns < 0) | (columns >= dem.n_base)).any():
        raise ValueError(
            f"faults must lie in [0, {dem.n_base}), the physical columns; got {columns}"
        )
    if np.unique(columns).size != columns.size:
        raise ValueError(f"faults must be distinct; got {columns}")
    vector = np.zeros(dem.h.shape[1], dtype=np.uint8)
    vector[columns] = 1
    return vector


def _mod2(counts: NDArray[np.generic]) -> NDArray[np.uint8]:
    # Sparse products of uint8 operands wrap modulo 256, which preserves parity.
    return (np.asarray(counts) % 2).astype(np.uint8)
