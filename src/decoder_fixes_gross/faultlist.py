"""Lists of fault-set orbits, the format shared by the list builders and the decoding runs."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from decoder_fixes_gross.symmetry import N_TRANSLATIONS


@dataclass(frozen=True, eq=False)
class FaultList:
    """Canonical representatives of fault-set orbits and the size of each orbit.

    Each row of ``sets`` is expected to be the canonical form of its orbit, as returned by
    :func:`decoder_fixes_gross.symmetry.canonicalize`; this is not checked, since checking
    it needs the model.

    Attributes
    ----------
    sets : numpy.ndarray
        int32 array of shape [N, w], w >= 1, of column indices; each row strictly
        increasing, rows in strictly increasing lexicographic order.
    orbit_sizes : numpy.ndarray
        int16 array of shape [N]; each entry a divisor of 72.

    Both arrays are made read-only in place, so :meth:`content_hash` stays valid for the
    life of the list.

    Raises
    ------
    ValueError
        If a dtype, shape or ordering condition above does not hold.
    """

    sets: NDArray[np.int32]
    orbit_sizes: NDArray[np.int16]

    def __post_init__(self) -> None:
        """Check dtypes, shapes, row order and orbit sizes, then freeze the arrays."""
        sets, sizes = self.sets, self.orbit_sizes
        if sets.dtype != np.int32 or sets.ndim != 2 or sets.shape[1] < 1:
            raise ValueError(
                f"sets must be int32 of shape [N, w] with w >= 1, got {sets.dtype} {sets.shape}"
            )
        if sizes.dtype != np.int16 or sizes.shape != (sets.shape[0],):
            raise ValueError(
                f"orbit_sizes must be int16 of shape ({sets.shape[0]},), "
                f"got {sizes.dtype} {sizes.shape}"
            )
        if np.any(sets < 0) or np.any(np.diff(sets, axis=1) <= 0):
            raise ValueError("each row of sets must be strictly increasing column indices")
        if not _rows_strictly_increasing(sets):
            raise ValueError("rows of sets must be in strictly increasing lexicographic order")
        if np.any(sizes <= 0) or np.any(N_TRANSLATIONS % sizes != 0):
            raise ValueError(f"every orbit size must divide {N_TRANSLATIONS}")
        sets.flags.writeable = False
        sizes.flags.writeable = False

    def save(self, path: Path) -> None:
        """Write the list to ``path`` as an uncompressed ``.npz`` with keys sets, orbit_sizes."""
        with path.open("wb") as f:
            np.savez(f, sets=self.sets, orbit_sizes=self.orbit_sizes)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a list written by :meth:`save`.

        Raises
        ------
        ValueError
            If the file lacks the key sets or orbit_sizes, or its arrays fail the checks of
            the class.
        """
        with np.load(path, allow_pickle=False) as data:
            missing = {"sets", "orbit_sizes"} - set(data.files)
            if missing:
                raise ValueError(f"{path} lacks the keys {sorted(missing)}")
            return cls(sets=data["sets"], orbit_sizes=data["orbit_sizes"])

    def content_hash(self) -> str:
        """Return the sha256 hex digest of the dtype, shape and bytes of both arrays.

        Equal contents give equal digests on every platform and in every session.
        """
        digest = hashlib.sha256()
        for name, array in (("sets", self.sets), ("orbit_sizes", self.orbit_sizes)):
            little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
            digest.update(f"{name}:{little.dtype.str}:{little.shape};".encode())
            digest.update(little.tobytes())
        return digest.hexdigest()


def _rows_strictly_increasing(sets: NDArray[np.int32]) -> bool:
    """Return whether each row is lexicographically greater than the one before."""
    diff = sets[1:].astype(np.int64) - sets[:-1]
    nonzero = diff != 0
    first = nonzero.argmax(axis=1)
    sign = diff[np.arange(len(diff)), first]
    return bool(np.all(nonzero.any(axis=1)) and np.all(sign > 0))
