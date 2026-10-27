"""Leafless elementary trapping sets (LETS) of the gross-code model and their support lists.

The input is the dataset of Pacenti, Raveendran and Vasić [1]_: one file ``a_<a>_b_<b>.csv``
per (a, b) class, each row a LETS as 0-based column indices of the model in
:mod:`decoder_fixes_gross.dem`. A LETS of class (a, b) has a columns, and the checks its
columns touch have induced degree at most 2, exactly b of them odd.

Rows are reduced to translation orbits with :func:`decoder_fixes_gross.symmetry.canonicalize`.
An orbit keeps its full size, from the group, even when some of its rows are missing from a
file, so a file cut short still yields every orbit it touches. Each orbit also carries the
number of rows of its file that fall in it, which is what the counts of [1]_ (Table I) sum.

References
----------
.. [1] M. Pacenti, N. Raveendran and B. Vasić, arXiv:2608.11516 (2026); dataset at Zenodo,
   record 21878685.
"""

import itertools
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from decoder_fixes_gross.faultlist import FaultList, hash_arrays
from decoder_fixes_gross.symmetry import N_TRANSLATIONS, canonicalize

logger = logging.getLogger(__name__)

# Rows canonicalized at once; canonicalize holds about 72 x 8 bytes per entry of a chunk.
_CHUNK_ROWS = 50_000
# Distinct orbits buffered from several chunks before they are merged into one table.
_MERGE_ROWS = 1_000_000
_FILE_NAME = re.compile(r"a_(\d+)_b_(\d+)\.csv")
# Supports of weight 1 to 4 make up the estimate; weight-5 LETS are stored but not injected.
MAX_SUPPORT_WEIGHT = 4
# A LETS fails when one of its supports of these weights fails.
INCIDENCE_WEIGHTS = (3, 4)
_ARRAY_NAMES = ("sets", "orbit_sizes", "published_counts", "u3", "u4")
_ARRAY_KEY = re.compile(rf"a(\d+)_b(\d+)/({'|'.join(_ARRAY_NAMES)})")


@dataclass(frozen=True, eq=False)
class LetsOrbits:
    """Translation orbits of the LETS of one (a, b) class.

    Attributes
    ----------
    a : int
        Number of columns of each LETS.
    b : int
        Number of odd-degree checks of each LETS.
    orbits : FaultList
        Canonical representatives, of width ``a``, and orbit sizes from the group.
    published_counts : numpy.ndarray
        int16 array of shape [N]; entry i is the number of rows of the source file in orbit
        i, between 1 and its orbit size.

    The array is made read-only in place.

    Raises
    ------
    ValueError
        If ``orbits`` does not have width ``a``, or ``published_counts`` has the wrong dtype
        or shape or an entry outside [1, orbit size].
    """

    a: int
    b: int
    orbits: FaultList
    published_counts: NDArray[np.int16]

    def __post_init__(self) -> None:
        """Check widths and counts against the orbits, then freeze the counts."""
        sets, sizes, counts = self.orbits.sets, self.orbits.orbit_sizes, self.published_counts
        if sets.shape[1] != self.a:
            raise ValueError(f"orbits must have width a = {self.a}, got {sets.shape[1]}")
        if counts.dtype != np.int16 or counts.shape != sizes.shape:
            raise ValueError(
                f"published_counts must be int16 of shape {sizes.shape}, "
                f"got {counts.dtype} {counts.shape}"
            )
        if np.any(counts < 1) or np.any(counts > sizes):
            raise ValueError(
                "every published count must be between 1 and its orbit size; a count above "
                "the orbit size means the file repeats a row"
            )
        counts.flags.writeable = False

    @property
    def rows(self) -> int:
        """Number of rows of the source file, the sum of ``published_counts``."""
        return int(self.published_counts.sum(dtype=np.int64))

    @property
    def missing(self) -> int:
        """Number of orbit members absent from the source file."""
        return int(self.orbits.orbit_sizes.sum(dtype=np.int64)) - self.rows


@dataclass(frozen=True)
class LetsFile:
    """What :func:`read_lets_file` read from one file.

    Attributes
    ----------
    orbits : LetsOrbits
        The orbits of the rows read.
    dropped_lines : int
        1 if the file ended in an unterminated line, which was not read, else 0.
    """

    orbits: LetsOrbits
    dropped_lines: int


def read_lets_file(path: Path) -> LetsFile:
    """Read one class file of the dataset and reduce its rows to translation orbits.

    Parameters
    ----------
    path : pathlib.Path
        File named ``a_<a>_b_<b>.csv`` whose lines each hold a comma-separated column
        indices in [0, 8784), no index repeated within a line. An unterminated last line, as
        left by a file cut mid-row, is dropped and counted.

    Returns
    -------
    LetsFile
        The orbits of the rows, sorted lexicographically by representative, and the number
        of dropped lines.

    Raises
    ------
    ValueError
        If the file name does not have the form above, a row does not have a entries or
        does not parse as integers, or a count of rows in one orbit exceeds its size.
    """
    key = _class_of(path)
    if key is None:
        raise ValueError(f"path must name a file a_<a>_b_<b>.csv, got {path.name}")
    a, b = key
    buffered: list[_OrbitTable] = []
    merged = _empty_table(a)
    dropped_lines = 0
    with path.open() as f:
        while lines := list(itertools.islice(f, _CHUNK_ROWS)):
            if not lines[-1].endswith("\n"):
                # Only the last line of a file can be unterminated.
                lines.pop()
                dropped_lines += 1
            if lines:
                buffered.append(
                    _reduce_chunk(np.loadtxt(lines, np.int32, delimiter=",", ndmin=2), a)
                )
            if sum(len(t.sets) for t in buffered) > _MERGE_ROWS:
                merged = _merge([merged, *buffered])
                buffered = []
    merged = _merge([merged, *buffered])
    if np.any(merged.counts > N_TRANSLATIONS):
        # Checked before the cast to int16; LetsOrbits checks counts against orbit sizes.
        raise ValueError(
            f"{path.name} has more than {N_TRANSLATIONS} rows in one orbit, so it repeats a row"
        )
    orbits = LetsOrbits(
        a=a,
        b=b,
        orbits=FaultList(sets=merged.sets, orbit_sizes=merged.orbit_sizes),
        published_counts=merged.counts.astype(np.int16),
    )
    logger.info(
        "%s: %d rows, %d orbits, %d missing, %d dropped lines",
        path.name,
        orbits.rows,
        len(merged.sets),
        orbits.missing,
        dropped_lines,
    )
    return LetsFile(orbits=orbits, dropped_lines=dropped_lines)


def read_lets_files(directory: Path) -> list[LetsFile]:
    """Read every class file ``a_<a>_b_<b>.csv`` in ``directory``, in increasing (a, b).

    Raises
    ------
    ValueError
        If ``directory`` holds no class file, or as :func:`read_lets_file`.
    """
    by_class = {key: path for path in directory.iterdir() if (key := _class_of(path)) is not None}
    if not by_class:
        raise ValueError(f"{directory} holds no file a_<a>_b_<b>.csv")
    return [read_lets_file(by_class[key]) for key in sorted(by_class)]


@dataclass(frozen=True)
class InducedClasses:
    """The class of each fault set in the Tanner graph of a check matrix.

    Attributes
    ----------
    b : numpy.ndarray
        int64 array of shape [N]; entry i is the number of checks that an odd number of the
        columns of set i touch.
    max_check_degree : numpy.ndarray
        int64 array of shape [N]; entry i is the largest number of columns of set i that
        touch one check. A set is elementary when this is at most 2.
    """

    b: NDArray[np.int64]
    max_check_degree: NDArray[np.int64]


def induced_classes(h: sp.csc_matrix, sets: NDArray[np.integer]) -> InducedClasses:
    """Return the number of odd-degree checks and the largest check degree of each set.

    Parameters
    ----------
    h : scipy.sparse.csc_matrix
        Check matrix with entries 0 and 1.
    sets : numpy.ndarray
        Integer array of shape [N, w] of column indices of ``h``, no index repeated within a
        row.

    Returns
    -------
    InducedClasses
        For each row, the checks' degrees in the subgraph induced by its columns, summarized
        as the number that are odd and the largest.

    Raises
    ------
    ValueError
        If ``sets`` is not a two-dimensional integer array or holds an index outside the
        columns of ``h``.
    """
    sets = np.asarray(sets)
    if not np.issubdtype(sets.dtype, np.integer) or sets.ndim != 2:
        raise ValueError(
            f"sets must be a two-dimensional integer array, got {sets.dtype} {sets.shape}"
        )
    n_columns = h.shape[1]
    if sets.size and not (sets.min() >= 0 and sets.max() < n_columns):
        raise ValueError(
            f"sets must hold column indices in [0, {n_columns}), "
            f"got values {sets.min()}..{sets.max()}"
        )
    n, w = sets.shape
    # Row i of the indicator selects the columns of set i; entry (i, d) of the product with
    # h^T is the number of those columns that touch check d.
    indicator = sp.csr_matrix(
        (np.ones(n * w, np.int32), sets.ravel(), np.arange(0, n * w + 1, w)), shape=(n, n_columns)
    )
    degrees = sp.csr_matrix(indicator @ h.T.astype(np.int32))
    odd = degrees.copy()
    odd.data %= 2
    return InducedClasses(
        b=np.asarray(odd.sum(axis=1), np.int64).reshape(-1),
        max_check_degree=degrees.max(axis=1).toarray().astype(np.int64).reshape(-1),
    )


@dataclass(frozen=True, eq=False)
class Supports:
    """The support lists U_1 to U_4 and the incidence of each LETS orbit into U_3 and U_4.

    Attributes
    ----------
    lists : dict of int to FaultList
        Key w, for w from 1 to 4, gives U_w: the orbits of the distinct weight-w subsets of
        every LETS with at least w columns, in lexicographic order of representative.
    incidence : dict of (int, int) to dict of int to numpy.ndarray
        Key (a, b), then w in (3, 4), gives an int32 array of shape [N, C(a, w)], N the
        orbits of the class. Entry (i, j) is the row of U_w holding the orbit of the j-th
        w-subset of representative i, subsets taken in the order of
        ``itertools.combinations(range(a), w)``. The width is 0 when a < w.
    """

    lists: dict[int, FaultList]
    incidence: dict[tuple[int, int], dict[int, NDArray[np.int32]]]


def build_supports(classes: Sequence[LetsOrbits]) -> Supports:
    """Build U_1 to U_4 from the orbits of LETS classes, with the incidence into U_3 and U_4.

    A subset of a translate is a translate of a subset, so the subsets of the representatives
    reach every orbit of subsets of every LETS in a class. A LETS with exactly w columns is a
    weight-w subset of itself, and subsets shared by several classes are listed once.

    Parameters
    ----------
    classes : sequence of LetsOrbits
        Classes with distinct (a, b).

    Returns
    -------
    Supports
        The lists and the incidence of every class.

    Raises
    ------
    ValueError
        If two classes have the same (a, b).
    """
    keys = [(c.a, c.b) for c in classes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"classes must have distinct (a, b), got {keys}")
    lists: dict[int, FaultList] = {}
    incidence: dict[tuple[int, int], dict[int, NDArray[np.int32]]] = {
        (c.a, c.b): {w: np.zeros((len(c.orbits.sets), 0), np.int32) for w in INCIDENCE_WEIGHTS}
        for c in classes
    }
    for w in range(1, MAX_SUPPORT_WEIGHT + 1):
        members = [c for c in classes if c.a >= w]
        parts = [_canonical_subsets(c.orbits.sets, w) for c in members]
        sets = np.concatenate([np.zeros((0, w), np.int32), *(canonical for canonical, _ in parts)])
        sizes = np.concatenate([np.zeros(0, np.int16), *(sizes for _, sizes in parts)])
        unique, first, inverse = np.unique(sets, axis=0, return_index=True, return_inverse=True)
        lists[w] = FaultList(sets=unique.astype(np.int32), orbit_sizes=sizes[first])
        if w not in INCIDENCE_WEIGHTS or not members:
            continue
        ends = np.cumsum([len(canonical) for canonical, _ in parts])[:-1]
        rows = np.split(inverse.reshape(-1).astype(np.int32), ends)
        for c, r in zip(members, rows, strict=True):
            incidence[(c.a, c.b)][w] = r.reshape(len(c.orbits.sets), comb(c.a, w))
    return Supports(lists=lists, incidence=incidence)


@dataclass(frozen=True, eq=False)
class LetsList:
    """Every LETS class with its incidence into U_3 and U_4, the stored form of the LETS.

    Attributes
    ----------
    classes : tuple of LetsOrbits
        Classes with distinct (a, b), in increasing order of (a, b).
    incidence : dict of (int, int) to dict of int to numpy.ndarray
        As the ``incidence`` field of :class:`Supports`, with one entry per class.

    The incidence arrays are made read-only in place.

    Raises
    ------
    ValueError
        If the classes are not in strictly increasing order of (a, b), or the incidence does
        not have, for each class and w in (3, 4), an int32 array of shape [N, C(a, w)].
    """

    classes: tuple[LetsOrbits, ...]
    incidence: dict[tuple[int, int], dict[int, NDArray[np.int32]]]

    def __post_init__(self) -> None:
        """Check the class order and the incidence shapes, then freeze the incidence."""
        keys = [(c.a, c.b) for c in self.classes]
        if keys != sorted(set(keys)):
            raise ValueError(f"classes must be in strictly increasing order of (a, b), got {keys}")
        if set(self.incidence) != set(keys):
            raise ValueError("incidence must have one entry per class")
        for c in self.classes:
            for w in INCIDENCE_WEIGHTS:
                rows = self.incidence[(c.a, c.b)].get(w)
                shape = (len(c.orbits.sets), comb(c.a, w))
                if rows is None or rows.dtype != np.int32 or rows.shape != shape:
                    raise ValueError(
                        f"incidence of class ({c.a}, {c.b}) into U_{w} must be int32 of shape "
                        f"{shape}, got {None if rows is None else (rows.dtype, rows.shape)}"
                    )
                rows.flags.writeable = False

    def save(self, path: Path) -> None:
        """Write the list to ``path`` as an uncompressed ``.npz``.

        Each class has the keys ``a<a>_b<b>/sets``, ``/orbit_sizes``, ``/published_counts``,
        ``/u3`` and ``/u4``.
        """
        with path.open("wb") as f:
            # Typed as Any: mypy matches a str-keyed dict against savez's allow_pickle flag.
            arrays: dict[str, Any] = dict(self._named_arrays())
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a list written by :meth:`save`.

        Raises
        ------
        ValueError
            If a key does not have the form above, a class lacks one of the five keys, or the
            arrays fail the checks of this class, :class:`LetsOrbits` or
            :class:`~decoder_fixes_gross.faultlist.FaultList`.
        """
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        by_class: dict[tuple[int, int], dict[str, NDArray[Any]]] = {}
        for key, array in arrays.items():
            match = _ARRAY_KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"{path} has a key {key!r} not of the form a<a>_b<b>/<name>")
            by_class.setdefault((int(match[1]), int(match[2])), {})[match[3]] = array
        classes, incidence = [], {}
        for (a, b), named in sorted(by_class.items()):
            missing = set(_ARRAY_NAMES) - set(named)
            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)} for class ({a}, {b})")
            orbits = FaultList(sets=named["sets"], orbit_sizes=named["orbit_sizes"])
            classes.append(
                LetsOrbits(a=a, b=b, orbits=orbits, published_counts=named["published_counts"])
            )
            incidence[(a, b)] = {3: named["u3"], 4: named["u4"]}
        return cls(classes=tuple(classes), incidence=incidence)

    def content_hash(self) -> str:
        """Return the sha256 hex digest of every array, by key in the order of :meth:`save`.

        The digest uses :func:`~decoder_fixes_gross.faultlist.hash_arrays`, as fault lists do.
        """
        return hash_arrays(self._named_arrays())

    def _named_arrays(self) -> list[tuple[str, NDArray[np.generic]]]:
        named: list[tuple[str, NDArray[np.generic]]] = []
        for c in self.classes:
            prefix = f"a{c.a}_b{c.b}/"
            rows = self.incidence[(c.a, c.b)]
            named += [
                (prefix + "sets", c.orbits.sets),
                (prefix + "orbit_sizes", c.orbits.orbit_sizes),
                (prefix + "published_counts", c.published_counts),
                (prefix + "u3", rows[3]),
                (prefix + "u4", rows[4]),
            ]
        return named


def _class_of(path: Path) -> tuple[int, int] | None:
    """Return (a, b) of a file named ``a_<a>_b_<b>.csv``, or None for any other name."""
    match = _FILE_NAME.fullmatch(path.name)
    return None if match is None else (int(match[1]), int(match[2]))


def _canonical_subsets(
    sets: NDArray[np.int32], w: int
) -> tuple[NDArray[np.int32], NDArray[np.int16]]:
    """Return the canonical forms and orbit sizes of the w-subsets of each row, row by row."""
    positions = list(itertools.combinations(range(sets.shape[1]), w))
    subsets = sets[:, positions].reshape(-1, w)
    canonical, sizes = np.empty_like(subsets), np.empty(len(subsets), np.int16)
    for start in range(0, len(subsets), _CHUNK_ROWS):
        chunk = slice(start, start + _CHUNK_ROWS)
        canonical[chunk], sizes[chunk] = canonicalize(subsets[chunk])
    return canonical, sizes


@dataclass(frozen=True)
class _OrbitTable:
    """Distinct canonical rows in lexicographic order, with orbit sizes and row counts."""

    sets: NDArray[np.int32]
    orbit_sizes: NDArray[np.int16]
    counts: NDArray[np.int64]


def _empty_table(width: int) -> _OrbitTable:
    return _OrbitTable(np.zeros((0, width), np.int32), np.zeros(0, np.int16), np.zeros(0, np.int64))


def _reduce_chunk(rows: NDArray[np.int32], a: int) -> _OrbitTable:
    if rows.shape[1] != a:
        raise ValueError(f"rows of class a = {a} must have {a} entries, got {rows.shape[1]}")
    canonical, orbit_sizes = canonicalize(rows)
    return _merge(
        [_OrbitTable(canonical.astype(np.int32), orbit_sizes, np.ones(len(rows), np.int64))]
    )


def _merge(tables: list[_OrbitTable]) -> _OrbitTable:
    """Merge tables of the same width, adding the counts of rows that coincide."""
    sets = np.concatenate([t.sets for t in tables])
    unique, first, inverse = np.unique(sets, axis=0, return_index=True, return_inverse=True)
    counts = np.zeros(len(unique), np.int64)
    np.add.at(counts, inverse.reshape(-1), np.concatenate([t.counts for t in tables]))
    orbit_sizes = np.concatenate([t.orbit_sizes for t in tables])[first]
    return _OrbitTable(unique.astype(np.int32), orbit_sizes, counts)
