"""The 72 translations of the gross code acting on detectors, columns and fault sets.

Detector ``72 * layer + 6 * a + b``, for layer 0 to 12, a in 0..11 and b in 0..5, is the
check at position (a, b) of the 12 x 6 torus of the bivariate bicycle code in that layer.
The group element ``g = 6 * s + u``, for s in 0..11 and u in 0..5, translates position
(a, b) to ((a + s) mod 12, (b + u) mod 6) in every layer. It acts on a column of the model
by mapping its detector set, whose image is again a column.

A fault set is a set of columns, stored as a sorted row of column indices. Its canonical
form is the lexicographically smallest sorted image over the 72 translations.
"""

from functools import cache

import numpy as np
from numpy.typing import NDArray

from decoder_fixes_gross.dem import dem_at

N_TRANSLATIONS = 72
_L, _M = 12, 6
_N_LAYERS = 13


def detector_permutation(g: int) -> NDArray[np.intp]:
    """Return the image of every detector under translation ``g``.

    Parameters
    ----------
    g : int
        Group element ``6 * s + u``, in [0, 72).

    Returns
    -------
    numpy.ndarray
        Integer array of shape [936] whose entry d is the image of detector d.

    Raises
    ------
    ValueError
        If ``g`` is not in [0, 72).
    """
    if not 0 <= g < N_TRANSLATIONS:
        raise ValueError(f"g must be in [0, {N_TRANSLATIONS}), got {g}")
    s, u = divmod(g, _M)
    layer, check = np.divmod(np.arange(_N_LAYERS * _L * _M), _L * _M)
    a, b = np.divmod(check, _M)
    return np.asarray(_L * _M * layer + _M * ((a + s) % _L) + (b + u) % _M, dtype=np.intp)


@cache
def column_permutations() -> NDArray[np.intp]:
    """Return the image of every column of the model under every translation.

    Returns
    -------
    numpy.ndarray
        Read-only integer array of shape [72, 8784]; row g is a permutation of the columns,
        and entry (g, c) is the column whose detector set is the image of that of column c.
    """
    h = dem_at(1e-3).h
    n_columns = h.shape[1]
    supports = [h.indices[h.indptr[c] : h.indptr[c + 1]] for c in range(n_columns)]
    column_of = {tuple(support.tolist()): c for c, support in enumerate(supports)}
    assert len(column_of) == n_columns, "detector sets of the columns are not distinct"
    images = np.empty((N_TRANSLATIONS, n_columns), dtype=np.intp)
    for g in range(N_TRANSLATIONS):
        sigma = detector_permutation(g)
        for c, support in enumerate(supports):
            # A KeyError here means translation g is not a symmetry of the model.
            images[g, c] = column_of[tuple(np.sort(sigma[support]).tolist())]
    images.flags.writeable = False
    return images


def translate(sets: NDArray[np.integer], g: NDArray[np.integer]) -> NDArray[np.integer]:
    """Return the image of each fault set under its translation, as a sorted row.

    Parameters
    ----------
    sets : numpy.ndarray
        Integer array of shape [N, w] of column indices in [0, 8784).
    g : numpy.ndarray
        Integer array of shape [N] of group elements in [0, 72); ``g[i]`` acts on ``sets[i]``.

    Returns
    -------
    numpy.ndarray
        Array of the dtype and shape of ``sets`` whose row i is the image of row i,
        sorted ascending.

    Raises
    ------
    ValueError
        If ``sets`` is not two-dimensional, ``g`` does not have one entry per row, or an
        entry of either is out of range.
    """
    sets, g = np.asarray(sets), np.asarray(g)
    _check_sets(sets)
    if g.shape != (sets.shape[0],):
        raise ValueError(f"g must have shape ({sets.shape[0]},), got {g.shape}")
    if g.size and not (g.min() >= 0 and g.max() < N_TRANSLATIONS):
        raise ValueError(f"g must be in [0, {N_TRANSLATIONS}), got values {g.min()}..{g.max()}")
    images: NDArray[np.intp] = column_permutations()[g[:, None], sets]
    return np.sort(images, axis=1).astype(sets.dtype)


def canonicalize(
    sets: NDArray[np.integer],
) -> tuple[NDArray[np.integer], NDArray[np.int16]]:
    """Return the canonical form and the orbit size of each fault set.

    Parameters
    ----------
    sets : numpy.ndarray
        Integer array of shape [N, w] of column indices in [0, 8784), with no column
        repeated within a row. Rows need not be sorted.

    Returns
    -------
    canonical : numpy.ndarray
        Array of the dtype and shape of ``sets``; row i is the lexicographically smallest
        sorted image of row i over the 72 translations, so two rows have the same canonical
        form exactly when they lie in the same orbit.
    orbit_sizes : numpy.ndarray
        int16 array of shape [N]; entry i is 72 divided by the number of translations that
        fix row i as a set.

    Raises
    ------
    ValueError
        If ``sets`` is not two-dimensional, an entry is out of range, or a row repeats a
        column.

    Notes
    -----
    All 72 sorted images are formed at once, so memory is proportional to 72 N w. The
    minimum is found by keeping, position by position, the images whose entry at that
    position is smallest among those still kept; the first image left is the
    lexicographic minimum over all 72 images.
    """
    sets = np.asarray(sets)
    _check_sets(sets)
    ordered = np.sort(sets, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("a row of sets repeats a column")
    # images[i, g] is the sorted image of row i under translation g.
    images = np.sort(column_permutations()[:, ordered].transpose(1, 0, 2), axis=2)
    kept = np.ones(images.shape[:2], dtype=bool)
    for k in range(images.shape[2]):
        entry = np.where(kept, images[:, :, k], np.iinfo(np.intp).max)
        kept &= entry == entry.min(axis=1, keepdims=True)
    first = kept.argmax(axis=1)
    canonical = images[np.arange(len(images)), first].astype(sets.dtype)
    stabilizer = (images == ordered[:, None, :]).all(axis=2).sum(axis=1)
    return canonical, (N_TRANSLATIONS // stabilizer).astype(np.int16)


def _check_sets(sets: NDArray[np.integer]) -> None:
    n_columns = column_permutations().shape[1]
    if sets.ndim != 2:
        raise ValueError(f"sets must be two-dimensional, got shape {sets.shape}")
    if sets.size and not (sets.min() >= 0 and sets.max() < n_columns):
        raise ValueError(
            f"sets must hold column indices in [0, {n_columns}), "
            f"got values {sets.min()}..{sets.max()}"
        )
