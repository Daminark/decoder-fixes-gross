"""Tests of the translations and the canonical form of fault sets."""

import igraph as ig
import numpy as np
import pytest
from gf2 import rank
from hypothesis import given
from hypothesis import strategies as st

from decoder_fixes_gross.dem import Dem, dem_at
from decoder_fixes_gross.symmetry import (
    N_TRANSLATIONS,
    canonicalize,
    column_permutations,
    detector_permutation,
    translate,
)

N_COLUMNS = 8784
# Translations checked in the fast suite: one along each axis of the torus and one along both.
FAST_TRANSLATIONS = {1, 6, 41}
ALL_TRANSLATIONS = [
    pytest.param(g, marks=() if g in FAST_TRANSLATIONS else pytest.mark.slow)
    for g in range(N_TRANSLATIONS)
]


@pytest.fixture(scope="module")
def dem() -> Dem:
    return dem_at(1e-3)


def test_translation_moves_corner_check_to_origin_in_its_layer() -> None:
    # g = (s, u) = (1, 1) sends (a, b) = (11, 5) in layer 2 to (0, 0) in layer 2.
    assert detector_permutation(7)[72 * 2 + 6 * 11 + 5] == 72 * 2


def test_translation_by_zero_is_identity() -> None:
    np.testing.assert_array_equal(detector_permutation(0), np.arange(936))
    np.testing.assert_array_equal(column_permutations()[0], np.arange(N_COLUMNS))


@pytest.mark.parametrize("g", [-1, 72])
def test_detector_permutation_rejects_element_outside_group(g: int) -> None:
    with pytest.raises(ValueError, match="g must be in"):
        detector_permutation(g)


@pytest.mark.parametrize("g", ALL_TRANSLATIONS)
def test_translation_permutes_columns_and_preserves_h_and_priors(dem: Dem, g: int) -> None:
    pi = column_permutations()[g]
    np.testing.assert_array_equal(np.sort(pi), np.arange(N_COLUMNS))
    h = dem.h.toarray()
    np.testing.assert_array_equal(h[np.ix_(detector_permutation(g), pi)], h)
    np.testing.assert_array_equal(dem.priors[pi], dem.priors)


@pytest.mark.parametrize("g", ALL_TRANSLATIONS)
def test_translated_observables_are_equivalent_modulo_h(dem: Dem, g: int) -> None:
    h, l = dem.h.toarray(), dem.l.toarray()
    l_g = l[:, column_permutations()[g]]
    # 942 = rank 930 of h plus 12 observables independent modulo h.
    assert rank(np.vstack([h, l])) == 942
    assert rank(np.vstack([h, l, l_g])) == 942
    assert rank(np.vstack([h, l_g])) == 942


def test_columns_form_122_orbits_of_size_72() -> None:
    canonical, orbit_sizes = canonicalize(np.arange(N_COLUMNS)[:, None])
    assert len(np.unique(canonical)) == 122
    assert set(orbit_sizes.tolist()) == {72}


def test_single_column_has_orbit_size_72() -> None:
    _, orbit_sizes = canonicalize(np.array([[0]]))
    assert orbit_sizes.tolist() == [72]


def test_pair_fixed_by_half_turn_has_orbit_size_36() -> None:
    # g = (6, 0) = 36 has order 2, so it swaps c and g c and the stabilizer is {0, 36}.
    c = 0
    pair = np.array([[c, column_permutations()[36, c]]])
    _, orbit_sizes = canonicalize(pair)
    assert orbit_sizes.tolist() == [36]


def _random_sets(seed: int, w: int, n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(N_COLUMNS, size=w, replace=False) for _ in range(n)])


@given(seed=st.integers(0, 2**32 - 1), w=st.integers(1, 5))
def test_canonical_form_is_invariant_under_translation(seed: int, w: int) -> None:
    sets = _random_sets(seed, w)
    g = np.random.default_rng(seed).integers(N_TRANSLATIONS, size=len(sets))
    canonical, orbit_sizes = canonicalize(sets)
    canonical_t, orbit_sizes_t = canonicalize(translate(sets, g))
    np.testing.assert_array_equal(canonical_t, canonical)
    np.testing.assert_array_equal(orbit_sizes_t, orbit_sizes)


@given(seed=st.integers(0, 2**32 - 1), w=st.integers(1, 5))
def test_canonical_form_is_idempotent(seed: int, w: int) -> None:
    canonical, orbit_sizes = canonicalize(_random_sets(seed, w))
    again, orbit_sizes_again = canonicalize(canonical)
    np.testing.assert_array_equal(again, canonical)
    np.testing.assert_array_equal(orbit_sizes_again, orbit_sizes)


@given(seed=st.integers(0, 2**32 - 1), w=st.integers(1, 5))
def test_canonical_form_is_smallest_image_by_brute_force(seed: int, w: int) -> None:
    sets = _random_sets(seed, w)
    canonical, orbit_sizes = canonicalize(sets)
    perms = column_permutations().tolist()
    for row, got, size in zip(sets.tolist(), canonical.tolist(), orbit_sizes, strict=True):
        images = {tuple(sorted(perm[c] for c in row)) for perm in perms}
        assert tuple(got) == min(images)
        assert size == len(images)


def test_translate_sorts_each_image() -> None:
    sets = np.array([[0, 1, 2], [5, 9, 100]], dtype=np.int32)
    images = translate(sets, np.array([3, 50]))
    assert images.dtype == np.int32
    assert np.all(np.diff(images, axis=1) > 0)


def test_translate_rejects_one_element_per_set_mismatch() -> None:
    with pytest.raises(ValueError, match="g must have shape"):
        translate(np.array([[0, 1]]), np.array([0, 1]))


def test_translate_rejects_element_outside_group() -> None:
    with pytest.raises(ValueError, match="g must be in"):
        translate(np.array([[0, 1]]), np.array([72]))


def test_canonicalize_rejects_column_outside_model() -> None:
    with pytest.raises(ValueError, match="column indices"):
        canonicalize(np.array([[0, N_COLUMNS]]))


def test_canonicalize_rejects_repeated_column() -> None:
    with pytest.raises(ValueError, match="repeats a column"):
        canonicalize(np.array([[4, 4]]))


def test_canonicalize_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        canonicalize(np.array([0, 1]))


@pytest.mark.slow
def test_automorphism_group_of_tanner_graph_with_priors_has_order_72(dem: Dem) -> None:
    # Vertices: detectors, then columns; columns are coloured by their prior, so an
    # automorphism must preserve h and the priors. Observables are not encoded, since they
    # are preserved only modulo h.
    h = dem.h.tocoo()
    n_detectors = h.shape[0]
    _, prior_class = np.unique(np.round(dem.priors, 12), return_inverse=True)
    graph = ig.Graph(
        n=n_detectors + h.shape[1],
        edges=list(zip(h.row.tolist(), (n_detectors + h.col).tolist(), strict=True)),
    )
    color = [0] * n_detectors + (1 + prior_class).tolist()
    assert graph.count_automorphisms(color=color) == 72
