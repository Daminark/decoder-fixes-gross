"""Tests of the fault-set probability, the floor estimate and the per-round rate."""

import itertools

import numpy as np
import pytest

from decoder_fixes_gross.dem import dem_at
from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.floor import (
    Estimate,
    Form,
    estimate,
    log_probabilities,
    orbit_masses,
    per_round,
)

TOY_PRIORS = np.array([0.1, 0.2, 0.3])


# Per subset of the toy priors: the exact-event probability, the occurrence probability (the
# priors in S), and their ratio, the product of (1 - prior) over the complement of S.
TOY_SUBSETS = [
    ((), 0.9 * 0.8 * 0.7, 1.0, 0.504),
    ((0,), 0.1 * 0.8 * 0.7, 0.1, 0.56),
    ((1,), 0.9 * 0.2 * 0.7, 0.2, 0.63),
    ((2,), 0.9 * 0.8 * 0.3, 0.3, 0.72),
    ((0, 1), 0.1 * 0.2 * 0.7, 0.1 * 0.2, 0.7),
    ((0, 2), 0.1 * 0.8 * 0.3, 0.1 * 0.3, 0.8),
    ((1, 2), 0.9 * 0.2 * 0.3, 0.2 * 0.3, 0.9),
    ((0, 1, 2), 0.1 * 0.2 * 0.3, 0.1 * 0.2 * 0.3, 1.0),
]


def _one(subset: tuple[int, ...]) -> np.typing.NDArray[np.int32]:
    return np.array([subset], dtype=np.int32).reshape(1, len(subset))


@pytest.mark.parametrize(("subset", "exact", "occurrence", "ratio"), TOY_SUBSETS)
def test_toy_set_probabilities_are_the_hand_worked_products(
    subset: tuple[int, ...], exact: float, occurrence: float, ratio: float
) -> None:
    found_exact = np.exp(log_probabilities(TOY_PRIORS, _one(subset), form="exact_event"))
    found_occurrence = np.exp(log_probabilities(TOY_PRIORS, _one(subset), form="occurrence"))
    np.testing.assert_allclose(found_exact, [exact])
    np.testing.assert_allclose(found_occurrence, [occurrence])
    np.testing.assert_allclose(found_exact / found_occurrence, [ratio])


def test_toy_exact_event_probabilities_of_all_subsets_sum_to_one() -> None:
    # The exact events are disjoint and exhaustive; the occurrence events are neither.
    total = 0.0
    for w in range(4):
        subsets = list(itertools.combinations(range(3), w))
        sets = np.array(subsets, dtype=np.int32).reshape(len(subsets), w)
        total += np.exp(log_probabilities(TOY_PRIORS, sets, form="exact_event")).sum()
    np.testing.assert_allclose(total, 1.0, rtol=1e-14)


def test_no_fault_probability_of_real_model_at_1e_3() -> None:
    # 2.02239e-5 is the product of (1 - prior) from the independent dem_trapping_sets build.
    empty = np.empty((1, 0), dtype=np.int32)
    np.testing.assert_allclose(
        np.exp(log_probabilities(dem_at(1e-3).priors, empty, form="exact_event")),
        [2.02239e-5],
        rtol=3e-6,
    )


def test_log_space_agrees_with_direct_product_on_real_sets() -> None:
    priors = dem_at(1e-3).priors
    rng = np.random.default_rng(8)
    for w in range(1, 5):
        sets = np.array([rng.choice(len(priors), w, replace=False) for _ in range(5)])
        occurrence = [np.prod(priors[s]) for s in sets]
        exact = [np.prod(priors[s]) * np.prod(np.delete(1 - priors, s)) for s in sets]
        pairs: list[tuple[Form, list[float]]] = [("occurrence", occurrence), ("exact_event", exact)]
        for form, direct in pairs:
            found = np.exp(log_probabilities(priors, sets, form=form))
            np.testing.assert_allclose(found, direct, rtol=1e-12)


def test_orbit_mass_is_orbit_size_times_set_probability() -> None:
    orbits = FaultList(
        sets=np.array([[0], [2]], dtype=np.int32), orbit_sizes=np.array([72, 36], dtype=np.int16)
    )
    np.testing.assert_allclose(
        orbit_masses(TOY_PRIORS, orbits, form="exact_event"), [72 * 0.056, 36 * 0.216]
    )
    np.testing.assert_allclose(
        orbit_masses(TOY_PRIORS, orbits, form="occurrence"), [72 * 0.1, 36 * 0.3]
    )


@pytest.mark.parametrize(
    ("priors", "sets", "match"),
    [
        (np.array([0.1, 0.0]), np.array([[0]]), "priors"),
        (np.array([0.1, 1.0]), np.array([[0]]), "priors"),
        (TOY_PRIORS, np.array([0, 1]), "two-dimensional"),
        (TOY_PRIORS, np.array([[0, 3]]), "column indices"),
        (TOY_PRIORS, np.array([[-1, 0]]), "column indices"),
        (TOY_PRIORS, np.array([[1, 0, 1]]), "distinct"),
    ],
)
def test_log_probabilities_rejects_invalid_input(
    priors: np.typing.NDArray[np.float64], sets: np.typing.NDArray[np.int64], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        log_probabilities(priors, sets, form="occurrence")


def test_log_probabilities_rejects_an_unknown_form() -> None:
    with pytest.raises(ValueError, match="form"):
        log_probabilities(TOY_PRIORS, _one((0,)), form="exact")  # type: ignore[arg-type]


def test_estimate_sums_failing_masses_and_their_squares() -> None:
    masses = np.array([1e-9, 2e-9, 3e-9, 4e-9])
    result = estimate(masses, np.array([True, False, True, False]))
    # 1 + 3 = 4; sqrt(1 + 9) = sqrt(10).
    np.testing.assert_allclose([result.value, result.error], [4e-9, 3.1622776601683795e-9])


def test_estimate_with_no_failure_is_zero() -> None:
    assert estimate(np.array([1e-9]), np.array([False])) == Estimate(0.0, 0.0)


@pytest.mark.parametrize(
    ("masses", "failed", "match"),
    [
        (np.array([1e-9, 2e-9]), np.array([True]), "shape"),
        (np.array([1e-9]), np.array([1]), "bool"),
        (np.array([-1e-9]), np.array([True]), "non-negative"),
    ],
)
def test_estimate_rejects_invalid_input(
    masses: np.typing.NDArray[np.float64], failed: np.typing.NDArray[np.bool_], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        estimate(masses, failed)


def test_per_round_inverts_twelve_rounds_of_a_known_rate() -> None:
    # 0.9**12 = 0.282429536481, so twelve rounds at 0.1 fail with probability 0.717570463519.
    np.testing.assert_allclose(per_round(0.717570463519), 0.1, rtol=1e-12)


def test_per_round_keeps_precision_at_small_probability() -> None:
    # Series: P / 12 + (11 / 288) P**2 = 1e-9 + 5.5e-18 at P = 1.2e-8. The naive
    # 1 - (1 - P)**(1 / 12) is off by 3e-8 relative here.
    np.testing.assert_allclose(per_round(1.2e-8), 1.0000000055e-9, rtol=1e-12)


def test_per_round_of_zero_is_zero() -> None:
    assert per_round(0.0) == 0.0


def test_estimate_converts_value_and_both_error_ends_per_round() -> None:
    result = Estimate(value=4e-9, error=3.1622776601683795e-9)
    # Each end divided by 12; the second-order terms are below 1e-18.
    assert result.per_round() == pytest.approx(
        (3.3333333e-10, 6.9810195e-11, 5.9685647e-10), rel=1e-7
    )


@pytest.mark.parametrize(("value", "error"), [(1e-9, 2e-9), (-1e-9, -2e-9)])
def test_estimate_rejects_an_error_outside_zero_to_value(value: float, error: float) -> None:
    with pytest.raises(ValueError, match=rf"value {value} and error {error}"):
        Estimate(value=value, error=error)


def test_estimate_per_round_is_none_where_an_end_reaches_one() -> None:
    value, low, high = Estimate(value=0.9, error=0.2).per_round()
    # 1 - 0.1**(1/12) and 1 - 0.3**(1/12); the high end 1.1 has no per-round rate.
    assert (value, low) == pytest.approx((0.1745958, 0.0954621), abs=1e-6)
    assert high is None


def test_estimate_per_round_is_none_at_exactly_one() -> None:
    # 0.8 + 0.2 is exactly 1.0 in float64.
    value, low, high = Estimate(value=0.8, error=0.2).per_round()
    assert value is not None
    assert low is not None
    assert high is None


def test_estimate_per_round_above_one_is_all_none() -> None:
    assert Estimate(value=1.5, error=0.2).per_round() == (None, None, None)


@pytest.mark.parametrize(("probability", "rounds"), [(-0.1, 12), (1.0, 12), (0.1, 0)])
def test_per_round_rejects_invalid_input(probability: float, rounds: int) -> None:
    with pytest.raises(ValueError, match=r"probability|rounds"):
        per_round(probability, rounds)
