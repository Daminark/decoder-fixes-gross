"""Probability of a fault set and the known-list floor estimated from it.

Two probabilities of a set S of columns are used (ADR 0005). The occurrence probability, the
product of the priors of the columns in S, is the probability that every fault in S occurs,
whatever else occurs; it is the headline floor, and it is consistent with Fig. 1 of Pacenti,
Raveendran and Vasić [1]_. The exact-event probability, their Eq. (1), multiplies it by the product
of one minus the prior over every other column, so it is the probability that exactly the
faults in S occur. The floor of a decoder is the sum of either over the supports it fails on;
ADR 0003 gives the estimator from one random translate per orbit and its sampling error. The
per-round rate is Eq. (7) of [1]_.

References
----------
.. [1] M. Pacenti, N. Raveendran and B. Vasić, arXiv:2608.11516 (2026).
"""

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

from decoder_fixes_gross.faultlist import FaultList

# Syndrome-extraction rounds of the stored circuit.
ROUNDS = 12

Form = Literal["occurrence", "exact_event"]
FORMS: tuple[Form, ...] = get_args(Form)


def log_probabilities(
    priors: NDArray[np.float64], sets: NDArray[np.integer], *, form: Form
) -> NDArray[np.float64]:
    """Return the natural log of the probability of each row S of ``sets``.

    Parameters
    ----------
    priors : numpy.ndarray
        float64 of shape [n]; the probability of each column, each in (0, 1).
    sets : numpy.ndarray
        Integer array of shape [N, w], w >= 0; each row distinct column indices in [0, n).
    form : {"occurrence", "exact_event"}
        "occurrence": the product of the priors in S. "exact_event": that product times the
        product of (1 - prior) over every column outside S, Eq. (1) of arXiv:2608.11516.

    Returns
    -------
    numpy.ndarray
        float64 of shape [N].

    Raises
    ------
    ValueError
        If ``form`` is not one of the two forms, ``priors`` is not one-dimensional with
        entries in (0, 1), or ``sets`` is not two-dimensional with distinct indices in [0, n)
        in each row.

    Notes
    -----
    The exact-event form is computed as the sum over S of log(pi / (1 - pi)) plus the sum over
    all columns of log(1 - pi), with ``log1p`` for the second term, so the factor common to
    every set is computed once and each row costs w additions.
    """
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    if priors.ndim != 1 or not np.all((priors > 0) & (priors < 1)):
        raise ValueError(
            "priors must be one-dimensional with entries in (0, 1), got shape "
            f"{priors.shape} with range [{priors.min(initial=np.inf)}, "
            f"{priors.max(initial=-np.inf)}]"
        )
    if sets.ndim != 2:
        raise ValueError(f"sets must be two-dimensional, got shape {sets.shape}")
    if sets.size and (sets.min() < 0 or sets.max() >= len(priors)):
        raise ValueError(f"sets must hold column indices in [0, {len(priors)})")
    if np.any(np.diff(np.sort(sets, axis=1), axis=1) == 0):
        raise ValueError("each row of sets must hold distinct column indices")
    result: NDArray[np.float64]
    if form == "occurrence":
        result = np.log(priors)[sets].sum(axis=1)
        return result
    log_none = np.log1p(-priors)
    log_odds = np.log(priors) - log_none
    result = log_none.sum() + log_odds[sets].sum(axis=1)
    return result


def orbit_masses(
    priors: NDArray[np.float64], orbits: FaultList, *, form: Form
) -> NDArray[np.float64]:
    """Return orbit size times the probability of S for each orbit, S its representative.

    Translations preserve the priors, so every member of an orbit has the probability of its
    representative.

    Parameters
    ----------
    priors : numpy.ndarray
        As for `log_probabilities`, with an entry for every column ``orbits`` uses.
    orbits : FaultList
        The orbits.
    form : {"occurrence", "exact_event"}
        As for `log_probabilities`.

    Returns
    -------
    numpy.ndarray
        float64 of shape [N].
    """
    log_p = log_probabilities(priors, orbits.sets, form=form)
    return orbits.orbit_sizes.astype(np.float64) * np.exp(log_p)


@dataclass(frozen=True)
class Estimate:
    """A floor per experiment and its sampling error (ADR 0003).

    Attributes
    ----------
    value : float
        Sum of the masses of the failing orbits.
    error : float
        Square root of the sum of their squared masses; never above ``value``.

    Raises
    ------
    ValueError
        If ``error`` is not in [0, ``value``].
    """

    value: float
    error: float

    def __post_init__(self) -> None:
        """Check that the error is in [0, value]."""
        # No tolerance: estimate() cannot produce error > value by rounding, since a single
        # term gives sqrt(fl(m * m)) == m exactly in binary round-to-nearest.
        if not 0 <= self.error <= self.value:
            raise ValueError(
                f"error must be in [0, value], got value {self.value} and error {self.error}"
            )

    def per_round(self) -> tuple[float | None, float | None, float | None]:
        """Return the value and the two ends of value -+ error, each converted by `per_round`.

        Returns
        -------
        tuple of float or None
            The per-round value, low end and high end. Each is None when the quantity it
            converts is 1 or more, where the per-round rate is undefined; an occurrence floor
            (ADR 0005) is not a probability and can reach 1.
        """

        def converted(end: float) -> float | None:
            return None if end >= 1 else per_round(end)

        return (
            converted(self.value),
            converted(self.value - self.error),
            converted(self.value + self.error),
        )


def estimate(masses: NDArray[np.float64], failed: NDArray[np.bool_]) -> Estimate:
    """Return the floor over the failing orbits and its sampling error.

    Parameters
    ----------
    masses : numpy.ndarray
        float64 of shape [N], non-negative; orbit size times Pr(S; p) of each orbit.
    failed : numpy.ndarray
        bool of shape [N]; whether the decode of the orbit's translate failed.

    Returns
    -------
    Estimate
        The sum of ``masses`` over failing orbits and the square root of the sum of their
        squares.

    Raises
    ------
    ValueError
        If the shapes differ, ``failed`` is not bool, or a mass is negative.
    """
    if failed.dtype != np.bool_ or masses.shape != failed.shape:
        raise ValueError(
            f"failed must be bool of the shape of masses {masses.shape}, "
            f"got {failed.dtype} {failed.shape}"
        )
    if np.any(masses < 0):
        raise ValueError("masses must be non-negative")
    terms = masses[failed]
    return Estimate(value=float(terms.sum()), error=float(np.sqrt(np.sum(terms**2))))


def per_round(probability: float, rounds: int = ROUNDS) -> float:
    """Return the per-round rate 1 - (1 - P)^(1/rounds), Eq. (7) of arXiv:2608.11516.

    Parameters
    ----------
    probability : float
        Failure probability P of an experiment of ``rounds`` rounds, in [0, 1).
    rounds : int, optional
        Number of rounds; positive.

    Returns
    -------
    float
        The rate, computed as ``-expm1(log1p(-P) / rounds)`` so that it keeps full relative
        precision when P is small.

    Raises
    ------
    ValueError
        If ``probability`` is not in [0, 1) or ``rounds`` is not positive.
    """
    if not 0 <= probability < 1:
        raise ValueError(f"probability must be in [0, 1), got {probability}")
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")
    return float(-np.expm1(np.log1p(-probability) / rounds))
