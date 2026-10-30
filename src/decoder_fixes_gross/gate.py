"""The week-1 gate: whether the Relay-BP baseline reproduces the trapping-set paper.

The rule was fixed before the run. It compares a tier-1 run's aggregates
(:func:`~decoder_fixes_gross.aggregate.summarize`) with Table II and Fig. 1 of Pacenti,
Raveendran and Vasić [1]_ on the first failing weight, the number N_fail of failing LETS
instances per (a, b) class and in total as published, and the largest failing class. The floor
check against Fig. 1 is informational and never changes the verdict.

References
----------
.. [1] M. Pacenti, N. Raveendran and B. Vasić, arXiv:2608.11516 (2026).
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

# N_fail of Relay-BP per (a, b) class, Table II of [1].
PAPER_N_FAIL = {
    (4, 0): 0,
    (4, 2): 0,
    (4, 3): 0,
    (4, 4): 23_751,
    (4, 5): 1_425,
    (5, 0): 0,
    (5, 2): 0,
    (5, 3): 68_239,
    (5, 4): 13_045,
    (5, 5): 46_397,
}
# The total stated in Sec. IV of [1]; it equals the sum of Table II.
PAPER_TOTAL = 152_857
# The class with the most failing instances in Table II.
PAPER_LARGEST = (5, 3)
# The paper's error-floor estimate per round, read off Fig. 1 of [1] by eye as (low, high) at
# two physical error rates; readings off the log-scale plot are good to about a factor of 2.
FIG1_READINGS = {4e-4: (3e-10, 4e-10), 1e-3: (1e-8, 2e-8)}
# Relative deviation of the total from PAPER_TOTAL: at most WIN_DEVIATION for a win, more than
# LOSS_DEVIATION for a loss.
WIN_DEVIATION = 0.10
LOSS_DEVIATION = 0.30
# Floor check: a factor from the read range up to CONSISTENT_FACTOR is consistent, one beyond
# RECHECK_FACTOR sends the probability model back for a recheck.
CONSISTENT_FACTOR = 3.0
RECHECK_FACTOR = 10.0
# Supports of this weight or below must not fail.
MAX_CORRECTED_WEIGHT = 3


class Verdict(StrEnum):
    """Outcome of the gate."""

    WIN = "win"
    INCONCLUSIVE = "inconclusive"
    LOSS = "loss"


class FloorLabel(StrEnum):
    """Outcome of the floor check at one Fig. 1 point."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    RECHECK = "recheck"


def evaluate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the gate's conditions, verdict and floor check for one run.

    Parameters
    ----------
    record : Mapping
        The aggregates of the run, as :func:`~decoder_fixes_gross.aggregate.summarize`
        returns them or as read back from their JSON file. Its ``lets`` entry must hold every
        class of Table II, and its ``floor`` entry the two physical error rates of
        ``FIG1_READINGS``.

    Returns
    -------
    dict
        JSON-serializable. ``verdict``: "win", "inconclusive" or "loss". ``conditions``: per
        condition of the rule, what was found and whether it holds: ``first_failing_weight``
        (None when nothing failed), ``zero_classes`` and ``paper_classes`` (N_fail as
        published per class), ``total`` (with the paper's total, the shortfall, i.e. the paper's
        total less ours, and the relative deviation) and ``largest_class`` (every class with the
        most failing instances). ``classes``: per Table II class, N_fail as published and
        closed, the paper's N_fail, the shortfall, and the ratio of published to paper (None
        where the paper's is 0). ``floor_check``: at each Fig. 1
        point, the reading and, for both forms, the per-round floor and its ``factor`` from the
        read range, max(low / floor, floor / high, 1), None when the floor is 0 or undefined;
        the occurrence form also has the ``label`` of the check. ``rule``: the thresholds.

    Raises
    ------
    ValueError
        If a support of weight ``MAX_CORRECTED_WEIGHT`` or below failed, since only tier 2
        (slice 11) can tell a loss from a seed effect there; if a Table II class or a Fig. 1
        rate is missing from ``record``.

    Notes
    -----
    A loss is a failure in a class where the paper has none, or a total more than
    ``LOSS_DEVIATION`` from the paper's. A win needs every condition to hold and the total
    within ``WIN_DEVIATION``. Anything else is inconclusive.
    """
    failing_weights = sorted(
        int(w) for w, entry in record["by_weight"].items() if entry["failing_orbits"] > 0
    )
    low = [w for w in failing_weights if w <= MAX_CORRECTED_WEIGHT]
    if low:
        raise ValueError(
            f"supports of weight {low} failed; tier 2 (slice 11) must confirm them before the "
            "gate can call them a loss or a seed effect"
        )
    missing = [_key(c) for c in PAPER_N_FAIL if _key(c) not in record["lets"]]
    if missing:
        raise ValueError(f"record lacks the Table II classes {missing}")

    published = {c: int(record["lets"][_key(c)]["published"]) for c in PAPER_N_FAIL}
    zero = {_key(c): n for c, n in published.items() if PAPER_N_FAIL[c] == 0}
    paper = {_key(c): n for c, n in published.items() if PAPER_N_FAIL[c] > 0}
    total = int(record["lets"]["total"]["published"])
    deviation = (total - PAPER_TOTAL) / PAPER_TOTAL
    most = max(published.values())
    largest = [_key(c) for c, n in published.items() if n == most and n > 0]

    conditions: dict[str, dict[str, Any]] = {
        "first_failing_weight": {
            "found": failing_weights[0] if failing_weights else None,
            # Failures at MAX_CORRECTED_WEIGHT or below were refused above.
            "holds": True,
        },
        "zero_classes": {"found": zero, "holds": not any(zero.values())},
        "paper_classes": {"found": paper, "holds": all(paper.values())},
        "total": {
            "found": total,
            "paper": PAPER_TOTAL,
            "shortfall": PAPER_TOTAL - total,
            "deviation": deviation,
            "holds": abs(deviation) <= WIN_DEVIATION,
        },
        "largest_class": {"found": largest, "holds": largest == [_key(PAPER_LARGEST)]},
    }
    if not conditions["zero_classes"]["holds"] or abs(deviation) > LOSS_DEVIATION:
        verdict = Verdict.LOSS
    elif all(condition["holds"] for condition in conditions.values()):
        verdict = Verdict.WIN
    else:
        verdict = Verdict.INCONCLUSIVE

    return {
        "verdict": verdict.value,
        "conditions": conditions,
        "classes": {
            _key(c): {
                "published": published[c],
                "closed": int(record["lets"][_key(c)]["closed"]),
                "paper": n,
                "shortfall": n - published[c],
                "ratio": published[c] / n if n else None,
            }
            for c, n in PAPER_N_FAIL.items()
        },
        "floor_check": [_floor_point(record, p, reading) for p, reading in FIG1_READINGS.items()],
        "rule": {
            "win_deviation": WIN_DEVIATION,
            "loss_deviation": LOSS_DEVIATION,
            "consistent_factor": CONSISTENT_FACTOR,
            "recheck_factor": RECHECK_FACTOR,
        },
    }


def _key(c: tuple[int, int]) -> str:
    return f"a{c[0]}_b{c[1]}"


def _floor_point(
    record: Mapping[str, Any], p: float, reading: tuple[float, float]
) -> dict[str, Any]:
    entries = [entry for entry in record["floor"] if entry["p"] == p]
    if not entries:
        raise ValueError(f"record has no floor at the Fig. 1 rate p = {p}")
    low, high = reading
    point: dict[str, Any] = {"p": p, "reading": {"low": low, "high": high}}
    for form in ("occurrence", "exact_event"):
        value = entries[0][form]["per_round"]["value"]
        factor = max(low / value, value / high, 1.0) if value else None
        point[form] = {"per_round": value, "factor": factor}
    factor = point["occurrence"]["factor"]
    if factor is None or factor > RECHECK_FACTOR:
        label = FloorLabel.RECHECK
    elif factor <= CONSISTENT_FACTOR:
        label = FloorLabel.CONSISTENT
    else:
        label = FloorLabel.INCONSISTENT
    point["occurrence"]["label"] = label.value
    return point
