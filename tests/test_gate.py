"""Tests of the gate's verdict and floor check, on records written by hand."""

from collections.abc import Mapping
from typing import Any

import pytest

from decoder_fixes_gross.gate import evaluate

# N_fail as published per class of the Relay-BP baseline run (results/floor_relay-bp-baseline.json).
BASELINE = {
    "a4_b0": 0,
    "a4_b2": 0,
    "a4_b3": 0,
    "a4_b4": 23_328,
    "a4_b5": 1_152,
    "a5_b0": 0,
    "a5_b2": 0,
    "a5_b3": 68_110,
    "a5_b4": 6_408,
    "a5_b5": 44_352,
}
# Inside both Fig. 1 readings: 3e-10 to 4e-10 at 4e-4 and 1e-8 to 2e-8 at 1e-3.
INSIDE = {4e-4: 3.5e-10, 1e-3: 1.5e-8}


def _record(
    published: dict[str, int] = BASELINE,
    total: int | None = None,
    failing_orbits_by_weight: dict[int, int] | None = None,
    occurrence: Mapping[float, float | None] = INSIDE,
) -> dict[str, Any]:
    by_weight = failing_orbits_by_weight or {1: 0, 2: 0, 3: 0, 4: 450}
    lets = {key: {"published": n, "closed": n + 1} for key, n in published.items()}
    # Classes outside Table II are ignored by the gate.
    lets["a3_b2"] = {"published": 0, "closed": 0}
    lets["total"] = {"published": sum(published.values()) if total is None else total}
    floor = [
        {
            "p": p,
            "occurrence": {"per_round": {"value": value}},
            "exact_event": {"per_round": {"value": 3e-12}},
        }
        for p, value in [(1e-4, 1e-12), *occurrence.items()]
    ]
    return {
        "by_weight": {str(w): {"failing_orbits": n} for w, n in by_weight.items()},
        "lets": lets,
        "floor": floor,
    }


def test_baseline_counts_are_a_win() -> None:
    gate = evaluate(_record())
    assert gate["verdict"] == "win"
    conditions = gate["conditions"]
    assert conditions["first_failing_weight"]["found"] == 4
    assert conditions["total"]["found"] == 143_350
    assert conditions["total"]["shortfall"] == 9_507
    # (143,350 - 152,857) / 152,857 = -9,507 / 152,857.
    assert conditions["total"]["deviation"] == pytest.approx(-0.062195, abs=1e-6)
    assert conditions["largest_class"]["found"] == ["a5_b3"]
    assert all(condition["holds"] for condition in conditions.values())


def test_class_table_holds_published_closed_paper_shortfall_and_ratio() -> None:
    classes = evaluate(_record())["classes"]
    assert list(classes) == list(BASELINE)
    assert classes["a5_b4"]["published"] == 6_408
    assert classes["a5_b4"]["closed"] == 6_409
    assert classes["a5_b4"]["paper"] == 13_045
    assert classes["a5_b4"]["shortfall"] == 6_637
    assert classes["a5_b4"]["ratio"] == pytest.approx(0.49122, abs=1e-5)
    assert classes["a4_b0"]["ratio"] is None


def test_total_twenty_percent_below_the_paper_is_inconclusive() -> None:
    # 0.8 x 152,857 = 122,285.6.
    gate = evaluate(_record(total=122_286))
    assert gate["verdict"] == "inconclusive"
    assert not gate["conditions"]["total"]["holds"]


def test_total_thirty_five_percent_below_the_paper_is_a_loss() -> None:
    # 0.65 x 152,857 = 99,357.05.
    assert evaluate(_record(total=99_357))["verdict"] == "loss"


def test_total_thirty_five_percent_above_the_paper_is_a_loss() -> None:
    # 1.35 x 152,857 = 206,356.95.
    assert evaluate(_record(total=206_357))["verdict"] == "loss"


def test_failure_in_a_zero_class_is_a_loss() -> None:
    gate = evaluate(_record(published=BASELINE | {"a4_b3": 72}, total=143_350))
    assert gate["verdict"] == "loss"
    assert gate["conditions"]["zero_classes"]["found"]["a4_b3"] == 72


def test_another_largest_class_is_inconclusive() -> None:
    gate = evaluate(_record(published=BASELINE | {"a5_b5": 70_000}, total=143_350))
    assert gate["verdict"] == "inconclusive"
    assert gate["conditions"]["largest_class"]["found"] == ["a5_b5"]


def test_a_paper_class_without_failures_is_inconclusive() -> None:
    gate = evaluate(_record(published=BASELINE | {"a4_b5": 0}, total=143_350))
    assert gate["verdict"] == "inconclusive"
    assert not gate["conditions"]["paper_classes"]["holds"]


def test_run_without_failures_is_a_loss() -> None:
    gate = evaluate(
        _record(published=dict.fromkeys(BASELINE, 0), failing_orbits_by_weight={3: 0, 4: 0})
    )
    assert gate["verdict"] == "loss"
    assert gate["conditions"]["first_failing_weight"]["found"] is None
    assert gate["conditions"]["largest_class"]["found"] == []


def test_failure_at_weight_three_is_refused_until_tier_two() -> None:
    with pytest.raises(ValueError, match="slice 11"):
        evaluate(_record(failing_orbits_by_weight={3: 1, 4: 450}))


def test_record_without_a_table_ii_class_is_refused() -> None:
    published = {key: n for key, n in BASELINE.items() if key != "a5_b0"}
    with pytest.raises(ValueError, match="a5_b0"):
        evaluate(_record(published=published))


def test_record_without_a_fig1_rate_is_refused() -> None:
    with pytest.raises(ValueError, match=r"p = 0\.0004"):
        evaluate(_record(occurrence={1e-3: 1.5e-8}))


@pytest.mark.parametrize(
    ("value", "factor", "label"),
    [
        # Inside the reading 3e-10 to 4e-10.
        (3.5e-10, 1.0, "consistent"),
        # 3e-10 / 2.5e-10 = 1.2.
        (2.5e-10, 1.2, "consistent"),
        # 2e-9 / 4e-10 = 5.
        (2e-9, 5.0, "inconsistent"),
        # 3e-10 / 1.5e-11 = 20.
        (1.5e-11, 20.0, "recheck"),
        # A floor of 1 or more has no per-round value.
        (None, None, "recheck"),
    ],
)
def test_floor_check_factor_from_the_read_range(
    value: float | None, factor: float | None, label: str
) -> None:
    point = evaluate(_record(occurrence={4e-4: value, 1e-3: 1.5e-8}))["floor_check"][0]
    assert point["p"] == 4e-4
    assert point["reading"] == {"low": 3e-10, "high": 4e-10}
    assert point["occurrence"]["factor"] == pytest.approx(factor)
    assert point["occurrence"]["label"] == label


def test_floor_check_reports_the_exact_event_factor_without_a_label() -> None:
    point = evaluate(_record())["floor_check"][1]
    assert point["p"] == 1e-3
    # 1e-8 / 3e-12.
    assert point["exact_event"] == {"per_round": 3e-12, "factor": pytest.approx(3333.33, rel=1e-5)}


def test_floor_check_does_not_change_the_verdict() -> None:
    gate = evaluate(_record(occurrence={4e-4: 1e-15, 1e-3: 1e-15}))
    assert [point["occurrence"]["label"] for point in gate["floor_check"]] == ["recheck"] * 2
    assert gate["verdict"] == "win"
