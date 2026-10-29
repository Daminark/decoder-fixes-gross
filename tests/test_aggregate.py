"""Tests of run aggregation, on a run written by hand in the format of the tier-1 driver."""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from decoder_fixes_gross.aggregate import ceiling, failing_lets, read_run, summarize
from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.lets import LetsList, LetsOrbits
from decoder_fixes_gross.runs import ListRecord, RunConfig, Shard, shard_name


def _list(sets: list[list[int]], sizes: list[int]) -> FaultList:
    return FaultList(
        sets=np.array(sets, dtype=np.int32), orbit_sizes=np.array(sizes, dtype=np.int16)
    )


# Fault sets over 10 columns. With uniform priors r, Pr(S) = r**w (1 - r)**(10 - w).
LISTS = {
    "u1": _list([[0], [1]], [72, 72]),
    "u3": _list([[0, 1, 2], [0, 1, 3], [0, 2, 5]], [72, 72, 36]),
    "u4": _list([[0, 1, 2, 3], [1, 2, 3, 4]], [72, 36]),
}
# Shards of 2: u1 rows 0-2; u3 rows 0-2 and 2-3; u4 rows 0-2.
SHARD_SIZE = 2
# u3 row 0 is a wrong logical, u3 row 2 has no solution, u4 row 1 is a wrong logical.
OUTCOMES = {"u1": [0, 0], "u3": [1, 0, 2], "u4": [0, 1]}
MICROS = {"u1": [10, 20], "u3": [30, 40, 50], "u4": [60, 1000]}
ITERATIONS = {"u1": [1, 2], "u3": [3, 4, 5], "u4": [6, 700]}
# The occurrence probability is r**w; the exact-event one multiplies it by (1 - r)**(10 - w).
# r = 0.1: exact 1e-3 x 0.9**7 = 4.782969e-4 at w = 3 and 1e-4 x 0.9**6 = 5.31441e-5 at w = 4.
# r = 0.05: exact 1.25e-4 x 0.95**7 = 1.25e-4 x 0.69833729609375 at w = 3 and
# 6.25e-6 x 0.95**6 = 6.25e-6 x 0.735091890625 at w = 4.
PRIORS_BY_P = {1e-3: np.full(10, 0.1), 2e-3: np.full(10, 0.05)}


def _lets() -> LetsList:
    def orbits(
        a: int, b: int, sets: list[list[int]], sizes: list[int], published: list[int]
    ) -> LetsOrbits:
        return LetsOrbits(
            a=a, b=b, orbits=_list(sets, sizes), published_counts=np.array(published, np.int16)
        )

    def rows(values: list[list[int]], width: int) -> np.typing.NDArray[np.int32]:
        return np.array(values, dtype=np.int32).reshape(len(values), width)

    classes = (
        orbits(2, 2, [[0, 1]], [72], [72]),
        orbits(3, 2, [[0, 1, 2], [0, 1, 3]], [72, 72], [70, 72]),
        orbits(4, 4, [[0, 1, 2, 5], [0, 1, 3, 4], [1, 2, 3, 4]], [72, 36, 72], [71, 36, 72]),
    )
    incidence = {
        (2, 2): {3: rows([[]], 0), 4: rows([[]], 0)},
        # Orbit 0 holds the failing u3 row 0; orbit 1 holds only passing u3 row 1.
        (3, 2): {3: rows([[0], [1]], 1), 4: rows([[], []], 0)},
        (4, 4): {
            # Orbit 0 holds two failing supports (u3 rows 0 and 2) and counts once; orbit 1
            # fails through u4 row 1 alone; orbit 2 holds only passing supports.
            3: rows([[0, 2, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 4),
            4: rows([[0], [1], [0]], 1),
        },
    }
    return LetsList(classes=classes, incidence=incidence)


def _write_run(
    run_dir: Path,
    iterations: dict[str, list[int]] = ITERATIONS,
    lists: dict[str, FaultList] = LISTS,
) -> None:
    config = RunConfig(
        decoder="double",
        decoder_settings={},
        p0=1e-3,
        run_seed=1,
        shard_size=SHARD_SIZE,
        lists=tuple(
            ListRecord(name, fl.content_hash(), len(fl.sets), fl.sets.shape[1])
            for name, fl in lists.items()
        ),
        versions={},
        commit="test",
    )
    run_dir.mkdir()
    config.save(run_dir)
    for spec in config.shards():
        rows = slice(spec.start, spec.stop)
        n = spec.stop - spec.start
        Shard(
            decoder_seed=spec.index,
            translates=np.zeros(n, dtype=np.uint8),
            outcomes=np.array(OUTCOMES[spec.list_name][rows], dtype=np.uint8),
            micros=np.array(MICROS[spec.list_name][rows], dtype=np.uint32),
            iterations=np.array(iterations[spec.list_name][rows], dtype=np.int32),
        ).save(run_dir / shard_name(spec.index))


@pytest.fixture
def record(tmp_path: Path) -> dict[str, Any]:
    _write_run(tmp_path / "run")
    return summarize(read_run(tmp_path / "run"), LISTS, _lets(), PRIORS_BY_P)


def test_read_run_joins_shards_in_list_order(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    run = read_run(tmp_path / "run")
    assert {name: a.tolist() for name, a in run.outcomes.items()} == OUTCOMES
    assert {name: a.tolist() for name, a in run.micros.items()} == MICROS
    assert {name: a.tolist() for name, a in run.iterations.items()} == ITERATIONS


def test_read_run_refuses_an_incomplete_run(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    (tmp_path / "run" / shard_name(2)).unlink()
    with pytest.raises(ValueError, match=r"shard_0002\.npz is missing"):
        read_run(tmp_path / "run")


def test_run_digest_changes_with_a_shard(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    before = read_run(tmp_path / "run").digest
    shard = Shard.load(tmp_path / "run" / shard_name(0))
    Shard(9, shard.translates, shard.outcomes, shard.micros, shard.iterations).save(
        tmp_path / "run" / shard_name(0)
    )
    assert read_run(tmp_path / "run").digest != before


def test_failures_by_list_and_outcome(record: dict[str, Any]) -> None:
    assert record["lists"] == {
        "u1": {
            "weight": 1,
            "orbits": 2,
            "failing_orbits": 0,
            "failing_instances": 0,
            "wrong_logical": {"orbits": 0, "instances": 0},
            "no_solution": {"orbits": 0, "instances": 0},
        },
        "u3": {
            "weight": 3,
            "orbits": 3,
            "failing_orbits": 2,
            "failing_instances": 72 + 36,
            "wrong_logical": {"orbits": 1, "instances": 72},
            "no_solution": {"orbits": 1, "instances": 36},
        },
        "u4": {
            "weight": 4,
            "orbits": 2,
            "failing_orbits": 1,
            "failing_instances": 36,
            "wrong_logical": {"orbits": 1, "instances": 36},
            "no_solution": {"orbits": 0, "instances": 0},
        },
    }
    assert record["by_weight"] == {
        "1": {"orbits": 2, "failing_orbits": 0, "failing_instances": 0},
        "3": {"orbits": 3, "failing_orbits": 2, "failing_instances": 108},
        "4": {"orbits": 2, "failing_orbits": 1, "failing_instances": 36},
    }


def test_failing_lets_counted_once_per_orbit_as_published_and_closed(
    record: dict[str, Any],
) -> None:
    assert record["lets"] == {
        "a2_b2": {"failing_orbits": 0, "published": 0, "closed": 0},
        "a3_b2": {"failing_orbits": 1, "published": 70, "closed": 72},
        "a4_b4": {"failing_orbits": 2, "published": 71 + 36, "closed": 72 + 36},
        "total": {"failing_orbits": 3, "published": 177, "closed": 180},
    }


def test_failing_lets_masks_per_class() -> None:
    failed = {3: np.array([True, False, True]), 4: np.array([False, True])}
    masks = failing_lets(_lets(), failed)
    assert {key: mask.tolist() for key, mask in masks.items()} == {
        (2, 2): [False],
        (3, 2): [True, False],
        (4, 4): [True, True, False],
    }


def test_floors_are_listed_by_increasing_p(record: dict[str, Any]) -> None:
    assert [entry["p"] for entry in record["floor"]] == [1e-3, 2e-3]


@pytest.mark.parametrize(
    ("index", "form", "pr3", "pr4"),
    [
        (0, "exact_event", 4.782969e-4, 5.31441e-5),
        (0, "occurrence", 1e-3, 1e-4),
        (1, "exact_event", 8.729216201171875e-5, 4.59432431640625e-6),
        (1, "occurrence", 1.25e-4, 6.25e-6),
    ],
)
def test_floor_and_error_from_hand_worked_probabilities(
    record: dict[str, Any], index: int, form: str, pr3: float, pr4: float
) -> None:
    entry = record["floor"][index][form]
    # Failing orbits: u3 row 0 (size 72) and row 2 (size 36), u4 row 1 (size 36).
    terms = [72 * pr3, 36 * pr3, 36 * pr4]
    value, error = sum(terms), np.sqrt(np.sum(np.square(terms)))
    np.testing.assert_allclose([entry["value"], entry["error"]], [value, error], rtol=1e-12)
    np.testing.assert_allclose(entry["lists"]["u3"]["value"], 108 * pr3, rtol=1e-12)
    np.testing.assert_allclose(entry["lists"]["u4"]["value"], 36 * pr4, rtol=1e-12)
    assert entry["lists"]["u1"] == {"value": 0.0, "error": 0.0}
    # Each end of value -+ error is converted on its own; at these magnitudes the plain power
    # form is accurate, an independent route to the per-round rate.
    np.testing.assert_allclose(
        [entry["per_round"][k] for k in ("value", "low", "high")],
        [1 - (1 - v) ** (1 / 12) for v in (value, value - error, value + error)],
        rtol=1e-8,
    )


def test_cost_statistics_over_every_orbit(record: dict[str, Any]) -> None:
    # Times 10, 20, 30, 40, 50, 60, 1000: mean 1210 / 7; the 99th percentile interpolates
    # 0.94 of the way from 60 to 1000 (position 0.99 x 6 = 5.94).
    assert record["cost"]["micros"] == pytest.approx({"mean": 1210 / 7, "median": 40, "p99": 943.6})
    assert record["cost"]["iterations"] == pytest.approx(
        {"mean": 103, "median": 4, "p99": 6 + 0.94 * 694}
    )


def test_cost_without_iteration_counts_is_none(tmp_path: Path) -> None:
    none = {name: [-1] * len(values) for name, values in ITERATIONS.items()}
    _write_run(tmp_path / "run", iterations=none)
    record = summarize(read_run(tmp_path / "run"), LISTS, _lets(), PRIORS_BY_P)
    assert record["cost"]["iterations"] is None


def test_mixed_iteration_counts_are_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", iterations=ITERATIONS | {"u1": [-1, 2]})
    with pytest.raises(ValueError, match="mix"):
        summarize(read_run(tmp_path / "run"), LISTS, _lets(), PRIORS_BY_P)


def test_inputs_name_the_run_lists_and_lets(
    tmp_path: Path,
    record: dict[str, Any],
) -> None:
    assert record["inputs"] == {
        "run_digest": read_run(tmp_path / "run").digest,
        "lists": {name: fl.content_hash() for name, fl in LISTS.items()},
        "lets": _lets().content_hash(),
    }


def test_list_differing_from_the_run_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    other = LISTS | {"u1": _list([[0], [2]], [72, 72])}
    with pytest.raises(ValueError, match="'u1' differs"):
        summarize(read_run(tmp_path / "run"), other, _lets(), PRIORS_BY_P)


def test_missing_list_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    lists = {name: fl for name, fl in LISTS.items() if name != "u1"}
    with pytest.raises(ValueError, match="lacks the run's list 'u1'"):
        summarize(read_run(tmp_path / "run"), lists, _lets(), PRIORS_BY_P)


def test_run_without_u3_and_u4_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", lists={"u1": LISTS["u1"]})
    with pytest.raises(ValueError, match="u3 and u4"):
        summarize(read_run(tmp_path / "run"), LISTS, _lets(), PRIORS_BY_P)


def test_ceiling_is_the_mass_of_every_orbit() -> None:
    # r = 0.1: exact-event Pr = 0.1 x 0.9**9 = 0.0387420489 at w = 1. The occurrence masses
    # are sums over overlapping events, so the one of u1 exceeds 1.
    (entry,) = ceiling(LISTS, {1e-3: PRIORS_BY_P[1e-3]})
    assert entry["p"] == 1e-3
    np.testing.assert_allclose(
        [entry["exact_event"][name] for name in ("u1", "u3", "u4")],
        [144 * 0.0387420489, 180 * 4.782969e-4, 108 * 5.31441e-5],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        [entry["occurrence"][name] for name in ("u1", "u3", "u4")],
        [144 * 0.1, 180 * 1e-3, 108 * 1e-4],
        rtol=1e-12,
    )


def _script_main() -> Callable[[list[str]], None]:
    path = Path(__file__).resolve().parent.parent / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main: Callable[[list[str]], None] = module.main
    return main


def _data_dir(tmp_path: Path, registered_lists: list[str]) -> list[str]:
    """Write lists, LETS, one run and a registry under ``tmp_path``; return the CLI arguments."""
    (tmp_path / "lists").mkdir()
    for name, fl in LISTS.items():
        fl.save(tmp_path / "lists" / f"{name}.npz")
    _lets().save(tmp_path / "lists" / "lets.npz")
    (tmp_path / "runs").mkdir()
    _write_run(tmp_path / "runs" / "small")
    registry = tmp_path / "runs.toml"
    registry.write_text(
        f'[[runs]]\nid = "small"\ndecoder = "double"\nlists = {json.dumps(registered_lists)}\n'
    )
    return [f"--data={tmp_path}", f"--runs={registry}", f"--results={tmp_path / 'results'}"]


@pytest.mark.slow
def test_aggregation_command_writes_floor_and_ceiling_with_input_hashes(tmp_path: Path) -> None:
    # Slow: the model is built at each of the 21 rates, about 2.5 s.
    _script_main()(_data_dir(tmp_path, ["u1", "u3", "u4"]))
    results = tmp_path / "results"
    assert sorted(p.name for p in results.iterdir()) == ["ceiling.json", "floor_small.json"]
    hashes = {name: fl.content_hash() for name, fl in LISTS.items()}

    floor = json.loads((results / "floor_small.json").read_text())
    assert floor["run_id"] == "small"
    assert floor["decoder"] == "double"
    assert floor["inputs"] == {
        "run_digest": read_run(tmp_path / "runs" / "small").digest,
        "lists": hashes,
        "lets": _lets().content_hash(),
    }
    grids = floor["p_grids"]
    assert grids["table"] == [1e-4, 3e-4, 4e-4, 1e-3]
    assert len(grids["curve"]) == 17
    np.testing.assert_allclose([grids["curve"][0], grids["curve"][-1]], [1e-4, 4e-3])
    assert [entry["p"] for entry in floor["floor"]] == sorted({*grids["table"], *grids["curve"]})

    ceiling_record = json.loads((results / "ceiling.json").read_text())
    assert ceiling_record["inputs"] == {"lists": hashes}
    assert len(ceiling_record["ceiling"]) == len(floor["floor"])


def test_aggregation_command_refuses_a_run_that_differs_from_the_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="registers double on"):
        _script_main()(_data_dir(tmp_path, ["u1", "u3"]))
