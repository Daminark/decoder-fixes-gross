"""Tests of the tier-1 driver, with decoder doubles on the real model."""

import dataclasses
import importlib.util
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from decoder_fixes_gross.decoders import Decoder, Decoding
from decoder_fixes_gross.dem import Dem, dem_at
from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.inject import decode_injected, syndrome
from decoder_fixes_gross.runs import CONFIG_NAME, RunConfig, Shard, shard_name
from decoder_fixes_gross.symmetry import canonicalize, translate
from decoder_fixes_gross.tier1 import DECODERS, DecoderSpec, run

# Five orbits of single columns and three of column pairs; with shards of 2 the run has
# shards 0 to 2 on the singles (rows 0-2, 2-4, 4-5) and 3 to 4 on the pairs (0-2, 2-3).
SHARD_SIZE = 2
SHARD_ROWS = [
    ("singles", 0, 2),
    ("singles", 2, 4),
    ("singles", 4, 5),
    ("pairs", 0, 2),
    ("pairs", 2, 3),
]

# A rule maps (list name, row of the orbit, decoder seed, calls before this one) to whether
# the double returns the injected set (else the zero correction) and its iteration count.
Rule = Callable[[str, int, int, int], tuple[bool, int | None]]
# Syndrome bytes to (list name, row of the orbit, injected fault set).
Table = dict[bytes, tuple[str, int, list[int]]]


@pytest.fixture(scope="module")
def dem() -> Dem:
    return dem_at(1e-3)


def _orbit_list(sets: list[list[int]], n: int) -> FaultList:
    canonical, sizes = canonicalize(np.array(sets, dtype=np.int32))
    rows, first = np.unique(canonical, axis=0, return_index=True)
    assert len(rows) >= n, "the chosen sets do not span enough orbits"
    return FaultList(sets=rows[:n].astype(np.int32), orbit_sizes=sizes[first][:n])


@pytest.fixture(scope="module")
def fault_lists() -> dict[str, FaultList]:
    return {
        "singles": _orbit_list([[c] for c in range(10)], 5),
        "pairs": _orbit_list([[0, 1], [0, 2], [1, 2], [0, 3], [2, 3]], 3),
    }


@pytest.fixture(scope="module")
def table(dem: Dem, fault_lists: dict[str, FaultList]) -> Table:
    """Every translate of every orbit, by syndrome."""
    found: Table = {}
    distinct: set[tuple[int, ...]] = set()
    for name, fl in fault_lists.items():
        n = len(fl.sets)
        images = translate(np.repeat(fl.sets, 72, axis=0), np.tile(np.arange(72), n))
        for k, faults in enumerate(images):
            distinct.add(tuple(faults.tolist()))
            found[syndrome(dem, faults).tobytes()] = (name, k // 72, faults.tolist())
    # One syndrome per fault set, so a syndrome identifies the set that was injected.
    assert len(found) == len(distinct)
    return found


@pytest.fixture
def list_paths(tmp_path: Path, fault_lists: dict[str, FaultList]) -> dict[str, Path]:
    paths = {}
    for name, fl in fault_lists.items():
        paths[name] = tmp_path / f"{name}.npz"
        fl.save(paths[name])
    return paths


class _Double:
    """Looks the injected set up by its syndrome and answers as its rule says."""

    def __init__(self, dem: Dem, seed: int, rule: Rule, table: Table) -> None:
        self.n_columns = dem.h.shape[1]
        self.seed, self.rule, self.table = seed, rule, table
        self.syndromes: list[NDArray[np.uint8]] = []

    def decode(self, syndrome: NDArray[np.uint8]) -> Decoding:
        name, row, faults = self.table[syndrome.tobytes()]
        correct, iterations = self.rule(name, row, self.seed, len(self.syndromes))
        self.syndromes.append(syndrome.copy())
        correction = np.zeros(self.n_columns, dtype=np.uint8)
        if correct:
            correction[faults] = 1
        return Decoding(correction, iterations)


def _spec(rule: Rule, table: Table, built: list[_Double] | None = None) -> DecoderSpec:
    def build(dem: Dem, seed: int) -> Decoder:
        double = _Double(dem, seed, rule, table)
        if built is not None:
            built.append(double)
        return double

    return DecoderSpec("double", {"rule": "test"}, build)


def _shards(run_dir: Path) -> list[Shard]:
    return [Shard.load(run_dir / shard_name(k)) for k in range(len(SHARD_ROWS))]


def _always(correct: bool) -> Rule:
    return lambda name, row, seed, calls: (correct, None)


@pytest.mark.parametrize(("correct", "outcome"), [(True, 0), (False, 2)])
def test_constant_double_gives_its_outcome_on_every_orbit(
    tmp_path: Path, list_paths: dict[str, Path], table: Table, correct: bool, outcome: int
) -> None:
    run(tmp_path / "run", list_paths, _spec(_always(correct), table), 1, shard_size=SHARD_SIZE)
    shards = _shards(tmp_path / "run")
    assert [len(s) for s in shards] == [stop - start for _, start, stop in SHARD_ROWS]
    assert all((s.outcomes == outcome).all() for s in shards)
    assert all((s.iterations == -1).all() for s in shards)


def test_outcomes_iterations_and_times_land_at_their_orbits(
    tmp_path: Path, list_paths: dict[str, Path], table: Table
) -> None:
    def rule(name: str, row: int, seed: int, calls: int) -> tuple[bool, int | None]:
        if name == "singles" and row == 3:
            # A double decodes in microseconds; 50 ms keeps a stall on a busy runner far
            # from the threshold.
            time.sleep(0.05)
        # Correct on even rows; iterations name the orbit on the singles, none on the pairs.
        return row % 2 == 0, (10 * row + 1 if name == "singles" else None)

    run(tmp_path / "run", list_paths, _spec(rule, table), 1, shard_size=SHARD_SIZE)
    shards = _shards(tmp_path / "run")
    assert [s.outcomes.tolist() for s in shards] == [[0, 2], [0, 2], [0], [0, 2], [0]]
    assert [s.iterations.tolist() for s in shards] == [[1, 11], [21, 31], [41], [-1, -1], [-1]]
    slow = [(s.micros >= 50_000).tolist() for s in shards]
    assert slow == [[False, False], [False, True], [False], [False, False], [False]]


def test_stored_translates_are_the_translates_decoded(
    tmp_path: Path,
    list_paths: dict[str, Path],
    fault_lists: dict[str, FaultList],
    dem: Dem,
    table: Table,
) -> None:
    built: list[_Double] = []
    run(tmp_path / "run", list_paths, _spec(_always(True), table, built), 1, shard_size=SHARD_SIZE)
    for shard, double, (name, start, stop) in zip(
        _shards(tmp_path / "run"), built, SHARD_ROWS, strict=True
    ):
        injected = translate(fault_lists[name].sets[start:stop], shard.translates)
        expected = [syndrome(dem, faults) for faults in injected]
        np.testing.assert_array_equal(double.syndromes, expected)
        assert double.seed == shard.decoder_seed


def test_same_seed_gives_same_draws_and_another_seed_other_translates(
    tmp_path: Path, list_paths: dict[str, Path], table: Table
) -> None:
    spec = _spec(_always(True), table)
    for run_id, seed in [("a", 5), ("b", 5), ("c", 6)]:
        run(tmp_path / run_id, list_paths, spec, seed, shard_size=SHARD_SIZE)
    a, b, c = (_shards(tmp_path / run_id) for run_id in "abc")
    assert [s.decoder_seed for s in a] == [s.decoder_seed for s in b]
    np.testing.assert_array_equal(
        np.concatenate([s.translates for s in a]), np.concatenate([s.translates for s in b])
    )
    # Eight translates agree by chance with probability 72**-8.
    assert (
        np.concatenate([s.translates for s in a]) != np.concatenate([s.translates for s in c])
    ).any()
    assert len({s.decoder_seed for s in a}) == len(a)


def _stateful(name: str, row: int, seed: int, calls: int) -> tuple[bool, int | None]:
    """Answers that depend on the decoder's seed and on the calls before, as Relay-BP's do."""
    state = (seed % 1000 + calls) % 1000
    return state % 3 != 0, state


def test_resumed_run_decodes_only_missing_shards_and_equals_an_uninterrupted_run(
    tmp_path: Path, list_paths: dict[str, Path], table: Table
) -> None:
    builds = 0

    def failing_build(dem: Dem, seed: int) -> Decoder:
        nonlocal builds
        builds += 1
        if builds == 3:
            raise RuntimeError("decoder failed on shard 2")
        return _Double(dem, seed, _stateful, table)

    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="shard 2"):
        run(
            interrupted,
            list_paths,
            dataclasses.replace(_spec(_stateful, table), build=failing_build),
            9,
            shard_size=SHARD_SIZE,
        )
    assert sorted(p.name for p in interrupted.iterdir()) == [
        CONFIG_NAME,
        shard_name(0),
        shard_name(1),
    ]

    spec = _spec(_stateful, table)
    assert run(interrupted, list_paths, spec, 9, shard_size=SHARD_SIZE) == [2, 3, 4]
    run(tmp_path / "whole", list_paths, spec, 9, shard_size=SHARD_SIZE)
    resumed, whole = _shards(interrupted), _shards(tmp_path / "whole")
    for r, w in zip(resumed, whole, strict=True):
        assert r.decoder_seed == w.decoder_seed
        for name in ("translates", "outcomes", "iterations"):
            np.testing.assert_array_equal(getattr(r, name), getattr(w, name))
    # The rule makes both outcomes occur, so the identity is not of constant arrays.
    assert {0, 2} <= set(np.concatenate([s.outcomes for s in whole]).tolist())


def test_complete_run_decodes_nothing_on_rerun(
    tmp_path: Path, list_paths: dict[str, Path], table: Table
) -> None:
    spec = _spec(_always(True), table)
    run(tmp_path / "run", list_paths, spec, 3, shard_size=SHARD_SIZE)
    assert run(tmp_path / "run", list_paths, spec, 3, shard_size=SHARD_SIZE) == []


@pytest.mark.parametrize(
    ("seed", "shard_size", "match"), [(4, SHARD_SIZE, "run_seed"), (3, 3, "shard_size")]
)
def test_resume_with_another_configuration_is_refused(
    tmp_path: Path,
    list_paths: dict[str, Path],
    table: Table,
    seed: int,
    shard_size: int,
    match: str,
) -> None:
    spec = _spec(_always(True), table)
    run(tmp_path / "run", list_paths, spec, 3, shard_size=SHARD_SIZE)
    with pytest.raises(ValueError, match=match):
        run(tmp_path / "run", list_paths, spec, seed, shard_size=shard_size)


def test_run_rejects_non_positive_workers(
    tmp_path: Path, list_paths: dict[str, Path], table: Table
) -> None:
    with pytest.raises(ValueError, match="workers"):
        run(tmp_path / "run", list_paths, _spec(_always(True), table), 3, workers=0)


def _script_main() -> Callable[[list[str]], None]:
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_tier1.py"
    spec = importlib.util.spec_from_file_location("run_tier1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main: Callable[[list[str]], None] = module.main
    return main


def test_cli_runs_a_tiny_list_with_relay_bp_on_two_workers(
    tmp_path: Path, fault_lists: dict[str, FaultList]
) -> None:
    (tmp_path / "lists").mkdir()
    singles = fault_lists["singles"]
    FaultList(sets=singles.sets[:4].copy(), orbit_sizes=singles.orbit_sizes[:4].copy()).save(
        tmp_path / "lists" / "tiny.npz"
    )
    _script_main()(
        [
            "--decoder=relay-bp-baseline",
            "--seed=11",
            "--run-id=cli",
            f"--data={tmp_path}",
            "--lists=tiny",
            "--shard-size=2",
            "--workers=2",
        ]
    )
    run_dir = tmp_path / "runs" / "cli"
    config = RunConfig.load(run_dir)
    assert config.decoder == "relay-bp-baseline"
    assert config.decoder_settings["gamma0"] == 0.125
    assert [(r.name, r.n_orbits, r.weight) for r in config.lists] == [("tiny", 4, 1)]
    shards = [Shard.load(run_dir / shard_name(k)) for k in range(2)]
    # Relay-BP corrects single faults (tests/test_decoders.py decodes 200 seeded ones).
    assert all((s.outcomes == 0).all() and (s.iterations >= 1).all() for s in shards)
    assert sorted(p.name for p in run_dir.iterdir()) == [CONFIG_NAME, shard_name(0), shard_name(1)]


def test_registered_baseline_corrects_a_single_fault(dem: Dem) -> None:
    decoder = DECODERS["relay-bp-baseline"].build(dem, 0)
    assert decode_injected(dem, decoder, [100]).outcome == 0


def test_worker_error_stops_a_pool_run_and_writes_no_shard(
    tmp_path: Path, list_paths: dict[str, Path]
) -> None:
    # A builtin, so that spawned workers can unpickle it; divmod(dem, seed) raises TypeError.
    spec = DecoderSpec("broken", {}, divmod)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run(tmp_path / "run", list_paths, spec, 1, shard_size=SHARD_SIZE, workers=2)
    assert [p.name for p in (tmp_path / "run").iterdir()] == [CONFIG_NAME]
