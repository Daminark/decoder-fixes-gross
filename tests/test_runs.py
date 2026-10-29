"""Tests of the run files: shards, the configuration record and the commit helper."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from decoder_fixes_gross.runs import (
    CONFIG_NAME,
    ListRecord,
    RunConfig,
    Shard,
    ShardSpec,
    commit_hash,
    package_versions,
)


def _shard() -> Shard:
    return Shard(
        decoder_seed=2**64 - 1,
        translates=np.array([0, 71, 6], dtype=np.uint8),
        outcomes=np.array([0, 1, 2], dtype=np.uint8),
        micros=np.array([5, 4_000_000_000, 0], dtype=np.uint32),
        iterations=np.array([-1, 18_080, 0], dtype=np.int32),
    )


def _config(**changes: Any) -> RunConfig:
    fields: dict[str, Any] = {
        "decoder": "toy",
        "decoder_settings": {"interval": (-0.24, 0.66), "legs": 301},
        "p0": 1e-3,
        "run_seed": 7,
        "shard_size": 2,
        "lists": (ListRecord("u1", "abc", 5, 1), ListRecord("u2", "def", 3, 2)),
        "versions": {"numpy": "2.4.6"},
        "commit": "0123abcd",
    }
    return RunConfig(**(fields | changes))


def test_shard_round_trips_through_npz(tmp_path: Path) -> None:
    original = _shard()
    path = tmp_path / "shard_0000.npz"
    original.save(path)
    loaded = Shard.load(path)
    assert loaded.decoder_seed == original.decoder_seed
    for name in ("translates", "outcomes", "micros", "iterations"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(original, name))
        assert getattr(loaded, name).dtype == getattr(original, name).dtype
    assert [p.name for p in tmp_path.iterdir()] == ["shard_0000.npz"]


def test_interrupted_shard_write_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def partial_write(f: Any, **arrays: Any) -> None:
        f.write(b"PK\x03\x04 half a zip")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez", partial_write)
    path = tmp_path / "shard_0000.npz"
    with pytest.raises(KeyboardInterrupt):
        _shard().save(path)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_shard_write_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "shard_0000.npz"
    _shard().save(path)
    before = path.read_bytes()

    def failing(f: Any, **arrays: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", failing)
    with pytest.raises(OSError, match="disk full"):
        _shard().save(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("translates", np.array([0, 72, 6], dtype=np.uint8)),
        ("outcomes", np.array([0, 3, 2], dtype=np.uint8)),
        ("iterations", np.array([-2, 0, 0], dtype=np.int32)),
        ("micros", np.array([5, 4, 0], dtype=np.int64)),
        ("micros", np.array([5, 4], dtype=np.uint32)),
    ],
)
def test_shard_rejects_out_of_range_or_misshaped_arrays(field: str, value: np.ndarray) -> None:
    fields = vars(_shard()) | {field: value}
    with pytest.raises(ValueError, match=field):
        Shard(**fields)


def test_shard_rejects_seed_outside_64_bits() -> None:
    with pytest.raises(ValueError, match="decoder_seed"):
        Shard(**(vars(_shard()) | {"decoder_seed": 2**64}))


def test_shard_arrays_are_read_only() -> None:
    with pytest.raises(ValueError, match="read-only"):
        _shard().outcomes[0] = 1


def test_shards_cut_each_list_separately() -> None:
    # u1 has 5 orbits and u2 has 3; size 2 gives u1 rows 0-2, 2-4, 4-5 and u2 rows 0-2, 2-3.
    assert _config().shards() == [
        ShardSpec(0, "u1", 0, 2),
        ShardSpec(1, "u1", 2, 4),
        ShardSpec(2, "u1", 4, 5),
        ShardSpec(3, "u2", 0, 2),
        ShardSpec(4, "u2", 2, 3),
    ]


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    original = _config()
    original.save(tmp_path)
    loaded = RunConfig.load(tmp_path)
    assert loaded.differing_fields(original) == []
    assert loaded.lists == original.lists
    assert loaded.decoder_settings == {"interval": [-0.24, 0.66], "legs": 301}


def test_config_file_holds_every_recorded_field(tmp_path: Path) -> None:
    _config().save(tmp_path)
    record = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert set(record) == {
        "decoder",
        "decoder_settings",
        "p0",
        "run_seed",
        "shard_size",
        "lists",
        "versions",
        "commit",
    }
    assert record["p0"] == 1e-3
    assert record["lists"][1] == {"name": "u2", "content_hash": "def", "n_orbits": 3, "weight": 2}


def test_differing_fields_names_each_changed_field() -> None:
    changed = _config(run_seed=8, commit="0123abcd-dirty")
    assert _config().differing_fields(changed) == ["run_seed", "commit"]


def test_config_load_names_a_missing_field(tmp_path: Path) -> None:
    record = _config().to_json()
    del record["commit"]
    (tmp_path / CONFIG_NAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="commit"):
        RunConfig.load(tmp_path)


def test_config_load_names_a_missing_list_record_field(tmp_path: Path) -> None:
    record = _config().to_json()
    del record["lists"][1]["weight"]
    (tmp_path / CONFIG_NAME).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="weight"):
        RunConfig.load(tmp_path)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"run_seed": -1}, "run_seed"),
        ({"shard_size": 0}, "shard_size"),
        ({"lists": (ListRecord("u1", "a", 1, 1), ListRecord("u1", "b", 1, 1))}, "distinct"),
    ],
)
def test_config_rejects_invalid_fields(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(**changes)


def test_commit_hash_is_unknown_outside_a_checkout(tmp_path: Path) -> None:
    assert commit_hash(tmp_path) == "unknown"


def test_commit_hash_of_this_checkout_is_a_full_hash() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", commit_hash(Path(__file__).parent))


def test_package_versions_name_the_decoder_packages() -> None:
    assert {"python", "numpy", "stim", "relay-bp", "ldpc"} <= set(package_versions())


def test_commit_hash_is_unknown_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert commit_hash(Path(__file__).parent) == "unknown"


def test_shard_load_names_a_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "shard_0000.npz"
    np.savez(path, decoder_seed=np.uint64(0), translates=np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="outcomes"):
        Shard.load(path)
