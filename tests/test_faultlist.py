"""Tests of the fault-list format."""

from pathlib import Path

import numpy as np
import pytest

from decoder_fixes_gross.faultlist import FaultList


def _small_list() -> FaultList:
    return FaultList(
        sets=np.array([[0, 5, 9], [0, 6, 7], [3, 4, 8000]], dtype=np.int32),
        orbit_sizes=np.array([72, 36, 72], dtype=np.int16),
    )


def test_fault_list_round_trips_through_npz(tmp_path: Path) -> None:
    original = _small_list()
    path = tmp_path / "list.npz"
    original.save(path)
    loaded = FaultList.load(path)
    np.testing.assert_array_equal(loaded.sets, original.sets)
    np.testing.assert_array_equal(loaded.orbit_sizes, original.orbit_sizes)
    assert loaded.sets.dtype == np.int32
    assert loaded.orbit_sizes.dtype == np.int16
    assert loaded.content_hash() == original.content_hash()


def test_content_hash_is_stable() -> None:
    # Recorded when the format was defined; a change here changes every list's provenance.
    assert (
        _small_list().content_hash()
        == "bb41fb38fd02f9f7c6613d171f71886e9340fc4853ef4026045ed5981902946c"
    )


def test_content_hash_changes_with_one_entry() -> None:
    changed = FaultList(
        sets=np.array([[0, 5, 9], [0, 6, 7], [3, 4, 8001]], dtype=np.int32),
        orbit_sizes=np.array([72, 36, 72], dtype=np.int16),
    )
    assert changed.content_hash() != _small_list().content_hash()


def test_empty_fault_list_is_valid() -> None:
    empty = FaultList(sets=np.zeros((0, 4), dtype=np.int32), orbit_sizes=np.zeros(0, np.int16))
    assert empty.sets.shape == (0, 4)


@pytest.mark.parametrize(
    ("sets", "orbit_sizes", "message"),
    [
        (np.array([[0, 1]], dtype=np.int64), np.array([72], np.int16), "sets must be int32"),
        (np.array([0, 1], dtype=np.int32), np.array([72], np.int16), "sets must be int32"),
        (np.array([[0, 1]], dtype=np.int32), np.array([72], np.int32), "orbit_sizes must be"),
        (np.array([[0, 1]], dtype=np.int32), np.array([72, 72], np.int16), "orbit_sizes must be"),
        (np.array([[1, 0]], dtype=np.int32), np.array([72], np.int16), "strictly increasing col"),
        (np.array([[2, 2]], dtype=np.int32), np.array([72], np.int16), "strictly increasing col"),
        (np.array([[-1, 2]], dtype=np.int32), np.array([72], np.int16), "strictly increasing col"),
        (
            np.array([[0, 2], [0, 1]], dtype=np.int32),
            np.array([72, 72], np.int16),
            "lexicographic",
        ),
        (
            np.array([[0, 1], [0, 1]], dtype=np.int32),
            np.array([72, 72], np.int16),
            "lexicographic",
        ),
        (np.array([[0, 1]], dtype=np.int32), np.array([5], np.int16), "divide 72"),
        (np.array([[0, 1]], dtype=np.int32), np.array([0], np.int16), "divide 72"),
    ],
)
def test_fault_list_rejects_invalid_arrays(
    sets: np.ndarray, orbit_sizes: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FaultList(sets=sets, orbit_sizes=orbit_sizes)


def test_load_rejects_file_without_orbit_sizes(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    with path.open("wb") as f:
        np.savez(f, sets=np.array([[0, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="lacks the keys"):
        FaultList.load(path)
