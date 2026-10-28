"""Tests of reading the LETS files, the support lists and the incidence."""

import itertools
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from decoder_fixes_gross import lets
from decoder_fixes_gross.dem import dem_at
from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.lets import (
    LetsFile,
    LetsList,
    LetsOrbits,
    Supports,
    build_supports,
    induced_classes,
    read_lets_file,
    read_lets_files,
)
from decoder_fixes_gross.symmetry import column_permutations

SAMPLE = Path(__file__).parent / "data" / "lets_sample"


@pytest.mark.parametrize(
    ("name", "rows", "orbits", "missing"),
    [
        # Rows are the line counts of the files; orbits and missing rows are the counts of an
        # independent orbit reduction of 2026-10-12, reproduced on 2026-10-23.
        ("a_2_b_2.csv", 5040, 72, 144),
        ("a_2_b_3.csv", 6048, 84, 0),
        ("a_2_b_4.csv", 6693, 96, 219),
        ("a_2_b_5.csv", 9360, 130, 0),
        ("a_3_b_0.csv", 8640, 120, 0),
    ],
)
def test_fixture_file_has_known_rows_orbits_and_missing_rows(
    name: str, rows: int, orbits: int, missing: int
) -> None:
    read = read_lets_file(SAMPLE / name)
    lets = read.orbits
    assert (lets.a, lets.b) == (int(name[2]), int(name[6]))
    assert len(lets.orbits.sets) == orbits
    assert int(lets.published_counts.sum()) == rows
    assert int(lets.orbits.orbit_sizes.sum()) == rows + missing
    assert (lets.rows, lets.missing) == (rows, missing)
    assert read.dropped_lines == 0


def test_every_fixture_orbit_has_size_72() -> None:
    for path in sorted(SAMPLE.glob("*.csv")):
        assert set(read_lets_file(path).orbits.orbits.orbit_sizes.tolist()) == {72}


def test_unterminated_last_line_is_dropped_and_counted(tmp_path: Path) -> None:
    lines = (SAMPLE / "a_2_b_3.csv").read_text().splitlines(keepends=True)
    path = tmp_path / "a_2_b_3.csv"
    # The file cut in the middle of its last row: "8585,8586\n" becomes "8585,85".
    path.write_text("".join(lines[:-1]) + lines[-1][:7])
    read = read_lets_file(path)
    assert read.dropped_lines == 1
    assert read.orbits.rows == 6047
    # The dropped row's orbit is still complete, from the group.
    assert len(read.orbits.orbits.sets) == 84
    assert read.orbits.missing == 1


def test_complete_last_row_without_newline_is_dropped(tmp_path: Path) -> None:
    text = (SAMPLE / "a_2_b_3.csv").read_text()
    path = tmp_path / "a_2_b_3.csv"
    # The last row is whole but lacks its newline, as in the dataset's (5,4) file.
    path.write_text(text.removesuffix("\n"))
    read = read_lets_file(path)
    assert read.dropped_lines == 1
    assert read.orbits.rows == 6047
    assert len(read.orbits.orbits.sets) == 84
    assert read.orbits.missing == 1


def test_deleted_row_still_yields_its_full_orbit(tmp_path: Path) -> None:
    lines = (SAMPLE / "a_2_b_3.csv").read_text().splitlines(keepends=True)
    path = tmp_path / "a_2_b_3.csv"
    path.write_text("".join(lines[:100] + lines[101:]))
    lets = read_lets_file(path).orbits
    assert len(lets.orbits.sets) == 84
    assert lets.rows == 6047
    assert lets.missing == 1
    assert sorted(lets.published_counts.tolist())[:2] == [71, 72]


def test_repeated_row_is_rejected(tmp_path: Path) -> None:
    lines = (SAMPLE / "a_2_b_3.csv").read_text().splitlines(keepends=True)
    path = tmp_path / "a_2_b_3.csv"
    path.write_text("".join([*lines, lines[0]]))
    with pytest.raises(ValueError, match="repeats a row"):
        read_lets_file(path)


def test_file_name_without_class_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lets.csv"
    path.write_text("0,1\n")
    with pytest.raises(ValueError, match=r"a_<a>_b_<b>\.csv"):
        read_lets_file(path)


def test_row_of_wrong_width_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a_3_b_0.csv"
    path.write_text("0,1\n")
    with pytest.raises(ValueError, match="must have 3 entries"):
        read_lets_file(path)


def _orbits(sets: list[list[int]]) -> FaultList:
    return FaultList(
        sets=np.array(sets, dtype=np.int32), orbit_sizes=np.full(len(sets), 72, np.int16)
    )


@pytest.mark.parametrize(
    ("a", "counts", "message"),
    [
        (3, [1], "width a = 3"),
        (2, [0], "between 1 and its orbit size"),
        (2, [73], "between 1 and its orbit size"),
    ],
)
def test_lets_orbits_rejects_inconsistent_parts(a: int, counts: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LetsOrbits(a=a, b=0, orbits=_orbits([[0, 1]]), published_counts=np.array(counts, np.int16))


def test_lets_orbits_rejects_counts_of_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="published_counts must be int16"):
        LetsOrbits(a=2, b=0, orbits=_orbits([[0, 1]]), published_counts=np.array([1]))


def test_induced_classes_of_hand_worked_sets() -> None:
    # Columns {0, 1}, {0, 2}, {1, 2} and {2} as sets of checks.
    h = sp.csc_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.uint8))
    # Check degrees: {0, 1, 2} gives 2, 2, 2, so b = 0; {1, 2, 3} gives 1, 1, 3, so b = 3.
    triples = induced_classes(h, np.array([[0, 1, 2], [1, 2, 3]]))
    assert triples.b.tolist() == [0, 3]
    assert triples.max_check_degree.tolist() == [2, 3]
    # {0, 1} gives 2, 1, 1, so b = 2.
    pair = induced_classes(h, np.array([[0, 1]]))
    assert (pair.b.tolist(), pair.max_check_degree.tolist()) == ([2], [2])


def test_every_fixture_representative_has_its_file_class_and_is_elementary() -> None:
    h = dem_at(1e-3).h
    for path in sorted(SAMPLE.glob("*.csv")):
        lets = read_lets_file(path).orbits
        classes = induced_classes(h, lets.orbits.sets)
        assert set(classes.b.tolist()) == {lets.b}, path.name
        assert classes.max_check_degree.max() <= 2, path.name


def test_fixture_lets_of_class_b0_flip_no_observable() -> None:
    l = dem_at(1e-3).l.toarray()
    sets = read_lets_file(SAMPLE / "a_3_b_0.csv").orbits.orbits.sets
    assert not np.any(l[:, sets].sum(axis=2) % 2)


def test_induced_classes_rejects_column_outside_h() -> None:
    h = sp.csc_matrix(np.eye(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="column indices"):
        induced_classes(h, np.array([[0, 3]]))


def test_induced_classes_rejects_one_dimensional_sets() -> None:
    h = sp.csc_matrix(np.eye(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="two-dimensional integer array"):
        induced_classes(h, np.array([0, 1]))


def _fixture_classes(pattern: str) -> list[LetsOrbits]:
    return [read_lets_file(path).orbits for path in sorted(SAMPLE.glob(pattern))]


def test_u2_from_the_a2_fixtures_has_382_orbits() -> None:
    # 72 + 84 + 96 + 130: each weight-2 LETS is its own only weight-2 subset, and LETS of
    # different classes differ in b, so no orbit is shared.
    supports = build_supports(_fixture_classes("a_2_*.csv"))
    assert len(supports.lists[2].sets) == 382
    assert set(supports.lists[2].orbit_sizes.tolist()) == {72}


def test_u1_from_the_fixtures_matches_a_count_of_column_orbits() -> None:
    supports = build_supports(_fixture_classes("*.csv"))
    perms = column_permutations()
    rows = [np.loadtxt(p, np.int64, delimiter=",", ndmin=2) for p in SAMPLE.glob("*.csv")]
    columns = np.unique(np.concatenate([r.ravel() for r in rows]))
    column_orbits = {frozenset(perms[:, c].tolist()) for c in columns}
    assert len(supports.lists[1].sets) == len(column_orbits)
    assert len(column_orbits) <= 122


def test_a3_fixture_lets_are_their_own_weight_3_supports() -> None:
    supports = build_supports(_fixture_classes("a_3_b_0.csv"))
    lets = _fixture_classes("a_3_b_0.csv")[0]
    u3 = supports.lists[3]
    assert len(u3.sets) == 120
    incidence = supports.incidence[(3, 0)]
    assert incidence[3].shape == (120, 1)
    assert incidence[4].shape == (120, 0)
    np.testing.assert_array_equal(u3.sets[incidence[3][:, 0]], lets.orbits.sets)


def _class(a: int, b: int, sets: list[list[int]]) -> LetsOrbits:
    return LetsOrbits(a=a, b=b, orbits=_orbits(sets), published_counts=np.ones(len(sets), np.int16))


def test_subsets_shared_across_classes_are_listed_once() -> None:
    # Columns 0, 1 and 4 lie in three different column orbits, so no translation maps one
    # of the pairs {0, 1}, {0, 4}, {1, 4} to another, and U_2 has exactly three orbits. The
    # weight-2 class holds a translate of {0, 1}, which must not add a fourth.
    perms = column_permutations()
    assert len({frozenset(perms[:, c].tolist()) for c in (0, 1, 4)}) == 3
    translate_of_01 = sorted(perms[7, [0, 1]].tolist())
    supports = build_supports([_class(3, 0, [[0, 1, 4]]), _class(2, 2, [translate_of_01])])
    assert len(supports.lists[1].sets) == 3
    assert len(supports.lists[2].sets) == 3
    assert len(supports.lists[3].sets) == 1
    assert supports.lists[4].sets.shape == (0, 4)
    assert supports.incidence[(3, 0)][3].tolist() == [[0]]
    assert supports.incidence[(2, 2)][3].shape == (1, 0)


def test_incidence_points_each_subset_to_its_orbit() -> None:
    rng = np.random.default_rng(5)
    # Rows sorted, in lexicographic order, as a FaultList requires.
    sets = np.unique(
        np.sort([rng.choice(8784, 5, replace=False) for _ in range(6)], axis=1), axis=0
    )
    supports = build_supports([_class(5, 0, sets.tolist())])
    perms = column_permutations()
    for w in (3, 4):
        rows = supports.incidence[(5, 0)][w]
        assert rows.shape == (6, comb(5, w))
        for i, row in enumerate(sets.tolist()):
            for j, subset in enumerate(itertools.combinations(row, w)):
                images = {tuple(sorted(perm[list(subset)].tolist())) for perm in perms}
                assert tuple(supports.lists[w].sets[rows[i, j]].tolist()) in images


def test_build_supports_rejects_repeated_class() -> None:
    with pytest.raises(ValueError, match="distinct"):
        build_supports([_class(2, 2, [[0, 1]]), _class(2, 2, [[0, 4]])])


def _small_lets_list() -> LetsList:
    classes = (_class(2, 2, [[0, 1]]), _class(3, 0, [[0, 1, 4], [0, 4, 5]]))
    return LetsList(classes=classes, incidence=build_supports(classes).incidence)


def test_lets_list_round_trips_through_npz(tmp_path: Path) -> None:
    original = _small_lets_list()
    path = tmp_path / "lets.npz"
    original.save(path)
    loaded = LetsList.load(path)
    assert [(c.a, c.b) for c in loaded.classes] == [(2, 2), (3, 0)]
    for got, want in zip(loaded.classes, original.classes, strict=True):
        np.testing.assert_array_equal(got.orbits.sets, want.orbits.sets)
        np.testing.assert_array_equal(got.orbits.orbit_sizes, want.orbits.orbit_sizes)
        np.testing.assert_array_equal(got.published_counts, want.published_counts)
    assert loaded.incidence[(3, 0)][3].tolist() == [[0], [1]]
    assert loaded.incidence[(2, 2)][4].shape == (1, 0)
    assert loaded.content_hash() == original.content_hash()


def test_lets_list_content_hash_is_stable() -> None:
    # Recorded when the format was defined; a change here changes the LETS file's provenance.
    assert _small_lets_list().content_hash() == (
        "57f0fcc72ebde3f9868c2577b7278c2b807f3e7d509b00e99f605ce548e4ba7f"
    )


def test_lets_list_content_hash_changes_with_one_count() -> None:
    changed = LetsList(
        classes=(
            _class(2, 2, [[0, 1]]),
            LetsOrbits(
                a=3,
                b=0,
                orbits=_orbits([[0, 1, 4], [0, 4, 5]]),
                published_counts=np.array([1, 2], np.int16),
            ),
        ),
        incidence=_small_lets_list().incidence,
    )
    assert changed.content_hash() != _small_lets_list().content_hash()


def test_lets_list_rejects_classes_out_of_order() -> None:
    lets = _small_lets_list()
    with pytest.raises(ValueError, match="increasing order"):
        LetsList(classes=lets.classes[::-1], incidence=lets.incidence)


def test_lets_list_rejects_incidence_of_wrong_shape() -> None:
    lets = _small_lets_list()
    incidence = {
        **lets.incidence,
        (3, 0): {3: np.zeros((2, 2), np.int32), 4: lets.incidence[(3, 0)][4]},
    }
    with pytest.raises(ValueError, match=r"into U_3 must be int32 of shape \(2, 1\)"):
        LetsList(classes=lets.classes, incidence=incidence)


def test_lets_list_rejects_missing_incidence() -> None:
    lets = _small_lets_list()
    with pytest.raises(ValueError, match="one entry per class"):
        LetsList(classes=lets.classes, incidence={(2, 2): lets.incidence[(2, 2)]})


def test_lets_list_load_rejects_missing_array(tmp_path: Path) -> None:
    path = tmp_path / "lets.npz"
    arrays: dict[str, Any] = {"a2_b2/sets": np.array([[0, 1]], np.int32)}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="lacks"):
        LetsList.load(path)


def test_lets_list_load_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "lets.npz"
    np.savez(path, other=np.zeros(1))
    with pytest.raises(ValueError, match="not of the form"):
        LetsList.load(path)


def test_read_lets_files_reads_the_class_files_in_order_of_class() -> None:
    files = read_lets_files(SAMPLE)
    assert [(f.orbits.a, f.orbits.b) for f in files] == [(2, 2), (2, 3), (2, 4), (2, 5), (3, 0)]


def test_read_lets_files_rejects_directory_without_class_files(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("")
    with pytest.raises(ValueError, match="holds no file"):
        read_lets_files(tmp_path)


LETS_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "lets"
    / "ets_supports_full_detector_algorithm1_full_parallel"
)


@pytest.fixture(scope="module")
def full_lets() -> tuple[list[LetsFile], Supports]:
    if not LETS_DIR.is_dir():
        pytest.skip(f"the Zenodo LETS files are not unpacked at {LETS_DIR}; see data/README.md")
    files = read_lets_files(LETS_DIR)
    return files, build_supports([f.orbits for f in files])


@pytest.mark.slow
def test_full_dataset_reproduces_the_orbit_and_list_counts(
    full_lets: tuple[list[LetsFile], Supports],
) -> None:
    # Counts of an independent orbit reduction of 2026-10-12, reproduced on 2026-10-23.
    files, supports = full_lets
    orbits = {(f.orbits.a, f.orbits.b): len(f.orbits.orbits.sets) for f in files}
    assert orbits == {
        (2, 2): 72, (2, 3): 84, (2, 4): 96, (2, 5): 130,
        (3, 0): 120, (3, 2): 293, (3, 3): 751, (3, 4): 873, (3, 5): 1223,
        (4, 0): 557, (4, 2): 2448, (4, 3): 7279, (4, 4): 19470, (4, 5): 34675,
        (5, 0): 2702, (5, 2): 17471, (5, 3): 94786, (5, 4): 308960, (5, 5): 787081,
    }  # fmt: skip
    assert sum(orbits.values()) == 1_279_071
    assert sum(n for (a, _), n in orbits.items() if a <= 4) == 68_071
    # Translations preserve h, so the representative's class is the class of every row.
    dem = dem_at(1e-3)
    l = dem.l.toarray()
    for f in files:
        induced = induced_classes(dem.h, f.orbits.orbits.sets)
        assert set(induced.b.tolist()) == {f.orbits.b}, (f.orbits.a, f.orbits.b)
        assert induced.max_check_degree.max() <= 2, (f.orbits.a, f.orbits.b)
        if f.orbits.b == 0:
            assert not np.any(l[:, f.orbits.orbits.sets].sum(axis=2) % 2), f.orbits.a
    sizes = {key: f.orbits.orbits.orbit_sizes for key, f in zip(orbits, files, strict=True)}
    assert int(np.count_nonzero(sizes[(4, 4)] == 36)) == 85
    assert all(set(s.tolist()) == {72} for key, s in sizes.items() if key != (4, 4))
    assert sum(f.orbits.rows for f in files) == 92_088_579
    # Table I of arXiv:2608.11516. The paper also counts the unterminated last line of four
    # files, where our complete rows are one fewer; the other four unterminated lines, in
    # (4,0), (5,0), (5,3) and (5,5), it does not count.
    table_i = {
        (2, 2): 5040, (2, 3): 6048, (2, 4): 6693, (2, 5): 9360,
        (3, 0): 8640, (3, 2): 20918, (3, 3): 53826, (3, 4): 62649, (3, 5): 88056,
        (4, 0): 39995, (4, 2): 176256, (4, 3): 524088, (4, 4): 1398780, (4, 5): 2496600,
        (5, 0): 194457, (5, 2): 1257912, (5, 3): 6824502, (5, 4): 22244970, (5, 5): 56669793,
    }  # fmt: skip
    assert sum(table_i.values()) == 92_088_583
    counted_by_paper = {(3, 2), (3, 3), (3, 4), (5, 4)}
    for key, f in zip(orbits, files, strict=True):
        offset = int(key in counted_by_paper)
        assert f.orbits.rows == table_i[key] - offset, key
        if offset:
            assert f.dropped_lines == 1, key
    assert sum(f.orbits.missing for f in files) == 1_473
    assert len(supports.lists[3].sets) == 2_515_853
    assert len(supports.lists[4].sets) == 3_702_251
    assert len(supports.lists[1].sets) <= 122


def test_reading_in_small_chunks_gives_the_same_orbits(monkeypatch: pytest.MonkeyPatch) -> None:
    whole = read_lets_file(SAMPLE / "a_2_b_4.csv").orbits
    # 6,693 rows in chunks of 500, merged whenever more than 1,000 distinct rows are buffered.
    monkeypatch.setattr(lets, "_CHUNK_ROWS", 500)
    monkeypatch.setattr(lets, "_MERGE_ROWS", 1000)
    chunked = read_lets_file(SAMPLE / "a_2_b_4.csv").orbits
    np.testing.assert_array_equal(chunked.orbits.sets, whole.orbits.sets)
    np.testing.assert_array_equal(chunked.orbits.orbit_sizes, whole.orbits.orbit_sizes)
    np.testing.assert_array_equal(chunked.published_counts, whole.published_counts)
