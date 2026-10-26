from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from decoder_fixes_gross.decoders import Decoding
from decoder_fixes_gross.dem import Dem
from decoder_fixes_gross.inject import Outcome, classify, decode_injected, syndrome


@pytest.fixture
def toy() -> Dem:
    """Three detectors, one observable, three physical columns and one added column.

    Column 3 has the detectors of columns 0 and 1 but not their observable, so {0, 1, 3}
    is a logical operator: undetectable and flipping the observable.
    """
    h = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    l = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    return Dem(h=sp.csc_matrix(h), l=sp.csc_matrix(l), priors=np.full(4, 0.1), p=1e-3, n_base=3)


def _vector(*columns: int) -> np.ndarray:
    v = np.zeros(4, dtype=np.uint8)
    v[list(columns)] = 1
    return v


def test_syndrome_of_two_faults_is_sum_of_their_columns(toy: Dem) -> None:
    # (1, 0, 0) + (1, 1, 0) = (0, 1, 0).
    assert syndrome(toy, [0, 1]).tolist() == [0, 1, 0]


def test_syndrome_of_single_fault_is_its_column(toy: Dem) -> None:
    assert syndrome(toy, [2]).tolist() == [0, 1, 1]


def test_syndrome_of_empty_set_is_zero(toy: Dem) -> None:
    assert syndrome(toy, []).tolist() == [0, 0, 0]


def test_injection_rejects_columns_beyond_n_base(toy: Dem) -> None:
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        syndrome(toy, [3])


@pytest.mark.parametrize("faults", [[-1], [0, 0], [[0, 1]], [0.0]])
def test_injection_rejects_invalid_fault_sets(toy: Dem, faults: list[Any]) -> None:
    with pytest.raises(ValueError, match="faults"):
        syndrome(toy, faults)


def test_correction_equal_to_faults_is_correct(toy: Dem) -> None:
    assert classify(toy, [0, 1], _vector(0, 1)) is Outcome.CORRECT


def test_correction_through_added_column_is_scored_on_full_model(toy: Dem) -> None:
    # Residual {0, 1, 3}: h r = (1,0,0) + (1,1,0) + (0,1,0) = 0 and l r = 1.
    assert classify(toy, [0, 1], _vector(3)) is Outcome.WRONG_LOGICAL


def test_correction_with_wrong_syndrome_is_no_solution(toy: Dem) -> None:
    # Residual {0, 2}: h r = (1,0,0) + (0,1,1) = (1,1,1).
    assert classify(toy, [2], _vector(0)) is Outcome.NO_SOLUTION


def test_zero_correction_of_nonzero_syndrome_is_no_solution(toy: Dem) -> None:
    assert classify(toy, [1], _vector()) is Outcome.NO_SOLUTION


def test_outcome_codes_match_run_output_convention() -> None:
    assert [Outcome.CORRECT, Outcome.WRONG_LOGICAL, Outcome.NO_SOLUTION] == [0, 1, 2]


@pytest.mark.parametrize("correction", [np.zeros(3), np.full(4, 2)])
def test_classify_rejects_malformed_corrections(toy: Dem, correction: np.ndarray) -> None:
    with pytest.raises(ValueError, match="correction"):
        classify(toy, [0], correction)


class _FixedDecoder:
    """Test double: records each syndrome and returns one fixed correction."""

    def __init__(self, correction: np.ndarray, iterations: int | None) -> None:
        self.correction = correction
        self.iterations = iterations
        self.syndromes: list[list[int]] = []

    def decode(self, syndrome: np.ndarray) -> Decoding:
        self.syndromes.append(syndrome.tolist())
        return Decoding(correction=self.correction, iterations=self.iterations)


def test_decode_injected_passes_syndrome_and_scores_correction(toy: Dem) -> None:
    decoder = _FixedDecoder(_vector(3), iterations=7)
    trial = decode_injected(toy, decoder, [0, 1])
    assert decoder.syndromes == [[0, 1, 0]]
    assert trial.outcome is Outcome.WRONG_LOGICAL
    assert trial.iterations == 7
    assert trial.seconds >= 0


def test_decode_injected_reports_missing_iteration_count(toy: Dem) -> None:
    trial = decode_injected(toy, _FixedDecoder(_vector(2), iterations=None), [2])
    assert trial.outcome is Outcome.CORRECT
    assert trial.iterations is None
