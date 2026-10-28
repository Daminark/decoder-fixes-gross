import numpy as np
import pytest

from decoder_fixes_gross.decoders import RelayBP
from decoder_fixes_gross.dem import Dem, dem_at
from decoder_fixes_gross.inject import Outcome, decode_injected, syndrome


@pytest.fixture(scope="module")
def dem() -> Dem:
    return dem_at(1e-3)


def test_relay_bp_corrects_200_seeded_single_faults(dem: Dem) -> None:
    columns = np.random.default_rng(20261026).choice(dem.n_base, size=200, replace=False)
    decoder = RelayBP(dem, seed=0)
    trials = [decode_injected(dem, decoder, [column]) for column in columns]
    assert [t.outcome for t in trials] == [Outcome.CORRECT] * 200
    assert all(t.iterations is not None and t.iterations >= 1 for t in trials)


def test_new_decoder_with_same_seed_reproduces_decode(dem: Dem) -> None:
    detectors = syndrome(dem, [0, 1000, 5000])
    first = RelayBP(dem, seed=3).decode(detectors)
    second = RelayBP(dem, seed=3).decode(detectors)
    assert np.array_equal(first.correction, second.correction)
    assert first.iterations == second.iterations


def test_zero_syndrome_converges_on_first_iteration(dem: Dem) -> None:
    # The all-zero estimate reproduces a zero syndrome, so the first iteration converges.
    decoding = RelayBP(dem, seed=0).decode(np.zeros(936, dtype=np.uint8))
    assert decoding.iterations == 1
    assert not decoding.correction.any()


def test_correction_covers_every_model_column(dem: Dem) -> None:
    decoding = RelayBP(dem, seed=0).decode(syndrome(dem, [42]))
    assert decoding.correction.shape == (8784,)
    assert decoding.correction.dtype == np.uint8


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_relay_bp_rejects_seed_out_of_range(dem: Dem, seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        RelayBP(dem, seed=seed)


@pytest.mark.parametrize("seed", [1.5, True])
def test_relay_bp_rejects_non_integer_seed(dem: Dem, seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        RelayBP(dem, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("detectors", [np.zeros(935, np.uint8), np.full(936, 2, np.uint8)])
def test_relay_bp_rejects_malformed_syndrome(dem: Dem, detectors: np.ndarray) -> None:
    with pytest.raises(ValueError, match="syndrome"):
        RelayBP(dem, seed=0).decode(detectors)


@pytest.mark.slow
def test_seed_dependent_weight_4_set_fails_for_some_seed(dem: Dem) -> None:
    # A weight-4 LETS support that failed in 7 of 10 seeds in an earlier run with relay-bp
    # 0.2.2; if each seed fails with probability 0.7, none of 10 failing has chance 6e-6.
    # A decode that finds no syndrome-valid estimate runs every leg: 80 + 300 x 60.
    trials = [
        decode_injected(dem, RelayBP(dem, seed=s), [7852, 8073, 8120, 8147]) for s in range(10)
    ]
    failures = [t for t in trials if t.outcome is not Outcome.CORRECT]
    assert failures
    assert all(t.iterations == 18_080 for t in failures if t.outcome is Outcome.NO_SOLUTION)
