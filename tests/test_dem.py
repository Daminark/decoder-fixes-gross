"""Tests of the stored circuit and the detector error model."""

from importlib.resources import files

import numpy as np
import pytest
import stim
from gf2 import rank

from decoder_fixes_gross.dem import Dem, dem_at


def _stored_circuit() -> stim.Circuit:
    text = (files("decoder_fixes_gross") / "data" / "gross_z_r12.stim").read_text()
    return stim.Circuit(text)


def test_stored_circuit_loads_as_package_data() -> None:
    circuit = _stored_circuit()
    assert circuit.num_detectors == 936
    assert circuit.num_observables == 12


def test_every_noise_argument_in_stored_circuit_is_1e_3() -> None:
    # Measurements count as noisy gates in Stim; an argument on one would be a flip probability.
    flat = _stored_circuit().flattened()
    args = []
    for k in range(len(flat)):
        instruction = flat[k]
        assert isinstance(instruction, stim.CircuitInstruction)
        if stim.gate_data(instruction.name).is_noisy_gate:
            args += instruction.gate_args_copy()
    assert args
    assert set(args) == {0.001}


@pytest.fixture(scope="module")
def dem() -> Dem:
    return dem_at(1e-3)


# The oracles below come from an independent build of the same model with the
# dem_trapping_sets preset (Error-Correction-Lab/dem_trapping_sets), 2026-10-12.


def test_model_has_936_detectors_12_observables_and_8784_columns(dem: Dem) -> None:
    assert dem.h.shape == (936, 8784)
    assert dem.l.shape == (12, 8784)
    assert dem.priors.shape == (8784,)
    assert dem.n_base == 8784


def test_column_weights_match_independent_build(dem: Dem) -> None:
    weights = np.diff(dem.h.indptr)
    values, counts = np.unique(weights, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {
        2: 864,
        3: 5328,
        4: 864,
        5: 864,
        6: 864,
    }


def test_every_column_has_a_distinct_detector_pattern(dem: Dem) -> None:
    patterns = {tuple(dem.h.indices[dem.h.indptr[j] : dem.h.indptr[j + 1]]) for j in range(8784)}
    assert len(patterns) == 8784


def test_rank_of_h_is_930(dem: Dem) -> None:
    assert rank(dem.h.toarray()) == 930


def test_priors_take_the_eight_values_of_independent_build(dem: Dem) -> None:
    # Sums of the same probabilities in different orders differ in the last bits.
    distinct = np.unique(np.round(dem.priors, 12))
    expected = [
        2.667e-4,
        5.333e-4,
        1.1998e-3,
        1.2662e-3,
        1.3323e-3,
        2.7279e-3,
        3.0636e-3,
        3.5899e-3,
    ]
    np.testing.assert_allclose(distinct, expected, rtol=1e-3)


def test_prior_sum_and_no_fault_probability_pin_the_merge_rule(dem: Dem) -> None:
    np.testing.assert_allclose(dem.priors.sum(), 10.79687079, rtol=1e-9)
    np.testing.assert_allclose(np.prod(1 - dem.priors), 2.02239e-5, rtol=1e-5)


def test_model_at_1e_4_has_same_columns_and_priors_near_one_tenth(dem: Dem) -> None:
    low = dem_at(1e-4)
    np.testing.assert_array_equal(low.h.toarray(), dem.h.toarray())
    np.testing.assert_array_equal(low.l.toarray(), dem.l.toarray())
    ratio = low.priors / dem.priors
    assert ratio.min() >= 0.095
    assert ratio.max() <= 0.105


@pytest.mark.parametrize("p", [0.0, -1e-3, 1.0])
def test_dem_at_rejects_rate_outside_unit_interval(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        dem_at(p)


def test_dem_rejects_priors_with_wrong_length(dem: Dem) -> None:
    with pytest.raises(ValueError, match="same number of columns"):
        Dem(h=dem.h, l=dem.l, priors=dem.priors[:-1], p=dem.p, n_base=dem.n_base)


@pytest.mark.parametrize("n_base", [0, 8785])
def test_dem_rejects_n_base_outside_columns(dem: Dem, n_base: int) -> None:
    with pytest.raises(ValueError, match="n_base must be in"):
        Dem(h=dem.h, l=dem.l, priors=dem.priors, p=dem.p, n_base=n_base)
