"""Tests of the stored circuit and the detector error model."""

from importlib.resources import files

import stim


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
