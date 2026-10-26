# Provenance of the stored circuit

`gross_z_r12.stim` is the syndrome-extraction circuit of the \[[144,12,12]\] bivariate bicycle (gross) code with l = 12 and m = 6, Z-basis memory, 12 rounds, every noise channel at probability 1e-3.

## Generator

- Repository: https://github.com/gongaa/SlidingWindowDecoder (MIT licence; see `third_party/gong-et-al.md`).
- Commit: `05d6b1f478f2b044effdc7477278647dfb99db07` (2025-05-13).
- Reference: A. Gong, S. Cammerer and J. M. Renes, "Toward Low-latency Iterative Decoding of QLDPC Codes Under Circuit-Level Noise", arXiv:2403.18901.

## Call

With `src/codes_q.py` and `src/build_circuit.py` of that commit, and Stim 1.16.0.
`codes_q.py` imports `.utils` relatively, so the modules must be loaded as part of a package, but the repository's `src/__init__.py` imports Cython extensions that would have to be built first.
The call therefore registers a stub package whose path is the repository's `src/` and loads the two modules from it, without running that `__init__.py`:

```python
import importlib
import sys
import types

package = types.ModuleType("gong")
package.__path__ = ["<clone of the repository>/src"]
sys.modules["gong"] = package
codes_q = importlib.import_module("gong.codes_q")
build_circuit = importlib.import_module("gong.build_circuit")

code, A_list, B_list = codes_q.create_bivariate_bicycle_codes(12, 6, [3], [1, 2], [1, 2], [3])
circuit = build_circuit.build_circuit(code, A_list, B_list, p=0.001, num_repeat=12)
text = str(circuit) + "\n"
```

The defaults `z_basis=True`, `use_both=False` and `HZH=False` give the Z-memory circuit with Z detectors only.
The stored file is `text`, 105,309 bytes, sha256 `ed17942e821ca2cda0e824994a0f5fc0bfd7fc5b4c71f3d2c3c853d22b4b77f0`.

## Equality check

The same call through the `build_circuit` of https://github.com/Error-Correction-Lab/dem_trapping_sets (commit `b9ac6a1133e1db71a99d253e523ebca524117142`, `examples/include_py`), which builds the model indexed by the published trapping-set lists, gives a circuit equal to this one under `stim.Circuit.__eq__`, and the same text byte for byte.
Checked on 2026-10-24.

## Noise arguments

Every noise argument in the file (`X_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`) equals 1e-3, so the circuit at physical error rate p is this circuit with every noise argument multiplied by p / 1e-3.
