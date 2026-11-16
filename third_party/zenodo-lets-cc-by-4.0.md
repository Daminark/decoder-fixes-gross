# Pacenti, Raveendran and Vasić, trapping-set supports of the gross code

The trapping-set supports read by `decoder_fixes_gross.lets` are the dataset "Trapping-set supports for the \[[144,12,12]\] bivariate-bicycle code", Zenodo record 21878685 (https://doi.org/10.5281/zenodo.21878685), archive `144_12_12_dem_LETS.zip`, md5 `1efaaa8083405c2b4fd0d8b090d26b98`.
It accompanies M. Pacenti, N. Raveendran and B. Vasić, arXiv:2608.11516, and was enumerated with https://github.com/Error-Correction-Lab/dem_trapping_sets.

Two committed files are derived from it:

- `tests/data/lets_sample/` holds five of its files unchanged: `a_2_b_2.csv`, `a_2_b_3.csv`, `a_2_b_4.csv`, `a_2_b_5.csv` and `a_3_b_0.csv`, each matching the archive's `SHA256SUMS`.
- `tests/data/relay_version_sample.npz` holds 2,654 of its supports, as column indices padded with -1, together with decoding outcomes computed in this project.

The dataset is distributed under the Creative Commons Attribution 4.0 International licence, https://creativecommons.org/licenses/by/4.0/.
The files in `tests/data/lets_sample/` are not modified; the sample in `relay_version_sample.npz` is a selection of rows with outcomes added.
