# 3. Estimate the known-list floor from one random translate per support orbit

Date: 2026-10-23

## Status

Accepted

Which probability of a support the estimate sums is set by [ADR 0005](0005-report-the-floor-as-an-occurrence-probability.md); the estimator and its error below apply to either form.

## Context

The trapping-set paper (Pacenti, Raveendran and Vasić, arXiv:2608.11516, Section IV) estimates a lower bound on a decoder's logical failure rate as a sum over fault supports.
For each weight w from 1 to 4, the supports are the distinct sets of w fault mechanisms that lie inside at least one enumerated leafless elementary trapping set.
Each support is injected as the exact fault set and decoded once, and the probability that exactly that set of faults occurs is added to the estimate when the decoder fails.
Weight-five sets are not part of the estimate.

The detector error model has 72 translation symmetries.
They preserve the check matrix and the priors exactly, and preserve the logical observables up to an invertible change of basis modulo the checks.
The probability of a support is therefore constant on a translation orbit, and a decoder that commutes with the translations fails on all or none of the supports in an orbit.
Orbit reduction takes the number of supports from several hundred million to a few million orbits.
Not every decoder commutes with the translations: randomized and schedule-dependent decoders can fail on some translates of an orbit and succeed on others.

## Decision

For each orbit we decode one translate, chosen uniformly from the 72 translations, and add the orbit size times the probability of the support when the decoder fails.
The orbit size is used because some orbits have only 36 elements.
A failure is any outcome other than a correction that reproduces both the syndrome and the logical observables.

For every decoder, the failure indicator on a uniformly random translate has as its mean the failure probability averaged over the orbit, so the estimate is unbiased for the paper's target, the sum over supports of failure probability times support probability.
The contribution of an orbit has variance at most its squared weight times q(1 - q), where q is the orbit-averaged failure probability, and variance zero for a deterministic decoder that commutes with the translations.
We report the square root of the sum of squared weights over the orbits that failed as the sampling error.
Its expectation is the sum of squared weights times q, which bounds the variance from above and is close to it when q is small, as it is for the failures of interest.

A second pass decodes all 72 translates, with several seeds, of every orbit that failed for any decoder, together with a seeded random sample of the orbits that passed.
It measures how far each decoder is from commuting with the translations and runs the expensive reference decoder.
Its failure fractions are never combined with the single-translate estimate.
Replacing the indicator by the second-pass fraction on orbits that failed once estimates q squared in place of q, and so biases the estimate low.

## Consequences

When failures are rare and a decoder does not commute with the translations, the variance of the estimate can exceed that of decoding every support by a factor up to the orbit size.
For decoders that do commute with the translations the reported error overstates the true sampling error, which is zero.
The estimate is an unbiased estimate of a lower bound: failures outside the tested supports, including every failure at weight five and above, are not counted.
Weight-five sets are also left out of the first pass because a circuit distance of 10 does not guarantee that they are correctable, so a failure there is not necessarily a failure of the decoder.
