# 5. Report the floor as an occurrence probability

Date: 2026-10-29

## Status

Accepted

## Context

The trapping-set paper (Pacenti, Raveendran and Vasić, arXiv:2608.11516) estimates the error floor of a decoder as a sum, over the supports it fails on, of a probability of the support.
Its Eq. (1) gives that probability as the product of the priors of the faults in S times the product of one minus the prior over every other mechanism, i.e. the probability that exactly the faults in S occur.
On this model the second product is nearly the probability that no fault occurs at all, which is 1.3e-2 at p = 4e-4 and 2.0e-5 at p = 1e-3, since the priors sum to 10.8 at p = 1e-3.
The exact-event probability of a weight-4 set therefore peaks near p = 3.7e-4 and falls steeply above it.

For the Relay-BP baseline run, the floor under Eq. (1) is 3.2e-12 per round at p = 4e-4 and 1.9e-13 at p = 1e-3.
The paper's Fig. 1 reads 3e-10 to 4e-10 and 1e-8 to 2e-8 at the same points, and its estimate rises with p.
Replacing Eq. (1) by the product of the priors in S alone gives 2.4e-10 and 9.5e-9 per round, just below the lower end of the readings at both points and within the factor of about 2 to which a log-scale plot can be read.
The figure is therefore consistent with this occurrence probability, and the printed Eq. (1) cannot produce it.
Two points cannot distinguish the occurrence probability from other forms that stay within that factor of it at both points, such as the product of π/(1 - π) over S, i.e. the exact-event probability divided by the probability of no fault, which differs from it by at most 1.5% at p up to 1e-3.
Of the forms that fit, the occurrence probability is the one with a direct meaning, the probability that every fault in S occurs.

The two quantities answer different questions.
The events "exactly the faults in S occur" are disjoint for different S, so the exact-event sum over failing supports is a rigorous lower bound on the probability of the failures it covers.
The occurrence probability is the probability that every fault in S occurs, whatever else occurs; these events overlap, and counting each as a failure assumes that the decoder fails on every fault pattern containing S.
That assumption is the one behind the classical trapping-set estimate of an error floor, and the occurrence floor is the quantity that a Monte Carlo simulation at low p can test.

## Decision

We report the floor in both forms, each with the estimator and sampling error of [ADR 0003](0003-estimate-the-floor-from-one-translate-per-orbit.md) and each converted to a rate per round.
The occurrence floor is the headline, and it is the floor compared with Fig. 1 of the paper.
The exact-event floor, Eq. (1) as printed, is reported beside it.
The ceiling, the floor of a decoder that fails on every support, is reported in both forms too.

## Consequences

The headline floor is an estimate resting on the assumption above, not a lower bound; the exact-event line remains the rigorous one.
At p = 1e-3 a 12-round experiment has about eleven faults on average, so the assumption is strong there, and the gap between the two lines measures how much the headline depends on it.
Comparisons between decoders at a fixed p hardly depend on the choice: the two forms differ by the product of one minus the prior over the columns outside S, which varies between sets of weight at most 4 by under 2% at p up to 1e-3.
The finding that the printed Eq. (1) cannot produce the paper's Fig. 1 is part of the reproduction report.
