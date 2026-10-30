# The floor estimate

A tier-1 run records, for each orbit of a support list, whether the decoder failed on one random translate of it.
This page explains how those outcomes become a failure probability.
We begin with the probability of a single fault set, which comes in two forms, and explain why the headline floor uses the one consistent with Fig. 1 of the trapping-set paper (Pacenti, Raveendran and Vasić, arXiv:2608.11516) rather than the one it prints.
We then turn to the estimator built from one translate per orbit and its sampling error, as decided in [ADR 0003](../adr/0003-estimate-the-floor-from-one-translate-per-orbit.md).
Finally we discuss the ceiling that bounds the estimate, the count of failing trapping sets, and the conversion to a rate per round.

## The probability of a fault set

Recall that the model has one column per fault mechanism, and that column i occurs independently with its prior π_i, the prior of the circuit at physical error rate p.
The decoder is always built from the model at p = 1e-3, as in the paper, so p enters only through the priors of the faults that occur.

Eq. (1) of the paper gives the probability that exactly the faults in a set S occur, and no other:

```{math}
\Pr{}_{\mathrm{exact}}(S; p) = \prod_{i \in S} \pi_i \prod_{j \notin S} (1 - \pi_j).
```

The second product runs over all 8,784 columns but S, so it is nearly the same for every set.
We therefore compute the log-probability as

```{math}
\log \Pr{}_{\mathrm{exact}}(S; p) = \sum_{i \in S} \log \frac{\pi_i}{1 - \pi_i} + \sum_{j} \log (1 - \pi_j),
```

in which the second sum is computed once per p, and each set costs w additions.
The logarithm of 1 - π_j is evaluated with `log1p`, which keeps its relative precision when π_j is small.

The second sum is the logarithm of the probability that no fault occurs at all, and on this model it is far from negligible.
At p = 1e-3 the priors sum to 10.8, so a 12-round experiment has about eleven faults on average, and the probability of no fault is 2.0e-5; at p = 4e-4 it is 1.3e-2.
A set of weight w is then an atypically quiet outcome, and its exact-event probability behaves like λ^w e^(-λ), where λ, the expected number of faults, is proportional to p.
This is largest at λ = w, which for w = 4 is near p = 3.7e-4, and falls off quickly above it.

This is incompatible with Fig. 1 of the paper, whose floor estimate rises with p.
For the Relay-BP baseline the exact-event floor is 3.2e-12 per round at p = 4e-4 and 1.9e-13 at p = 1e-3, while the figure reads 3e-10 to 4e-10 and 1e-8 to 2e-8 at the same points.
Dropping the product over the other columns leaves the occurrence probability,

```{math}
\Pr{}_{\mathrm{occ}}(S; p) = \prod_{i \in S} \pi_i ,
```

the probability that every fault in S occurs, whatever else occurs.
With it the same failures give 2.4e-10 and 9.5e-9 per round.
Values read off a log-scale plot are accurate to about a factor of 2, and both sit just below the lower end of the readings, at 0.8 and 0.95 of it.
The figure is therefore consistent with the occurrence probability, while the printed Eq. (1) falls where the figure rises and is about 100 times too low at p = 4e-4 and 5e4 to 1e5 times too low at p = 1e-3.
Two points cannot single out the occurrence form, however: any form that stays within that factor of it at both points fits as well, for instance the exact-event probability divided by the probability of no fault, i.e. the product of π_i / (1 - π_i) over S, which differs from the occurrence form by at most 1.5% at p up to 1e-3.
Which form the authors computed remains open.

The two forms answer different questions.
The events "exactly the faults in S occur" are disjoint for different S, so the exact-event floor is a rigorous lower bound on the probability of the failures it covers.
The occurrence events overlap, and counting each of them as a failure assumes that the decoder fails on every fault pattern containing S, which is the assumption behind the classical trapping-set estimate of an error floor.
We report both, with the occurrence floor as the headline: of the forms that fit the figure, it is the one with a direct meaning, the probability that every fault in S occurs, and it is the quantity a Monte Carlo simulation can test ([ADR 0005](../adr/0005-report-the-floor-as-an-occurrence-probability.md)).
The rest of this page applies to either form, and writes Pr(S; p) for the one in use.

## One translate per orbit

The quantity of interest is the sum, over every support S in the lists, of Pr(S; p) times the probability f(S) that the decoder fails on S.
The 72 translations preserve the priors, so Pr(S; p) is constant on an orbit.
If the decoder commuted with the translations, f would be constant on orbits too, and decoding the representative would suffice.
Relay-BP and BP+OSD do not commute with them, so f can vary within an orbit.

Tier 1 therefore decodes one translate per orbit, drawn uniformly from the 72 group elements.
Let m_O be the mass of orbit O, i.e. its size times Pr(S; p) for any S in it, and let X_O be 1 when the decode failed and 0 otherwise.
The estimate is

```{math}
\hat P = \sum_O m_O X_O .
```

The translate is uniform, so the expectation of X_O is q_O, the failure probability averaged over the orbit, and the expectation of the estimate is the sum over orbits of m_O q_O, which is the target.
This holds for any decoder, equivariant or not.
The orbit size enters because each orbit stands for that many supports, and it is not a constant factor because not every orbit has 72 members: 410 support orbits of weight 2 and 4 have 36, and 12 of weight 3 have 24.

## The sampling error

Each term m_O X_O is an independent Bernoulli variable scaled by m_O, so its variance is m_O^2 q_O (1 - q_O), which is at most m_O^2 q_O.
The q_O are unknown, but the sum over orbits of m_O^2 X_O has expectation exactly the sum of m_O^2 q_O.
We report its square root,

```{math}
\sigma = \Big( \sum_{O \text{ failed}} m_O^2 \Big)^{1/2},
```

as the sampling error.
Its square is therefore an upwardly biased estimate of the variance: nearly unbiased when the q_O are small, and increasingly an overestimate as they approach 1.
The q_O of the failing orbits are not measured here; tier 2, which decodes all 72 translates, measures them.
Nothing similar holds for σ itself: by Jensen's inequality its expectation is at most the square root of that of σ², so σ is a plug-in estimate of the standard deviation rather than a bound on it.
How far to trust it depends on how many orbits carry the estimate.
The ratio (P̂/σ)² is the number of equal-mass failing orbits that would give the same ratio of estimate to error, and for the Relay-BP baseline it is about 32 at every p, out of 450 failing orbits.
With so few effective terms, σ is itself a noisy estimate, so the band is best read as a rough error and not as a confidence interval.
For a decoder that commutes with the translations the true variance is zero, so σ then overstates the error.

Notice that σ never exceeds the estimate, since the square root of a sum of squares of non-negative terms is at most their sum.
The lower end of the band, the estimate minus σ, is therefore never negative, and it is zero exactly when at most one orbit failed.

## The ceiling

The estimate cannot exceed the mass of the whole list, the sum of m_O over every orbit, which is the floor of a decoder that failed everywhere.
We report this ceiling for each list and each p, in both forms.
The occurrence ceiling is a sum over overlapping events, so it can exceed one: at p = 1e-4 the weight-1 list alone has an occurrence ceiling equal to the sum of the priors.
It is the first quantity to check when a floor looks wrong, since a floor above it is an arithmetic error, and a value from the literature above it cannot be a sum of the form above over these supports.

## Failing trapping sets

The paper counts failures by trapping set as well as by support (its N_fail).
A trapping-set orbit fails when one of its subsets of weight 3 or 4, as listed by the incidence of [the support lists](failure-lists.md), failed at tier 1.
It counts once, however many of its supports failed.
For a decoder that commutes with the translations this is exact.
Otherwise it is an orbit-level approximation, since the members of a trapping-set orbit contain different translates of each support, and tier 1 decoded only one of them.

Each class reports N_fail twice.
The published count adds up, over failing orbits, the rows of the published files that fall in each orbit, and is the number to compare with the paper, which counted those rows.
The closed count adds up the orbit sizes, and differs from it only where a file was cut short.
The floor uses orbit sizes, i.e. the closed lists.

## The rate per round

The estimate is a probability per experiment of 12 rounds.
The paper converts it to a rate per round by its Eq. (7),

```{math}
p_L = 1 - (1 - \hat P)^{1/12}.
```

For an estimate far below one, the direct formula loses most of its digits to cancellation, since 1 - P̂ rounds away the low digits of P̂; we evaluate it as -expm1(log1p(-P̂) / 12).
The conversion is not linear, so we convert the two ends of the band P̂ - σ and P̂ + σ separately rather than converting σ.
Both forms of the floor are converted the same way.
The occurrence floor is not a probability, since its events overlap, so for a decoder that fails often it can reach 1 at the top of the p grid.
The per-round rate is undefined there, and the record stores null for each of the value and the two ends that reaches 1, while the per-experiment value and error are still written and may exceed 1.

The numbers themselves are written by the aggregation command to `results/`, one file per run and one for the ceiling; [reproducing the figures](../howto/reproduce-figures.md) lists the commands.
