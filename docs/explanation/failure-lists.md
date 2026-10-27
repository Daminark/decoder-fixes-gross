# Trapping sets and the support lists

The known-list floor is a sum over a fixed universe of fault sets, and this page describes where that universe comes from.
We begin with the trapping sets that Pacenti, Raveendran and Vasić enumerate for the gross-code model (arXiv:2608.11516) and the (a, b) classes they are sorted into.
We then reduce the published files to translation orbits, which is complicated by the fact that ten of the files are cut short.
Finally we define the support lists U_1 to U_4 and the incidence that links each trapping set to its supports.

## Trapping sets and their classes

Recall that the model is a check matrix `h` whose columns are fault mechanisms and whose rows are detectors, i.e. a Tanner graph with a variable node per column and a check node per detector.
A set S of columns induces a subgraph: the columns of S, the checks they touch, and the edges of `h` between them.
The degree of a check in this subgraph is the number of columns of S that flip it, so a check of odd degree is a detector that the fault set S lights up.

A set S belongs to the class (a, b) when it has a columns and exactly b of its induced checks have odd degree (Pacenti et al., Sec. III.A).
The set is *elementary* when every induced check has degree one or two.
An elementary set has a compact picture: its columns are points, each degree-two check is an edge joining the two columns it touches, and the degree-one checks are dropped.
The set is a *leafless elementary trapping set* (LETS) when, in addition, every point of this picture lies on at least two edges.

The interest of such a set is that its syndrome is small compared with its size, so an iterative decoder sees little local evidence against the faults inside it.
An (a, 0) set flips no detector at all, and is a degeneracy of the model rather than an error the decoder could see.

## The published enumeration

The dataset (Zenodo record 21878685) lists every LETS with a from 2 to 5 and b in {0, 2, 3, 4, 5}, one file per class, each row a LETS as column indices in the column order of the model.
There is no (2, 0) class, and the 19 files hold 92,088,579 complete rows.

The model has 72 translation symmetries, described in [the model and its symmetry](model-and-symmetry.md).
Translations preserve `h`, so the subgraph induced by a translate of S is the subgraph induced by S with its nodes relabelled, and the class, the check degrees and the leafless condition are the same on a whole orbit.
The enumeration is exhaustive, so it contains whole orbits, and we store one canonical representative per orbit.
This gives 1,279,071 orbits, 68,071 of them with a at most 4.
Every orbit has 72 members except 85 orbits in class (4, 4), which have 36.

The same invariance makes the files cheap to check.
For every representative we recompute b and the largest induced check degree from `h`, and require b to be the class of its file and the degree to be at most 2.
Checking the representative checks every row of its orbit, so this tests all 92 million rows against our model and its column order, at the cost of 1.28 million checks.

## Cut files and orbit closure

Ten of the files end on a 4,096-byte boundary, and eight of these end in the middle of a row.
The cut loses the rows that would have followed, so an orbit near the end of such a file may have only some of its members in the file.
In all, 1,473 rows are absent from orbits that the files otherwise contain.
We drop the unterminated last line of a file, since it is a fragment of a row in all but one case, and we count it.

Orbit closure restores the missing rows without reading them.
A row that is present determines its orbit through its canonical form, and the size of the orbit comes from the group, not from the rows that happen to be present.
Every orbit therefore keeps its full size even when some of its translates are missing.

The paper's class counts (Table I) are the rows present in the files, so the paper itself worked from the cut files.
The one difference is that Table I also counts the unterminated last line in four classes, (3, 2), (3, 3), (3, 4) and (5, 4), so our published counts there sit one below it; we count complete rows only, since three of those four lines are fragments rather than rows.
We therefore keep two counts per orbit: the orbit size, which is the closed count, and the number of rows of the file that fall in the orbit, which we call the published count.
A comparison with the paper uses published counts, and closure is reported as a separate line.
The difference is 1,473 of 92 million rows, so closure changes the totals by under 0.002%.

## The support lists

The estimate of the paper (Sec. IV) injects fault sets of weight 1 to 4 that lie inside enumerated LETS.
For each weight w, the support list U_w is the set of distinct weight-w subsets of all LETS with at least w columns.
A LETS with exactly w columns is a weight-w subset of itself, so the LETS of classes (2, b), (3, b) and (4, b) are in U_2, U_3 and U_4 respectively.

The union matters because a small set can lie inside many LETS, for instance when one LETS contains another.
Each fault set is a distinct physical event with its own probability, so it must be counted once, however many LETS contain it.

The lists are again stored as orbits.
A subset of a translate of a LETS is a translate of a subset of the LETS, so the subsets of the representatives alone reach every orbit of subsets.
We canonicalize each subset, deduplicate across all classes, and obtain 122, 166,051, 2,515,853 and 3,702,251 orbits in U_1 to U_4.
U_1 is every column orbit of the model, so every fault mechanism lies in some LETS.

The weight-5 LETS contribute their subsets to U_1 to U_4 but are not injected themselves.
They lie outside the paper's estimator, and [ADR 0003](../adr/0003-estimate-the-floor-from-one-translate-per-orbit.md) records why we leave them out.

## From supports back to trapping sets

The paper counts a LETS as failing when at least one of its weight-3 or weight-4 subsets fails when injected.
To evaluate this after the supports have been decoded, we record, for every LETS orbit, the rows of U_3 and U_4 that hold the orbits of its subsets.
A LETS with five columns has ten weight-3 subsets and five weight-4 subsets, one with four columns has four and one, and one with three columns is its own single weight-3 support.

The decoded supports are translates chosen at random, so the outcome of a LETS is read off the orbits of its subsets rather than off the subsets of a particular translate.
For a decoder that commutes with the translations the two agree.
For any other decoder it is an orbit-level approximation, which the floor estimate does not depend on, since the floor is a sum over supports and not over LETS.
