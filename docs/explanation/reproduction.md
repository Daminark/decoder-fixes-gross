---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Reproducing the Relay-BP result

Every comparison of decoder fixes in this project rests on the harness: the support lists, the injection of a fault set, the scoring of the decode, and the floor built from the outcomes.
Before any fix is scored, we therefore check that the harness reproduces the one published result it can be held to, the Relay-BP numbers of Pacenti, Raveendran and Vasić (arXiv:2608.11516): the count of failing trapping sets per class in their Table II, and the error-floor estimate of their Fig. 1.
The rule deciding the comparison was fixed before our run.

We begin with the rule and the number found for each of its conditions, which together give the verdict.
We then compare the failing trapping sets class by class, where one class falls well short of the paper, and the floor against Fig. 1.
Finally we list every difference between our setup and the paper's.

```{code-cell} ipython3
---
tags: [remove-cell]
---
import io
import json
from pathlib import Path

from IPython.display import SVG, Markdown
from myst_nb import glue

from decoder_fixes_gross.plots import floor_against_fig1

RESULTS = Path("../../results")
gate = json.loads((RESULTS / "gate_relay_bp.json").read_text(encoding="utf-8"))
floor = json.loads((RESULTS / gate["inputs"]["floor"]).read_text(encoding="utf-8"))
build = json.loads((RESULTS / "lists.json").read_text(encoding="utf-8"))


def label(key):
    """Write a class key such as a5_b4 as (5,4)."""
    a, b = key.removeprefix("a").split("_b")
    return f"({a},{b})"


def sci(x, digits=2):
    """Write a small number as 9.5e-9, without the padded exponent of Python's format."""
    mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


conditions = gate["conditions"]
rule = gate["rule"]
glue("verdict", gate["verdict"], display=False)
glue("first_weight", conditions["first_failing_weight"]["found"], display=False)
glue("total", conditions["total"]["found"], display=False)
glue("paper_total", conditions["total"]["paper"], display=False)
glue("total_shortfall", conditions["total"]["shortfall"], display=False)
glue("deviation", conditions["total"]["deviation"], display=False)
glue("closed_total", floor["lets"]["total"]["closed"], display=False)
glue("failing_orbits", floor["by_weight"]["4"]["failing_orbits"], display=False)
glue("u4_orbits", floor["by_weight"]["4"]["orbits"], display=False)
glue("win_deviation", rule["win_deviation"], display=False)
glue("loss_deviation", rule["loss_deviation"], display=False)
glue("consistent_factor", rule["consistent_factor"], display=False)
glue("recheck_factor", rule["recheck_factor"], display=False)
for key, entry in gate["classes"].items():
    for field in ("published", "closed", "paper", "shortfall", "ratio"):
        glue(f"{key}_{field}", entry[field], display=False)
for point in gate["floor_check"]:
    tag = f"{point['p']:.0e}".replace("-0", "-")
    glue(f"reading_low_{tag}", sci(point["reading"]["low"], 1), display=False)
    glue(f"reading_high_{tag}", sci(point["reading"]["high"], 1), display=False)
    for form in ("occurrence", "exact_event"):
        glue(f"{form}_{tag}", sci(point[form]["per_round"]), display=False)
        glue(f"{form}_factor_{tag}", point[form]["factor"], display=False)
    glue(f"label_{tag}", point["occurrence"]["label"], display=False)
    (entry,) = [e for e in floor["floor"] if e["p"] == point["p"]]
    glue(f"occurrence_high_{tag}", sci(entry["occurrence"]["per_round"]["high"]), display=False)
glue("missing_total", build["totals"]["missing"], display=False)
glue("a5_b4_missing", build["classes"]["a5_b4"]["missing"], display=False)
settings = floor["decoder_settings"]
glue("legs", settings["num_sets"] + 1, display=False)
glue("pre_iter", settings["pre_iter"], display=False)
glue("set_max_iter", settings["set_max_iter"], display=False)
glue("gamma0", settings["gamma0"], display=False)
glue("gamma_low", settings["gamma_dist_interval"][0], display=False)
glue("gamma_high", settings["gamma_dist_interval"][1], display=False)
```

## The rule and the verdict

The rule compares four things with the paper: the lowest weight at which a support fails, the classes in which trapping sets fail, the total number N_fail of failing trapping sets, and the class with the most failures.
A trapping set fails when the decoder fails on any of its supports of weight 3 or 4, which is the paper's definition, and N_fail counts trapping sets as published in the paper's data files.
The run is a win when no support of weight 3 or below fails, the failures fall in exactly the five classes where the paper has them, the total is within {glue:text}`win_deviation:.0%` of the paper's, and (5,3) is the largest failing class.
It is a loss when a trapping set fails in one of the five classes where the paper has none, or when the total is more than {glue:text}`loss_deviation:.0%` from the paper's.
A failure at weight 3 or below would also be a loss, but only once every translate of it, decoded with several seeds, confirms it, since a randomized decoder can fail a single decode by chance; no such failure occurred.
Any other outcome is inconclusive.

```{code-cell} ipython3
---
tags: [remove-input]
---
def classes(found):
    return ", ".join(f"{label(key)} {n:,}" for key, n in found.items())


rows = [
    ("First failing weight", "4 or more", f"{conditions['first_failing_weight']['found']}"),
    ("Classes without failures in the paper", "none failing", classes(conditions["zero_classes"]["found"])),
    ("Classes with failures in the paper", "every one failing", classes(conditions["paper_classes"]["found"])),
    (
        "Total N_fail",
        f"within {rule['win_deviation']:.0%} of {conditions['total']['paper']:,}",
        f"{conditions['total']['found']:,} ({conditions['total']['deviation']:+.1%})",
    ),
    ("Largest failing class", "(5,3)", ", ".join(label(k) for k in conditions["largest_class"]["found"])),
]
holds = [conditions[key]["holds"] for key in conditions]
table = ["| Condition | Required | Found | Holds |", "| --- | --- | --- | --- |"]
table += [f"| {name} | {required} | {found} | {'yes' if ok else 'no'} |" for (name, required, found), ok in zip(rows, holds)]
Markdown("\n".join(table))
```

Every condition holds.
The first failure is at weight {glue:text}`first_weight:d`, in {glue:text}`failing_orbits:,d` of the {glue:text}`u4_orbits:,d` orbits of weight-4 supports, and the total of {glue:text}`total:,d` failing trapping sets is {glue:text}`total_shortfall:,d` short of the paper's {glue:text}`paper_total:,d`, a deviation of {glue:text}`deviation:.1%`.
The verdict is therefore a {glue:text}`verdict:s`.

## Failing trapping sets by class

The total hides how the deficit is shared among the classes, so we compare them one by one.
The table lists, for each class of Table II, our N_fail as published, the same count over the closed orbits, the paper's count, how far ours falls short of it, and the ratio of ours to the paper's.

```{code-cell} ipython3
---
tags: [remove-input]
---
table = [
    "| Class | Ours, as published | Ours, closed | Paper | Shortfall | Ratio |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for key, entry in gate["classes"].items():
    ratio = "" if entry["ratio"] is None else f"{entry['ratio']:.3f}"
    table.append(
        f"| {label(key)} | {entry['published']:,} | {entry['closed']:,} | {entry['paper']:,} "
        f"| {entry['shortfall']:,} | {ratio} |"
    )
Markdown("\n".join(table))
```

Three of the paper's five failing classes agree closely: (4,4) at {glue:text}`a4_b4_ratio:.1%` of the paper's count, (5,3) at {glue:text}`a5_b3_ratio:.1%` and (5,5) at {glue:text}`a5_b5_ratio:.1%`.
The class (4,5) is at {glue:text}`a4_b5_ratio:.0%`, a shortfall of {glue:text}`a4_b5_shortfall:,d` trapping sets, which is small in absolute terms.
The class (5,4) is not: we find {glue:text}`a5_b4_published:,d` failing trapping sets against the paper's {glue:text}`a5_b4_paper:,d`, i.e. {glue:text}`a5_b4_ratio:.0%` of them, and its shortfall of {glue:text}`a5_b4_shortfall:,d` is most of the total shortfall of {glue:text}`total_shortfall:,d`.
The rule conditions on the classes only through the five zero classes and the largest one, so the verdict does not depend on this gap, but the gap is the largest disagreement with the paper that we find.
We have not determined its cause.
The candidates are the following, and the differences of setup listed below say more about each:

- we decode one random translate per orbit, where the paper decodes every support once, and Relay-BP does not commute with the translations;
- the paper's Table I counts one more trapping set in (5,4) than the complete rows of its data file;
- the data file of (5,4) is one of those cut short in the published archive, and lacks {glue:text}`a5_b4_missing:,d` rows of its orbits.

The paper has no failing trapping set with 2 or 3 variable nodes and omits those classes from Table II; we find none either.

## The floor against Fig. 1

We now turn to the floor, i.e. the probability per round of the decoder failures on the supports, which the paper plots against a Monte Carlo simulation in Fig. 1.
The figure gives no numbers, so before our run we read the estimate off it at the two physical error rates where it can be read reliably: {glue:text}`reading_low_4e-4:s` to {glue:text}`reading_high_4e-4:s` per round at p = 4e-4 and {glue:text}`reading_low_1e-3:s` to {glue:text}`reading_high_1e-3:s` at p = 1e-3.
Readings off a log-scale plot are good to about a factor of 2, so the rule calls the floor consistent with a reading when it is within a factor of {glue:text}`consistent_factor:.0f` of the read range, and sends the probability model back for a recheck beyond a factor of {glue:text}`recheck_factor:.0f`; this check is informational and does not enter the verdict.
The figure below shows our floor over the whole grid of rates, in the two forms explained in [the floor estimate](floor-estimate.md), with the readings as vertical ranges.

```{code-cell} ipython3
---
tags: [remove-input]
---
figure = floor_against_fig1(floor, gate)
svg = io.StringIO()
figure.savefig(svg, format="svg")
SVG(svg.getvalue())
```

The occurrence floor, our headline, is {glue:text}`occurrence_4e-4:s` per round at p = 4e-4, a factor of {glue:text}`occurrence_factor_4e-4:.2f` below the read range, and {glue:text}`occurrence_1e-3:s` at p = 1e-3, a factor of {glue:text}`occurrence_factor_1e-3:.2f` below it.
The floor check is therefore {glue:text}`label_4e-4:s` at p = 4e-4 and {glue:text}`label_1e-3:s` at p = 1e-3, and at p = 1e-3 the band of one sampling error ([ADR 0003](../adr/0003-estimate-the-floor-from-one-translate-per-orbit.md)) reaches {glue:text}`occurrence_high_1e-3:s`, inside the read range.
The exact-event floor, the paper's printed Eq. (1), is {glue:text}`exact_event_4e-4:s` and {glue:text}`exact_event_1e-3:s` per round at the same points, a factor of {glue:text}`exact_event_factor_4e-4:.0f` and {glue:text}`exact_event_factor_1e-3:,.0f` below the readings.
It peaks near p = 4e-4 and falls above it, while the figure rises, so the printed formula cannot produce Fig. 1, while the occurrence probability is consistent with it.
Two read points do not single out the occurrence form among the forms that agree with it to a factor of 2, so which formula the authors computed remains open ([ADR 0005](../adr/0005-report-the-floor-as-an-occurrence-probability.md)).

## Differences from the paper's setup

The verdict is reached under a setup that differs from the paper's in the following ways, each of which could move a number on this page.

- **Closure.**
  Several of the paper's data files are cut short, so a few orbits are held only in part ({glue:text}`missing_total:,d` rows are missing in all), and a count as published can differ from one over closed orbits.
  The rule compares counts as published, as the paper's are, and we report the closed counts beside them: {glue:text}`closed_total:,d` in total against {glue:text}`total:,d` as published.
- **The unterminated last line.**
  Eight of the paper's data files end in a line without a newline.
  The paper's Table I counts that line in (3,2), (3,3), (3,4) and (5,4), and we do not, so our tested trapping sets in those classes are one fewer than Table I ([the failure lists](failure-lists.md)).
- **One translate per orbit.**
  The paper decodes every support in its files once.
  We decode one uniformly random translate per orbit of supports and count every member of the orbit as failing when it fails, as [ADR 0003](../adr/0003-estimate-the-floor-from-one-translate-per-orbit.md) explains.
  The floor is unbiased either way, but Relay-BP does not commute with the translations, so our class counts are an orbit-level approximation of the paper's, and each failing orbit adds all its trapping sets at once.
- **The error term.**
  The paper bounds the variance of its estimate by a decoder-independent sum over all supports (its Eq. (6)); we report the sampling error of ADR 0003, computed from the failing orbits.
- **The Relay-BP implementation.**
  We use the relay-bp package at version 0.2.0, pinned; the paper does not name its implementation or version.
- **Settings the paper does not state.**
  The paper uses {glue:text}`legs:d` relay legs and stops at the first syndrome-valid estimate, which it identifies with Relay-BP-1 of Müller et al. (arXiv:2506.01779); both match our settings.
  It does not state the remaining settings, for which we take that paper's values for the gross code: {glue:text}`pre_iter:d` iterations in the first leg and {glue:text}`set_max_iter:d` in each later one, a first memory strength of {glue:text}`gamma0:.3f`, and later strengths drawn from \[{glue:text}`gamma_low:.2f`, {glue:text}`gamma_high:.2f`\].
  The paper does not state its decoder seeds either; ours are drawn from the run seed, one stream per shard of orbits.
- **The form of the floor.**
  Our headline floor is the occurrence probability, while the paper prints the exact-event probability as its Eq. (1); the section above explains why.

None of these differences changes the verdict.
The harness reproduces the paper's Relay-BP result under the rule: the gate is a {glue:text}`verdict:s`.
