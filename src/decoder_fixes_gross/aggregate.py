"""Aggregates of a tier-1 run: failures by list, weight, outcome and LETS class, the floor, and cost.

A run (:mod:`decoder_fixes_gross.runs`) holds one outcome per orbit of each of its fault
lists. Every aggregate here is a function of those outcomes, the lists, the LETS incidence
(:mod:`decoder_fixes_gross.lets`) and the priors at each physical error rate. The floor and
its sampling error follow ADR 0003 (:mod:`decoder_fixes_gross.floor`), and both are reported
in the occurrence and the exact-event form of ADR 0005. A LETS orbit fails
when any support orbit in its weight-3 or weight-4 incidence failed, the definition of N_fail
of Pacenti, Raveendran and Vasić (arXiv:2608.11516); this is exact for decoders that commute
with the translations and an orbit-level approximation otherwise.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.floor import FORMS, estimate, orbit_masses
from decoder_fixes_gross.inject import Outcome
from decoder_fixes_gross.lets import INCIDENCE_WEIGHTS, LetsList
from decoder_fixes_gross.runs import CONFIG_NAME, RunConfig, Shard, shard_name

# Percentile reported as the tail of the decode cost.
TAIL_PERCENTILE = 99


@dataclass(frozen=True, eq=False)
class RunOutcomes:
    """The per-orbit results of a complete run, joined across shards.

    Attributes
    ----------
    config : RunConfig
        The run's configuration.
    outcomes, micros, iterations : Mapping of str to numpy.ndarray
        By list name, one entry per orbit in list order, with the dtypes of
        :class:`~decoder_fixes_gross.runs.Shard`.
    digest : str
        sha256 hex digest of the bytes of ``config.json`` and of every shard file, in shard
        order; it identifies the run's files as an input of an aggregate.
    """

    config: RunConfig
    outcomes: Mapping[str, NDArray[np.uint8]]
    micros: Mapping[str, NDArray[np.uint32]]
    iterations: Mapping[str, NDArray[np.int32]]
    digest: str


def read_run(run_dir: Path) -> RunOutcomes:
    """Read a complete run.

    Raises
    ------
    ValueError
        If a shard of the configuration has no file, or a shard's length differs from the
        number of orbits it covers.
    """
    config = RunConfig.load(run_dir)
    digest = hashlib.sha256()
    digest.update((run_dir / CONFIG_NAME).read_bytes())
    parts: dict[str, list[Shard]] = {record.name: [] for record in config.lists}
    for spec in config.shards():
        path = run_dir / shard_name(spec.index)
        if not path.exists():
            raise ValueError(f"{run_dir} is incomplete: {path.name} is missing")
        digest.update(path.read_bytes())
        shard = Shard.load(path)
        if len(shard) != spec.stop - spec.start:
            raise ValueError(
                f"{path} holds {len(shard)} orbits, but covers rows {spec.start} to {spec.stop}"
            )
        parts[spec.list_name].append(shard)

    def joined(name: str) -> dict[str, NDArray[Any]]:
        return {
            list_name: np.concatenate([getattr(s, name) for s in shards])
            for list_name, shards in parts.items()
        }

    return RunOutcomes(
        config=config,
        outcomes=joined("outcomes"),
        micros=joined("micros"),
        iterations=joined("iterations"),
        digest=digest.hexdigest(),
    )


def failing_lets(
    lets: LetsList, failed: Mapping[int, NDArray[np.bool_]]
) -> dict[tuple[int, int], NDArray[np.bool_]]:
    """Return, per (a, b) class, whether each LETS orbit has a failing support orbit.

    Parameters
    ----------
    lets : LetsList
        The LETS orbits and their incidence into U_3 and U_4.
    failed : Mapping of int to numpy.ndarray
        For w = 3 and 4, a bool array over the orbits of U_w.

    Returns
    -------
    dict of (int, int) to numpy.ndarray
        bool of shape [N] per class; a class with a < 3 has no support of weight 3 or 4 and
        never fails.
    """
    return {
        (c.a, c.b): np.logical_or.reduce(
            [failed[w][lets.incidence[(c.a, c.b)][w]].any(axis=1) for w in INCIDENCE_WEIGHTS]
        )
        for c in lets.classes
    }


def summarize(
    run: RunOutcomes,
    lists: Mapping[str, FaultList],
    lets: LetsList,
    priors_by_p: Mapping[float, NDArray[np.float64]],
) -> dict[str, Any]:
    """Return the aggregates of a run as a JSON-serializable record.

    Parameters
    ----------
    run : RunOutcomes
        The run; its lists must include ``u3`` and ``u4``, the lists the LETS incidence
        indexes.
    lists : Mapping of str to FaultList
        Every list of the run, by name.
    lets : LetsList
        The LETS, with incidence into the run's ``u3`` and ``u4``.
    priors_by_p : Mapping of float to numpy.ndarray
        The priors of the model at each physical error rate p at which to evaluate the floor.

    Returns
    -------
    dict
        ``inputs``: the hashes of the run's files, of each list and of the LETS file.
        ``lists`` and ``by_weight``: orbits, failing orbits and failing instances (the sum of
        their orbit sizes); per list also both counts split by outcome.
        ``lets``: per (a, b) class and in ``total``, failing LETS orbits and N_fail twice, as
        published (the sum of published counts) and closed (the sum of orbit sizes).
        ``floor``: per p in increasing order and, under ``occurrence`` and ``exact_event``, for
        each form of :func:`~decoder_fixes_gross.floor.log_probabilities`: the floor per
        experiment with its error, the per-round value and the per-round ends of value -+
        error, and the floor of each list.
        ``cost``: mean, median and 99th percentile of the decode time in microseconds and of
        the iteration count over every orbit, or None for iterations when the decoder reports
        none.

    Raises
    ------
    ValueError
        If ``lists`` lacks a list of the run or holds one whose content hash differs from the
        run's record, the run lacks ``u3`` or ``u4``, or the iteration counts mix counts
        with -1.
    """
    for record in run.config.lists:
        if record.name not in lists:
            raise ValueError(f"lists lacks the run's list {record.name!r}")
        if lists[record.name].content_hash() != record.content_hash:
            raise ValueError(f"list {record.name!r} differs from the one the run decoded")
    if not {"u3", "u4"} <= set(run.outcomes):
        raise ValueError(f"the run must include u3 and u4, got {sorted(run.outcomes)}")
    names = [record.name for record in run.config.lists]
    failed = {name: run.outcomes[name] != Outcome.CORRECT for name in names}
    lets_failed = failing_lets(lets, {3: failed["u3"], 4: failed["u4"]})
    return {
        "inputs": {
            "run_digest": run.digest,
            "lists": {record.name: record.content_hash for record in run.config.lists},
            "lets": lets.content_hash(),
        },
        "lists": {
            record.name: {"weight": record.weight}
            | _failures(run.outcomes[record.name], lists[record.name])
            for record in run.config.lists
        },
        "by_weight": _by_weight(run, lists),
        "lets": _lets_counts(lets, lets_failed),
        "floor": [_floor(p, priors_by_p[p], lists, failed, names) for p in sorted(priors_by_p)],
        "cost": _cost(run, names),
    }


def ceiling(
    lists: Mapping[str, FaultList], priors_by_p: Mapping[float, NDArray[np.float64]]
) -> list[dict[str, Any]]:
    """Return, per p in increasing order, the total mass of each list: the floor if every orbit failed.

    Returns
    -------
    list of dict
        Entries ``{"p": p, form: {name: mass}}``, for each form of
        :func:`~decoder_fixes_gross.floor.log_probabilities`, the mass being the sum over the
        orbits of the list of orbit size times the probability of S in that form.
    """
    return [
        {"p": p}
        | {
            form: {
                name: float(orbit_masses(priors_by_p[p], fault_list, form=form).sum())
                for name, fault_list in lists.items()
            }
            for form in FORMS
        }
        for p in sorted(priors_by_p)
    ]


def _failures(outcomes: NDArray[np.uint8], fault_list: FaultList) -> dict[str, Any]:
    sizes = fault_list.orbit_sizes.astype(np.int64)

    def count(mask: NDArray[np.bool_]) -> dict[str, int]:
        return {"orbits": int(mask.sum()), "instances": int(sizes[mask].sum())}

    failing = count(outcomes != Outcome.CORRECT)
    return {
        "orbits": len(outcomes),
        "failing_orbits": failing["orbits"],
        "failing_instances": failing["instances"],
        "wrong_logical": count(outcomes == Outcome.WRONG_LOGICAL),
        "no_solution": count(outcomes == Outcome.NO_SOLUTION),
    }


def _by_weight(run: RunOutcomes, lists: Mapping[str, FaultList]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for record in run.config.lists:
        found = _failures(run.outcomes[record.name], lists[record.name])
        entry = totals.setdefault(
            str(record.weight), {"orbits": 0, "failing_orbits": 0, "failing_instances": 0}
        )
        for key in entry:
            entry[key] += found[key]
    return dict(sorted(totals.items()))


def _lets_counts(
    lets: LetsList, lets_failed: Mapping[tuple[int, int], NDArray[np.bool_]]
) -> dict[str, dict[str, int]]:
    counts = {}
    for c in lets.classes:
        mask = lets_failed[(c.a, c.b)]
        counts[f"a{c.a}_b{c.b}"] = {
            "failing_orbits": int(mask.sum()),
            "published": int(c.published_counts[mask].sum(dtype=np.int64)),
            "closed": int(c.orbits.orbit_sizes[mask].sum(dtype=np.int64)),
        }
    counts["total"] = {
        key: sum(entry[key] for entry in counts.values())
        for key in ("failing_orbits", "published", "closed")
    }
    return counts


def _floor(
    p: float,
    priors: NDArray[np.float64],
    lists: Mapping[str, FaultList],
    failed: Mapping[str, NDArray[np.bool_]],
    names: list[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {"p": p}
    for form in FORMS:
        masses = {name: orbit_masses(priors, lists[name], form=form) for name in names}
        total = estimate(
            np.concatenate([masses[name] for name in names]),
            np.concatenate([failed[name] for name in names]),
        )
        value, low, high = total.per_round()
        by_list = {name: estimate(masses[name], failed[name]) for name in names}
        entry[form] = {
            "value": total.value,
            "error": total.error,
            "per_round": {"value": value, "low": low, "high": high},
            "lists": {name: {"value": e.value, "error": e.error} for name, e in by_list.items()},
        }
    return entry


def _cost(run: RunOutcomes, names: list[str]) -> dict[str, Any]:
    micros = np.concatenate([run.micros[name] for name in names])
    iterations = np.concatenate([run.iterations[name] for name in names])
    reported = iterations >= 0
    if reported.any() and not reported.all():
        raise ValueError("iteration counts mix counts with -1; a run has one decoder")
    return {
        "micros": _stats(micros),
        "iterations": _stats(iterations) if reported.all() else None,
    }


def _stats(values: NDArray[np.integer]) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        f"p{TAIL_PERCENTILE}": float(np.percentile(values, TAIL_PERCENTILE)),
    }
