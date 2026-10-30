"""Regenerate the aggregate results in results/ from the runs listed in scripts/runs.toml.

For each registered run, reads ``<data>/runs/<id>/`` and the lists it decoded from
``<data>/lists/``, with ``<data>/lists/lets.npz`` for the LETS incidence, and writes
``<results>/floor_<id>.json``: failures by list, weight, outcome and LETS class, the floor at
every p of the two grids below, and the decode cost. ``<results>/ceiling.json`` holds the
mass of every list the runs use, at the same p. When the registry has a ``[gate]`` table, the
run it names, which must be registered with the Relay-BP baseline decoder, is evaluated against
the trapping-set paper and ``<results>/gate_relay_bp.json`` is written
(:func:`decoder_fixes_gross.gate.evaluate`). Each file records the hashes of its inputs and the
commit of this code. Takes seconds.

Before computing anything, the command reads ``<results>/lists.json``, the record of the build
that wrote ``u3``, ``u4`` and ``lets.npz`` together, and refuses to run unless the LETS file and
the ``u3`` and ``u4`` it loaded have the hashes recorded there, since the LETS incidence holds
row indices into those two lists. It never writes ``lists.json``.

Two records in results/ come from their own one-off commands and are not regenerated here:
``lists.json``, by ``scripts/build_lists.py`` from the Zenodo files (about 7 minutes), and
``relay_version_check.json``, by ``scripts/check_relay_version.py`` (about a minute).

Usage: ``uv run nox -s figures``, which runs ``uv run python scripts/make_figures.py``.
"""

import argparse
import json
import logging
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from decoder_fixes_gross.aggregate import ceiling, read_run, summarize
from decoder_fixes_gross.dem import dem_at
from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.gate import evaluate
from decoder_fixes_gross.lets import INCIDENCE_WEIGHTS, LetsList
from decoder_fixes_gross.runs import commit_hash

ROOT = Path(__file__).resolve().parent.parent
# Record of the build of the lists and the LETS file, read from the results directory.
BUILD_RECORD = "lists.json"
# Rates of the tables; 4e-4 and 1e-3 are where the trapping-set paper's Fig. 1 is read.
TABLE_P = [1e-4, 3e-4, 4e-4, 1e-3]
# Rates of the floor curve: 17 points on a log scale.
CURVE_P = np.geomspace(1e-4, 4e-3, 17).tolist()
# The gate compares this decoder with the paper, and writes its verdict to this file.
GATE_DECODER = "relay-bp-baseline"
GATE_FILE = "gate_relay_bp.json"
logger = logging.getLogger(__name__)


def write(path: Path, record: dict[str, Any]) -> None:
    """Write ``record`` as indented JSON."""
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def check_build(results: Path, lists: dict[str, FaultList], lets: LetsList) -> None:
    """Exit unless ``lets`` and the lists it indexes have the hashes of the recorded build."""
    path = results / BUILD_RECORD
    remedy = "rebuild the lists with scripts/build_lists.py, which writes them and the record"
    if not path.exists():
        raise SystemExit(f"{path} is missing; {remedy}")
    build = json.loads(path.read_text(encoding="utf-8"))
    found = {"lets.npz": (lets.content_hash(), build["lets_content_hash"])} | {
        f"u{w}.npz": (lists[f"u{w}"].content_hash(), build["lists"][f"u{w}"]["content_hash"])
        for w in INCIDENCE_WEIGHTS
        if f"u{w}" in lists
    }
    for name, (actual, recorded) in found.items():
        if actual != recorded:
            raise SystemExit(
                f"{name} has hash {actual}, but {path} records {recorded} for the build that "
                f"wrote the LETS incidence; {remedy}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Aggregate every registered run and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--runs", type=Path, default=ROOT / "scripts" / "runs.toml")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    registry_file = tomllib.loads(args.runs.read_text(encoding="utf-8"))
    registry = registry_file["runs"]
    gate_run = registry_file.get("gate", {}).get("run")
    if gate_run is not None and not any(
        entry["id"] == gate_run and entry["decoder"] == GATE_DECODER for entry in registry
    ):
        raise SystemExit(
            f"the gate reads run {gate_run}, which {args.runs.name} must register with "
            f"decoder {GATE_DECODER}"
        )
    names = list(dict.fromkeys(name for entry in registry for name in entry["lists"]))
    lists = {name: FaultList.load(args.data / "lists" / f"{name}.npz") for name in names}
    lets = LetsList.load(args.data / "lists" / "lets.npz")
    check_build(args.results, lists, lets)
    runs = [read_run(args.data / "runs" / entry["id"]) for entry in registry]
    for entry, run in zip(registry, runs, strict=True):
        recorded = [record.name for record in run.config.lists]
        if run.config.decoder != entry["decoder"] or recorded != entry["lists"]:
            raise SystemExit(
                f"run {entry['id']} holds decoder {run.config.decoder} on {recorded}, "
                f"but {args.runs.name} registers {entry['decoder']} on {entry['lists']}"
            )
    priors_by_p = {p: dem_at(p).priors for p in sorted({*TABLE_P, *CURVE_P})}
    provenance = {"commit": commit_hash(ROOT), "p_grids": {"table": TABLE_P, "curve": CURVE_P}}

    args.results.mkdir(exist_ok=True)
    for entry, run in zip(registry, runs, strict=True):
        run_lists = {name: lists[name] for name in entry["lists"]}
        summary = summarize(run, run_lists, lets, priors_by_p)
        record = {
            "run_id": entry["id"],
            "decoder": run.config.decoder,
            "decoder_settings": dict(run.config.decoder_settings),
            "run_seed": run.config.run_seed,
            "run_commit": run.config.commit,
        } | provenance
        floor_name = f"floor_{entry['id']}.json"
        write(args.results / floor_name, record | summary)
        if entry["id"] == gate_run:
            gate = {
                "run_id": gate_run,
                "commit": provenance["commit"],
                "inputs": {"floor": floor_name, "run_digest": summary["inputs"]["run_digest"]},
            }
            write(args.results / GATE_FILE, gate | evaluate(summary))
    write(
        args.results / "ceiling.json",
        provenance
        | {
            "inputs": {"lists": {name: fl.content_hash() for name, fl in lists.items()}},
            "ceiling": ceiling(lists, priors_by_p),
        },
    )


if __name__ == "__main__":
    main()
