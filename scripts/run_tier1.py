"""Decode one random translate of every orbit in the support lists with a named decoder.

Reads ``<data>/lists/<name>.npz`` for each list and writes ``<data>/runs/<run-id>/``:
``config.json`` and one ``shard_<k>.npz`` per shard. Rerunning the same command resumes an
interrupted run; see ``docs/howto/run-tier1.md``. Usage:
``uv run python scripts/run_tier1.py --decoder relay-bp-baseline --seed <n> --run-id <id>``.
"""

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from decoder_fixes_gross.tier1 import DECODERS, SHARD_SIZE, run

ROOT = Path(__file__).resolve().parent.parent
LISTS = ("u1", "u2", "u3", "u4")


def physical_cores() -> int:
    """Return the number of physical cores from ``/proc/cpuinfo``, else the logical CPUs."""
    cores, physical = set(), None
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip() == "physical id":
            physical = value.strip()
        elif key.strip() == "core id":
            cores.add((physical, value.strip()))
    return len(cores) or os.cpu_count() or 1


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments and run, or resume, the decoding run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decoder", required=True, choices=sorted(DECODERS))
    parser.add_argument("--seed", required=True, type=int, help="run seed, non-negative")
    parser.add_argument("--run-id", required=True, help="directory name under <data>/runs/")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data",
        help="folder holding lists/ and runs/; from a separate worktree, the main checkout's data/",
    )
    parser.add_argument("--lists", nargs="+", default=list(LISTS), help="list names, in order")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    # One per physical core: the decoders are single-threaded, and a busy SMT sibling adds
    # noise to the per-orbit decode times.
    parser.add_argument("--workers", type=int, default=physical_cores())
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = args.data.resolve()
    run(
        data / "runs" / args.run_id,
        {name: data / "lists" / f"{name}.npz" for name in args.lists},
        DECODERS[args.decoder],
        args.seed,
        shard_size=args.shard_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
