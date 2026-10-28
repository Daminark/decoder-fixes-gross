"""Build the LETS orbits and the support lists U_1 to U_4 from the Zenodo trapping-set files.

Input: the 19 files ``a_<a>_b_<b>.csv`` of Zenodo record 21878685, unpacked under
``data/lets/`` as ``data/README.md`` describes; each is checked against the archive's
``SHA256SUMS`` before it is read.

Writes ``data/lists/u1.npz`` to ``u4.npz`` (fault lists), ``data/lists/lets.npz`` (the LETS
orbits with their incidence into U_3 and U_4), and ``results/lists.json`` with the counts,
the hashes of the inputs and outputs, and the configuration. Takes about 7 minutes on one
core. Usage: ``uv run python scripts/build_lists.py``.
"""

import hashlib
import json
import logging
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from decoder_fixes_gross.dem import dem_at
from decoder_fixes_gross.lets import LetsList, build_supports, induced_classes, read_lets_files

ROOT = Path(__file__).resolve().parent.parent
LETS_DIR = ROOT / "data" / "lets" / "ets_supports_full_detector_algorithm1_full_parallel"
LISTS_DIR = ROOT / "data" / "lists"
OUT = ROOT / "results" / "lists.json"
ZENODO_RECORD = 21878685
logger = logging.getLogger(__name__)


def commit_hash() -> str:
    """Return the short hash of HEAD, or "unknown" outside a git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def verified_checksums(directory: Path) -> dict[str, str]:
    """Return the sha256 of each CSV in ``directory``, after checking it against SHA256SUMS."""
    expected = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split()
        expected[name] = checksum
    found = {}
    for path in sorted(directory.glob("a_*_b_*.csv")):
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while block := f.read(1 << 24):
                digest.update(block)
        found[path.name] = digest.hexdigest()
        if expected.get(path.name) != found[path.name]:
            raise SystemExit(f"{path.name} does not match SHA256SUMS")
    if set(found) != set(expected):
        raise SystemExit(f"files and SHA256SUMS differ: {sorted(set(found) ^ set(expected))}")
    return found


def main() -> None:
    """Build the lists and write them with their record."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = time.perf_counter()
    checksums = verified_checksums(LETS_DIR)
    files = read_lets_files(LETS_DIR)
    classes = [f.orbits for f in files]

    # Translations preserve h, so checking each representative checks every row of its orbit.
    h = dem_at(1e-3).h
    outside = {}
    for c in classes:
        induced = induced_classes(h, c.orbits.sets)
        outside[f"a{c.a}_b{c.b}"] = int(
            np.count_nonzero((induced.b != c.b) | (induced.max_check_degree > 2))
        )
    if any(outside.values()):
        raise SystemExit(f"representatives outside their file's class: {outside}")
    logger.info("every representative has its file's class and is elementary")

    supports = build_supports(classes)
    lets = LetsList(classes=tuple(classes), incidence=supports.incidence)
    LISTS_DIR.mkdir(parents=True, exist_ok=True)
    for w, fault_list in supports.lists.items():
        fault_list.save(LISTS_DIR / f"u{w}.npz")
    lets.save(LISTS_DIR / "lets.npz")

    per_class = {
        f"a{c.a}_b{c.b}": {
            "rows": c.rows,
            "missing": c.missing,
            "dropped_lines": f.dropped_lines,
            "orbits": len(c.orbits.sets),
            "orbits_of_size_36": int(np.count_nonzero(c.orbits.orbit_sizes == 36)),
        }
        for f, c in zip(files, classes, strict=True)
    }
    record = {
        "zenodo_record": ZENODO_RECORD,
        "input_sha256": checksums,
        "commit": commit_hash(),
        "package_versions": {name: version(name) for name in ("numpy", "scipy", "stim")},
        "classes": per_class,
        "class_check": {
            "rule": "(a, b) recomputed from h and every induced check degree at most 2, "
            "on every orbit representative",
            "representatives": sum(len(c.orbits.sets) for c in classes),
            "outside_class": 0,
        },
        "totals": {
            "rows": sum(c.rows for c in classes),
            "missing": sum(c.missing for c in classes),
            "dropped_lines": sum(f.dropped_lines for f in files),
            "orbits": sum(len(c.orbits.sets) for c in classes),
            "orbits_a_at_most_4": sum(len(c.orbits.sets) for c in classes if c.a <= 4),
            "orbits_a_5": sum(len(c.orbits.sets) for c in classes if c.a == 5),
            "orbits_of_size_36": sum(v["orbits_of_size_36"] for v in per_class.values()),
        },
        "lists": {
            f"u{w}": {
                "orbits": len(fault_list.sets),
                "supports": int(fault_list.orbit_sizes.sum(dtype=np.int64)),
                "content_hash": fault_list.content_hash(),
            }
            for w, fault_list in supports.lists.items()
        },
        "lets_content_hash": lets.content_hash(),
        "seconds": round(time.perf_counter() - start),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s and the lists in %s", OUT, LISTS_DIR)


if __name__ == "__main__":
    main()
