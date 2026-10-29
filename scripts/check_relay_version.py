"""Check that the pinned relay-bp reproduces an earlier relay-bp 0.2.2 run.

Input: ``tests/data/relay_version_sample.npz``, 2,654 LETS supports of weight 2 to 5 from
the Zenodo trapping-set dataset (record 21878685, CC-BY-4.0), as column indices of the
gross-code model padded with -1 (``sets``), in four file groups (``group``), with their
outcome codes under relay-bp 0.2.2 (``outcome_0_2_2``). That run used the baseline
settings with seed 0 and a fresh decoder per file group, decoding each group in file order.
The sets are an arbitrary fixed selection, two of them repeated; they exist so that both
versions decode the same input in the same order, not to estimate a failure rate.

The check repeats that procedure with the installed relay-bp. The pin stands if the
weight-4 sets of group 0 (355 sets, 1 failure under 0.2.2) fail at most 4 times and the
weight-5 sets (1,170 sets, 13 failures under 0.2.2) fail between 4 and 22 times, the 99%
binomial range around 13 of 1,170. Otherwise the fallback is relay-bp 0.2.2 built from
source.

Writes ``results/relay_version_check.json``. Usage: ``uv run python scripts/check_relay_version.py``.
"""

import dataclasses
import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np

from decoder_fixes_gross.decoders import RELAY_BP_BASELINE, RelayBP
from decoder_fixes_gross.dem import dem_at
from decoder_fixes_gross.inject import decode_injected
from decoder_fixes_gross.runs import commit_hash

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "data" / "relay_version_sample.npz"
OUT = ROOT / "results" / "relay_version_check.json"
P = 1e-3
SEED = 0
# Pass conditions, fixed before the check was run.
MAX_WEIGHT_4_FAILURES = 4
WEIGHT_5_FAILURE_RANGE = (4, 22)
logger = logging.getLogger(__name__)


def decode_sample(sets: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Return the outcome code of each set, one fresh seed-0 decoder per group, in order."""
    dem = dem_at(P)
    outcomes = np.zeros(len(sets), dtype=np.uint8)
    for g in np.unique(group):
        decoder = RelayBP(dem, seed=SEED)
        for k in np.flatnonzero(group == g):
            outcomes[k] = decode_injected(dem, decoder, sets[k][sets[k] >= 0]).outcome
        logger.info("group %d: %d sets decoded", g, np.count_nonzero(group == g))
    return outcomes


def failures(outcomes: np.ndarray, mask: np.ndarray) -> int:
    """Return the number of nonzero outcome codes under ``mask``."""
    return int(np.count_nonzero(outcomes[mask]))


def main() -> None:
    """Run the check and write its record."""
    logging.basicConfig(level=logging.INFO)
    sample = np.load(SAMPLE)
    sets, group, reference = sample["sets"], sample["group"], sample["outcome_0_2_2"]
    outcomes = decode_sample(sets, group)

    weight = np.count_nonzero(sets >= 0, axis=1)
    weight_4 = (weight == 4) & (group == 0)
    weight_5 = weight == 5
    by_weight = {
        int(w): {
            "sets": int(np.count_nonzero(weight == w)),
            "failures_installed": failures(outcomes, weight == w),
            "failures_0_2_2": failures(reference, weight == w),
        }
        for w in np.unique(weight)
    }
    n_4 = failures(outcomes, weight_4)
    n_5 = failures(outcomes, weight_5)
    passed = n_4 <= MAX_WEIGHT_4_FAILURES and (
        WEIGHT_5_FAILURE_RANGE[0] <= n_5 <= WEIGHT_5_FAILURE_RANGE[1]
    )
    record = {
        "relay_bp_version": version("relay-bp"),
        "settings": dataclasses.asdict(RELAY_BP_BASELINE),
        "p": P,
        "seed": SEED,
        "sample_sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
        "commit": commit_hash(ROOT),
        "weight_4_group_0": {
            "sets": int(np.count_nonzero(weight_4)),
            "failures_installed": n_4,
            "failures_0_2_2": failures(reference, weight_4),
            "max_failures": MAX_WEIGHT_4_FAILURES,
        },
        "weight_5": {
            "sets": int(np.count_nonzero(weight_5)),
            "failures_installed": n_5,
            "failures_0_2_2": failures(reference, weight_5),
            "failure_range": list(WEIGHT_5_FAILURE_RANGE),
        },
        "by_weight": by_weight,
        "sets_with_different_outcome": int(np.count_nonzero(outcomes != reference)),
        "passed": passed,
        "decision": "pin the installed version" if passed else "fall back to 0.2.2 from source",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s: passed=%s", OUT, passed)


if __name__ == "__main__":
    main()
