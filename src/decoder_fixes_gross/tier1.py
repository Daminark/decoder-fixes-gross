"""Tier 1: decode one uniformly random translate of every orbit in a set of fault lists.

The estimator is that of ADR 0003 and the run layout that of ADR 0004. The lists are cut
into shards (`decoder_fixes_gross.runs.RunConfig.shards`). Shard k draws from its own stream,
``SeedSequence(run_seed).spawn(n_shards)[k]``: first the seed of a decoder built for that
shard alone, then one translate per orbit, uniform over the 72 group elements. Each
representative is translated, injected, decoded and scored in list order, so a shard's
outcomes are reproducible from the run seed even for a decoder whose results depend on the
calls before them.
"""

import dataclasses
import logging
import multiprocessing
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decoder_fixes_gross.decoders import RELAY_BP_BASELINE, Decoder, RelayBP
from decoder_fixes_gross.dem import Dem, dem_at
from decoder_fixes_gross.faultlist import FaultList
from decoder_fixes_gross.inject import decode_injected
from decoder_fixes_gross.runs import (
    CONFIG_NAME,
    ListRecord,
    RunConfig,
    Shard,
    ShardSpec,
    commit_hash,
    package_versions,
    shard_name,
)
from decoder_fixes_gross.symmetry import N_TRANSLATIONS, translate

# Physical error rate of the model every decoder is built from, as in the trapping-set paper.
P0 = 1e-3
# ADR 0004: shards of at most 50,000 orbits.
SHARD_SIZE = 50_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecoderSpec:
    """A named decoder: its settings as recorded in a run, and how to build it.

    Attributes
    ----------
    name : str
        Name recorded in the run configuration.
    settings : Mapping
        JSON-serializable settings recorded in the run configuration.
    build : callable
        ``build(dem, seed)`` returns a `Decoder` for ``dem`` seeded with ``seed`` in
        [0, 2**64). For a run with several workers it must be picklable, such as a
        module-level function.
    """

    name: str
    settings: Mapping[str, Any]
    build: Callable[[Dem, int], Decoder]


def _relay_bp_baseline(dem: Dem, seed: int) -> Decoder:
    return RelayBP(dem, seed=seed, settings=RELAY_BP_BASELINE)


DECODERS: Mapping[str, DecoderSpec] = {
    spec.name: spec
    for spec in [
        DecoderSpec("relay-bp-baseline", dataclasses.asdict(RELAY_BP_BASELINE), _relay_bp_baseline),
    ]
}


def run(
    run_dir: Path,
    lists: Mapping[str, Path],
    decoder: DecoderSpec,
    run_seed: int,
    *,
    shard_size: int = SHARD_SIZE,
    workers: int = 1,
) -> list[int]:
    """Decode every shard of a run that ``run_dir`` does not yet hold.

    A new run writes ``config.json`` first. A run whose ``config.json`` exists resumes: the
    configuration of this call must equal the stored one in every field, provenance
    included, and only the shards without a file are decoded.

    Parameters
    ----------
    run_dir : Path
        Directory of the run; created if absent.
    lists : Mapping of str to Path
        Fault lists by name, in the order they are sharded.
    decoder : DecoderSpec
        The decoder.
    run_seed : int
        Non-negative seed of the run.
    shard_size : int, optional
        Maximum number of orbits per shard.
    workers : int, optional
        Number of processes; 1 decodes in this process.

    Returns
    -------
    list of int
        Indices of the shards decoded by this call, ascending.

    Raises
    ------
    ValueError
        If ``workers`` is not positive, a configuration field is invalid, or the stored
        configuration differs from this call's.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    fault_lists = {name: FaultList.load(path) for name, path in lists.items()}
    config = RunConfig(
        decoder=decoder.name,
        decoder_settings=decoder.settings,
        p0=P0,
        run_seed=run_seed,
        shard_size=shard_size,
        lists=tuple(
            ListRecord(name, fl.content_hash(), fl.sets.shape[0], fl.sets.shape[1])
            for name, fl in fault_lists.items()
        ),
        versions=package_versions(),
        commit=commit_hash(Path(__file__).parent),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / CONFIG_NAME).exists():
        differing = RunConfig.load(run_dir).differing_fields(config)
        if differing:
            raise ValueError(
                f"{run_dir} holds a run with a different configuration in {differing}; "
                "resume with the same code, packages and arguments, or start a new run id"
            )
    else:
        config.save(run_dir)

    specs = config.shards()
    streams = np.random.SeedSequence(run_seed).spawn(len(specs))
    pending = [spec for spec in specs if not (run_dir / shard_name(spec.index)).exists()]
    started = time.perf_counter()
    done = len(specs) - len(pending)
    logger.info(
        "%s: %d shards, %d complete, %d to decode on %d workers",
        run_dir,
        len(specs),
        done,
        len(pending),
        workers,
    )

    def finish(spec: ShardSpec, shard: Shard) -> None:
        nonlocal done
        shard.save(run_dir / shard_name(spec.index))
        done += 1
        logger.info(
            "shard %d (%s rows %d to %d): %d failures in %d orbits; %d of %d shards; %.0f s",
            spec.index,
            spec.list_name,
            spec.start,
            spec.stop,
            np.count_nonzero(shard.outcomes),
            len(shard),
            done,
            len(specs),
            time.perf_counter() - started,
        )

    def task(spec: ShardSpec) -> tuple[NDArray[np.int32], np.random.SeedSequence]:
        return fault_lists[spec.list_name].sets[spec.start : spec.stop], streams[spec.index]

    if workers == 1:
        for spec in pending:
            finish(spec, _decode_shard(*task(spec), decoder.build))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = {
                pool.submit(_decode_shard, *task(spec), decoder.build): spec for spec in pending
            }
            try:
                for future in as_completed(futures):
                    finish(futures[future], future.result())
            except BaseException:
                # Queued shards do not start; shards already running finish and are discarded.
                pool.shutdown(cancel_futures=True)
                raise
    return [spec.index for spec in pending]


def _decode_shard(
    representatives: NDArray[np.int32],
    stream: np.random.SeedSequence,
    build: Callable[[Dem, int], Decoder],
) -> Shard:
    """Decode one random translate of each representative, in order, with one decoder.

    ``representatives`` are rows of a `FaultList`; ``stream`` gives the decoder seed, then
    the translates.
    """
    rng = np.random.default_rng(stream)
    decoder_seed = int(rng.integers(2**64, dtype=np.uint64))
    translates = rng.integers(N_TRANSLATIONS, size=len(representatives), dtype=np.uint8)
    dem = _model()
    decoder = build(dem, decoder_seed)
    fault_sets = translate(representatives, translates)
    outcomes = np.empty(len(fault_sets), dtype=np.uint8)
    micros = np.empty(len(fault_sets), dtype=np.uint32)
    iterations = np.empty(len(fault_sets), dtype=np.int32)
    for i, faults in enumerate(fault_sets):
        trial = decode_injected(dem, decoder, faults)
        outcomes[i] = trial.outcome
        micros[i] = round(trial.seconds * 1e6)
        iterations[i] = -1 if trial.iterations is None else trial.iterations
    return Shard(decoder_seed, translates, outcomes, micros, iterations)


@cache
def _model() -> Dem:
    """Return the model at `P0`, built once per process."""
    return dem_at(P0)
