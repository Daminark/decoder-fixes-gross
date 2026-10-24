# 4. Keep per-support outcomes outside the repository and run in resumable shards

Date: 2026-10-23

## Status

Accepted

## Context

One pass of one decoder produces an outcome code, a decode time and an iteration count for each of a few million support orbits, which is about 100 MB.
A pass takes hours, and a decoder that fails to converge takes several seconds on a single support, so a pass has to survive interruption.
The documentation is built in CI from committed results, and the repository has to stay small.
Several decoders are randomized, so the seed of every random choice has to be recorded.

## Decision

Raw outcomes are written under `data/`, which is not committed, in shards of at most 50,000 orbits.
Each run has a directory with one configuration record and one file per shard.
The record holds the decoder and its settings, the error rate of the decoder priors, the run seed, the shard size, the hashes of the fault lists, the package versions and the commit.
A shard is written under a temporary name and renamed when complete, so a run holds only complete shards and a rerun skips them.
Within a shard, the random translates and the decoder seed come from a stream spawned from the run seed with `SeedSequence.spawn`, and one decoder object decodes the whole shard.
Results are therefore reproducible per shard and not per support.

The aggregation command, `uv run nox -s figures`, reads the shards listed in a committed registry and writes small JSON files to `results/`, each recording the hashes of its inputs.
The directory `results/` is committed and the shards are not.

## Consequences

The committed aggregates can be checked against the shards by hash, and the figures rebuild without the raw data.
Reproducing a single support's outcome requires rerunning its shard.
Reproducing the aggregates from scratch requires either the shards or the long commands listed in the how-to pages.
If constructing a decoder turns out to cost under a millisecond, per-support seeding becomes affordable, and a new record would supersede this one.
