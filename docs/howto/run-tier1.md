# Run a decoder over the support lists

This page shows how to launch the tier-1 pass of one decoder over the support lists U_1 to U_4, how to watch it, and how to resume it after an interruption.
The pass decodes one uniformly random translate of every orbit, as [ADR 0003](../adr/0003-estimate-the-floor-from-one-translate-per-orbit.md) describes, and writes per-orbit outcomes under `data/runs/`, which is not committed ([ADR 0004](../adr/0004-keep-raw-outcomes-outside-the-repository.md)).
It needs the lists in `data/lists/`, built by `scripts/build_lists.py`.

## Launch from a separate worktree

Launch a long run from a checkout that nobody edits while it runs.
Worker processes import the package when they start, so an edit made in the same folder after launch could reach some shards and not others, and the commit recorded in `config.json` would no longer describe the code that ran.

From the main checkout, create a worktree at `main` and install the environment there:

```bash
git worktree add --detach ../decoder-fixes-gross-run main
```

```bash
cd ../decoder-fixes-gross-run && uv sync
```

Pick a run seed once, for example with NumPy's own entropy source:

```bash
uv run python -c "import numpy; print(numpy.random.SeedSequence().entropy)"
```

Then start the run from the worktree, pointing `--data` at the main checkout's `data/`, so that the lists are read from there and the run is written there:

```bash
mkdir -p ~/Projects/decoder-fixes-gross/data/runs && uv run python scripts/run_tier1.py --decoder relay-bp-baseline --seed SEED --run-id relay-bp-baseline --data ~/Projects/decoder-fixes-gross/data 2>&1 | tee -a ~/Projects/decoder-fixes-gross/data/runs/relay-bp-baseline.log
```

The terminal must stay open while the run lasts: closing it ends `tee`, and the run stops at its next log line.
To run unattended instead, detach the run from the terminal and send its output straight to the log, then follow the log as under Watch below:

```bash
mkdir -p ~/Projects/decoder-fixes-gross/data/runs && nohup uv run python scripts/run_tier1.py --decoder relay-bp-baseline --seed SEED --run-id relay-bp-baseline --data ~/Projects/decoder-fixes-gross/data >> ~/Projects/decoder-fixes-gross/data/runs/relay-bp-baseline.log 2>&1 &
```

The run directory `data/runs/relay-bp-baseline/` receives `config.json` first and then one `shard_<k>.npz` per shard of at most 50,000 orbits.
`--lists` chooses other lists (default `u1 u2 u3 u4`), `--shard-size` the shard size and `--workers` the number of processes.

By default there is one worker per physical core, 8 on a Ryzen 7 9800X3D.
The decoders are single-threaded, and a busy hyperthread sibling slows the decode it shares a core with, which adds noise to the per-orbit decode times.
`--workers 16` likely buys more throughput (not measured here), at the price of noisier times.

For the Relay-BP baseline, expect roughly 2 to 4 hours on 8 workers.
Weight-2 supports decode at about 0.5 ms each, and the cost is dominated by the rare decodes that never converge, each of which runs all 18,080 iterations in 6 to 7 s.
The estimate is extrapolated from samples and is good to a factor of two.

When the run is complete, remove the worktree from the main checkout:

```bash
git worktree remove ../decoder-fixes-gross-run
```

## Watch

The first log line states the number of shards, how many are already complete and how many remain.
Each finished shard then logs one line with its list, its rows, the failures it found, the shards done so far and the time since launch.
Shards finish out of order, since each worker takes the next shard as it becomes free.

To follow the log, or to count finished shards against the total in the first line:

```bash
tail -f ~/Projects/decoder-fixes-gross/data/runs/relay-bp-baseline.log
```

```bash
ls ~/Projects/decoder-fixes-gross/data/runs/relay-bp-baseline/shard_*.npz | wc -l
```

## Resume

An interrupted run, whether by Ctrl-C, a crash or a shutdown, keeps every shard that was complete.
A shard is written under a temporary name ending in `.tmp` and renamed once it is on disk, so a file named `shard_<k>.npz` is always complete; a leftover `.tmp` file is overwritten.

To resume, run the same command again from the same worktree at the same commit.
The driver compares the configuration of the new command with the stored `config.json` and decodes only the shards that have no file.
Each shard draws its translates and its decoder seed from its own stream of the run seed, and builds its own decoder, so a resumed shard is identical to the one an uninterrupted run would have written.

A difference in any field of the configuration refuses to resume, by design: shards decoded by different code would be indistinguishable afterwards.
This covers the decoder and its settings, the seed, the shard size, the content of the lists, the package versions, and the commit, including a `-dirty` marker when tracked files differ from it.
The remedy is a new run id.
