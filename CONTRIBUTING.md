# Contributing

## Setup

```
uv sync
uv run pre-commit install
```

## Commands

- `uv run nox` runs lint, type check, and fast tests.
- `uv run nox -s tests_full` runs every test, including slow ones, with coverage.
- `uv run nox -s docs` builds the documentation site; notebooks execute during the build.
- `uv run nox -s figures` regenerates the cached results in `results/`, which the docs build renders into figures.
- `uv run nox -s review` runs sp-repo-review.

## Conventions

- Source layout: package in `src/`, tests in `tests/` mirroring it, experiment entry points in `scripts/`, documentation in `docs/`.
- Docstrings follow numpydoc and state the contract: parameters, returns, and the conditions under which the function raises.
- Stochastic functions take a seed or a generator; tests fix it.
- Slow tests carry the `slow` marker and run only with `--runslow`.
- Notebooks are stored under `docs/` as MyST Markdown through Jupytext and executed by the docs build.
- Design decisions a maintainer would otherwise have to reverse-engineer are recorded in `docs/adr/`.
- Material borrowed from another project gets one notice file in `third_party/` naming its source, commit and licence.
- Files shipped as package data carry a provenance file next to them recording how they were produced.
- A test that needs data under `data/` carries the `slow` marker and skips with a stated reason when the data is absent.
- Large inputs and raw outputs go in `data/`, which is not committed; `data/README.md` records the source or the command that regenerates each. Fixtures that tests need are small and live in `tests/data/`.
- Every figure in the documentation is rendered at build time from `results/`, which one command regenerates; see the how-to page on reproducing figures.
- Commit messages: an imperative subject line, a blank line, then a body saying why.
- Work happens on a branch and lands on `main` through a pull request.
