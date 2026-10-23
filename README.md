# decoder-fixes-gross

[![ci](https://github.com/Daminark/decoder-fixes-gross/actions/workflows/ci.yml/badge.svg)](https://github.com/Daminark/decoder-fixes-gross/actions/workflows/ci.yml)
[![Project Status: WIP](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Comparing decoder fixes for sub-distance failures on the Gross code

Work in progress.
Start with the [tutorial](docs/tutorials/index.md); the method and the results, with their limitations, are in the [explanation pages](docs/explanation/results.md).

## Installation

```
uv sync
```

## Usage

```python
import decoder_fixes_gross
```

## Documentation

Built with `uv run nox -s docs`; the results it renders regenerate with `uv run nox -s figures` (see the how-to page on reproducing figures).

## Relation to similar work

The paper implemented, and the packages this one is compared against, with what differs.

## Citation

See `CITATION.cff`; GitHub renders it as "Cite this repository".
