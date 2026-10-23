# Reproduce the figures

Results and figures are produced in two stages.

The experiment scripts write result data to `results/`, each file next to a JSON record of the configuration that produced it: parameters, seed, and git commit.

```
uv run nox -s figures
```

The documentation build then renders every figure from those results, so no figure image is committed for the documentation.

```
uv run nox -s docs
```

The results folder is committed while it is small, so the documentation builds quickly in CI; it is never edited by hand, only regenerated.
