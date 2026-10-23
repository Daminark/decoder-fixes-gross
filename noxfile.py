#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox"]
# ///
"""Task runner.

Local runs and CI execute the same sessions.
uv manages the environment, so sessions run without a nox-created venv.
"""

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "none"


@nox.session
def lint(session: nox.Session) -> None:
    """Run every pre-commit hook on every file."""
    session.run("uv", "run", "pre-commit", "run", "--all-files", external=True)


@nox.session
def types(session: nox.Session) -> None:
    """Type-check against the locked environment."""
    session.run("uv", "run", "mypy", external=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Fast tests."""
    session.run("uv", "run", "pytest", *session.posargs, external=True)


@nox.session(default=False)
def tests_full(session: nox.Session) -> None:
    """All tests, including slow ones, with coverage against the floor."""
    session.run(
        "uv",
        "run",
        "pytest",
        "--runslow",
        "--cov",
        "--cov-report=term-missing",
        *session.posargs,
        env={"HYPOTHESIS_PROFILE": "ci"},
    )


@nox.session(default=False)
def docs(session: nox.Session) -> None:
    """Build the documentation site; notebooks execute and warnings fail the build."""
    session.run(
        "uv",
        "run",
        "--group",
        "docs",
        "sphinx-build",
        "-W",
        "--keep-going",
        "-b",
        "html",
        "docs",
        "docs/_build/html",
    )


@nox.session(default=False)
def figures(session: nox.Session) -> None:
    """Regenerate the cached experiment results under results/."""
    session.run("uv", "run", "python", "scripts/make_figures.py", external=True)


@nox.session(default=False)
def review(session: nox.Session) -> None:
    """Run sp-repo-review; waivers live in pyproject.toml under [tool.repo-review]."""
    session.run("uvx", "--with", "repo-review[cli]", "sp-repo-review", ".", external=True)


if __name__ == "__main__":
    nox.main()
