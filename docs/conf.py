project = "decoder-fixes-gross"
author = "Amin Idelhaj"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"

# Every unresolved cross-reference fails the build.
nitpicky = True
# Libraries without a Sphinx inventory are ignored by a pattern scoped to their namespace.
nitpick_ignore_regex: list[tuple[str, str]] = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Notebooks are stored as MyST Markdown (Jupytext) and executed during the build; cached when unchanged.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True

numpydoc_show_class_members = False
