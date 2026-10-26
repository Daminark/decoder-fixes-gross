project = "decoder-fixes-gross"
author = "Amin Idelhaj"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
exclude_patterns = ["_build"]

# Module pages come from a template with :members:, since the default template lists no members.
templates_path = ["_templates"]
# API pages are generated for every module, so a new module needs no edit to api.rst.
autosummary_generate = True
# Types are stated in the numpydoc sections; signature hints would render numpy's private alias paths.
autodoc_typehints = "none"

html_theme = "pydata_sphinx_theme"

# Every unresolved cross-reference fails the build.
nitpicky = True
# Libraries without a Sphinx inventory are ignored by a pattern scoped to their namespace.
# Stim publishes no Sphinx inventory, so references to its classes cannot resolve.
nitpick_ignore_regex: list[tuple[str, str]] = [("py:class", r"stim\..*")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Notebooks are stored as MyST Markdown (Jupytext) and executed during the build; cached when unchanged.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True

numpydoc_show_class_members = False
