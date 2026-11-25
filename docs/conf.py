"""Sphinx configuration of the kpimpc documentation."""

import datetime

import kpimpc

project = "kpimpc"
author = "Ifremer LOPS"
copyright = f"2021-{datetime.date.today().year}, {author}"
version = release = kpimpc.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

nitpicky = True
nitpick_ignore = [("py:class", "optional")]

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = f"kpimpc {release}"
html_static_path = ["_static"]
