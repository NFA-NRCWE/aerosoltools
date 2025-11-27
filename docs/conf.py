# docs/conf.py
from __future__ import annotations
import os, sys
from datetime import date

# -- Project info -----------------------------------------------------
project = "aerosoltools"
author = "NFA / NRCWE"
copyright = f"{date.today().year}, {author}"
root_doc = "index"

# -- General config ---------------------------------------------------
extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "nbsphinx",  # to render .ipynb in docs/examples
]

# Treat Markdown as first-class
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["colon_fence", "linkify"]

# Exclude build artifacts and notebook checkpoints
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

# Put repo /src on path so type hints and cross-refs can import when needed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# -- AutoAPI (generate API pages without .rst stubs) ------------------
autoapi_type = "python"
autoapi_dirs = [SRC]  # parse this package
autoapi_root = "api"                 # generated into docs/api/*
autoapi_keep_files = True
autoapi_add_toctree_entry = False     # add “API Reference” to the TOC
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"  # include class + __init__ docstrings

# Include all public members + inherited methods; exclude private (“_”)
autoapi_options = [
    "members",
    "undoc-members",
    "inherited-members",
    "show-inheritance",
]

autoapi_ignore = []


# -- Autodoc/Napoleon/typing polish ----------------------------------
autodoc_default_options = {"members": True}
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "aerosoltools"
