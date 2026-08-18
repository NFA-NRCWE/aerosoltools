# docs/conf.py
from __future__ import annotations

import os
import sys
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
    "nbsphinx",  # renders (and executes) the .ipynb files in docs/examples
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

# -- Notebooks (nbsphinx) ---------------------------------------------
# The example notebooks are *always* re-executed at build time, so the docs can
# never show output from an API that no longer exists. Any error in any
# notebook fails the build rather than silently publishing a broken example.
# The notebooks read sample files from the repo's ``tests/data/`` folder using
# paths relative to ``docs/examples/``, so the docs build needs a full checkout.
nbsphinx_execute = "always"
nbsphinx_allow_errors = False
nbsphinx_timeout = 300

# -- AutoAPI (generate API pages without .rst stubs) ------------------
autoapi_type = "python"
autoapi_dirs = [SRC]  # parse this package
autoapi_root = "api"  # generated into docs/api/*
# Generated pages are build artifacts, not source: they are regenerated from
# scratch on every build and are gitignored. Keeping them caused deleted
# modules (e.g. the old ``aerosolalt`` / ``utility`` packages) to linger and be
# republished, because AutoAPI never prunes stale files it did not write.
autoapi_keep_files = False
autoapi_add_toctree_entry = True  # let AutoAPI own the API Reference toctree
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"  # include class + __init__ docstrings

# Include all public members + inherited methods; exclude private ("_")
autoapi_options = [
    "members",
    "undoc-members",
    "inherited-members",
    "show-inheritance",
]

# -- Hide internal implementation packages from the API reference -----
# The public classes (Aerosol1D / Aerosol2D / Aerosol3d and the instrument
# subclasses) compose topic mixins from the internal ``_core`` package. Those
# mixins are still *parsed* by AutoAPI, so ``inherited-members`` documents every
# method on the class pages as before; but the mixin modules themselves are
# internal, so we keep them out of the generated API pages to avoid orphaned,
# noisy reference pages.
#
# The PyQt5 GUI is skipped at *parse* time instead: it is an application, not a
# library API, and parsing it emitted ~40 "Cannot resolve import of
# aerosoltools.gui.qt.*" warnings (the Qt names are re-exported through the
# gui.qt binding module, which AutoAPI's static analysis cannot follow). Those
# warnings would defeat the -W flag the docs workflow builds with.
autoapi_ignore = ["*/gui/*"]

_AUTOAPI_INTERNAL = ("._core",)


def _autoapi_skip_internal(app, what, name, obj, skip, options):
    if what in ("module", "package") and any(p in name for p in _AUTOAPI_INTERNAL):
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", _autoapi_skip_internal)


# -- Autodoc/Napoleon/typing polish ----------------------------------
autodoc_default_options = {"members": True}
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# Render docstring "Attributes:" sections as an :ivar: field list rather than
# standalone `.. attribute::` directives. Without this, a documented property
# (e.g. Environmental1D.pressure) is described twice on the same page -- once
# from the class docstring's Attributes section and once from AutoAPI's own
# entry for the property -- which Sphinx reports as a duplicate object
# description, and the docs workflow builds with -W.
napoleon_use_ivar = True

# -- Intersphinx ------------------------------------------------------
# Turns references to third-party types in signatures/docstrings into links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_timeout = 15

# NOTE: an unreachable inventory (upstream outage, or a corporate proxy that
# intercepts TLS) warns but is not a defect in these docs, and Sphinx logs that
# particular warning without a type, so ``suppress_warnings`` cannot target it.
# The docs workflow therefore treats warnings as fatal via an explicit filter
# over the captured warning log rather than via ``-W``; see
# .github/workflows/_docs-build.yml.

# -- HTML -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "aerosoltools"
