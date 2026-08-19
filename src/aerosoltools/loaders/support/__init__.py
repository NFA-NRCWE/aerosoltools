"""Shared support code for the instrument loaders.

This subpackage holds the pieces the per-instrument loaders reuse, kept separate
from the instrument modules themselves so the loader package reads as "one file
per instrument, plus a clearly-marked support corner":

* :mod:`~aerosoltools.loaders.support.parsing` — text-file encoding/delimiter
  sniffing and the folder batch-loader (formerly ``loaders/Common.py``).
* :mod:`~aerosoltools.loaders.support.exceptions` — the loader exception
  hierarchy (:class:`~aerosoltools.loaders.support.exceptions.LoaderError` and
  its subclasses).

Nothing here is instrument-specific; the dispatch/auto-detection registry stays
at the loader-package top level because it depends on every instrument module.
"""
