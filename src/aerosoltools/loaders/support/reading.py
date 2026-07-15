"""Shared file-reading helpers for the instrument loaders.

Every loader used to hand-roll the same "detect encoding/delimiter, then read"
dance — often calling :func:`~aerosoltools.loaders.support.parsing._detect_delimiter`
several times and re-parsing the file with :func:`numpy.genfromtxt`/
:func:`pandas.read_csv` more than once. These helpers give one place to sniff a
file and read its data table, so a loader detects once and reads once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parsing import _detect_delimiter


def sniff(
    file: str,
    encoding: str | None = None,
    delimiter: str | None = None,
    **detect_kwargs,
):
    """Return ``(encoding, delimiter)`` for ``file``, detecting what is missing.

    Pass a value you already know to skip detecting it; pass neither to detect
    both. ``detect_kwargs`` (e.g. ``sample_lines=30``) are forwarded to
    :func:`~aerosoltools.loaders.support.parsing._detect_delimiter` so loaders
    that need a larger sample keep their exact detection behaviour.
    """
    if encoding is not None and delimiter is not None:
        return encoding, delimiter
    enc, delim = _detect_delimiter(file, **detect_kwargs)
    return (encoding or enc), (delimiter or delim)


def read_table(
    file: str,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read ``file`` as a delimited table with :func:`pandas.read_csv`.

    Detects the encoding and delimiter (via :func:`sniff`) when they are not
    supplied, then forwards ``read_csv_kwargs`` (``header``, ``usecols``,
    ``skipfooter``, ``engine``, ``nrows``, …) unchanged to
    :func:`pandas.read_csv`.
    """
    encoding, delimiter = sniff(file, encoding, delimiter)
    return pd.read_csv(file, encoding=encoding, delimiter=delimiter, **read_csv_kwargs)


def read_cells(
    file: str,
    *,
    skip_header: int,
    max_rows: int = 1,
    encoding: str | None = None,
    delimiter: str | None = None,
    **genfromtxt_kwargs,
) -> np.ndarray:
    """Read a small block of header cells as strings via :func:`numpy.genfromtxt`.

    A thin wrapper for the ``genfromtxt(..., skip_header=N, max_rows=1,
    dtype=str)`` metadata pluck the loaders repeat, with encoding/delimiter
    detection folded in. Keeps genfromtxt's exact tokenising, so results are
    identical to the previous inline calls.
    """
    encoding, delimiter = sniff(file, encoding, delimiter)
    genfromtxt_kwargs.setdefault("dtype", str)
    return np.genfromtxt(
        file,
        delimiter=delimiter,
        encoding=encoding,
        skip_header=skip_header,
        max_rows=max_rows,
        **genfromtxt_kwargs,
    )
