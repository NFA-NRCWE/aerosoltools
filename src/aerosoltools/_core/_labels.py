"""Small shared helpers for plot/axis labels.

Centralises the one-line conventions that were previously inlined at several
call sites, so the whole codebase derives axis labels the same way.
"""

from __future__ import annotations


def base_dtype(dtype: str) -> str:
    """Return the base distribution type, dropping any normalisation suffix.

    For example ``"dN/dlogDp" -> "dN"`` and ``"dM" -> "dM"``. Used wherever a
    y-axis label or a dtype selector needs the underlying quantity rather than
    its log-diameter-normalised form.
    """
    if not dtype:
        return dtype
    return dtype.split("/")[0]
