"""Shared time/frequency helpers used across the utility submodules."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd


def _ts(x) -> pd.Timestamp:
    """Coerce an input into a :class:`pandas.Timestamp`.

    This helper accepts strings, Python datetime objects, NumPy datetime64, and
    already-constructed :class:`pandas.Timestamp` objects and returns a
    normalized ``Timestamp`` instance.

    Args:
        x: A value representing a point in time (e.g. ``str``,
            :class:`datetime.datetime`, :class:`datetime.date`,
            :class:`numpy.datetime64`, or :class:`pandas.Timestamp`).

    Returns:
        pandas.Timestamp: The input converted to a ``Timestamp``.
    """
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (dt.datetime, dt.date, np.datetime64, str)):
        return pd.to_datetime(x)
    return pd.to_datetime(x)


def _infer_freq(idx: pd.DatetimeIndex) -> Optional[str]:
    """Infer a reasonable resampling rule from a time index.

    The function first tries :func:`pandas.infer_freq`. If that fails, it falls
    back to estimating the cadence from the median inter-sample spacing and
    returns a rule like ``"1S"``, ``"5T"`` or ``"1H"``.

    Args:
        idx: Datetime index from which to infer a sampling frequency.

    Returns:
        str | None: A pandas offset alias representing the inferred cadence
        (e.g. ``"1S"``, ``"30T"``, ``"1H"``), or ``None`` if it cannot be
        determined.
    """
    if len(idx) < 3:
        return None
    f = pd.infer_freq(idx)
    if f:
        return f
    d = np.diff(idx.view("i8"))  # ns
    if d.size == 0:
        return None
    sec = int(round(np.median(d) / 1e9))
    if sec < 60:
        return f"{max(1, sec)}S"
    if sec < 3600:
        return f"{max(1, sec // 60)}T"
    return f"{max(1, sec // 3600)}H"


def _coarser(rule_a: str, rule_b: str) -> str:
    """Return the coarser (slower) cadence between two resampling rules.

    The rules are interpreted as simple second-, minute-, hour- or day-based
    frequencies (e.g. ``"S"``, ``"10S"``, ``"5T"``, ``"1H"``, ``"1D"``), and
    compared by their corresponding period length in seconds.

    Args:
        rule_a: First pandas-style frequency string.
        rule_b: Second pandas-style frequency string.

    Returns:
        str: The rule corresponding to the larger time step (coarser cadence).
    """

    def to_s(rule: str) -> float:
        r = rule.upper()
        num = "".join(ch for ch in r if ch.isdigit())
        n = int(num) if num else 1
        unit = "".join(ch for ch in r if ch.isalpha()) or "S"
        return n * {"S": 1, "T": 60, "MIN": 60, "H": 3600, "D": 86400}.get(unit, 1)

    return rule_a if to_s(rule_a) >= to_s(rule_b) else rule_b
