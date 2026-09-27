"""Activity periods to and from clipboard text (GitHub #38).

Periods often already exist in a spreadsheet or a field log. Copying two
columns (start and end) from Excel puts tab-separated text on the clipboard;
:func:`parse_periods` reads that — or semicolon/comma-separated text, ISO or
day-first dates, bare clock times, Excel serial numbers — back into
``(start, end)`` pairs. :func:`format_periods` writes periods the same way, so
they paste into Excel as columns, or into another activity.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import pandas as pd

Period = tuple[pd.Timestamp, pd.Timestamp]

#: Clipboard timestamps are written like the period editor displays them.
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_TIME_ONLY = re.compile(r"^\d{1,2}[:.]\d{2}([:.]\d{2}([.,]\d+)?)?$")
# A date followed by a clock time written with dots (Danish "03.09.2026 10.00").
_DOTTED_CLOCK = re.compile(r"(\s)(\d{1,2})\.(\d{2})(?:\.(\d{2}))?$")
_YEAR_FIRST = re.compile(r"^\d{4}[-/.]")
# Separators tried in order when a row holds no tab/semicolon/comma: a dash or
# "to" between two timestamps, or a run of two or more spaces.
_LOOSE_SEPARATOR = re.compile(r"\s+[-–—]\s+|\s+to\s+|\s{2,}", re.IGNORECASE)
# Excel serial days that are plausible measurement dates (1954–2119).
_EXCEL_RANGE = (20_000.0, 80_000.0)
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


@dataclass
class ParsedPeriods:
    """What :func:`parse_periods` found in a block of text.

    Attributes:
        periods: The ``(start, end)`` pairs, in the order they appeared.
        skipped: Rows with numbers but no valid start and end (e.g. a single
            time, or an end not after its start). Rows without any digit —
            blank lines, headers — are ignored and not counted.
        dayfirst: How ambiguous dates such as ``03-09-2026`` were read —
            ``True`` day-first, ``False`` month-first — or ``None`` when the
            text held no such date.
    """

    periods: list[Period] = field(default_factory=list)
    skipped: int = 0
    dayfirst: Optional[bool] = None


def _cells(line: str) -> list[str]:
    """Split one row into cells (tab, then semicolon, then comma, then loose)."""
    for sep in ("\t", ";", ","):
        if sep in line:
            return [c.strip().strip('"').strip() for c in line.split(sep)]
    return [c.strip() for c in _LOOSE_SEPARATOR.split(line.strip())]


def _parse_cell(cell: str, dayfirst: bool, base: Optional[pd.Timestamp]):
    """One cell as ``(timestamp, time_only)``, or ``None`` if it is not a time."""
    if not cell or not any(ch.isdigit() for ch in cell):
        return None
    if _TIME_ONLY.match(cell):
        if base is None:
            return None
        clock = cell.replace(",", ".")
        if clock.count(".") and ":" not in clock:  # 12.30 or 12.30.15
            parts = clock.split(".")
            clock = ":".join(parts[:3]) + (f".{parts[3]}" if len(parts) > 3 else "")
        try:
            return pd.Timestamp(f"{base.date()} {clock}"), True
        except ValueError:
            return None
    try:
        number = float(cell.replace(",", "."))
    except ValueError:
        number = None
    if number is not None:
        if _EXCEL_RANGE[0] <= number <= _EXCEL_RANGE[1]:
            days = pd.to_timedelta(number, unit="D")
            return (_EXCEL_EPOCH + days).round("s"), False
        return None  # a row number or a duration, not a time
    cell = _DOTTED_CLOCK.sub(
        lambda m: f"{m[1]}{m[2]}:{m[3]}" + (f":{m[4]}" if m[4] else ""), cell
    )
    # Year-first dates (ISO) are never day-first; dateutil would swap 09-03.
    dayfirst = dayfirst and not _YEAR_FIRST.match(cell)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(cell, dayfirst=dayfirst)
    except (ValueError, OverflowError, TypeError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts), False


def _read(text: str, dayfirst: bool, base: Optional[pd.Timestamp]):
    """Periods and the skipped-row count for one reading of ``dayfirst``."""
    periods: list[Period] = []
    skipped = 0
    for line in text.splitlines():
        if not any(ch.isdigit() for ch in line):
            continue  # blank, or a header such as "Start  End"
        found = [_parse_cell(c, dayfirst, base) for c in _cells(line)]
        found = [f for f in found if f is not None]
        if len(found) < 2:
            skipped += 1
            continue
        (start, start_clock), (end, end_clock) = found[0], found[1]
        if start_clock and end_clock and end <= start:
            end += pd.Timedelta(days=1)  # clock times across midnight
        if end <= start:
            skipped += 1
            continue
        periods.append((start, end))
    return periods, skipped


def _overlapping(periods: Iterable[Period], lo, hi) -> int:
    """How many periods overlap ``[lo, hi]`` (all of them without a range)."""
    periods = list(periods)
    if lo is None or hi is None:
        return len(periods)
    return sum(1 for s, e in periods if s <= hi and e >= lo)


def parse_periods(
    text: str,
    data_start: Optional[pd.Timestamp] = None,
    data_end: Optional[pd.Timestamp] = None,
) -> ParsedPeriods:
    """Read ``(start, end)`` pairs from pasted text.

    Each row gives one period: its first two cells that read as a time are
    the start and the end, so a leading name column and trailing columns are
    ignored, and a header row is simply skipped.

    Args:
        text: Rows separated by newlines; cells by tabs (as Excel copies),
            semicolons, commas, a spaced dash, or two or more spaces.
        data_start: Start of the data the periods are for. Clock times without
            a date (``12:30``) are placed on this day, and it decides between
            day-first and month-first readings of dates like ``03-09-2026``.
        data_end: End of that data (used with ``data_start`` to pick the reading
            under which most periods overlap the data).

    Returns:
        The periods found, the number of rows skipped, and the date order used.
        Ambiguous dates default to day-first (the European ``dd-mm-yyyy``) when
        both readings fit the data equally well.
    """
    base = None if data_start is None else pd.Timestamp(data_start)
    lo = base
    hi = None if data_end is None else pd.Timestamp(data_end)
    day_first = _read(text, True, base)
    month_first = _read(text, False, base)
    if day_first == month_first:
        return ParsedPeriods(*day_first, dayfirst=None)

    def score(result):
        periods, _skipped = result
        return (len(periods), _overlapping(periods, lo, hi))

    if score(month_first) > score(day_first):
        return ParsedPeriods(*month_first, dayfirst=False)
    return ParsedPeriods(*day_first, dayfirst=True)


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime(TIME_FORMAT)


def format_periods(periods: Iterable[Period], header: bool = True) -> str:
    """Tab-separated ``Start``/``End`` rows, ready to paste into a spreadsheet."""
    rows = [f"{_fmt(s)}\t{_fmt(e)}" for s, e in periods]
    return "\n".join((["Start\tEnd"] if header else []) + rows) + "\n"


def format_activities(activities: Mapping[str, Iterable[Period]]) -> str:
    """Tab-separated ``Activity``/``Start``/``End`` rows for several activities."""
    rows = ["Activity\tStart\tEnd"]
    for name, periods in activities.items():
        rows.extend(f"{name}\t{_fmt(s)}\t{_fmt(e)}" for s, e in periods)
    return "\n".join(rows) + "\n"
