"""Occupational exposure limits (OELs) for particulate substances.

The Danish limits are read from the legal source itself — *Bekendtgørelse om
grænseværdier for stoffer og materialer (kemiske agenser) i arbejdsmiljøet* on
Retsinformation — so an exposure summary or plot threshold can be filled from
the order rather than typed by hand, and every list carries the order's number
and date (e.g. ``BEK nr 613 af 29/06/2026``) so it is always clear which version
a measurement was compared against.

A parsed snapshot ships with the package; :func:`load_exposure_limits` returns
it without any network access. :func:`find_current_order` checks Retsinformation
for a newer order and :func:`fetch_exposure_limits` downloads and parses one
(``python -m aerosoltools.exposure_limits`` does both from the command line).

Scope: only Bilag 2, **Afsnit B** (dust) is parsed, without its fibre entries,
because aerosoltools measures mass and number concentrations, not fibre counts.
Limits are in mg/m³; convert with :meth:`ExposureLimit.twa_in` /
:meth:`ExposureLimit.stel_in`. Where the order gives no separate short-term
value it refers to § 3, stk. 2 — the short-term limit is then twice the 8-hour
limit, and :attr:`ExposureLimit.stel_rule` records that reference.

Example:
    >>> from aerosoltools import load_exposure_limits
    >>> limits = load_exposure_limits()
    >>> limits.source.label
    'BEK nr 613 af 29/06/2026'
    >>> quartz = limits["Kvarts, respirabel"]
    >>> quartz.twa_in("µg/m³"), quartz.stel_in("µg/m³")
    (100.0, 200.0)
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from ._model import REMARK_CODES, ExposureLimit, ExposureLimitList, LimitSource
from .retsinformation import (
    BUNDLED_ELI,
    CurrentOrder,
    OrderStatus,
    RetsinformationError,
    eli_url,
    fetch_exposure_limits,
    fetch_order_status,
    find_current_order,
    parse_exposure_limits_xml,
)

#: File name of the bundled snapshot (in this package's ``data/`` folder).
BUNDLED_FILE = "dk_bek_2026_613.json"


def load_exposure_limits(path: str | Path | None = None) -> ExposureLimitList:
    """Load an exposure-limit list — by default the snapshot bundled with the package.

    Args:
        path: A list saved with :meth:`ExposureLimitList.to_json` (e.g. a newer
            order fetched with :func:`fetch_exposure_limits`). ``None`` loads
            the bundled snapshot.

    Returns:
        The limits together with the order they were parsed from.
    """
    if path is not None:
        return ExposureLimitList.from_json(path)
    bundled = resources.files(__name__) / "data" / BUNDLED_FILE
    return ExposureLimitList.from_dict(json.loads(bundled.read_text(encoding="utf-8")))


__all__ = [
    "BUNDLED_ELI",
    "BUNDLED_FILE",
    "REMARK_CODES",
    "CurrentOrder",
    "ExposureLimit",
    "ExposureLimitList",
    "LimitSource",
    "OrderStatus",
    "RetsinformationError",
    "eli_url",
    "fetch_exposure_limits",
    "fetch_order_status",
    "find_current_order",
    "load_exposure_limits",
    "parse_exposure_limits_xml",
]
