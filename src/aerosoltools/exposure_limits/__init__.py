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

Scope: the particulate entries of Bilag 2 — all of **Afsnit B** (dust) and the
particles of **Afsnit A** (metals and their compounds, fumes, mists, carbon
black, diesel exhaust, …), without fibres, which are counted rather than
weighed; see :mod:`.retsinformation` for exactly which entries count. Limits
are in mg/m³; convert with :meth:`ExposureLimit.twa_in` /
:meth:`ExposureLimit.stel_in`. Where the order gives no separate short-term
value it refers to § 3, stk. 2 — the short-term limit is then twice the 8-hour
limit, and :attr:`ExposureLimit.stel_rule` records that reference; a listed
short-term value (or none) is used as the order gives it.

Each limit applies to one size fraction (respirable, thoracic, inhalable or
total dust). :func:`check_fraction` tells whether it applies to a metric such as
PM4 or a spectrometer's total mass, and :func:`applicable_limit` picks the
fraction variant of a substance that does (``"Kvarts, total"`` for a Total
channel, ``"Kvarts, respirabel"`` for PM4).

Example:
    >>> from aerosoltools import load_exposure_limits
    >>> from aerosoltools.exposure_limits import check_fraction, metric_size_cut
    >>> limits = load_exposure_limits()
    >>> limits.source.label
    'BEK nr 613 af 29/06/2026'
    >>> quartz = limits["Kvarts, respirabel"]
    >>> quartz.twa_in("µg/m³"), quartz.stel_in("µg/m³")
    (100.0, 200.0)
    >>> check_fraction(quartz, "PM4", metric_size_cut("PM4")).applies
    True
    >>> check_fraction(quartz, "PM1", metric_size_cut("PM1")).status
    'underestimates'
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from ._fractions import (
    FRACTION_CUT_UM,
    FractionCheck,
    applicable_limit,
    basis_note,
    check_fraction,
    metric_size_cut,
)
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
    "FRACTION_CUT_UM",
    "REMARK_CODES",
    "CurrentOrder",
    "ExposureLimit",
    "ExposureLimitList",
    "FractionCheck",
    "LimitSource",
    "OrderStatus",
    "RetsinformationError",
    "applicable_limit",
    "basis_note",
    "check_fraction",
    "eli_url",
    "fetch_exposure_limits",
    "fetch_order_status",
    "find_current_order",
    "load_exposure_limits",
    "metric_size_cut",
    "parse_exposure_limits_xml",
]
