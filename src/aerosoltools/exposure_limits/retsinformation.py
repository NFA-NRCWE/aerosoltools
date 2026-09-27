"""Fetch and parse the Danish limit-value order from Retsinformation.

The limits live in *Bekendtgørelse om grænseværdier for stoffer og materialer
(kemiske agenser) i arbejdsmiljøet*, Bilag 2. Retsinformation publishes every
order under a stable European Legislation Identifier (ELI), e.g.
``https://www.retsinformation.dk/eli/lta/2026/613``, with two machine-readable
views that this module uses:

* ``<eli>/xml`` — the full order as LexDania XML, tables included. This is what
  :func:`parse_exposure_limits_xml` reads.
* ``<eli>.rdfa`` — a small ELI-metadata record (JSON) with the order's in-force
  status and its ``changed_by`` / ``changes`` relations. A revised limit list is
  issued as a **new order** that repeals the old one, so there is no fixed URL
  for "the latest"; :func:`find_current_order` instead follows ``changed_by``
  from a known order to the one currently in force.

The official *høsteservice* API (``api.retsinformation.dk``) is not used: it is a
change feed of the last 10 days for nightly harvesting and cannot look up a
given document.

Only **Bilag 2, Afsnit B** (dust) is parsed, and its fibre entries (limits in
fibres/cm³) are left out, since aerosoltools reports mass and number
concentrations rather than fibre counts. Afsnit A (gases, vapours *and* some
particulate substances) and Afsnit C (process-specific welding limits) are not
parsed.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone

from ._model import ExposureLimit, ExposureLimitList, LimitSource

#: ELI root for Danish administrative regulations (*lovtidende A*).
ELI_BASE = "https://www.retsinformation.dk/eli/lta"

#: Title prefix shared by every full limit-value order. Amending orders start
#: "Bekendtgørelse om ændring af …" and so do not match.
ORDER_TITLE = "Bekendtgørelse om grænseværdier for stoffer og materialer"

_USER_AGENT = "aerosoltools (+https://nfa-nrcwe.github.io/aerosoltools/)"

_MONTHS = {
    name: i
    for i, name in enumerate(
        (
            "januar",
            "februar",
            "marts",
            "april",
            "maj",
            "juni",
            "juli",
            "august",
            "september",
            "oktober",
            "november",
            "december",
        ),
        start=1,
    )
}

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_PLAIN_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_FOOTNOTE_DIGITS = "0-9⁰¹²³⁴⁵⁶⁷⁸⁹"
#: A trailing footnote marker on a name or value: ``"Glasuldsfibre 1)"``.
_TRAILING_FOOTNOTE = re.compile(rf"\s+([{_FOOTNOTE_DIGITS}]+)\)$")
#: The first cell of a footnote row at the foot of a table: ``"1)"``.
_FOOTNOTE_ROW = re.compile(rf"^([{_FOOTNOTE_DIGITS}]+)\)$")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
#: § 3, stk. 2 of the order: where Bilag 2 refers to it, the short-term limit
#: is twice the 8-hour limit. Checked against the text, not assumed.
_DOUBLING_RULE = re.compile(
    r"korttidsgrænseværdien\s+to\s+gange\s+8-timers\s+grænseværdien", re.IGNORECASE
)
_IN_FORCE = re.compile(
    r"træder\s+i\s+kraft\s+den\s+(\d{1,2})\.\s*([a-zæøå]+)\s+(\d{4})", re.IGNORECASE
)


class RetsinformationError(RuntimeError):
    """Retsinformation could not be reached, or returned something unexpected."""


# -- ELI identifiers -----------------------------------------------------------
def eli_url(eli: str) -> str:
    """Normalise an order reference to its canonical ELI URL.

    Accepts ``"2026/613"``, an ELI URL (with or without ``www.``, ``/xml`` or
    ``.rdfa``), or a citation such as ``"BEK nr 613 af 29/06/2026"``.

    Args:
        eli: The order reference.

    Returns:
        ``"https://www.retsinformation.dk/eli/lta/<year>/<number>"``.

    Raises:
        ValueError: If no year and number can be read from ``eli``.
    """
    text = (eli or "").strip()
    cited = re.search(r"nr\.?\s*(\d+)\s+af\s+\d{1,2}/\d{1,2}/(\d{4})", text, re.I)
    if cited:
        return f"{ELI_BASE}/{cited.group(2)}/{int(cited.group(1))}"
    m = re.search(r"(?:^|/)(\d{4})/(\d+)(?=$|[/.?#])", text)
    if m:
        return f"{ELI_BASE}/{m.group(1)}/{int(m.group(2))}"
    raise ValueError(
        f"Could not read an order from {eli!r}; give it as 'year/number' "
        "(e.g. '2026/613'), an ELI link, or 'BEK nr 613 af 29/06/2026'."
    )


def _eli_parts(url: str) -> tuple[int, int]:
    """``(year, number)`` of a canonical ELI URL."""
    year, number = url.rstrip("/").rsplit("/", 2)[-2:]
    return int(year), int(number)


#: The order the bundled list was parsed from.
BUNDLED_ELI = f"{ELI_BASE}/2026/613"


def _http_get(url: str, timeout: float) -> bytes:
    """GET ``url`` and return the body, raising :class:`RetsinformationError`."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        hint = " — check the year and number" if exc.code == 404 else ""
        raise RetsinformationError(
            f"Retsinformation returned HTTP {exc.code} for {url}{hint}."
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RetsinformationError(
            f"Could not reach Retsinformation ({url}): {exc}"
        ) from exc


# -- order status / successor chain --------------------------------------------
@dataclass(frozen=True)
class OrderStatus:
    """In-force status and relations of one order (from its ELI metadata).

    Attributes:
        eli: Canonical ELI URL of the order.
        title: The order's title.
        in_force: Whether the order is currently in force.
        date: Date the order was issued (ISO), when known.
        no_longer_in_force: When the order stopped being in force (ISO
            timestamp), if it has.
        changed_by: ELI URLs of later orders that change or replace it.
        changes: ELI URLs of earlier orders it changes or replaces.
    """

    eli: str
    title: str
    in_force: bool
    date: str | None = None
    no_longer_in_force: str | None = None
    changed_by: tuple[str, ...] = ()
    changes: tuple[str, ...] = ()

    @property
    def number(self) -> int:
        """The order's number (*BEK nr*)."""
        return _eli_parts(self.eli)[1]

    @property
    def label(self) -> str:
        """Citation in Retsinformation's style, e.g. ``BEK nr 613 af 29/06/2026``."""
        if self.date:
            issued = date.fromisoformat(self.date).strftime("%d/%m/%Y")
            return f"BEK nr {self.number} af {issued}"
        year, number = _eli_parts(self.eli)
        return f"BEK nr {number} ({year})"

    @property
    def is_limit_order(self) -> bool:
        """Whether this is a full limit-value order (not an amending order)."""
        return _is_order_title(self.title)


@dataclass(frozen=True)
class CurrentOrder:
    """Result of :func:`find_current_order`.

    Attributes:
        current: The limit-value order in force now.
        chain: The orders walked through, from the starting order to
            :attr:`current` (just ``(current,)`` when nothing replaced it).
        amendments: Orders in force that amend :attr:`current` without
            replacing it. Their changes are **not** in the parsed list.
    """

    current: OrderStatus
    chain: tuple[OrderStatus, ...]
    amendments: tuple[OrderStatus, ...] = ()

    @property
    def superseded(self) -> bool:
        """Whether the starting order has been replaced by a newer one."""
        return len(self.chain) > 1


def _is_order_title(title: str) -> bool:
    """Whether ``title`` is that of a full limit-value order."""
    return " ".join((title or "").split()).casefold().startswith(ORDER_TITLE.casefold())


def _dk_date(text: str | None) -> str | None:
    """``"29-06-2026 00:00:00"`` → ``"2026-06-29"``."""
    if not text:
        return None
    try:
        return datetime.strptime(text.split()[0], "%d-%m-%Y").date().isoformat()
    except ValueError:
        return None


def _dk_datetime(text: str | None) -> str | None:
    """``"29-06-2026 16:42:28"`` → ``"2026-06-29T16:42:28"``."""
    if not text:
        return None
    try:
        return datetime.strptime(text, "%d-%m-%Y %H:%M:%S").isoformat()
    except ValueError:
        return _dk_date(text)


def parse_order_status(payload: object, eli: str) -> OrderStatus:
    """Build an :class:`OrderStatus` from an order's ELI-metadata record.

    Args:
        payload: The decoded JSON of ``<eli>.rdfa`` — a list of
            ``{"property": …, "content"|"resource": …}`` statements.
        eli: The order the record describes.

    Raises:
        RetsinformationError: If ``payload`` is not such a record.
    """
    if not isinstance(payload, list):
        raise RetsinformationError(f"Unexpected ELI metadata for {eli}.")
    props: dict[str, list[str]] = {}
    for statement in payload:
        if not isinstance(statement, dict) or not statement.get("property"):
            continue
        value = statement.get("content") or statement.get("resource")
        if value:
            props.setdefault(statement["property"], []).append(str(value))

    def first(key: str) -> str | None:
        values = props.get(key)
        return values[0] if values else None

    return OrderStatus(
        eli=eli_url(eli),
        title=first("eli:title") or "",
        in_force=any(
            v.endswith("InForce-inForce") for v in props.get("eli:in_force", [])
        ),
        date=_dk_date(first("eli:date_document")),
        no_longer_in_force=_dk_datetime(first("eli:date_no_longer_in_force")),
        changed_by=tuple(eli_url(u) for u in props.get("eli:changed_by", [])),
        changes=tuple(eli_url(u) for u in props.get("eli:changes", [])),
    )


def fetch_order_status(eli: str, timeout: float = 30.0) -> OrderStatus:
    """Download an order's ELI metadata and return its status (one small request).

    Args:
        eli: The order (see :func:`eli_url` for accepted forms).
        timeout: Network timeout in seconds.

    Raises:
        RetsinformationError: On a network error or an unexpected response.
    """
    url = eli_url(eli)
    raw = _http_get(url + ".rdfa", timeout)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetsinformationError(f"Unexpected ELI metadata for {url}.") from exc
    return parse_order_status(payload, url)


def find_current_order(
    eli: str = BUNDLED_ELI, timeout: float = 30.0, max_steps: int = 25
) -> CurrentOrder:
    """Find the limit-value order currently in force, starting from ``eli``.

    Follows each order's ``changed_by`` links to the full limit-value order that
    replaced it (amending orders are skipped), until reaching one in force. Any
    amending orders still attached to that one are reported in
    :attr:`CurrentOrder.amendments`, since their changes are not part of the
    parsed list.

    Args:
        eli: Order to start from — by default the one the bundled list uses.
        timeout: Network timeout per request, in seconds.
        max_steps: Safety cap on the number of orders followed.

    Returns:
        The in-force order and the chain that led to it.

    Raises:
        RetsinformationError: On a network error, or when an order is no longer
            in force but no replacement can be found.
    """
    seen: dict[str, OrderStatus] = {}

    def status(url: str) -> OrderStatus:
        if url not in seen:
            seen[url] = fetch_order_status(url, timeout)
        return seen[url]

    current = status(eli_url(eli))
    chain = [current]
    while not current.in_force:
        if len(chain) > max_steps:
            raise RetsinformationError(
                f"Gave up after following {max_steps} replaced orders."
            )
        successors = [
            s
            for s in (status(u) for u in current.changed_by if u not in seen)
            if s.is_limit_order
        ]
        if not successors:
            raise RetsinformationError(
                f"{current.label} is no longer in force, but Retsinformation lists "
                "no limit-value order that replaces it. Look it up on "
                "retsinformation.dk and fetch it by its number."
            )
        # Prefer a successor in force; otherwise the newest, and keep following.
        current = max(successors, key=lambda s: (s.in_force, s.date or ""))
        chain.append(current)
    amendments = tuple(status(u) for u in current.changed_by)
    return CurrentOrder(current=current, chain=tuple(chain), amendments=amendments)


# -- XML parsing ---------------------------------------------------------------
def _local(tag: str) -> str:
    """Tag name without any XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _children(elem: ET.Element, name: str) -> list[ET.Element]:
    """Direct children of ``elem`` named ``name``."""
    return [c for c in elem if _local(c.tag) == name]


def _descendants(elem: ET.Element, name: str):
    """All descendants of ``elem`` named ``name``."""
    return (e for e in elem.iter() if _local(e.tag) == name)


def _cell_text(td: ET.Element) -> str:
    """A table cell's text, rendering sub/superscript digits (SiO₂, mg/m³)."""
    lines = []
    for linea in _descendants(td, "Linea"):
        parts = []
        for char in _descendants(linea, "Char"):
            text = "".join(char.itertext())
            style = (char.get("formaChar") or "").lower()
            if "subscript" in style:
                text = text.translate(_SUBSCRIPT)
            elif "superscript" in style:
                text = text.translate(_SUPERSCRIPT)
            parts.append(text)
        lines.append("".join(parts))
    text = " ".join(lines) if lines else "".join(td.itertext())
    return " ".join(text.split())


def _row_cells(tr: ET.Element) -> tuple[list[str], list[tuple[int, int]]]:
    """A row's cell texts with ``colspan`` expanded, plus each cell's (start, span)."""
    cells: list[str] = []
    spans: list[tuple[int, int]] = []
    for td in _children(tr, "Td"):
        try:
            span = max(1, int(td.get("colspan") or 1))
        except ValueError:
            span = 1
        spans.append((len(cells), span))
        cells.append(_cell_text(td))
        cells.extend([""] * (span - 1))
    return cells, spans


def _number(text: str) -> float | None:
    """A Danish-formatted number (``"0,05"``), ignoring a footnote marker."""
    text = _TRAILING_FOOTNOTE.sub("", text or "").strip()
    return float(text.replace(",", ".")) if _NUMBER.match(text) else None


def _split_footnote(text: str) -> tuple[str, str | None]:
    """Split a trailing footnote marker off: ``"Glasuld 1)"`` → ``("Glasuld", "1")``."""
    m = _TRAILING_FOOTNOTE.search(text)
    if not m:
        return text, None
    return text[: m.start()].rstrip(), m.group(1).translate(_PLAIN_DIGITS)


def _find_section_table(root: ET.Element, section: str) -> ET.Element:
    """The Bilag 2 table whose title row reads ``"Afsnit <section> …"``."""
    pattern = re.compile(rf"^Afsnit\s+{re.escape(section)}\b")
    for table in _descendants(root, "Table"):
        rows = list(_descendants(table, "Tr"))
        if rows and pattern.match(" ".join(c for c in _row_cells(rows[0])[0] if c)):
            return table
    raise RetsinformationError(
        f"No 'Afsnit {section}' table was found — the order's layout may have "
        "changed; the parser needs updating."
    )


def _header_columns(tr: ET.Element) -> dict[str, tuple[int, int]]:
    """Map the header row's cells to ``{field: (start column, span)}``."""
    cells, spans = _row_cells(tr)
    keys = (
        ("cas", "cas"),
        ("name", "stof"),
        ("year", "årstal"),
        ("twa", "8-timers"),
        ("stel", "korttid"),
        ("remarks", "anmærkning"),
    )
    columns: dict[str, tuple[int, int]] = {}
    for start, span in spans:
        head = cells[start].lower()
        for key, word in keys:
            if key not in columns and head.startswith(word):
                columns[key] = (start, span)
    return columns


def _parse_section_b(
    table: ET.Element, doubling_rule: bool
) -> tuple[list[ExposureLimit], list[str]]:
    """Parse the Afsnit B (dust) table into limits and excluded fibre names."""
    rows = list(_descendants(table, "Tr"))
    header = next(
        (
            i
            for i, tr in enumerate(rows)
            if any(c.lower().startswith("cas") for c in _row_cells(tr)[0])
        ),
        None,
    )
    if header is None:
        raise RetsinformationError("The Afsnit B table has no header row.")
    columns = _header_columns(rows[header])
    missing = {"name", "twa"} - columns.keys()
    if missing:
        raise RetsinformationError(
            f"The Afsnit B header lacks {sorted(missing)} — the parser needs updating."
        )

    def col(cells: list[str], key: str, offset: int = 0) -> str:
        if key not in columns:
            return ""
        i = columns[key][0] + offset
        return cells[i] if i < len(cells) else ""

    # The 8-hour header spans two columns: the value, then its unit.
    twa_has_unit_column = columns["twa"][1] >= 2

    records: list[dict] = []
    footnotes: dict[str, str] = {}
    for tr in rows[header + 1 :]:
        cells, _spans = _row_cells(tr)
        if not any(cells):
            continue
        marker = _FOOTNOTE_ROW.match(cells[0])
        if marker:
            text = next((c for c in cells[1:] if c), "")
            footnotes[marker.group(1).translate(_PLAIN_DIGITS)] = text
            continue
        cas, name = col(cells, "cas"), col(cells, "name")
        value = col(cells, "twa")
        unit = col(cells, "twa", 1) if twa_has_unit_column else ""
        if not twa_has_unit_column and value:
            value, _, unit = value.partition(" ")
        if not name:
            # A continuation row carrying another CAS number of the entry above.
            if cas and records:
                records[-1]["cas"].append(cas)
            continue
        if name.lower().startswith("fra d") and not cas and records:
            # A later-dated value for the entry above ("Fra den 21. december …").
            records[-1]["notes"].append(f"{name}: {value} {unit}".strip())
            continue
        records.append(
            {
                "name": name,
                "cas": [cas] if cas else [],
                "year": col(cells, "year"),
                "value": value,
                "unit": unit,
                "stel": col(cells, "stel"),
                "remarks": col(cells, "remarks"),
                "notes": [],
            }
        )

    limits: list[ExposureLimit] = []
    excluded: list[str] = []
    for rec in records:
        name, marker = _split_footnote(rec["name"])
        notes = list(rec["notes"])
        if marker and marker in footnotes:
            notes.insert(0, footnotes[marker])
        # The order's legend: particulate pollution is given in mg/m³.
        unit = rec["unit"] or "mg/m³"
        if "fiber" in unit.lower() or "fibre" in unit.lower():
            excluded.append(name)
            continue
        twa = _number(rec["value"])
        stel_text = rec["stel"].strip()
        stel_rule = ""
        if stel_text.lower().startswith("jf"):
            stel_rule = stel_text
            stel = 2.0 * twa if (doubling_rule and twa is not None) else None
        else:
            stel = _number(stel_text)
        year = int(rec["year"]) if rec["year"].isdigit() else None
        limits.append(
            ExposureLimit(
                name=name,
                cas=tuple(rec["cas"]),
                twa=twa,
                stel=stel,
                unit=unit,
                stel_rule=stel_rule,
                remarks=rec["remarks"],
                year=year,
                notes=tuple(notes),
                section="B",
            )
        )
    return limits, excluded


def _in_force_from(text: str) -> str | None:
    """The date the order takes effect (``"træder i kraft den 1. juli 2026"``)."""
    m = _IN_FORCE.search(text)
    if not m or m.group(2).lower() not in _MONTHS:
        return None
    try:
        return date(
            int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))
        ).isoformat()
    except ValueError:
        return None


def parse_exposure_limits_xml(
    data: bytes | str, eli: str | None = None, retrieved: str | None = None
) -> ExposureLimitList:
    """Parse the particulate limits out of a limit-value order's XML.

    Args:
        data: The order as LexDania XML (``<eli>/xml``).
        eli: The order's ELI URL, recorded as the source; read from the
            document's number and year when omitted.
        retrieved: When the XML was downloaded (ISO timestamp); defaults to now.

    Returns:
        The Afsnit B (dust) limits, fibres excluded, with the order's details.

    Raises:
        RetsinformationError: If ``data`` is not the limit-value order or its
            layout is not recognised.
    """
    try:
        root = ET.fromstring(data.encode("utf-8") if isinstance(data, str) else data)
    except ET.ParseError as exc:
        raise RetsinformationError("Retsinformation did not return valid XML.") from exc
    meta = next(_descendants(root, "Meta"), None)

    def meta_text(name: str) -> str:
        found = next(_descendants(meta, name), None) if meta is not None else None
        return (found.text or "").strip() if found is not None else ""

    title = meta_text("DocumentTitle")
    if not _is_order_title(title):
        raise RetsinformationError(
            f"{title or 'This document'!r} is not the limit-value order "
            f"({ORDER_TITLE} …)."
        )
    number = int(meta_text("Number") or 0)
    issued = meta_text("DiesSigni") or meta_text("DiesEdicti")
    text = " ".join(root.itertext())
    limits, excluded = _parse_section_b(
        _find_section_table(root, "B"), bool(_DOUBLING_RULE.search(text))
    )
    source = LimitSource(
        title=" ".join(title.split()),
        number=number,
        date=issued,
        eli=eli_url(eli) if eli else f"{ELI_BASE}/{issued[:4]}/{number}",
        status=meta_text("Status"),
        in_force_from=_in_force_from(text),
        accession_number=meta_text("AccessionNumber"),
        retrieved=retrieved
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        sections=("B",),
    )
    return ExposureLimitList(
        source=source, limits=tuple(limits), excluded=tuple(excluded)
    )


def fetch_exposure_limits(
    eli: str = BUNDLED_ELI, timeout: float = 60.0
) -> ExposureLimitList:
    """Download a limit-value order from Retsinformation and parse its dust limits.

    Use :func:`find_current_order` first to learn which order is in force.

    Args:
        eli: The order (see :func:`eli_url` for accepted forms).
        timeout: Network timeout in seconds (the XML is a few MB).

    Raises:
        RetsinformationError: On a network error, or if the document is not the
            limit-value order.
    """
    url = eli_url(eli)
    return parse_exposure_limits_xml(_http_get(url + "/xml", timeout), eli=url)
