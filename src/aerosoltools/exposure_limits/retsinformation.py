"""Fetch and parse the Danish limit-value order from Retsinformation.

The limits live in *Bekendtgørelse om grænseværdier for stoffer og materialer
(kemiske agenser) i arbejdsmiljøet*, Bilag 2. Retsinformation publishes every
order under a stable European Legislation Identifier (ELI), e.g.
``https://www.retsinformation.dk/eli/lta/2026/613``, with two machine-readable
views that this module uses:

* ``<eli>/dan/xml`` — the full order as LexDania XML, tables included, which
  the order's ELI metadata declares an *official* embodiment (publisher
  Civilstyrelsen). This is what :func:`parse_exposure_limits_xml` reads; no web
  page is scraped.
* ``<eli>.rdfa`` — the order's ELI-metadata record (JSON) with its in-force
  status and ``changed_by`` / ``changes`` relations. A revised limit list is
  issued as a **new order** that repeals the old one, so there is no fixed URL
  for "the latest"; :func:`find_current_order` instead follows ``changed_by``
  from a known order to the one currently in force.

The official *høsteservice* API (``api.retsinformation.dk``) is not used: it is a
change feed of the last 10 days for nightly harvesting and cannot look up a
given document.

Parsed are the particulate entries of Bilag 2: all of **Afsnit B** (dust), and
the entries of **Afsnit A** (gases, vapours and particulate pollution) that are
particles — a dust, powder, fume, mist or particle form, a size fraction, a
metal or metalloid compound (*beregnet som* a metal), or one of a few named
substances (carbon black, diesel exhaust, …). Left out, with a reason in
:attr:`ExposureLimitList.excluded`: fibres (counted per cm³), mercury (also a
vapour), volatile metal compounds, and the remaining Afsnit A entries that have
no ppm limit but are not clearly particulate (mostly organic compounds), which
need a person's assessment. Afsnit C (process-specific welding limits) is not
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
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
#: A trailing footnote marker on a name or value: ``"Glasuldsfibre 1)"``, or a
#: superscript one attached as in ``"0,005⁴)"``.
_TRAILING_FOOTNOTE = re.compile(rf"(?:\s+([0-9]+)|\s*([{_SUPERSCRIPT_DIGITS}]+))\)$")
#: The first cell of a footnote row at the foot of a table: ``"1)"``.
_FOOTNOTE_ROW = re.compile(rf"^([0-9{_SUPERSCRIPT_DIGITS}]+)\)$")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
#: A row named only by a size fraction ("Inhalerbar"), sub-entry of the row above.
_BARE_FRACTION = re.compile(r"(?:respirabel|inhalerbar|thorakal|total)(?:\s+fraktion)?")

# -- which Afsnit A entries are particles --------------------------------------
#: Name words marking a particulate form (dust, powder, fume, mist, particles,
#: aerosol) or a size fraction.
_PARTICLE_WORDS = re.compile(
    r"støv|pulver|røg|tåge|partik|aerosol|respirab|inhalerb|thorak|\btotal",
    re.IGNORECASE,
)
#: "beregnet som <metal or metalloid>": such compounds are airborne as particles
#: (their gaseous hydrides are separate entries with a ppm limit).
_METAL_BASIS = re.compile(
    r"beregnet som\s+(?:Ag|Al|As|Ba|Be|Bi|Ca|Cd|Co|Cr|Cs|Cu|Fe|Hf|In|Ir|Li|Mg|Mn|"
    r"Mo|Nb|Ni|Os|Pb|Pd|Pt|Rh|Ru|Sb|Se|Sn|Sr|Ta|Te|Ti|Tl|U|V|W|Y|Zn|Zr)\b"
)
#: Particulate Afsnit A entries whose name gives neither of the above.
_PARTICULATE_NAMES = frozenset(
    {
        "carbon black",
        "emissioner fra dieseludstødning",
        "calciumhydroxid",
        "calciumoxid",
        "lithiumhydrid",
        "silicium",
        "boroxid",
        "vismuttellurid",
        "vismuttellurid, tilsat selen",
    }
)
#: Metal compounds that are volatile — airborne as vapour, not particles.
_VOLATILE_METAL = re.compile(
    r"carbonyl|alkyl|tetraethyl|tetramethyl|cyclopentadienyl", re.IGNORECASE
)
#: Matches of the rules above that still need a person's judgement.
_ASSESS_PREFIXES = ("tinforbindelser, organiske",)  # organotins: often volatile

#: Reasons recorded in :attr:`ExposureLimitList.excluded`.
REASON_FIBRE = "fibre: limit in fibres per cm³"
REASON_MERCURY = "mercury: also present as vapour"
REASON_VOLATILE = "volatile metal compound (vapour)"
REASON_UNASSESSED = "not classified as particulate: needs assessment"
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
    return " ".join(text.replace("\xad", "").split())  # drop soft hyphens


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
    text = _split_footnote(text or "")[0].strip()
    return float(text.replace(",", ".")) if _NUMBER.match(text) else None


def _split_footnote(text: str) -> tuple[str, str | None]:
    """Split a trailing footnote marker off: ``"Glasuld 1)"`` → ``("Glasuld", "1")``."""
    m = _TRAILING_FOOTNOTE.search(text)
    if not m:
        return text, None
    marker = (m.group(1) or m.group(2)).translate(_PLAIN_DIGITS)
    return text[: m.start()].rstrip(), marker


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


def _header_columns(
    rows: list[tuple[list[str], list[tuple[int, int]]]], h: int
) -> dict[str, int]:
    """Column of each field, read from header row ``h`` (and the group row above).

    Two layouts occur. Afsnit B has one header row whose "8-timers grænseværdi"
    spans the value and its unit. Afsnit A has group headers ("8-timers
    grænseværdi", "Korttidsgrænseværdi") over sub-headers "ppm" and "mg/m³".
    """
    cells, spans = rows[h]
    cols: dict[str, int] = {}
    for start, span in spans:
        head = cells[start].strip().lower()
        for key, word in (
            ("cas", "cas"),
            ("name", "stof"),
            ("year", "årstal"),
            ("remarks", "anmærkning"),
        ):
            if head.startswith(word):
                cols.setdefault(key, start)
    if any(c.strip().lower() == "ppm" for c in cells) and h > 0:
        group_cells, group_spans = rows[h - 1]
        groups = [
            (s, n, group_cells[s].strip().lower())
            for s, n in group_spans
            if group_cells[s].strip()
        ]
        for start, _span in spans:
            head = cells[start].strip().lower()
            if head not in ("ppm", "mg/m³", "mg/m3"):
                continue
            group = next((g for s, n, g in groups if s <= start < s + n), "")
            period = (
                "twa"
                if group.startswith("8-timers")
                else "stel" if group.startswith("korttid") else None
            )
            if period:
                cols.setdefault(
                    f"{period}_{'ppm' if head == 'ppm' else 'value'}", start
                )
        for s, _n, g in groups:
            if g.startswith("anmærkning"):
                cols.setdefault("remarks", s)
    else:
        for start, span in spans:
            head = cells[start].strip().lower()
            if head.startswith("8-timers"):
                cols.setdefault("twa_value", start)
                if span >= 2:
                    cols.setdefault("twa_unit", start + 1)
            elif head.startswith("korttid"):
                cols.setdefault("stel_value", start)
    return cols


def _read_table(table: ET.Element, section: str) -> tuple[list[dict], dict[str, str]]:
    """Read a Bilag 2 table into raw row records plus its footnotes.

    Handles the table's quirks: rows that only add a CAS number to the entry
    above, later-dated values ("Fra den 21. december 2029"), value-less heading
    rows whose sub-rows are named only by a fraction ("Manganrøg, beregnet som
    Mn" → "Inhalerbar", "Respirabel"), and the footnote rows at the foot.
    """
    rows = [_row_cells(tr) for tr in _descendants(table, "Tr")]
    h = next(
        (
            i
            for i, (cells, _spans) in enumerate(rows)
            if any(c.strip().lower().startswith("cas") for c in cells)
        ),
        None,
    )
    if h is None:
        raise RetsinformationError(f"The Afsnit {section} table has no header row.")
    cols = _header_columns(rows, h)
    missing = {"name", "twa_value"} - cols.keys()
    if missing:
        raise RetsinformationError(
            f"The Afsnit {section} header lacks {sorted(missing)} — the parser "
            "needs updating."
        )

    def cell(cells: list[str], key: str) -> str:
        i = cols.get(key)
        return cells[i].strip() if i is not None and i < len(cells) else ""

    records: list[dict] = []
    footnotes: dict[str, str] = {}
    heading: dict | None = None
    for cells, _spans in rows[h + 1 :]:
        if not any(c.strip() for c in cells):
            continue
        marker = _FOOTNOTE_ROW.match(cells[0].strip())
        if marker:
            text = next((c.strip() for c in cells[1:] if c.strip()), "")
            footnotes[marker.group(1).translate(_PLAIN_DIGITS)] = text
            continue
        rec = {
            key: cell(cells, key)
            for key in ("twa_value", "twa_unit", "stel_value", "twa_ppm", "stel_ppm")
        }
        cas, name = cell(cells, "cas"), cell(cells, "name")
        if not name:
            # A continuation row carrying another CAS number of the entry above.
            if cas and records:
                records[-1]["cas"].append(cas)
            continue
        if name.lower().startswith("fra d") and not cas and records:
            # A later-dated value for the entry above ("Fra den 21. december …").
            later = " ".join(
                v for v in (rec["twa_value"], rec["twa_unit"]) if v and v != "-"
            )
            records[-1]["notes"].append(f"{name}: {later}".strip(": "))
            continue
        rec.update(
            name=name,
            cas=[cas] if cas else [],
            year=cell(cells, "year"),
            remarks=cell(cells, "remarks"),
            notes=[],
            section=section,
        )
        valued = any(
            _number(rec[k]) is not None
            for k in ("twa_value", "stel_value", "twa_ppm", "stel_ppm")
        )
        if _BARE_FRACTION.fullmatch(name.lower()) and heading is not None:
            # A fraction sub-row of a value-less heading row above.
            rec["name"] = f"{heading['name']}, {name.lower()}"
            rec["cas"] = rec["cas"] or list(heading["cas"])
            rec["year"] = rec["year"] or heading["year"]
            rec["remarks"] = rec["remarks"] or heading["remarks"]
        elif not valued:
            heading = rec
        else:
            heading = None
        records.append(rec)
    return records, footnotes


def _build_limit(
    rec: dict, footnotes: dict[str, str], doubling_rule: bool
) -> tuple[ExposureLimit, bool] | None:
    """Turn a raw row into an :class:`ExposureLimit` (+ whether it has a ppm limit).

    Returns ``None`` for a row without any mass limit (a heading or a
    cross-reference such as "…, se bitumenrøg").
    """
    name, marker = _split_footnote(rec["name"])
    twa_text, twa_marker = _split_footnote(rec["twa_value"])
    stel_text, stel_marker = _split_footnote(rec["stel_value"])
    notes = []
    for mark in (marker, twa_marker, stel_marker):
        note = footnotes.get(mark or "")
        if note and note not in notes:
            notes.append(note)
    notes.extend(rec["notes"])
    unit = rec["twa_unit"]
    if not unit:
        # Afsnit A puts a fibre limit's unit in the mg/m³ column ("1 fiber/cm³");
        # otherwise the order's legend applies: particulate limits are mg/m³.
        m = re.match(r"^([\d.,]+)\s+(\S.*)$", twa_text)
        if m:
            twa_text, unit = m.group(1), m.group(2)
    unit = unit or "mg/m³"
    twa = _number(twa_text)
    stel_rule = ""
    if stel_text.lower().startswith("jf"):
        stel_rule = stel_text
        stel = 2.0 * twa if (doubling_rule and twa is not None) else None
    else:
        stel = _number(stel_text)
    if twa is None and stel is None:
        return None
    has_ppm = (
        _number(rec["twa_ppm"]) is not None or _number(rec["stel_ppm"]) is not None
    )
    limit = ExposureLimit(
        name=name,
        cas=tuple(rec["cas"]),
        twa=twa,
        stel=stel,
        unit=unit,
        stel_rule=stel_rule,
        remarks=rec["remarks"],
        year=int(rec["year"]) if rec["year"].isdigit() else None,
        notes=tuple(notes),
        section=rec["section"],
    )
    return limit, has_ppm


def _exclusion(limit: ExposureLimit, has_ppm: bool) -> str | None:
    """Why an entry is left out (``""``: silently, as a gas), or ``None`` to keep it."""
    if "fiber" in limit.unit.lower() or "fibre" in limit.unit.lower():
        return REASON_FIBRE
    if limit.section != "A":
        return None  # Afsnit B is dust throughout
    if has_ppm:
        return ""  # a gas or vapour
    name = limit.name.casefold()
    if "kviksølv" in name:
        return REASON_MERCURY
    if _VOLATILE_METAL.search(name):
        return REASON_VOLATILE
    if name.startswith(_ASSESS_PREFIXES):
        return REASON_UNASSESSED
    if (
        _PARTICLE_WORDS.search(limit.name)
        or _METAL_BASIS.search(limit.name)
        or name in _PARTICULATE_NAMES
    ):
        return None
    return REASON_UNASSESSED


def _same_entry(a: ExposureLimit, b: ExposureLimit) -> bool:
    """Whether two entries (one per section) are the same limit.

    Afsnit A repeats many Afsnit B dusts, sometimes spelled differently
    ("Christobalit" / "Cristobalit"); the same CAS numbers, fraction and values
    then identify them.
    """
    if a.name.casefold() == b.name.casefold():
        return True
    return (
        bool(a.cas)
        and set(a.cas) == set(b.cas)
        and a.fraction == b.fraction
        and (a.twa, a.stel) == (b.twa, b.stel)
    )


def _parse_sections(
    root: ET.Element, sections: tuple[str, ...], doubling_rule: bool
) -> tuple[list[ExposureLimit], list[tuple[str, str]]]:
    """Parse the particulate limits of ``sections``, Afsnit B first."""
    limits: list[ExposureLimit] = []
    excluded: list[tuple[str, str]] = []
    for section in sorted(sections, key=lambda s: s != "B"):
        records, footnotes = _read_table(_find_section_table(root, section), section)
        for rec in records:
            built = _build_limit(rec, footnotes, doubling_rule)
            if built is None:
                continue
            limit, has_ppm = built
            if any(_same_entry(limit, kept) for kept in limits):
                continue  # Afsnit A repeating an Afsnit B dust
            reason = _exclusion(limit, has_ppm)
            if reason is None:
                limits.append(limit)
            elif reason and all(name != limit.name for name, _r in excluded):
                excluded.append((limit.name, reason))
    return sorted(limits, key=lambda lim: lim.name.casefold()), excluded


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
    data: bytes | str,
    eli: str | None = None,
    retrieved: str | None = None,
    sections: tuple[str, ...] = ("A", "B"),
) -> ExposureLimitList:
    """Parse the particulate limits out of a limit-value order's XML.

    Args:
        data: The order as LexDania XML (``<eli>/dan/xml``).
        eli: The order's ELI URL, recorded as the source; read from the
            document's number and year when omitted.
        retrieved: When the XML was downloaded (ISO timestamp); defaults to now.
        sections: Which Bilag 2 sections to parse (``"A"``, ``"B"``).

    Returns:
        The particulate limits (see the module docstring for which entries
        count), with the order's details and the entries left out.

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
    limits, excluded = _parse_sections(
        root, tuple(sections), bool(_DOUBLING_RULE.search(text))
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
        sections=tuple(sorted(sections)),
    )
    return ExposureLimitList(
        source=source, limits=tuple(limits), excluded=tuple(excluded)
    )


def fetch_exposure_limits(
    eli: str = BUNDLED_ELI, timeout: float = 60.0
) -> ExposureLimitList:
    """Download a limit-value order from Retsinformation and parse its particle limits.

    Reads the order's official XML embodiment (``<eli>/dan/xml``). Use
    :func:`find_current_order` first to learn which order is in force.

    Args:
        eli: The order (see :func:`eli_url` for accepted forms).
        timeout: Network timeout in seconds (the XML is a few MB).

    Raises:
        RetsinformationError: On a network error, or if the document is not the
            limit-value order.
    """
    url = eli_url(eli)
    return parse_exposure_limits_xml(_http_get(url + "/dan/xml", timeout), eli=url)
