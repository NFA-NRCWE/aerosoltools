"""Tests for the Danish occupational-exposure-limit parser and bundled list.

``tests/data/exposure_limits/Sample_BEK_graensevaerdier.xml`` is the real BEK
nr 613 af 29/06/2026 (Retsinformation ``/xml``) trimmed to its metadata,
paragraphs and Bilag 2 Afsnit B table, so everything here runs offline. (It sits
in a subfolder because ``tests/data`` itself must hold only instrument files.)
"""

from __future__ import annotations

import os

import pytest

from aerosoltools.exposure_limits import (
    ExposureLimitList,
    RetsinformationError,
    eli_url,
    load_exposure_limits,
    parse_exposure_limits_xml,
)
from aerosoltools.exposure_limits import retsinformation as ri

_FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "exposure_limits",
    "Sample_BEK_graensevaerdier.xml",
)


@pytest.fixture(scope="module")
def xml_bytes() -> bytes:
    with open(_FIXTURE, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def parsed(xml_bytes) -> ExposureLimitList:
    return parse_exposure_limits_xml(xml_bytes, eli="2026/613", retrieved="test")


def test_bundled_snapshot_matches_parser(parsed):
    """The shipped JSON is exactly what the parser makes of BEK 613/2026."""
    bundled = load_exposure_limits()
    assert bundled.limits == parsed.limits
    assert bundled.excluded == parsed.excluded
    for field in ("title", "number", "date", "eli", "in_force_from", "sections"):
        assert getattr(bundled.source, field) == getattr(parsed.source, field)
    assert bundled.source.label == "BEK nr 613 af 29/06/2026"
    assert bundled.source.short_label == "BEK 613/2026"


def test_section_b_rows(parsed):
    """Afsnit B dust limits parse with CAS, derived STEL, remarks and fraction."""
    src = parsed.source
    assert (src.number, src.date, src.status) == (613, "2026-06-29", "Valid")
    assert src.in_force_from == "2026-07-01"
    assert src.eli == "https://www.retsinformation.dk/eli/lta/2026/613"

    assert len(parsed) == 24
    quartz = parsed["Kvarts, respirabel"]
    assert quartz.cas == ("14808-60-7",)
    assert (quartz.twa, quartz.stel, quartz.unit) == (0.1, 0.2, "mg/m³")
    assert quartz.stel_rule == "Jf. § 3, stk. 2" and quartz.stel_derived
    assert quartz.remarks == "K" and quartz.remark_codes == ("K",)
    assert quartz.fraction == "respirable" and quartz.year == 2020

    wood = parsed["Træstøv, inhalerbart"]
    assert wood.remark_codes == ("E", "K") and wood.fraction == "inhalable"
    # Subscripts render; a limit naming no fraction applies to total dust.
    silica = parsed["Kiselsyre, SiO₂, amorf"]
    assert silica.twa == 5.0 and silica.fraction == "total"
    # A footnote marker is split off the name and its text attached as a note.
    tobacco = next(lim for lim in parsed if lim.name.startswith("Tobaksstøv"))
    assert tobacco.name.endswith("håndteres)")
    assert "organiske fraktion" in tobacco.notes[0]
    # "(råbomuld)" is part of the name, not a footnote.
    assert parsed.get("Bomuldstøv (råbomuld)") is not None


def test_fibres_are_excluded(parsed):
    """Fibre limits (fibres/cm³) are dropped and listed as excluded."""
    assert all(lim.unit == "mg/m³" for lim in parsed)
    assert "Asbest" in parsed.excluded and "Keramiske fibre" in parsed.excluded
    assert len(parsed.excluded) == 9
    assert not any("fibre" in lim.name.lower() for lim in parsed)


def test_unit_conversion(parsed):
    """Limits convert across mass scales but never into another dimension."""
    quartz = parsed["Kvarts, respirabel"]
    assert quartz.twa_in("µg/m³") == pytest.approx(100.0)
    assert quartz.stel_in("ug/m3") == pytest.approx(200.0)
    assert quartz.twa_in("mg/m³") == pytest.approx(0.1)
    assert quartz.is_comparable_to("µg/m³")
    assert not quartz.is_comparable_to("cm⁻³") and not quartz.is_comparable_to(None)
    with pytest.raises(ValueError):
        quartz.twa_in("cm⁻³")


def test_lookup_suggests_close_names(parsed):
    """A near-miss name raises KeyError naming the likely substance."""
    assert parsed.get("kvarts, RESPIRABEL") is parsed["Kvarts, respirabel"]
    with pytest.raises(KeyError, match="Kvarts, respirabel"):
        parsed["Kvarts respirabel"]


def test_json_roundtrip(parsed, tmp_path):
    """A saved list reloads identically, source included."""
    path = tmp_path / "limits.json"
    parsed.to_json(path)
    again = load_exposure_limits(path)
    assert again == parsed
    assert len(again.to_dataframe()) == len(parsed)


def test_doubling_rule_is_read_not_assumed(xml_bytes):
    """Without § 3, stk. 2's doubling sentence no short-term value is invented."""
    text = xml_bytes.decode("utf-8").replace("to gange 8-timers", "tre gange 8-timers")
    limits = parse_exposure_limits_xml(text, eli="2026/613")
    quartz = limits["Kvarts, respirabel"]
    assert quartz.stel is None and quartz.stel_rule == "Jf. § 3, stk. 2"


def test_rejects_other_documents(xml_bytes):
    """Only the limit-value order is accepted."""
    other = xml_bytes.decode("utf-8").replace(
        "Bekendtgørelse om grænseværdier", "Bekendtgørelse om ændring af"
    )
    with pytest.raises(RetsinformationError, match="not the limit-value order"):
        parse_exposure_limits_xml(other)
    with pytest.raises(RetsinformationError):
        parse_exposure_limits_xml(b"<html>not xml")


@pytest.mark.parametrize(
    "text",
    [
        "2026/613",
        "lta/2026/613",
        "https://www.retsinformation.dk/eli/lta/2026/613",
        "https://retsinformation.dk/eli/lta/2026/613/xml",
        "https://www.retsinformation.dk/eli/lta/2026/613.rdfa",
        "BEK nr 613 af 29/06/2026",
    ],
)
def test_eli_url_forms(text):
    assert eli_url(text) == "https://www.retsinformation.dk/eli/lta/2026/613"


def test_eli_url_rejects_garbage():
    with pytest.raises(ValueError):
        eli_url("the newest one")


def test_parse_order_status():
    """The ELI metadata record yields status, dates and relations."""
    about = "https://retsinformation.dk/eli/lta/2026/583"
    payload = [
        {"about": about, "property": "eli:title", "content": ri.ORDER_TITLE + " …"},
        {
            "about": about,
            "property": "eli:in_force",
            "resource": "eli:InForce-notInForce",
        },
        {
            "about": about,
            "property": "eli:date_document",
            "content": "25-06-2026 00:00:00",
        },
        {
            "about": about,
            "property": "eli:date_no_longer_in_force",
            "content": "29-06-2026 16:42:28",
        },
        {"about": about, "property": "eli:changed_by", "content": about[:-3] + "613"},
    ]
    status = ri.parse_order_status(payload, about)
    assert not status.in_force and status.is_limit_order
    assert status.label == "BEK nr 583 af 25/06/2026"
    assert status.no_longer_in_force == "2026-06-29T16:42:28"
    assert status.changed_by == ("https://www.retsinformation.dk/eli/lta/2026/613",)
    with pytest.raises(RetsinformationError):
        ri.parse_order_status({"not": "a list"}, about)


def _status(n, in_force, changed_by=(), title=ri.ORDER_TITLE, when="2026-01-01"):
    return ri.OrderStatus(
        eli=eli_url(f"2026/{n}"),
        title=title,
        in_force=in_force,
        date=when,
        changed_by=tuple(eli_url(f"2026/{c}") for c in changed_by),
    )


def test_find_current_order_follows_replacements(monkeypatch):
    """Replacements are followed, amending orders skipped then reported."""
    amending = "Bekendtgørelse om ændring af bekendtgørelse om grænseværdier"
    statuses = {
        1: _status(1, False, changed_by=(2, 3)),
        2: _status(2, True, title=amending),  # amends 1, does not replace it
        3: _status(3, True, changed_by=(4,)),
        4: _status(4, True, title=amending),  # amends the current order
    }
    monkeypatch.setattr(
        ri,
        "fetch_order_status",
        lambda eli, timeout=30.0: statuses[int(eli.rsplit("/", 1)[1])],
    )
    found = ri.find_current_order("2026/1")
    assert found.superseded
    assert [s.number for s in found.chain] == [1, 3]
    assert found.current.number == 3
    assert [a.number for a in found.amendments] == [4]


def test_find_current_order_without_replacement(monkeypatch):
    """A replaced order with no findable successor raises a clear error."""
    monkeypatch.setattr(
        ri, "fetch_order_status", lambda eli, timeout=30.0: _status(1, False)
    )
    with pytest.raises(RetsinformationError, match="no longer in force"):
        ri.find_current_order("2026/1")


def test_cli_show_prints_bundled_list(capsys):
    """``python -m aerosoltools.exposure_limits show`` prints the bundled list."""
    from aerosoltools.exposure_limits.__main__ import main

    assert main(["show"]) == 0
    out = capsys.readouterr().out
    assert "BEK nr 613 af 29/06/2026" in out and "Kvarts, respirabel" in out
