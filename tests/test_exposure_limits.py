"""Tests for the Danish occupational-exposure-limit parser and bundled list.

``tests/data/exposure_limits/Sample_BEK_graensevaerdier.xml`` is the real BEK
nr 613 af 29/06/2026 (Retsinformation ``/dan/xml``) trimmed to its metadata,
paragraphs and the Bilag 2 Afsnit A and B tables, so everything here runs
offline. (It sits in a subfolder because ``tests/data`` itself must hold only
instrument files.)
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import pytest

from aerosoltools.exposure_limits import (
    ExposureLimit,
    ExposureLimitList,
    RetsinformationError,
    applicable_limit,
    basis_note,
    check_fraction,
    eli_url,
    load_exposure_limits,
    metric_size_cut,
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
    assert src.sections == ("A", "B", "C")

    assert len(parsed) == 95
    assert sum(lim.section == "B" for lim in parsed) == 24
    assert parsed.names() == sorted(parsed.names(), key=str.casefold)
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


def test_afsnit_a_particles(parsed):
    """Afsnit A contributes metals, fumes, carbon black, diesel exhaust, mists."""
    carbon = parsed["Carbon black"]
    assert (carbon.section, carbon.twa, carbon.remarks) == ("A", 3.5, "K")
    diesel = parsed["Emissioner fra dieseludstødning"]
    assert diesel.twa == 0.005 and diesel.basis == "elemental carbon"
    assert "elementært kulstof" in diesel.notes[0]
    tio2 = parsed["Titandioxid, beregnet som Ti"]
    assert (tio2.twa, tio2.stel, tio2.basis) == (6.0, 12.0, "Ti")
    borax = parsed["Natriumtetraborat, decahydrat"]
    assert (borax.section, borax.twa, borax.remarks) == ("A", 2.0, "H")
    # Value-less heading + fraction-only sub-rows become two named entries.
    fume = parsed["Manganrøg, beregnet som Mn, respirabel"]
    assert (fume.twa, fume.remarks, fume.fraction) == (0.05, "E", "respirable")
    assert parsed.get("Manganrøg, beregnet som Mn, inhalerbar").twa == 0.2
    assert parsed.get("Respirabel") is None and parsed.get("Manganrøg") is None


def test_short_term_exceptions(parsed, xml_bytes):
    """Listed short-term values are used as given; none is invented."""
    root = ET.fromstring(xml_bytes)
    assert parsed["Bly og dets uorganiske forbindelser, beregnet som Pb"].stel is None
    lime = parsed["Calciumhydroxid, respirabel fraktion"]
    assert (lime.twa, lime.stel, lime.stel_derived) == (1.0, 4.0, False)
    # A row with only a short-term limit parses (it is then excluded).
    records, notes = ri._read_table(ri._find_section_table(root, "A"), "A")
    rec = next(r for r in records if r["name"] == "Lithiumhydrid, inhalerbar")
    hydride, _has_ppm = ri._build_limit(rec, notes, True)
    assert (hydride.twa, hydride.stel) == (None, 0.02)
    assert all(lim.stel_minutes == 15 for lim in parsed)


def test_exclusions(parsed):
    """Fibres, mercury, volatile metals and non-aerosol substances are left out."""
    reasons = dict(parsed.excluded)
    assert reasons["Asbest"] == ri.REASON_FIBRE
    assert reasons["Keramiske fibre"] == ri.REASON_FIBRE
    assert sum(r == ri.REASON_FIBRE for r in reasons.values()) == 9
    assert reasons[
        "Kviksølv og uorganiske forbindelser inkl. dampe, beregnet som Hg"
    ] == (ri.REASON_MERCURY)
    assert reasons["Cobaltcarbonyl, beregnet som Co"] == ri.REASON_VOLATILE
    # The maintainer's assessment: substances the measured aerosol would not
    # be assumed to consist of — reactive chemicals, organics, soluble
    # compounds of metals that also have a dust entry, …
    for name in (
        "Natriumhydroxid",
        "Dibutylphthalat",
        "Terephthalsyre",
        "Lithiumhydrid",
        "Diquat, respirabel",
        "Jernsalte, opløselige, beregnet som Fe",
    ):
        assert reasons[name] == ri.REASON_NOT_AEROSOL, name
        assert parsed.get(name) is None
    assert ri.REASON_UNASSESSED not in reasons.values()
    # Soluble Ba/Tl compounds are the only limits for those metals: kept.
    assert parsed.get("Bariumforbindelser, opløselige, beregnet som Ba")
    assert parsed.get("Bitumenrøg, cyclohexanholdig fraktion af totalstøv")
    # A substance new in a later order is flagged, not silently dropped.
    new = ExposureLimit(name="Nyt stof", twa=1.0, section="A")
    assert ri._exclusion(new, has_ppm=False) == ri.REASON_UNASSESSED
    # Gases (ppm limits) are neither kept nor listed; Afsnit A repeats of
    # Afsnit B dusts are dropped silently (even when spelled differently).
    assert "Acetaldehyd" not in reasons and parsed.get("Acetaldehyd") is None
    assert "Christobalit, respirabel" not in reasons
    assert parsed.get("Christobalit, respirabel") is None
    assert parsed["Cristobalit, respirabel"].section == "B"
    assert all(lim.unit == "mg/m³" for lim in parsed)
    assert not any("fibre" in lim.name.lower() for lim in parsed)


def test_fraction_variants(parsed):
    """Fraction variants of one substance share a base name."""
    quartz = parsed["Kvarts, respirabel"]
    assert quartz.base_name == "Kvarts"
    assert {lim.name for lim in parsed.siblings(quartz)} == {
        "Kvarts, respirabel",
        "Kvarts, total",
    }
    nickel = parsed["Nikkelforbindelser, respirabel fraktion, beregnet som Ni"]
    assert nickel.base_name == "Nikkelforbindelser, beregnet som Ni"
    assert len(parsed.siblings(nickel)) == 2
    assert parsed.siblings(parsed["Carbon black"]) == [parsed["Carbon black"]]


def test_metric_size_cut():
    """PM names give their cut, capped by the instrument's size range."""
    assert metric_size_cut("PM2.5") == 2.5
    assert metric_size_cut("PM10", upper_um=0.42) == 0.42  # NanoScan "PM10"
    assert metric_size_cut("Total") == math.inf
    assert metric_size_cut("MASS", upper_um=10.0) == 10.0
    assert metric_size_cut("Mass concentration") is None
    assert metric_size_cut("Org") is None


@pytest.mark.parametrize(
    "name, metric, status",
    [
        ("Kvarts, respirabel", "PM4", "match"),
        ("Kvarts, respirabel", "PM1", "underestimates"),
        ("Kvarts, respirabel", "PM10", "conservative"),
        ("Kvarts, respirabel", "Total", "conservative"),
        ("Kvarts, total", "Total", "match"),
        ("Kvarts, total", "PM10", "match"),
        ("Kvarts, total", "PM1", "underestimates"),
        ("Emissioner fra dieseludstødning", "IR BCc", "match"),
        ("Emissioner fra dieseludstødning", "PM2.5", "conservative"),
        ("Kvarts, respirabel", "Org", "unknown"),
    ],
)
def test_check_fraction(parsed, name, metric, status):
    """A limit applies only to a metric covering its size fraction."""
    check = check_fraction(parsed[name], metric, metric_size_cut(metric))
    assert check.status == status
    assert check.applies == (status == "match")
    if status != "match":
        assert check.message


def test_applicable_limit_picks_the_matching_variant(parsed):
    """The fraction variant that fits the metric is chosen, if any."""
    quartz = parsed["Kvarts, respirabel"]
    candidates = [quartz, *parsed.siblings(quartz)]
    limit, check = applicable_limit(candidates, "Total", math.inf)
    assert limit.name == "Kvarts, total" and check.applies
    limit, check = applicable_limit(candidates, "PM4", 4.0)
    assert limit is quartz and check.applies
    limit, check = applicable_limit(candidates, "PM1", 1.0)
    assert limit is quartz and check.status == "underestimates"


def test_basis_note(parsed):
    """Limits 'beregnet som' an element carry a conservative-comparison caveat."""
    assert "Ti content" in basis_note(parsed["Titandioxid, beregnet som Ti"])
    assert basis_note(parsed["Kvarts, respirabel"]) == ""
    assert basis_note(parsed["Emissioner fra dieseludstødning"]) == ""


def test_short_term_reference_period_from_note():
    """A footnoted reference period overrides the default 15 minutes."""
    lim = ExposureLimit(
        name="X", twa=1.0, notes=("Med en referenceperiode på 1 minut.",)
    )
    assert lim.stel_minutes == 1


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
    # Early files listed excluded entries as bare (fibre) names.
    legacy = {**parsed.to_dict(), "excluded": ["Asbest"]}
    assert ExposureLimitList.from_dict(legacy).excluded == (("Asbest", "fibre"),)


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


@pytest.mark.parametrize(
    "metric, status",
    [("PM10", "match"), ("PM2.5", "underestimates"), ("Total", "conservative")],
)
def test_check_fraction_thoracic(metric, status):
    """A thoracic limit fits PM10 (none is in the current list; rule kept)."""
    mist = ExposureLimit(name="Tåge, thorakal fraktion", twa=0.05)
    assert check_fraction(mist, metric, metric_size_cut(metric)).status == status


def test_welding_limits(parsed):
    """Afsnit C's process-specific welding limits, '–' read as 'as above'."""
    welding = [lim for lim in parsed if lim.section == "C"]
    assert len(welding) == 5
    mig = parsed["Svejserøg, MIG/MAG, almindeligt konstruktionsstål, sædvanlig primer"]
    assert (mig.twa, mig.stel, mig.fraction) == (1.6, None, "total")
    assert any("row above" in note for note in mig.notes)
    assert any("erfaringsdatamateriale" in note for note in mig.notes)
    tig = parsed["Svejserøg, TIG, rustfast og syrebestandigt stål"]
    assert tig.twa == 1.1 and not any("row above" in n for n in tig.notes)
    assert (
        parsed["Svejserøg, elektrodesvejsning, rustfast og syrebestandigt stål"].twa
        == 0.5
    )


def test_applicable_limit_can_accept_a_conservative_variant(parsed):
    """With no matching variant, a conservative one is used only if allowed."""
    silica = parsed["Krystallinsk siliciumdioxid, respirabelt støv"]
    limit, check = applicable_limit([silica], "PM10", 10.0)
    assert limit is silica and check.status == "conservative" and not check.applies
    limit, check = applicable_limit([silica], "PM10", 10.0, allow_conservative=True)
    assert limit is silica and check.conservative
    # A match still wins over a conservative variant.
    quartz = parsed["Kvarts, respirabel"]
    limit, check = applicable_limit(
        [quartz, *parsed.siblings(quartz)], "Total", math.inf, allow_conservative=True
    )
    assert limit.name == "Kvarts, total" and check.applies
