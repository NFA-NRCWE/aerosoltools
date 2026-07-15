"""Unit-registry tests: canonicalization + vocabulary-gated header parsing.

Foundation for centralized unit handling across instruments. These are pure
(no I/O, no Qt): they pin that recognised units from any of the header
conventions found in real exports collapse to one canonical glyph, and that
non-unit bracket/dot content is never mistaken for a unit.
"""

from __future__ import annotations

import pytest

from aerosoltools._core.metrics import (
    canonical_unit,
    classify_unit,
    convert_value,
    parse_header_unit,
)


@pytest.mark.parametrize(
    "raw, canonical",
    [
        # temperature spellings collapse to °C
        ("C", "°C"),
        ("degC", "°C"),
        ("°C", "°C"),
        ("celsius", "°C"),
        # humidity
        ("%", "%"),
        ("%RH", "%"),
        # pressure -> hPa (== mbar)
        ("hPa", "hPa"),
        ("mBar", "hPa"),
        ("Pa", "hPa"),
        ("kPa", "hPa"),
        ("atm", "hPa"),
        # flow -> l/min
        ("l/min", "l/min"),
        ("L/min", "l/min"),
        ("mL/min", "l/min"),
        # size -> nm
        ("nm", "nm"),
        ("um", "nm"),
        ("µm", "nm"),
        # number / mass / gas (pre-existing, still canonical)
        ("#/cm3", "cm⁻³"),
        ("ug/m3", "µg/m³"),
        ("ng/m³", "µg/m³"),
        ("ppb", "ppm"),
    ],
)
def test_canonical_unit_collapses_spellings(raw, canonical):
    assert canonical_unit(raw) == canonical


def test_pressure_scales_are_consistent():
    """Same-dimension pressures convert by scale (canonical hPa)."""
    # 1 kPa == 10 hPa == 1000 Pa
    assert convert_value(1.0, "kPa", "hPa") == pytest.approx(10.0)
    assert convert_value(1000.0, "Pa", "hPa") == pytest.approx(10.0)
    # different dimensions never cross-convert
    assert convert_value(5.0, "°C", "hPa") == 5.0


def test_kelvin_is_not_merged_with_celsius():
    """Kelvin needs an offset, not a scale, so it must not share °C's dimension."""
    dim_k, _canon_k, _ = classify_unit("K")
    dim_c, _canon_c, _ = classify_unit("°C")
    assert dim_k != dim_c


@pytest.mark.parametrize(
    "header, name, unit",
    [
        # paren, no space
        ("Temp(C)", "Temp", "°C"),
        ("PM1(ug/m3)", "PM1", "µg/m³"),
        ("Pressure(atm)", "Pressure", "hPa"),
        ("Total Conc. (#/cm3)", "Total Conc.", "cm⁻³"),
        # paren, space
        ("Sample temp (C)", "Sample temp", "°C"),
        ("Sample RH (%)", "Sample RH", "%"),
        ("Flow total (mL/min)", "Flow total", "l/min"),
        ("Internal pressure (Pa)", "Internal pressure", "hPa"),
        ("GPS speed (km/h)", "GPS speed", "km/h"),
        # square brackets
        ("Pressure [mBar]", "Pressure", "hPa"),
        ("Sample Temp [°C]", "Sample Temp", "°C"),
        ("Elapsed [s]", "Elapsed", "s"),
        # dot suffix
        ("Temperature.C", "Temperature", "°C"),
    ],
)
def test_parse_header_unit_extracts_recognised_units(header, name, unit):
    assert parse_header_unit(header) == (name, unit)


def test_column_units_meta_drives_unit_of_without_breaking_scalar_unit():
    """A ``column_units`` map resolves per-column units; ``.unit`` stays scalar."""
    import aerosoltools as at

    # extra_data=False leaves the map empty — the plumbing is inert until set.
    ops = at.load_ops_file(_ops_sample(), extra_data=False)
    assert ops.column_units == {}
    assert ops.unit == "cm⁻³"

    ops._meta["column_units"] = {"Temperature (C)": "°C", "Total_conc": "cm⁻³"}
    assert ops.unit_of("Temperature (C)") == "°C"  # resolved from the map
    assert ops.unit_of("Total_conc") == "cm⁻³"
    assert ops.unit == "cm⁻³"  # scalar .unit unchanged (back-compat)
    # A copy carries the map (it lives in _meta, which is deep-copied).
    assert ops.copy_self().column_units == ops.column_units


def _ops_sample():
    import os

    return os.path.join(os.path.dirname(__file__), "data", "Sample_OPS2.txt")


@pytest.mark.parametrize(
    "header",
    [
        "[1] Conc",  # a leading index, not a unit
        "GPS lat (ddmm.mmmmm)",  # a coordinate format
        "Concentration (dW)",  # a distribution weighting
        "UB with RI",  # a note
        "RH.p",  # dot form but 'p' is not a recognised unit
        "PM2.5",  # numeric dot suffix
        "Bin0.35",  # numeric dot suffix
        "Accel X",
        "Status",
        "GSD",
        "Firmware Version",
    ],
)
def test_parse_header_unit_rejects_non_units(header):
    name, unit = parse_header_unit(header)
    assert unit is None and name == header
