"""Axis labels of single-channel core plots (GitHub #39).

``plot_total_conc(parameter=...)`` labelled every channel with the *primary*
series' dtype and unit: a PM2.5 mass series read "dN, cm⁻³", an aethalometer's
timezone column "dM, ng/m³", and a Partector's LDSA "Unknown dtype, …". A
channel is now labelled by its own name and unit; the total concentration of a
particle instrument keeps its dtype label.
"""

from __future__ import annotations

import contextlib
import io
import os

import matplotlib
import pytest

import aerosoltools as at
from aerosoltools._core._labels import channel_label, channel_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _load(name, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return at.load_file(os.path.join(_DATA, name), **kwargs)


def _ylabel(obj, parameter) -> str:
    fig, ax = obj.plot_total_conc(parameter=parameter)
    label = ax.get_ylabel()
    plt.close(fig)
    return label


@pytest.mark.parametrize(
    "name, parameter, expected",
    [
        # A particle instrument's total keeps its dtype label (unchanged).
        ("Sample_CPC_AIM.txt", 0, "dN, cm⁻³"),
        ("Sample_OPS.csv", "Total_conc", "dN, cm⁻³"),
        # Other channels: own name, own unit (or none on record).
        ("Sample_CPC_AIM.txt", "Sample Length", "Sample Length, s"),
        ("Sample_CPC_AIM.txt", "Comments", "Comments"),
        ("Sample_OPS.csv", "337.0", "dN (337.0 nm), cm⁻³"),
        ("Sample_DustTrak.csv", "PM2.5", "PM2.5, µg/m³"),
        ("Sample_DustTrak.csv", "Alarms", "Alarms"),
        ("Sample_Discmini.txt", "LDSA", "LDSA, nm²/cm³"),
        ("Sample_Discmini.txt", "Filter", "Filter"),
        ("Sample_ACSM.csv", "SO4", "SO4, µg/m³"),
        # Main series of instruments without a total concentration.
        ("Sample_Partector.txt", "Flow", "Flow, l/min"),
        ("Sample_Aethalometer.csv", "IR BCc", "IR BCc, ng/m³"),
        ("Sample_Aethalometer.csv", "Timezone offset (mins)", "Timezone offset (mins)"),
        ("Sample_Fourtec.xlsx", "Temperature", "Temperature, °C"),
        # A gas monitor is named by its gas.
        ("Ranger_2605-1000088-A_20260617-0712.csv", 0, "Cl₂, ppm"),
        ("Sample_Tiger.csv", 0, "TVOC, ppb"),
    ],
)
def test_plot_total_conc_labels_the_plotted_channel(name, parameter, expected):
    assert _ylabel(_load(name), parameter) == expected


def test_partector_ldsa_is_not_unknown_dtype():
    part = _load("Sample_Partector.txt")
    label = _ylabel(part, "LDSA")
    assert label.startswith("LDSA, ")
    assert "Unknown" not in label


def test_derived_size_fraction_gets_its_own_unit():
    """PM2.5 from PM_calc is a mass series, not "dN, cm⁻³"."""
    ops = _load("Sample_OPS.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        ops.PM_calc(PM=2.5)
        ops.PM_calc(dtype="dN", PM=1, lower_lim=0.5)
    assert _ylabel(ops, "PM2.5") == "PM2.5, µg/m³"
    assert channel_unit(ops, "PN0.5-1") == "cm⁻³"
    assert channel_label(ops, "MASS") == "MASS, µg/m³"


def test_channel_unit_defaults_to_the_primary():
    part = _load("Sample_Partector.txt")
    assert channel_unit(part) == channel_unit(part, "LDSA")
    assert channel_unit(part, "no such column") is None


def test_wind_rose_colorbar_is_labelled_by_the_parameter():
    """The colour bar indexed dtype/unit dicts, raising for a dtype of None."""
    import numpy as np
    import pandas as pd

    from aerosoltools.intercomparison import wind_rose

    part = _load("Sample_Partector.txt")
    t = part.time
    rng = np.random.default_rng(0)
    weather = at.Environmental1D(
        pd.DataFrame(
            {
                "Temperature": 20.0,
                "W_direction": rng.uniform(0, 360, len(t)),
                "W_speed": rng.uniform(0.5, 6, len(t)),
            },
            index=t,
        )
    )
    weather._meta["unit"] = {"Temperature": "°C", "W_direction": "°", "W_speed": "m/s"}
    fig, ax = wind_rose(weather, part, parameter="LDSA", min_observations=1)
    labels = [a.get_ylabel() for a in fig.axes]
    plt.close(fig)
    assert any(lbl.startswith("LDSA, ") for lbl in labels), labels
