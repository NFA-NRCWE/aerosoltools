"""Tests for ``delete_activity`` on the data classes (GitHub #35).

Deleting an activity used to be possible only through a GUI helper that
edited the private bookkeeping; the classes now have a public method, and the
GUI helper delegates to it.
"""

from __future__ import annotations

import os

import pytest

import aerosoltools as at

_DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def cpc():
    data = at.load_cpc_file(os.path.join(_DATA, "Sample_CPC_AIM.txt"))
    data.mark_activities(
        {"Task": [(data.time[10], data.time[20]), (data.time[40], data.time[50])]}
    )
    data.peak_finder()
    return data


def test_delete_activity_removes_mask_and_periods(cpc):
    cpc.delete_activity("Task")

    assert "Task" not in cpc.activities
    assert "Task" not in cpc.activity_periods
    assert "Task" not in cpc.data.columns
    # Everything else is left alone.
    assert cpc.activities == ["All data", "Peak"]
    assert "Peak" in cpc.data.columns


def test_delete_activity_then_mark_again(cpc):
    """A deleted name is free to be used again."""
    cpc.delete_activity("Peak")
    cpc.peak_finder(ratio=3.0)
    assert cpc.activities.count("Peak") == 1
    assert list(cpc.data.columns).count("Peak") == 1


def test_delete_activity_refuses_all_data_and_unknown_names(cpc):
    with pytest.raises(ValueError, match="All data"):
        cpc.delete_activity("All data")
    with pytest.raises(ValueError, match="No activity named"):
        cpc.delete_activity("Nope")
    assert "All data" in cpc.data.columns


def test_delete_activity_on_a_size_resolved_dataset():
    smps = at.load_smps_file(os.path.join(_DATA, "Sample_SMPS.txt"))
    smps.mark_activities({"Emission": (smps.time[3], smps.time[9])})
    smps.delete_activity("Emission")
    assert "Emission" not in smps.activities
    assert "Emission" not in smps.data.columns


def test_gui_helper_is_lenient_and_delegates(cpc):
    """The GUI helper stays a no-op for "All data" and names a dataset lacks."""
    from aerosoltools.gui.logic import helpers

    helpers.delete_activity(cpc, "All data")
    helpers.delete_activity(cpc, "Only on another dataset")
    assert "All data" in cpc.activities

    helpers.delete_activity(cpc, "Task")
    assert "Task" not in cpc.activities
    assert "Task" not in cpc.data.columns
