"""Tests for the always-a-list loading entry point (GitHub #34).

``load_file`` inherits its return type from the per-instrument loader, so a
multi-component Ranger export yields a list while a single-component one yields
a bare object -- and ``detect_instrument`` returns ``"Ranger"`` for both, so a
generic caller cannot branch in advance. ``load_all`` gives that caller one
predictable shape.
"""

import os
from pathlib import Path

import pytest

import aerosoltools as at

DATA = Path(os.path.dirname(__file__)) / "data"
MULTI = DATA / "Ranger_2605-1000088-A_20260612-0603.csv"
SINGLE = DATA / "Ranger_2605-1000088-A_20260617-0712.csv"


def test_the_two_ranger_files_are_indistinguishable_before_loading():
    """The premise of the issue: detection cannot tell the two cases apart."""
    assert at.detect_instrument(MULTI) == at.detect_instrument(SINGLE) == "Ranger"


def test_load_file_return_type_still_varies():
    """``load_file``'s documented behaviour is unchanged (additive fix)."""
    assert isinstance(at.load_file(MULTI), list)
    assert not isinstance(at.load_file(SINGLE), list)


@pytest.mark.parametrize("path,expected", [(MULTI, 3), (SINGLE, 1)])
def test_load_all_always_returns_a_list(path, expected):
    objs = at.load_all(path)
    assert isinstance(objs, list)
    assert len(objs) == expected
    for obj in objs:
        # The attribute that used to raise AttributeError on the list.
        assert obj.time is not None
        assert obj.metadata["instrument"] == "Ranger"


def test_load_all_forwards_instrument_and_kwargs():
    objs = at.load_all(MULTI, instrument="Ranger", extra_data=True)
    assert len(objs) == 3
    assert all(not o.extra_data.empty for o in objs)


def test_load_all_exposes_each_component():
    assert {o.metadata["measurement"] for o in at.load_all(MULTI)} == {
        "PM",
        "Cl₂",
        "NO₂",
    }


def test_generic_folder_walk_loads_every_sample_file():
    """The batch loader the issue was blocked on: no file needs special-casing."""
    loaded = 0
    for path in sorted(DATA.iterdir()):
        if path.is_dir():
            continue
        for obj in at.load_all(path):
            assert obj.time is not None
            loaded += 1
    assert loaded >= 34


def test_load_file_accepts_a_path_object_for_every_instrument():
    """`load_file` is typed `str | Path`; dispatch must honour that.

    ``load_fourtec_file`` calls ``path.lower()``, so a Path raised
    AttributeError for that one instrument.
    """
    obj = at.load_file(DATA / "Sample_Fourtec.xlsx")
    assert obj.time is not None
