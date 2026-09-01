"""Regression tests for the shared time-alignment helpers.

Covers the two bugs that made ``combine_size_ranges`` unusable without an
explicit ``rebin_freq``: a nanosecond-resolution assumption in ``_infer_freq``
(GitHub #31) and the uppercase pandas offset aliases removed in pandas 3
(GitHub #32).
"""

import os

import numpy as np
import pandas as pd
import pytest

import aerosoltools as at
from aerosoltools.intercomparison._alignment import _coarser, _infer_freq

DATA = os.path.join(os.path.dirname(__file__), "data")


def _irregular(index):
    """Drop one sample so ``pd.infer_freq`` fails and the fallback path runs."""
    return index.delete(5)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_infer_freq_is_resolution_agnostic(unit):
    """A 60 s cadence must read as one minute at every index resolution.

    Before the fix the spacing was computed as ``.view("i8") / 1e9``, so a
    ``datetime64[us]`` index (which several loaders produce) reported a cadence
    1000x too small -- ``"1s"`` instead of one minute.
    """
    ns = pd.date_range("2026-01-01", periods=50, freq="60s")
    idx = pd.DatetimeIndex(pd.array(ns.to_numpy().astype(f"datetime64[{unit}]")))
    assert idx.unit == unit

    rule = _infer_freq(_irregular(idx))
    assert pd.Timedelta(rule) == pd.Timedelta("60s")


def test_infer_freq_emits_current_pandas_aliases():
    """The fallback must not emit ``S``/``T``/``H`` (removed in pandas 3)."""
    cases = {
        "5s": pd.date_range("2026-01-01", periods=30, freq="5s"),
        "1min": pd.date_range("2026-01-01", periods=30, freq="60s"),
        "2h": pd.date_range("2026-01-01", periods=30, freq="2h"),
    }
    for expected, index in cases.items():
        rule = _infer_freq(_irregular(index))
        assert rule == expected
        # The rule must be usable as-is by pandas.
        pd.Series(1.0, index=index).resample(rule).mean()


def test_coarser_still_accepts_legacy_uppercase_rules():
    """``_coarser`` compares old and new spellings interchangeably."""
    assert _coarser("1s", "1min") == "1min"
    assert _coarser("1T", "30s") == "1T"
    assert _coarser("2h", "90min") == "2h"


def test_combine_size_ranges_without_explicit_rebin_freq():
    """The headline NanoScan+OPS combine must run on the inferred cadence.

    This is the call that raised on pandas >= 2.2 (#32) and, before that,
    silently rebinned onto a 1000x too fine grid (#31). The guard is the
    sample count: the result must stay in the same ballpark as its inputs
    rather than exploding by orders of magnitude.
    """
    ns = at.load_file(os.path.join(DATA, "Combine_example_NS.csv"))
    ops = at.load_file(os.path.join(DATA, "Combine_example_OPS - file 1.csv"))

    combined = at.combine_size_ranges(ns, ops, crossover=355, match="rebin")

    n_in = max(len(ns.time), len(ops.time))
    assert len(combined.time) <= 2 * n_in
    # A 1 s grid over the same span would be ~98 % NaN.
    assert float(np.asarray(combined.size_data.isna()).mean()) < 0.5
