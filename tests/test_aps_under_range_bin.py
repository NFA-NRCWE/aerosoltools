"""The APS under-range catch-all bin in the aero/optical views (GitHub #29).

TSI's APS AIM export opens its size columns with a ``"<x"`` catch-all holding
everything detected below the instrument's lower channel bound. It is a real
measurement but not a sized bin, and it routinely carries most of the counts --
on the bundled sample, 63 % of them, 70x the next bin -- so it flattens the
colour scale of any size-resolved view of the correlated record.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import aerosoltools as at  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def aps():
    return at.load_file(os.path.join(DATA, "Sample_APS_correlated.txt"))


def test_loader_flags_the_catch_all_bin(aps):
    assert aps.under_range_bin == 0
    # The synthetic log-spaced optical axis has no such bin.
    assert aps.optical.metadata.get("under_range_bin") is None


def test_aero_only_export_is_flagged_too():
    plain = at.load_file(os.path.join(DATA, "Sample_APS_aero.txt"))
    assert plain.metadata.get("under_range_bin") == 0


def test_the_catch_all_bin_really_does_dominate(aps):
    """Guards the premise -- if this stops holding, the default is worth revisiting."""
    per_bin = aps.size_data.sum(axis=0).to_numpy()
    assert per_bin[0] / per_bin.sum() > 0.5
    assert per_bin[0] > 10 * per_bin[1]


def test_correlation_cube_keeps_every_bin_by_default(aps):
    """The numeric accessor is unchanged -- only the plots trim."""
    cube = aps.correlation_cube()
    assert cube.values.shape[2] == len(aps.bin_mids)


def test_correlation_cube_can_drop_the_catch_all_bin(aps):
    full = aps.correlation_cube(include_under_range=True)
    trimmed = aps.correlation_cube(include_under_range=False)

    assert trimmed.values.shape[2] == full.values.shape[2] - 1
    assert len(trimmed.aero_edges) == len(full.aero_edges) - 1
    assert trimmed.aero_edges[0] == full.aero_edges[1]
    # The physical total is summed before trimming, so it must not change.
    assert np.allclose(full.total, trimmed.total)


def test_trimming_removes_the_dynamic_range_spike(aps):
    full = aps.correlation_cube(include_under_range=True)
    trimmed = aps.correlation_cube(include_under_range=False)
    assert np.nanmax(full.values) > 10 * np.nanmax(trimmed.values)


def test_plot_aero_vs_optical_omits_it_by_default(aps):
    _, ax_default = aps.plot_aero_vs_optical()
    _, ax_kept = aps.plot_aero_vs_optical(include_under_range=True)
    # Dropping bin 0 lifts the low end of the aerodynamic axis.
    assert ax_default.get_xlim()[0] > ax_kept.get_xlim()[0]


def test_plot_aero_optical_3d_omits_it_by_default(aps):
    fig, ax = aps.plot_aero_optical_3d()
    assert fig is not None and ax is not None
    fig2, ax2 = aps.plot_aero_optical_3d(include_under_range=True)
    assert fig2 is not None and ax2 is not None


def test_under_range_bin_is_none_for_non_aps_records():
    """Nothing else in the package sets the flag."""
    ops = at.load_file(os.path.join(DATA, "Sample_OPS.csv"))
    assert ops.metadata.get("under_range_bin") is None
