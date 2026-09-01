"""Regression test for the ``plot_timeseries`` colorbar placement (GitHub #33).

The shared colorbar used to be attached with ``fig.colorbar(mesh, ax=[ax1, ax2])``,
which shrinks the panels at call time but leaves the colorbar outside any layout
engine. Any later layout pass re-expanded the panels back over it, so the
colorbar was drawn through the middle of both panels, on top of the data.
"""

import os
import warnings

import matplotlib
import pytest

matplotlib.use("Agg")

import aerosoltools as at  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def ops():
    return at.load_file(os.path.join(DATA, "Sample_OPS.csv"))


def _panels(fig):
    """Return ``(ax1, ax2, colorbar_axes)`` of a plot_timeseries figure."""
    return fig.axes[0], fig.axes[1], fig.axes[-1]


def _assert_clear(fig):
    """The colorbar must sit entirely to the right of both data panels."""
    fig.canvas.draw()
    ax1, ax2, cax = _panels(fig)
    for name, ax in (("ax1", ax1), ("ax2", ax2)):
        assert ax.get_position().x1 <= cax.get_position().x0 + 1e-9, (
            f"colorbar (x0={cax.get_position().x0:.3f}) overlaps {name} "
            f"(x1={ax.get_position().x1:.3f})"
        )


def test_colorbar_clear_of_panels_on_creation(ops):
    fig, _ = ops.plot_timeseries(log=True)
    _assert_clear(fig)


def test_colorbar_survives_tight_layout(ops):
    """The exact reproduction from the issue."""
    fig, _ = ops.plot_timeseries(log=True)
    fig.canvas.draw()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.tight_layout()
    incompatible = [
        w for w in caught if "not compatible with tight_layout" in str(w.message)
    ]
    assert not incompatible, "colorbar axes are still outside the layout"
    _assert_clear(fig)


def test_colorbar_survives_resize(ops):
    """An embedding canvas that re-lays out on resize hits the same path."""
    fig, _ = ops.plot_timeseries(log=True)
    fig.canvas.draw()
    fig.set_size_inches(6, 4)
    _assert_clear(fig)


def test_caller_supplied_axes_still_supported(ops):
    """The ax1/ax2 branch (used by the GUI) keeps working unchanged."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    out_fig, axes = ops.plot_timeseries(log=True, ax1=ax1, ax2=ax2)
    assert out_fig is fig
    assert len(axes) == 3
    _assert_clear(fig)
