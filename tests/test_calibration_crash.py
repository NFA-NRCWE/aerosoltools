"""Regression tests for GitHub #36 ("Crash when doing calibration").

The Correlation tab's *Calibrate…* button imported its dialog from a module that
the GUI reorganisation had moved, so the click raised ``ModuleNotFoundError`` —
and PyQt5 aborts the application on an exception escaping a slot. Reproducing
it with two NanoScan datasets also exposed two alignment bugs underneath: a
``match="rebin"`` alignment looked up the whole ``(x, y)`` parameter tuple as a
single column, and ``fit_data`` failed with its own default ``parameter=0``.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib

import pytest

import aerosoltools as at
from aerosoltools.intercomparison import fit_calibration, plot_correlation
from aerosoltools.intercomparison.correlation import fit_data

_DATA = os.path.join(os.path.dirname(__file__), "data")
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def nanoscans():
    """The two NanoScan samples, which overlap in time."""
    a = at.load_file(os.path.join(_DATA, "Combine_example_NS.csv"))
    b = at.load_file(os.path.join(_DATA, "Correlation_example_NS2.csv"))
    return a, b


def _internal_imports():
    """Every ``from aerosoltools… import …`` in the package, function-local too."""
    for path in sorted((_SRC / "aerosoltools").rglob("*.py")):
        # The package a relative import starts from (a module's or __init__'s).
        package = list(path.relative_to(_SRC).with_suffix("").parts)[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith("aerosoltools"):
                yield path, node.lineno, target, [a.name for a in node.names]


def test_every_internal_import_resolves():
    """Lazy imports inside functions only fail when clicked — check them all."""
    has_qt = importlib.util.find_spec("PyQt5") is not None
    if has_qt:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    broken = []
    for path, line, target, names in _internal_imports():
        if target.startswith("aerosoltools.gui") and not has_qt:
            continue
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            broken.append(f"{path.name}:{line}: {target} ({exc})")
            continue
        for name in names:
            if name == "*" or hasattr(module, name):
                continue
            try:
                importlib.import_module(f"{target}.{name}")
            except ImportError:
                broken.append(f"{path.name}:{line}: {target} has no {name}")
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("match", ["nearest", "rebin"])
def test_correlation_aligns_in_every_match_mode(nanoscans, match):
    """``rebin`` used to look up ``("Total_conc", "Total_conc")`` as one column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = nanoscans
    fig, ax = plt.subplots()
    plot_correlation(a, b, parameter="Total_conc", match=match, ax_in=ax)
    plt.close(fig)


@pytest.mark.parametrize("basis", ["total", "per_bin"])
def test_fit_calibration_with_rebin_alignment(nanoscans, basis):
    """A rebinned calibration fits the pair instead of failing to align."""
    a, b = nanoscans
    model = fit_calibration(b, a, basis=basis, match="rebin")
    assert max(model.n_obs) > 100


def test_fit_data_default_parameter_is_the_first_column(nanoscans):
    """``parameter=0`` is documented as a position in ``data.columns``."""
    a, _b = nanoscans
    coeffs, _errors, r2 = fit_data(a, a)
    assert coeffs["m"] == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from aerosoltools.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_calibrate_button_opens_the_dialog(app, monkeypatch):
    """Clicking *Calibrate…* must open the dialog, not abort the application."""
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.logic.calibration import CalibrationDialog
    from aerosoltools.gui.tabs.correlation import CorrelationTab

    opened = []
    monkeypatch.setattr(CalibrationDialog, "exec_", lambda self: opened.append(self))
    win = MainWindow()
    win.load_file(os.path.join(_DATA, "Combine_example_NS.csv"))
    win.load_file(os.path.join(_DATA, "Correlation_example_NS2.csv"))
    tab = win.findChildren(CorrelationTab)[0]
    tab.refresh()
    tab.match.setCurrentText("rebin")
    tab.calibrate_btn.click()
    assert len(opened) == 1

    dialog = opened[0]
    dialog._preview()
    assert dialog._model is not None, dialog.preview.text()


def test_uncaught_errors_are_shown_not_fatal(app, monkeypatch):
    """An exception escaping a slot is reported in a dialog; the app lives on."""
    import sys

    from aerosoltools.gui.app.excepthook import install_excepthook
    from aerosoltools.gui.qt import QtWidgets

    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda box: shown.append(box))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored afterwards
    install_excepthook()
    try:
        raise ModuleNotFoundError("No module named 'aerosoltools.gui.calibration'")
    except ModuleNotFoundError:
        sys.excepthook(*sys.exc_info())
    assert len(shown) == 1
    assert "ModuleNotFoundError" in shown[0].text()
    assert "Traceback" in shown[0].detailedText()
