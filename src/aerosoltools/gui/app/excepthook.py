"""Report unexpected errors in a dialog instead of letting Qt close the app.

PyQt5 (5.5 and later) aborts the whole application when a Python exception
escapes a slot and no :data:`sys.excepthook` is installed — so a single bug in a
button handler closed the program, and the unsaved project with it, with no
message (GitHub #36). :func:`install_excepthook` keeps the app running, shows
what went wrong, and still prints the traceback to the console.
"""

from __future__ import annotations

import sys
import traceback

from ..qt import QtWidgets

_showing = False  # guards against an error raised while the dialog is open


def _report(exc_type, exc, tb) -> None:
    """Show one unexpected error, with its traceback under *Show Details*."""
    global _showing
    if _showing or QtWidgets.QApplication.instance() is None:
        return
    _showing = True
    try:
        box = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow())
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("Unexpected error")
        box.setText(
            "Something went wrong, and that action was not completed. The "
            "program is still running — consider saving your project.\n\n"
            f"{exc_type.__name__}: {exc}"
        )
        box.setDetailedText("".join(traceback.format_exception(exc_type, exc, tb)))
        box.exec_()
    finally:
        _showing = False


def install_excepthook() -> None:
    """Route uncaught exceptions to a dialog (and stderr) instead of aborting.

    Ctrl+C (:class:`KeyboardInterrupt`) and interpreter exits still go to the
    previous hook unchanged.
    """
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            previous(exc_type, exc, tb)
            return
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        _report(exc_type, exc, tb)

    sys.excepthook = hook
