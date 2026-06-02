"""Interactive PyQt5 viewer for aerosoltools data.

This submodule is optional and only imported on demand, so the GUI's Qt
dependency does not affect the rest of :mod:`aerosoltools`. Install the extra
dependencies with::

    pip install aerosoltools[gui]

and launch the viewer from Python::

    from aerosoltools.gui import launch
    launch("path/to/Sample_ELPI.txt")          # instrument auto-guessed
    launch("data.txt", instrument="ELPI")        # or specify explicitly

or from the command line::

    python -m aerosoltools.gui path/to/Sample_ELPI.txt
"""

from __future__ import annotations

from typing import Optional

__all__ = ["launch", "main"]

_MISSING_QT_MSG = (
    "The aerosoltools GUI requires PyQt5. Install it with:\n"
    "    pip install aerosoltools[gui]\n"
    "or\n"
    "    pip install PyQt5\n"
)


def launch(path: Optional[str] = None, instrument: Optional[str] = None) -> int:
    """Launch the GUI, optionally pre-loading a file.

    Args:
        path: Optional path to a data file to open on startup.
        instrument: Optional instrument name (a key of
            :data:`aerosoltools.gui.loaders.LOADERS`). If omitted, the
            instrument is guessed from the file name.

    Returns:
        The Qt application's exit code.
    """
    try:
        from .qt import QtWidgets
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MISSING_QT_MSG) from exc

    from .main_window import MainWindow
    from .theme import apply_mpl_theme, apply_qt_theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_qt_theme(app)
    apply_mpl_theme()  # after aerosoltools import, to override its big defaults
    window = MainWindow(path=path, instrument=instrument)
    window.show()
    return app.exec_()


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point. See ``python -m aerosoltools.gui --help``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="aerosoltools.gui",
        description="Interactive viewer for aerosoltools data files.",
    )
    parser.add_argument("path", nargs="?", help="Data file to open on startup.")
    parser.add_argument(
        "-i",
        "--instrument",
        default=None,
        help="Instrument loader to use (e.g. ELPI, CPC, SMPS). "
        "Defaults to a guess based on the file name.",
    )
    args = parser.parse_args(argv)
    return launch(path=args.path, instrument=args.instrument)
