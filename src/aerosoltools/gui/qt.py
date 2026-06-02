"""Qt binding and Matplotlib backend setup for the aerosoltools GUI.

Importing this module configures Matplotlib to use the Qt backend bound to
PyQt5 and re-exports the Qt symbols and Matplotlib-Qt canvas/toolbar used
throughout the GUI. Centralising the imports here keeps the rest of the GUI
binding-agnostic and ensures the backend is selected exactly once.
"""

from __future__ import annotations

import os

# Force a deterministic binding/backend *before* importing the canvas, so that
# Matplotlib's qt_compat does not auto-pick a different (also-installed) binding.
os.environ.setdefault("QT_API", "PyQt5")

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

__all__ = [
    "QtCore",
    "QtGui",
    "QtWidgets",
    "FigureCanvas",
    "NavigationToolbar",
    "Figure",
]
