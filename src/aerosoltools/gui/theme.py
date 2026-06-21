"""Visual theming for the aerosoltools GUI.

Two themes are supported and switchable at runtime:

* **dark** (default) — deep-navy with luminous cyan accents, matching the
  application icon. Embedded plots are dark, with the line cycle brightened so
  series stay legible.
* **light** — a clean light theme. Because exported figures use a *light*
  profile, switching to the light theme also previews roughly what a saved plot
  will look like.

Each theme provides a Qt style sheet (applied on top of the "Fusion" base style)
and a matching Matplotlib rcParams profile. The Matplotlib profile also overrides
the very large default font sizes / figure size that the core :mod:`aerosoltools`
modules set at import time.

Saved/exported figures always use the light :func:`export_rc` profile.
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
_DARK = {
    "BG_DEEP": "#07111f",
    "BG": "#0a1626",
    "GRAD_TOP": "#0d2138",
    "GRAD_BOT": "#07111f",
    "PANEL": "#10243f",
    "PANEL_ALT": "#0c1d34",
    "BORDER": "#1f3d61",
    "BORDER_SOFT": "#16304e",
    "TEXT": "#e7f1fb",
    "MUTED": "#90a6c1",
    "ACCENT": "#29c5f6",
    "ACCENT_HOVER": "#4ad6ff",
    "ACCENT_DEEP": "#1391c4",
    "ON_ACCENT": "#04263e",
    "ACCENT_SOFT": "#173a5b",
    "TAB_BG": "#12304f",
    "TAB_TEXT": "#aebfd6",
    "TAB_BG_HOVER": "#193e63",
    "TAB_SELECTED": "#18406a",
    "TAB_SEL_TEXT": "#4ad6ff",
    "SECONDARY": "#15304e",
    "SECONDARY_HOVER": "#1c3d62",
    "SECONDARY_PRESS": "#24496f",
    "DISABLED_BG": "#122236",
    "DISABLED_TEXT": "#51637a",
    "DISABLED_BORDER": "#1a2f49",
    "PRIMARY_TOP": "#4ad6ff",
    "PRIMARY_BOT": "#1391c4",
    "PRIMARY_HOVER_TOP": "#6fe0ff",
    "PRIMARY_HOVER_BOT": "#29c5f6",
    "PRIMARY_PRESS": "#1391c4",
    "PRIMARY_DIS_BG": "#1d4257",
    "PRIMARY_DIS_TEXT": "#5f7e90",
    "TOOLBAR_BG": "#c9d7e8",
    "TOOLBAR_HOVER": "#a9c4e2",
    # Matplotlib-only
    "PLOT_BG": "#0c1f38",
    "AXES_EDGE": "#3a5a80",
}

_LIGHT = {
    "BG_DEEP": "#dfe5ee",
    "BG": "#e9edf3",
    "GRAD_TOP": "#f3f6fb",
    "GRAD_BOT": "#e3e9f1",
    "PANEL": "#ffffff",
    "PANEL_ALT": "#f4f7fb",
    "BORDER": "#d3dae4",
    "BORDER_SOFT": "#e6ebf1",
    "TEXT": "#1b2430",
    "MUTED": "#5d6775",
    "ACCENT": "#2563eb",
    "ACCENT_HOVER": "#1d4ed8",
    "ACCENT_DEEP": "#1b3fa8",
    "ON_ACCENT": "#ffffff",
    "ACCENT_SOFT": "#dce8ff",
    "TAB_BG": "#3a72e6",
    "TAB_TEXT": "#eaf1ff",
    "TAB_BG_HOVER": "#2c61d6",
    "TAB_SELECTED": "#ffffff",
    "TAB_SEL_TEXT": "#2563eb",
    "SECONDARY": "#eef2f8",
    "SECONDARY_HOVER": "#e1e9f4",
    "SECONDARY_PRESS": "#d4deee",
    "DISABLED_BG": "#f0f2f5",
    "DISABLED_TEXT": "#aab2bd",
    "DISABLED_BORDER": "#e3e7ed",
    "PRIMARY_TOP": "#3b7bf0",
    "PRIMARY_BOT": "#2563eb",
    "PRIMARY_HOVER_TOP": "#2563eb",
    "PRIMARY_HOVER_BOT": "#1d4ed8",
    "PRIMARY_PRESS": "#1b3fa8",
    "PRIMARY_DIS_BG": "#b9c4d6",
    "PRIMARY_DIS_TEXT": "#eef2f7",
    "TOOLBAR_BG": "#eef2f8",
    "TOOLBAR_HOVER": "#dbe7ff",
    # Matplotlib-only
    "PLOT_BG": "#ffffff",
    "AXES_EDGE": "#aeb6c2",
}

PALETTES = {"dark": _DARK, "light": _LIGHT}

# Matplotlib line color cycles per theme.
_CYCLE = {
    "dark": [
        "#33d2ff",
        "#ffb454",
        "#5fe6a8",
        "#c08bff",
        "#ff7a90",
        "#8ad4ff",
        "#ffd166",
        "#f78fb3",
    ],
    "light": [
        "#2563eb",
        "#ef6c00",
        "#16a34a",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#ca8a04",
        "#db2777",
    ],
}

# Module-level record of the theme currently applied (read by the tabs to decide
# whether to brighten plot colours for legibility on a dark background).
_MODE = "dark"


# ---------------------------------------------------------------------------
# Style sheet (one template, filled per palette via string.Template so the
# literal CSS braces need no escaping)
# ---------------------------------------------------------------------------
_QSS = Template(
    """
QMainWindow { background: ${BG_DEEP}; }
QWidget {
    background: ${BG};
    color: ${TEXT};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
}
QWidget#Central {
    background: qlineargradient(x1:0, y1:0, x2:0.4, y2:1,
                               stop:0 ${GRAD_TOP}, stop:1 ${GRAD_BOT});
}

QLabel { color: ${TEXT}; background: transparent; }
QToolTip {
    background: ${PANEL}; color: ${TEXT};
    border: 1px solid ${ACCENT}; padding: 4px 6px; border-radius: 4px;
}
QMenuBar { background: transparent; color: ${TEXT}; }
QMenuBar::item { background: transparent; padding: 5px 12px; border-radius: 6px; }
QMenuBar::item:selected { background: ${ACCENT_SOFT}; }
QMenu {
    background: ${PANEL}; color: ${TEXT};
    border: 1px solid ${BORDER}; border-radius: 8px; padding: 4px;
}
QMenu::item { padding: 6px 22px; border-radius: 5px; }
QMenu::item:selected { background: ${ACCENT_SOFT}; }
QMenu::separator { height: 1px; background: ${BORDER}; margin: 4px 8px; }

/* Top bar — a raised card holding the primary controls */
QFrame#TopBar {
    background: ${PANEL};
    border: 1px solid ${BORDER};
    border-radius: 12px;
}

/* Buttons — subtle/secondary by default */
QPushButton {
    background: ${SECONDARY};
    color: ${TEXT};
    border: 1px solid ${BORDER};
    border-radius: 8px;
    padding: 7px 15px;
    font-weight: 600;
}
QPushButton:hover { background: ${SECONDARY_HOVER}; border-color: ${ACCENT}; }
QPushButton:pressed { background: ${SECONDARY_PRESS}; }
QPushButton:disabled {
    background: ${DISABLED_BG}; color: ${DISABLED_TEXT};
    border-color: ${DISABLED_BORDER};
}

/* Primary buttons (key actions): accent gradient */
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 ${PRIMARY_TOP}, stop:1 ${PRIMARY_BOT});
    color: ${ON_ACCENT};
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 700;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 ${PRIMARY_HOVER_TOP}, stop:1 ${PRIMARY_HOVER_BOT});
}
QPushButton#primary:pressed { background: ${PRIMARY_PRESS}; }
QPushButton#primary:disabled { background: ${PRIMARY_DIS_BG}; color: ${PRIMARY_DIS_TEXT}; }

/* Toggle button (e.g. "Mark activities") — stays accent-filled when active */
QPushButton#toggle:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 ${PRIMARY_TOP}, stop:1 ${PRIMARY_BOT});
    color: ${ON_ACCENT};
    border: 1px solid ${ACCENT};
}
QPushButton#toggle:checked:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 ${PRIMARY_HOVER_TOP}, stop:1 ${PRIMARY_HOVER_BOT});
}

/* Inputs */
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QDateTimeEdit {
    background: ${PANEL_ALT};
    border: 1px solid ${BORDER};
    border-radius: 8px;
    padding: 5px 9px;
    min-height: 20px;
    color: ${TEXT};
    selection-background-color: ${ACCENT};
    selection-color: ${ON_ACCENT};
}
QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover,
QSpinBox:hover, QDateTimeEdit:hover { border-color: ${ACCENT}; }
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus,
QSpinBox:focus, QDateTimeEdit:focus { border: 1px solid ${ACCENT}; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: ${PANEL};
    color: ${TEXT};
    border: 1px solid ${BORDER};
    border-radius: 6px;
    selection-background-color: ${ACCENT_SOFT};
    selection-color: ${TEXT};
    outline: none;
}

/* Checkboxes */
QCheckBox { spacing: 6px; background: transparent; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid ${BORDER};
    border-radius: 4px;
    background: ${PANEL_ALT};
}
QCheckBox::indicator:hover { border-color: ${ACCENT}; }
QCheckBox::indicator:checked {
    background: ${ACCENT};
    border-color: ${ACCENT};
    image: none;
}

/* Tabs — large buttons; the active tab is lighter ("you are here") */
QTabWidget::pane {
    border: 1px solid ${BORDER};
    border-radius: 12px;
    background: ${PANEL};
    top: -1px;
}
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab {
    background: ${TAB_BG};
    color: ${TAB_TEXT};
    padding: 11px 28px;
    margin-right: 5px;
    border: 1px solid ${BORDER_SOFT};
    border-bottom: none;
    border-top-left-radius: 11px;
    border-top-right-radius: 11px;
    font-size: 10.5pt;
    font-weight: 600;
    min-width: 84px;
}
QTabBar::tab:hover { background: ${TAB_BG_HOVER}; color: ${TEXT}; }
QTabBar::tab:selected {
    background: ${TAB_SELECTED};
    color: ${TAB_SEL_TEXT};
    border: 1px solid ${BORDER};
    border-bottom: 3px solid ${ACCENT};
}

/* Tables */
QTableView {
    background: ${PANEL};
    alternate-background-color: ${PANEL_ALT};
    gridline-color: ${BORDER_SOFT};
    color: ${TEXT};
    border: 1px solid ${BORDER};
    border-radius: 10px;
    selection-background-color: ${ACCENT_SOFT};
    selection-color: ${TEXT};
}
QHeaderView::section {
    background: ${PANEL_ALT};
    color: ${MUTED};
    padding: 6px 8px;
    border: none;
    border-right: 1px solid ${BORDER_SOFT};
    border-bottom: 1px solid ${BORDER};
    font-weight: 600;
}
QTableCornerButton::section { background: ${PANEL_ALT}; border: none; }

/* Lists (dataset sidebar, activities) */
QListWidget {
    background: ${PANEL_ALT};
    border: 1px solid ${BORDER};
    border-radius: 8px;
    padding: 3px;
    color: ${TEXT};
}
QListWidget::item {
    padding: 6px 8px; border-radius: 6px;
    border-left: 3px solid transparent;
}
QListWidget::item:hover { background: ${ACCENT_SOFT}; }
QListWidget::item:selected {
    background: ${ACCENT_SOFT}; color: ${TEXT};
    border-left: 3px solid ${ACCENT};
}

/* Group boxes */
QGroupBox {
    border: 1px solid ${BORDER};
    border-radius: 10px;
    margin-top: 12px;
    padding: 10px;
    background: ${PANEL};
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 2px 6px;
    color: ${ACCENT};
    background: transparent;
    text-transform: uppercase;
    font-size: 8.5pt;
    letter-spacing: 1px;
}

/* Dock widget (the detachable datasets sidebar) */
QDockWidget {
    color: ${TEXT};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background: ${PANEL_ALT};
    color: ${ACCENT};
    padding: 8px 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 8.5pt;
    border: 1px solid ${BORDER};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}

/* Status bar */
QStatusBar {
    background: transparent;
    color: ${MUTED};
    border-top: 1px solid ${BORDER};
}
QStatusBar::item { border: none; }
QStatusBar QLabel { color: ${MUTED}; padding: 2px 6px; }

/* Scrollbars */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: ${SECONDARY_PRESS}; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: ${ACCENT}; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: ${SECONDARY_PRESS}; border-radius: 5px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background: ${ACCENT}; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }

/* Matplotlib navigation toolbar — light strip so dark icons stay legible */
QToolBar {
    background: ${TOOLBAR_BG};
    border: 1px solid ${BORDER};
    border-radius: 8px;
    padding: 2px;
    spacing: 2px;
}
QToolButton { background: transparent; border-radius: 4px; padding: 3px; }
QToolButton:hover { background: ${TOOLBAR_HOVER}; }
QToolButton:checked { background: ${TOOLBAR_HOVER}; }
QToolBar QLabel { color: #1b2430; background: transparent; }
"""
)


def build_qss(mode: str = "dark") -> str:
    """Return the Qt style sheet for ``mode`` ("dark" or "light")."""
    return _QSS.substitute(PALETTES.get(mode, _DARK))


def apply_qt_theme(app, mode: str = "dark") -> None:
    """Apply the Fusion base style and the ``mode`` style sheet to ``app``."""
    global _MODE
    _MODE = mode if mode in PALETTES else "dark"
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setStyleSheet(build_qss(_MODE))


def apply_mpl_theme(mode: str = "dark") -> None:
    """Apply the Matplotlib rcParams profile for ``mode``.

    Must be called *after* :mod:`aerosoltools` is imported so it overrides the
    oversized defaults that the core modules set at import time.
    """
    import matplotlib as mpl
    from cycler import cycler

    p = PALETTES.get(mode, _DARK)
    mpl.rcParams.update(
        {
            "figure.facecolor": p["PANEL"],
            "figure.edgecolor": p["PANEL"],
            "savefig.facecolor": p["PANEL"],
            "figure.figsize": (8, 5),
            "axes.facecolor": p["PLOT_BG"],
            "axes.edgecolor": p["AXES_EDGE"],
            "axes.labelcolor": p["TEXT"],
            "axes.titlecolor": p["TEXT"],
            "text.color": p["TEXT"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "600",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=_CYCLE.get(mode, _CYCLE["dark"])),
            "grid.color": p["BORDER_SOFT"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": p["MUTED"],
            "ytick.color": p["MUTED"],
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.facecolor": p["PANEL"],
            "legend.edgecolor": p["BORDER"],
            "legend.labelcolor": p["TEXT"],
            "font.size": 10,
        }
    )


# -- runtime accessors used by the widgets ---------------------------------
def current_mode() -> str:
    """Return the currently applied theme mode ("dark" or "light")."""
    return _MODE


def is_dark() -> bool:
    """Return True when the dark theme is active."""
    return _MODE == "dark"


def fig_facecolor() -> str:
    """Figure background colour for the current theme."""
    return PALETTES.get(_MODE, _DARK)["PANEL"]


def axes_facecolor() -> str:
    """Axes background colour for the current theme."""
    return PALETTES.get(_MODE, _DARK)["PLOT_BG"]


def mpl_cycle() -> list:
    """Return a copy of the current theme's Matplotlib colour cycle."""
    return list(_CYCLE.get(_MODE, _CYCLE["dark"]))


def shadow_rgba() -> tuple:
    """RGBA for the top-bar glow (cyan on dark, soft blue on light)."""
    return (41, 197, 246, 90) if is_dark() else (37, 99, 235, 45)


# ---------------------------------------------------------------------------
# Export profile (always light — report/publication friendly)
# ---------------------------------------------------------------------------
EXPORT_DPI = 200
EXPORT_FIGSIZE = (11.0, 6.5)  # single-panel plots
EXPORT_FIGSIZE_TALL = (11.0, 8.5)  # two-panel heatmap (total + size/time)


def export_rc() -> dict:
    """Return the rcParams overrides used when rendering a figure for export."""
    from cycler import cycler

    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#1f2933",
        "axes.titlecolor": "#1f2933",
        "text.color": "#1f2933",
        "axes.titlesize": 16,
        "axes.titleweight": "600",
        "axes.labelsize": 14,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=_CYCLE["light"]),
        "axes.linewidth": 1.1,
        "grid.color": "#dde3ea",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        # Pin the legend box light too: otherwise it inherits the dark theme's
        # panel colour and the dark label text becomes unreadable on the white
        # export figure (rc_context only overrides the keys listed here).
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "legend.labelcolor": "#1f2933",
        "font.size": 12,
    }
