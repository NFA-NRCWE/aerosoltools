"""Visual theming for the aerosoltools GUI.

Provides a modern light theme via a Qt style sheet (applied on top of the
"Fusion" base style) and a matching Matplotlib rcParams profile so embedded
figures blend in with the rest of the interface. The Matplotlib profile also
overrides the very large default font sizes / figure size that the core
:mod:`aerosoltools` modules set at import time.
"""

from __future__ import annotations

# Palette ------------------------------------------------------------------
BG = "#eef2f7"  # window background
PANEL = "#ffffff"  # inputs / panels
BORDER = "#d0d7de"  # subtle borders
TEXT = "#1f2933"  # primary text
MUTED = "#5b6573"  # secondary text
ACCENT = "#2563eb"  # primary accent (blue)
ACCENT_HOVER = "#1d4ed8"
ACCENT_SOFT = "#dbe7ff"  # selected/hover fills

QSS = f"""
QMainWindow, QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
}}

QLabel {{ color: {TEXT}; }}

/* Buttons */
QPushButton {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton:pressed {{ background: {ACCENT_HOVER}; }}
QPushButton:disabled {{ background: #c2c9d2; color: #eef2f7; }}

/* Inputs */
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QDateTimeEdit {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover,
QSpinBox:hover, QDateTimeEdit:hover {{ border-color: {ACCENT}; }}
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus,
QSpinBox:focus, QDateTimeEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
}}

/* Checkboxes */
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}

/* Tabs */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {PANEL};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 8px 18px;
    margin-right: 2px;
    border: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; background: {ACCENT_SOFT}; }}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {ACCENT};
    border: 1px solid {BORDER};
    border-bottom: 2px solid {ACCENT};
}}

/* Tables */
QTableView {{
    background: {PANEL};
    alternate-background-color: #f4f7fb;
    gridline-color: #e6ebf1;
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: #f0f4f9;
    color: {MUTED};
    padding: 6px 8px;
    border: none;
    border-right: 1px solid #e6ebf1;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QTableCornerButton::section {{ background: #f0f4f9; border: none; }}

/* Lists */
QListWidget {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 2px;
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}

/* Group boxes */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    background: {PANEL};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {MUTED};
}}

/* Scrollbars */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c2cad4; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c2cad4; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

/* Matplotlib navigation toolbar */
QToolBar {{ background: transparent; border: none; spacing: 2px; }}
QToolButton {{ background: transparent; border-radius: 4px; padding: 3px; }}
QToolButton:hover {{ background: {ACCENT_SOFT}; }}
QToolButton:checked {{ background: {ACCENT_SOFT}; }}
"""

# Matplotlib line color cycle (colour-blind friendly, matches the accent).
_MPL_CYCLE = [
    "#2563eb",
    "#ef6c00",
    "#16a34a",
    "#9333ea",
    "#dc2626",
    "#0891b2",
    "#ca8a04",
    "#db2777",
]


def apply_qt_theme(app) -> None:
    """Apply the Fusion base style and the custom style sheet to ``app``."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setStyleSheet(QSS)


def apply_mpl_theme() -> None:
    """Apply a clean Matplotlib rcParams profile for embedded figures.

    Must be called *after* :mod:`aerosoltools` is imported so it overrides the
    oversized defaults that the core modules set at import time.
    """
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.facecolor": PANEL,
            "figure.figsize": (8, 5),
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#aeb6c2",
            "axes.labelcolor": TEXT,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "600",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=_MPL_CYCLE),
            "grid.color": "#e3e8ef",
            "grid.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "font.size": 10,
        }
    )


# ---------------------------------------------------------------------------
# Export profile
# ---------------------------------------------------------------------------
# Fixed, non-configurable settings for saved figures. The key idea: pick a
# deliberate figure size *and* matching font/line sizes, then raise DPI only
# for crispness. This keeps every element (titles, labels, legend, activity
# shading, ticks) well proportioned — unlike bumping DPI or figure size alone.
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
        "axes.titlesize": 16,
        "axes.titleweight": "600",
        "axes.labelsize": 14,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=_MPL_CYCLE),
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
        "font.size": 12,
    }
