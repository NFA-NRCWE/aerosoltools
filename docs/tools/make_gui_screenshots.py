"""Regenerate the screenshots used by the GUI documentation.

The GUI docs (``docs/gui/``) illustrate every tab with a PNG committed under
``docs/_static/gui/``. This script produces all of them from scripted demo
projects, so refreshing the whole set after a UI change is one command rather
than a morning of manual screenshotting and cropping.

Run it from the repository root::

    python docs/tools/make_gui_screenshots.py

Why this is not part of the Sphinx build
----------------------------------------
It needs a **real desktop session**. Qt's ``offscreen`` platform plugin has no
font engine here, so it renders the layout and the Matplotlib canvas but draws
no text at all — every label, button and table comes out blank. The GitHub
Actions runners the docs are built on have no display, so the PNGs are generated
on a maintainer's machine and committed; the docs build itself stays
display-free. See ``docs/README.md`` for the full procedure.

Demo data
---------
No single project shows every tab well, so the captures run in three passes,
each built from the sample files in ``tests/data/`` that suit it:

``campaign``
    One coherent 3.5-day campaign (23-26 Oct 2023): four consecutive OPS files
    joined into a single record, plus the two NanoScans that ran alongside it.
    Drives most tabs. The two NanoScans agree closely (y = 0.94x, r^2 = 0.81
    over this window), which is what makes the Correlation capture worth
    looking at — an OPS against a NanoScan measures different size ranges and
    correlates poorly, so the pair is chosen deliberately.

``nanoscan``
    ``Sample_NS.csv``, which suits two panes at once: it contains one very
    well-defined concentration peak (a flat ~47k baseline, a spike to ~410k,
    then a clean decay) for the emission + decay fit, and one clean, distinct
    PSD mode for the lognormal fit.

``aps``
    ``Sample_APS_correlated.txt``, the only sample that produces a correlated
    :class:`~aerosoltools.Aerosol3d` and therefore the Aero <-> Optical tab.

Adding a tab
------------
Add an entry to ``CAMPAIGN_CAPTURES`` with the tab's title as it appears in the
GUI, and a ``setup`` callable if the tab needs driving (clicking Compute,
creating a fit) before it shows anything worth photographing.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, Optional

import matplotlib.dates as mdates
import pandas as pd

# --- The campaign project ----------------------------------------------------
# The period both instrument sets were actively measuring. Everything is cropped
# to it so no plot is mostly empty axis.
CAMPAIGN_WINDOW = (
    pd.Timestamp("2023-10-23 13:10:00"),
    pd.Timestamp("2023-10-26 22:51:00"),
)

# Four consecutive days from one OPS, joined into a single record the way the
# sidebar's "Join same instrument" does.
OPS_FILES = [f"tests/data/Combine_example_OPS - file {i}.csv" for i in (1, 2, 3, 4)]

# The two NanoScans that ran through the same campaign. Instrument keys must
# match aerosoltools.gui.logic.loaders.LOADERS exactly.
NANOSCANS = [
    ("tests/data/Combine_example_NS.csv", "NanoScan (NS)", "NanoScan A"),
    ("tests/data/Correlation_example_NS2.csv", "NanoScan (NS)", "NanoScan B"),
]

# Named purely by clock time — these are demo marks to show activity handling,
# not real logged tasks.
ACTIVITIES = [
    ("Day 1", "2023-10-23 14:00", "2023-10-23 16:00"),
    ("Night 1", "2023-10-23 23:00", "2023-10-24 05:00"),
    ("Day 2", "2023-10-24 09:00", "2023-10-24 15:00"),
]

# Crossover for the cross-instrument size-range combine (NanoScan tops out near
# 365 nm, the OPS starts near 337 nm).
COMBINE_CROSSOVER = 350.0

# --- The NanoScan sample project ---------------------------------------------
# One file that happens to suit two panes: it has a single sharp concentration
# peak (for the decay fit) and one clean, distinct PSD mode (for the fit).
NANOSCAN_FILE = ("tests/data/Sample_NS.csv", "NanoScan (NS)", "NanoScan")

# Spans the flat baseline, the spike and the full decay back towards baseline.
DECAY_WINDOW = (
    pd.Timestamp("2023-09-11 14:45:00"),
    pd.Timestamp("2023-09-11 15:55:00"),
)

# Starting mode for the PSD fit, on the observed peak (the mean distribution
# tops out near 1.1e5 dN/dlogDp between the 115 and 154 nm bins).
PSD_SEED = {"mu": 113.0, "sigma": 1.64, "peak": 1.1e5}

# --- The APS project ---------------------------------------------------------
APS_FILE = ("tests/data/Sample_APS_correlated.txt", "APS", "APS (correlated)")

# Wide enough that no tab's controls are clipped. The Decay and Correlation
# panes are the widest — check those two first if this is ever reduced.
WINDOW_SIZE = (1960, 1000)


# --- Qt helpers --------------------------------------------------------------
def settle(app, rounds: int = 8, ms: int = 220) -> None:
    """Let Qt lay out and Matplotlib draw before grabbing a frame."""
    from aerosoltools.gui.qt import QtCore

    for _ in range(rounds):
        app.processEvents()
    QtCore.QThread.msleep(ms)
    for _ in range(rounds):
        app.processEvents()


def grab(widget, out_dir: str, stem: str, note: str = "") -> Optional[str]:
    """Save ``widget`` to ``out_dir/stem.png``; return the path on success."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.png")
    if not widget.grab().save(path):
        print(f"  !! failed to write {stem}.png")
        return None
    print(f"  {stem + '.png':<30} {note}")
    return path


def make_window(app, theme_mode: str):
    """Create and show a MainWindow at the documented size."""
    from aerosoltools.gui.app.main_window import MainWindow
    from aerosoltools.gui.view import theme

    theme.apply_qt_theme(app, theme_mode)
    theme.apply_mpl_theme(theme_mode)
    win = MainWindow()
    win.resize(*WINDOW_SIZE)
    win.show()
    settle(app)
    return win


def load_one(win, path: str, instrument: str, label: str):
    """Load one file as a dataset and label it, failing loudly if it does not."""
    if not os.path.exists(path):
        raise SystemExit(f"Missing sample file: {path}\nRun this from the repo root.")
    before = len(win.project.datasets)
    win.load_file(path, instrument)
    if len(win.project.datasets) == before:
        # Without this guard a failed load silently relabels the *previous*
        # dataset, producing captures like "NanoScan (OPS)".
        raise SystemExit(
            f"Failed to load {path} as '{instrument}'.\n"
            "Check the key against aerosoltools.gui.logic.loaders.LOADERS."
        )
    ds = win.project.datasets[-1]
    ds.label = label
    return ds


# --- Tab setup callbacks -----------------------------------------------------
# Each receives (window, tab_widget) and prepares the tab so its capture shows a
# result rather than an empty pane with a "click Compute" hint.
def setup_summary(win, tab) -> None:
    """Build the combined summary table.

    Datasets are included by default (``Dataset.summary_on`` starts True), so
    only the Compute step is needed.
    """
    tab.refresh()
    tab._compute()


def _select_nanoscans(win, tab) -> None:
    """Point the correlation X/Y combos at the two NanoScans."""
    wanted = ("NanoScan A", "NanoScan B")
    for combo, label in zip((tab.x_combo, tab.y_combo), wanted):
        idx = combo.findText(label)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    # The two units log on different seconds, so 'exact' would pair nothing.
    idx = tab.match.findText("nearest")
    if idx >= 0:
        tab.match.setCurrentIndex(idx)


def setup_correlation(win, tab) -> None:
    """Correlate the two NanoScans on nearest timestamps."""
    _select_nanoscans(win, tab)
    tab._draw()  # what the Compute button is wired to


def setup_bland_altman(win, tab) -> None:
    """Same pair, shown as a Bland-Altman agreement plot."""
    _select_nanoscans(win, tab)
    idx = tab.analysis.findText("Bland–Altman")
    if idx >= 0:
        tab.analysis.setCurrentIndex(idx)
    tab._draw()


def setup_psd(win, tab) -> None:
    """Seed one mode on the peak and optimise it.

    ``_add_mode`` seeds at the geometric mid-point of the whole size range,
    which fits badly here: this NanoScan's smallest bin (11.5 nm) carries an
    anomalous ~62k spike that drags a full-range fit off the real mode. Seeding
    on the peak instead — and leaving "Local" ticked, so each mode is fitted
    only to the bins near it — converges on the actual distribution.
    """
    tab._add_mode()
    tab._modes[-1].update(PSD_SEED)
    tab._write_modes_to_table()
    tab.local_chk.setChecked(True)
    # Log weighting is the better default when several modes share a
    # distribution, but with a single dominant mode it chases the sparse tails
    # and leaves the curve short of the peak (R^2 0.90 vs 0.98). Fit linearly.
    tab.log_scaling.setChecked(False)
    tab._run_fit()


def setup_overlay(win, tab) -> None:
    """Put a second metric in slot 2 so both y-axes are in use.

    All three datasets are ticked by default, but they all default to metric
    slot 1 (number concentration) on axis 1 — where the OPS, at tens per cm3
    against the NanoScans' tens of thousands, flattens onto the baseline.
    Adding mass concentration as a second metric is what the pane is for: a
    different unit is placed on its own axis automatically.
    """
    combo = tab.metric_combos[1]
    idx = combo.findText("Mass concentration")
    if idx >= 0:
        combo.setCurrentIndex(idx)
    tab.refresh()


def setup_psd_comparison(win, tab) -> None:
    """Select two activities so the dataset x activity grid is visible.

    Datasets are ticked by default (``Dataset.psd_on``), and the activity list
    defaults to "All data" alone; selecting two named tasks instead shows what
    the pane is actually for — one curve per dataset per task.
    """
    for i in range(tab.act_list.count()):
        item = tab.act_list.item(i)
        item.setSelected(item.text() in ("Day 1", "Night 1"))
    tab.refresh()


# --- What to capture from the campaign project -------------------------------
# (tab title as shown in the GUI, output file stem, optional setup callable)
CAMPAIGN_CAPTURES: list[tuple[str, str, Optional[Callable]]] = [
    ("Raw data", "tab-raw-data", None),
    ("Metadata", "tab-metadata", None),
    ("Time series", "tab-timeseries", None),
    ("2D heatmap", "tab-heatmap", None),
    ("PM bands", "tab-pm-bands", None),
    ("Summary", "tab-summary", setup_summary),
    ("Overlay", "tab-overlay", setup_overlay),
    ("Correlation", "tab-correlation", setup_correlation),
    ("Correlation", "tab-correlation-bland-altman", setup_bland_altman),
    ("PSD comparison", "tab-psd-comparison", setup_psd_comparison),
]


# --- Passes ------------------------------------------------------------------
def build_campaign(win) -> None:
    """Load and join the OPS files, add both NanoScans, crop and mark tasks."""
    from aerosoltools.intercomparison import combine_measurements

    ops_sets = [
        load_one(win, path, "OPS", f"OPS part {i}")
        for i, path in enumerate(OPS_FILES, start=1)
    ]
    # Mirror the sidebar's "Join same instrument" without its modal dialog.
    joined = combine_measurements([d.obj for d in ops_sets], require_same_serial=False)
    win._add_derived_dataset(
        joined,
        "OPS",
        "OPS 3330",
        remove_ids=[d.id for d in ops_sets],
        source_files=[f for d in ops_sets for f in d.contributing_files],
    )

    for path, instrument, label in NANOSCANS:
        load_one(win, path, instrument, label)

    # Crop everything to the window both instrument sets were running.
    for ds in win.project.datasets:
        ds.obj = ds.obj.timecrop(*CAMPAIGN_WINDOW, inplace=False, focus=True)
        # Replacing .obj is a mutation, and bumping the generation is the only
        # signal the derived-copy cache gets. Without this, any basis-converted
        # view (dM for PM bands or an overlay mass metric) is served from the
        # entry built at load time — i.e. from the *uncropped* object.
        ds.touch()

    for name, start, end in ACTIVITIES:
        win.project.add_activity(name, pd.Timestamp(start), pd.Timestamp(end))

    win.project.set_active(win.project.datasets[0].id)
    # The crop fields are filled at load time, so refresh them to match the
    # cropped objects rather than showing each file's original full span.
    win.adjust_box.sync_crop_fields()
    win._build_tabs()
    win.refresh_all(reset_view=True)
    win._refresh_sidebar()


def capture_campaign(app, out_dir: str, theme_mode: str) -> list[str]:
    """Capture the full window plus every tab driven by the campaign project."""
    written: list[str] = []
    win = make_window(app, theme_mode)
    build_campaign(win)

    # Full window on the Time series tab — the landing page's anatomy figure.
    win.tabs.select_text("Time series")
    settle(app)
    written.append(grab(win, out_dir, "main-window", "(full window)"))

    for title, stem, setup in CAMPAIGN_CAPTURES:
        if not win.tabs.select_text(title):
            print(f"  !! tab not found: {title}")
            continue
        settle(app)
        tab = win.tabs.currentWidget()
        if setup is not None:
            try:
                setup(win, tab)
            except Exception:
                print(f"  !! setup failed for {title}:")
                traceback.print_exc(limit=3)
            settle(app)
        written.append(grab(win, out_dir, stem, f"({title})"))

    written.append(capture_combined_psd(app, win, out_dir))
    win.close()
    return written


def capture_combined_psd(app, win, out_dir: str) -> Optional[str]:
    """Stitch the OPS and a NanoScan, then capture the resulting broad PSD.

    Illustrates the sidebar's "Combine size ranges…" action for the landing
    page: the two instruments together span roughly 11 nm to 9 um.
    """
    from aerosoltools.intercomparison import combine_size_ranges

    by_label = {d.label: d for d in win.project.datasets}
    ops, ns = by_label.get("OPS 3330"), by_label.get("NanoScan A")
    if ops is None or ns is None:
        print("  !! combine skipped: expected datasets not found")
        return None
    try:
        combined = combine_size_ranges(ns.obj, ops.obj, crossover=COMBINE_CROSSOVER)
    except Exception:
        print("  !! combine_size_ranges failed:")
        traceback.print_exc(limit=3)
        return None
    win._add_derived_dataset(
        combined,
        "Combined",
        f"{ns.label} + {ops.label} (combined)",
        source_files=ns.contributing_files + ops.contributing_files,
    )
    if not win.tabs.select_text("PSD"):
        return None
    settle(app)
    return grab(win, out_dir, "feature-combined-psd", "(combined size ranges)")


def capture_nanoscan(app, out_dir: str, theme_mode: str) -> list[str]:
    """Capture the Decay and PSD tabs on the sample that suits both."""
    written: list[str] = []
    win = make_window(app, theme_mode)
    load_one(win, *NANOSCAN_FILE)
    win.refresh_all(reset_view=True)
    win._refresh_sidebar()
    settle(app)

    if win.tabs.select_text("Decay / Source"):
        settle(app)
        tab = win.tabs.currentWidget()
        try:
            # Create a fit over the peak, exactly as dragging on the plot would.
            tab._on_span(*(mdates.date2num(t) for t in DECAY_WINDOW))
            # Leave Adjust off: its drag handles are for editing, and a clean
            # fit curve reads better in a screenshot.
            tab.adjust_btn.setChecked(False)
            tab._draw()
        except Exception:
            print("  !! decay fit failed:")
            traceback.print_exc(limit=3)
        settle(app)
        written.append(grab(win, out_dir, "tab-decay", "(Decay / Source)"))
    else:
        print("  !! tab not found: Decay / Source")

    if win.tabs.select_text("PSD"):
        settle(app)
        tab = win.tabs.currentWidget()
        try:
            setup_psd(win, tab)
        except Exception:
            print("  !! PSD fit failed:")
            traceback.print_exc(limit=3)
        settle(app)
        written.append(grab(win, out_dir, "tab-psd", "(PSD)"))
    else:
        print("  !! tab not found: PSD")

    win.close()
    return written


def capture_aps(app, out_dir: str, theme_mode: str) -> list[str]:
    """Capture the Aero <-> Optical tab, which needs a correlated APS."""
    path, instrument, label = APS_FILE
    if not os.path.exists(path):
        print(f"  !! missing {path}; skipping Aero <-> Optical")
        return []

    win = make_window(app, theme_mode)
    load_one(win, path, instrument, label)
    win.refresh_all(reset_view=True)
    win._refresh_sidebar()
    settle(app)

    title = "Aero ↔ Optical"
    out = None
    if win.tabs.select_text(title):
        settle(app)
        tab = win.tabs.currentWidget()
        try:
            # Park the time cursor on the concentration peak. It starts at the
            # first sample, where this record is still near zero and the 3-D
            # panel is almost empty.
            peak = win.project.datasets[-1].obj.total_concentration.idxmax()
            tab._set_time(mdates.date2num(peak))
        except Exception:
            print("  !! could not move the time cursor:")
            traceback.print_exc(limit=3)
        # The 3-D bar plot is slow to render; give it extra time.
        settle(app, rounds=10, ms=400)
        out = grab(win, out_dir, "tab-aero-optical", f"({title})")
    else:
        print(f"  !! tab not found: {title} (is the APS file correlated?)")
    win.close()
    return [out]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        default=os.path.join("docs", "_static", "gui"),
        help="Output folder for the PNGs (default: docs/_static/gui).",
    )
    parser.add_argument(
        "--theme",
        default="dark",
        choices=["dark", "light"],
        help="GUI theme to capture (default: dark, the app's own default).",
    )
    parser.add_argument(
        "--only",
        choices=["campaign", "nanoscan", "aps"],
        help="Run a single pass instead of all three.",
    )
    args = parser.parse_args(argv)

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        print(
            "ERROR: QT_QPA_PLATFORM=offscreen renders no text — every label and\n"
            "       button would come out blank. Unset it and run on a real desktop.",
            file=sys.stderr,
        )
        return 2

    from aerosoltools.gui.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    passes = {
        "campaign": capture_campaign,
        "nanoscan": capture_nanoscan,
        "aps": capture_aps,
    }
    chosen = [args.only] if args.only else list(passes)

    print(f"Writing {args.theme}-theme screenshots to {args.out}/")
    written: list = []
    for name in chosen:
        print(f"\n[{name}]")
        written += passes[name](app, args.out, args.theme)

    ok = [w for w in written if w]
    print(f"\nWrote {len(ok)} screenshots.")
    return 0 if len(ok) == len(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
