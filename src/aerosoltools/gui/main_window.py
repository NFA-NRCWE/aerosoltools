"""Main application window for the aerosoltools GUI."""

from __future__ import annotations

import inspect
import os
import traceback
from typing import List, Optional

import pandas as pd

from ..utility import Combine_NS_OPS, combine_measurements
from . import helpers, theme
from .assets import icon_path
from .loaders import LOADERS, guess_instrument
from .project import Dataset, Project
from .projectio import load_project, save_project
from .qt import QtCore, QtGui, QtWidgets
from .sidebar import DatasetSidebar
from .tabs import (
    HeatmapTab,
    OverlayTab,
    PMBandsTab,
    PSDTab,
    RawDataTab,
    SummaryTab,
    TimeSeriesTab,
)


class _SlackTabBar(QtWidgets.QTabBar):
    """Tab bar that pads each tab's width hint so labels never clip.

    Qt's default size hint for a stylesheet-padded tab can under-allocate
    width, clipping the first/last characters of the label. Adding a fixed
    slack to the hint guarantees the full text is shown.
    """

    def tabSizeHint(self, index):  # noqa: N802
        size = super().tabSizeHint(index)
        size.setWidth(size.width() + 28)
        return size


class _CombineNSOPSDialog(QtWidgets.QDialog):
    """Pick a NanoScan and an OPS dataset (+ time-match) to combine."""

    def __init__(self, parent, datasets):
        super().__init__(parent)
        self.setWindowTitle("Combine NS + OPS")
        self._datasets = datasets

        form = QtWidgets.QFormLayout(self)
        self.ns_combo = QtWidgets.QComboBox()
        self.ops_combo = QtWidgets.QComboBox()
        for d in datasets:
            label = f"{d.label}  ({d.instrument})"
            self.ns_combo.addItem(label, d.id)
            self.ops_combo.addItem(label, d.id)
        self._preselect(self.ns_combo, ("nano", "ns"))
        self._preselect(self.ops_combo, ("ops",))

        self.match_combo = QtWidgets.QComboBox()
        self.match_combo.addItems(["rebin", "nearest", "exact"])

        form.addRow("NanoScan (smaller sizes):", self.ns_combo)
        form.addRow("OPS (larger sizes):", self.ops_combo)
        form.addRow("Time match:", self.match_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _preselect(self, combo, keys) -> None:
        for i in range(combo.count()):
            ds_id = combo.itemData(i)
            d = next(x for x in self._datasets if x.id == ds_id)
            text = f"{d.instrument} {d.instrument_key}".lower()
            if any(k in text for k in keys):
                combo.setCurrentIndex(i)
                return

    def result(self):
        ns = next(d for d in self._datasets if d.id == self.ns_combo.currentData())
        ops = next(d for d in self._datasets if d.id == self.ops_combo.currentData())
        return ns, ops, self.match_combo.currentText()


class MainWindow(QtWidgets.QMainWindow):
    """Top-level window: a load bar, dtype/density controls, and data tabs."""

    def __init__(self, path: Optional[str] = None, instrument: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("aerosoltools viewer")
        self.resize(1340, 860)
        _icon = icon_path()
        if _icon:
            self.setWindowIcon(QtGui.QIcon(_icon))

        # The project owns the loaded datasets and the shared task registry.
        self.project = Project()
        self._project_path: Optional[str] = None  # folder of the saved project
        self._theme: str = theme.current_mode()
        self._tabs: List = []
        # Signature ("1d"/"2d") of the tab set currently built, so switching the
        # active dataset only rebuilds tabs when the data shape actually changes.
        self._tab_sig: Optional[str] = None
        # When True, plot tabs autoscale on the next refresh; when False they
        # preserve the user's current zoom/pan (e.g. after marking a task).
        self._reset_view: bool = True

        self._build_ui()

        if path:
            self.load_file(path, instrument)

    # -- active-dataset views ---------------------------------------------
    # The single-view tabs read these (unchanged) attributes; they now resolve
    # to whichever dataset is active in the sidebar.
    @property
    def obj(self):
        ds = self.project.active
        return ds.obj if ds is not None else None

    @property
    def source_path(self) -> Optional[str]:
        ds = self.project.active
        return ds.source_path if ds is not None else None

    @property
    def source_instrument(self) -> Optional[str]:
        ds = self.project.active
        return ds.instrument_key if ds is not None else None

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        self._build_menu()
        self._build_sidebar()

        central = QtWidgets.QWidget()
        central.setObjectName("Central")
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(12)

        layout.addWidget(self._build_top_bar())

        # The crop / resample / smooth / time-shift controls are built here but
        # *not* added to the main layout: the Time series tab embeds them, so
        # processing always happens where the data is visible.
        self.adjust_box = self._build_adjust_box()

        # Tabs sit directly below the top bar.
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabBar(_SlackTabBar())
        self.tabs.tabBar().setExpanding(False)
        self.tabs.setElideMode(QtCore.Qt.ElideNone)
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs, stretch=1)

        # Wrap the central content in a scroll area, so when a pane is too small
        # to fit, a scrollbar appears instead of enforcing a hard minimum size
        # (which would block the datasets splitter from being dragged).
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("CentralScroll")
        scroll.setWidget(central)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setCentralWidget(scroll)

        # File / dataset info lives in the status bar at the bottom.
        self.info = QtWidgets.QLabel("No file loaded. Use 'Open file…' to begin.")
        self.statusBar().addWidget(self.info)

        self._set_2d_controls_enabled(False)
        self._set_adjust_enabled(False)

    def _build_top_bar(self) -> QtWidgets.QWidget:
        """Build the raised top control bar (open / instrument / dtype / density)."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("TopBar")
        bar = QtWidgets.QHBoxLayout(frame)
        bar.setContentsMargins(14, 10, 14, 10)
        bar.setSpacing(8)

        open_btn = QtWidgets.QPushButton("Open file…")
        open_btn.setObjectName("primary")
        open_btn.clicked.connect(self._open_dialog)
        bar.addWidget(open_btn)

        bar.addWidget(QtWidgets.QLabel("Instrument:"))
        self.instrument_combo = QtWidgets.QComboBox()
        self.instrument_combo.addItems(list(LOADERS.keys()))
        bar.addWidget(self.instrument_combo)

        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.reload_btn.setToolTip(
            "Reload the current file (discards conversions and activities)"
        )
        self.reload_btn.clicked.connect(self._reload)
        self.reload_btn.setEnabled(False)
        bar.addWidget(self.reload_btn)

        bar.addSpacing(20)

        # dtype / density controls (only meaningful for size-resolved data).
        self.dtype_label = QtWidgets.QLabel("dtype:")
        bar.addWidget(self.dtype_label)
        self.dtype_combo = QtWidgets.QComboBox()
        self.dtype_combo.addItems(["dN", "dM", "dS", "dV"])
        self.dtype_combo.currentIndexChanged.connect(self._on_dtype_change)
        bar.addWidget(self.dtype_combo)

        self.density_label = QtWidgets.QLabel("density (g/cm³):")
        bar.addWidget(self.density_label)
        self.density_spin = QtWidgets.QDoubleSpinBox()
        self.density_spin.setRange(0.1, 25.0)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setValue(1.0)
        self.density_spin.editingFinished.connect(self._on_density_change)
        bar.addWidget(self.density_spin)

        bar.addStretch(1)

        # Soft glow for a raised, high-end "card" feel (cyan on the dark theme).
        shadow = QtWidgets.QGraphicsDropShadowEffect(frame)
        shadow.setBlurRadius(26)
        shadow.setColor(QtGui.QColor(*theme.shadow_rgba()))
        shadow.setOffset(0, 0)
        frame.setGraphicsEffect(shadow)
        self._topbar_shadow = shadow
        return frame

    def _build_sidebar(self) -> None:
        """Build the detachable left-hand datasets panel."""
        self.sidebar = DatasetSidebar()
        self.sidebar.add_requested.connect(self._open_dialog)
        self.sidebar.dataset_selected.connect(self.set_active_dataset)
        self.sidebar.remove_requested.connect(self._remove_dataset)
        self.sidebar.rename_requested.connect(self._rename_dataset)
        self.sidebar.join_requested.connect(self._join_same_instrument)
        self.sidebar.combine_ns_ops_requested.connect(self._combine_ns_ops)

        # Scroll the sidebar when the dock is dragged too narrow, instead of
        # enforcing a minimum width (which would freeze the splitter).
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.sidebar)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        dock = QtWidgets.QDockWidget("Datasets", self)
        dock.setObjectName("DatasetsDock")
        dock.setWidget(scroll)
        # Detachable (float / re-dock) but not closable, so it cannot be lost.
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        # A small floor so the dock can be dragged narrow (scrollbar appears)
        # but never collapses to an ungrabbable sliver.
        dock.setMinimumWidth(80)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [240], QtCore.Qt.Horizontal)
        self.datasets_dock = dock

    def _build_menu(self) -> None:
        """Build the classic top menu bar (File / View / Help)."""
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction("New project", self._new_project)
        file_menu.addAction("Open project…", self._open_project)
        file_menu.addSeparator()
        file_menu.addAction("Save project", self._save_project)
        file_menu.addAction("Save project as…", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Add data file…", self._open_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        view_menu = mb.addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")
        self._theme_group = QtWidgets.QActionGroup(self)
        for label, mode in (("Dark", "dark"), ("Light", "light")):
            act = QtWidgets.QAction(label, self, checkable=True)
            act.setChecked(mode == self._theme)
            act.triggered.connect(lambda _c, m=mode: self.set_theme(m))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)
        view_menu.addSeparator()
        self._dock_action = QtWidgets.QAction(
            "Datasets panel", self, checkable=True, checked=True
        )
        self._dock_action.triggered.connect(
            lambda c: self.datasets_dock.setVisible(c)
        )
        view_menu.addAction(self._dock_action)

        help_menu = mb.addMenu("&Help")
        help_menu.addAction("About", self._about)

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About aerosoltools viewer",
            "aerosoltools viewer\n\nInteractive viewer for aerosol instrument "
            "data.\nLoad multiple datasets, mark shared tasks, and explore.",
        )

    def _build_adjust_box(self) -> QtWidgets.QWidget:
        """Build the compact one-column "Data adjustments" box.

        Each operation gets its own full-width row (crop, resample, smooth, time
        shift), stacked vertically, so action buttons always have room for their
        full label. The widgets are owned by :class:`MainWindow` (their handlers
        operate on the active dataset) but the box is embedded in the Time series
        tab so adjustments happen where the data is visible.
        """
        methods = ["mean", "median", "min", "max", "sum"]
        box = QtWidgets.QGroupBox("Data adjustments")
        grid = QtWidgets.QVBoxLayout(box)
        grid.setSpacing(6)

        def _row(label: str) -> QtWidgets.QHBoxLayout:
            r = QtWidgets.QHBoxLayout()
            tag = QtWidgets.QLabel(label)
            tag.setFixedWidth(72)
            r.addWidget(tag)
            grid.addLayout(r)
            return r

        # --- crop ---------------------------------------------------------
        fmt = "yyyy-MM-dd HH:mm:ss"
        crop = _row("Crop")
        crop.addWidget(QtWidgets.QLabel("from"))
        self.crop_start = QtWidgets.QDateTimeEdit()
        self.crop_start.setDisplayFormat(fmt)
        self.crop_start.setCalendarPopup(True)
        self.crop_start.setFixedWidth(150)
        crop.addWidget(self.crop_start)
        crop.addWidget(QtWidgets.QLabel("to"))
        self.crop_end = QtWidgets.QDateTimeEdit()
        self.crop_end.setDisplayFormat(fmt)
        self.crop_end.setCalendarPopup(True)
        self.crop_end.setFixedWidth(150)
        crop.addWidget(self.crop_end)
        self.crop_btn = QtWidgets.QPushButton("Apply crop")
        self.crop_btn.setObjectName("primary")
        self.crop_btn.clicked.connect(self._apply_crop)
        crop.addWidget(self.crop_btn)
        self.crop_view_btn = QtWidgets.QPushButton("Crop to view")
        self.crop_view_btn.setToolTip(
            "Crop to the time window currently shown on the active plot "
            "(Time series, 2D heatmap, or PM bands)"
        )
        self.crop_view_btn.clicked.connect(self._crop_to_view)
        crop.addWidget(self.crop_view_btn)
        crop.addStretch(1)

        # --- resample -----------------------------------------------------
        res = _row("Resample")
        res.addWidget(QtWidgets.QLabel("to"))
        self.resample_freq = QtWidgets.QLineEdit("1min")
        self.resample_freq.setFixedWidth(80)
        self.resample_freq.setToolTip(
            "Target time step as a pandas offset, e.g. 30s, 1min, 5min, 1H"
        )
        res.addWidget(self.resample_freq)
        self.resample_method = QtWidgets.QComboBox()
        self.resample_method.addItems(methods)
        res.addWidget(self.resample_method)
        self.resample_btn = QtWidgets.QPushButton("Apply resampling")
        self.resample_btn.setObjectName("primary")
        self.resample_btn.clicked.connect(self._apply_resampling)
        res.addWidget(self.resample_btn)
        res.addStretch(1)

        # --- smooth -------------------------------------------------------
        sm = _row("Smooth")
        sm.addWidget(QtWidgets.QLabel("window"))
        self.smooth_window = QtWidgets.QSpinBox()
        self.smooth_window.setRange(2, 999)
        self.smooth_window.setValue(5)
        self.smooth_window.setToolTip("Rolling window size, in number of samples")
        sm.addWidget(self.smooth_window)
        self.smooth_method = QtWidgets.QComboBox()
        self.smooth_method.addItems(methods)
        sm.addWidget(self.smooth_method)
        self.smooth_btn = QtWidgets.QPushButton("Apply smoothing")
        self.smooth_btn.setObjectName("primary")
        self.smooth_btn.clicked.connect(self._apply_smoothing)
        sm.addWidget(self.smooth_btn)
        sm.addStretch(1)

        # --- time shift ---------------------------------------------------
        sh = _row("Time shift")
        sh.addWidget(QtWidgets.QLabel("by"))
        self.shift_value = QtWidgets.QDoubleSpinBox()
        self.shift_value.setRange(-100000.0, 100000.0)
        self.shift_value.setDecimals(1)
        self.shift_value.setSingleStep(1.0)
        self.shift_value.setValue(0.0)
        sh.addWidget(self.shift_value)
        self.shift_unit = QtWidgets.QComboBox()
        self.shift_unit.addItems(["seconds", "minutes", "hours"])
        self.shift_unit.setCurrentText("minutes")
        sh.addWidget(self.shift_unit)
        self.shift_btn = QtWidgets.QPushButton("Apply time shift")
        self.shift_btn.setObjectName("primary")
        self.shift_btn.setToolTip(
            "Permanently shift the active dataset's time axis. Shared tasks keep "
            "their absolute times, so this dataset's summaries change accordingly."
        )
        self.shift_btn.clicked.connect(self._apply_timeshift)
        sh.addWidget(self.shift_btn)
        sh.addStretch(1)

        hint = QtWidgets.QLabel("Tip: use Reload to undo resample / smooth / shift.")
        hint.setStyleSheet("color: palette(mid);")
        grid.addWidget(hint)

        # Hug the content width so the box stays compact and its right-aligned
        # action buttons sit close to the settings (instead of far to the right),
        # leaving more horizontal room for the activities panel.
        box.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred
        )
        return box

    def _set_adjust_enabled(self, enabled: bool) -> None:
        for w in (
            self.crop_start, self.crop_end, self.crop_btn, self.crop_view_btn,
            self.resample_freq, self.resample_method, self.resample_btn,
            self.smooth_window, self.smooth_method, self.smooth_btn,
            self.shift_value, self.shift_unit, self.shift_btn,
        ):
            w.setEnabled(enabled)

    def _set_2d_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.dtype_label,
            self.dtype_combo,
            self.density_label,
            self.density_spin,
        ):
            w.setEnabled(enabled)

    # -- loading -----------------------------------------------------------
    def _open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open aerosol data file",
            "",
            "Data files (*.txt *.csv *.xlsx *.xls);;All files (*)",
        )
        if not path:
            return
        guess = guess_instrument(os.path.basename(path))
        if guess:
            idx = self.instrument_combo.findText(guess)
            if idx >= 0:
                self.instrument_combo.setCurrentIndex(idx)
        self.load_file(path, self.instrument_combo.currentText())

    def _reload(self) -> None:
        """Reload the active dataset's file in place (keeps shared activities)."""
        ds = self.project.active
        if ds is None or not ds.source_path:
            return
        obj, instrument = self._load_obj(ds.source_path, ds.instrument_key)
        if obj is None:
            return
        ds.obj = obj
        ds.instrument_key = instrument
        # Re-project the shared project activities onto the freshly loaded data.
        self.project._apply_activities(ds)
        self._sync_crop_fields()
        self._build_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def _load_obj(self, path: str, instrument: Optional[str] = None):
        """Resolve a loader and load ``path``; return ``(obj, instrument)``.

        Returns ``(None, None)`` (after showing a message) if the instrument is
        unknown or the file fails to load.
        """
        if instrument is None:
            instrument = (
                guess_instrument(os.path.basename(path))
                or self.instrument_combo.currentText()
            )
        if instrument not in LOADERS:
            QtWidgets.QMessageBox.warning(
                self, "Unknown instrument", f"No loader registered for '{instrument}'."
            )
            return None, None

        idx = self.instrument_combo.findText(instrument)
        if idx >= 0:
            self.instrument_combo.setCurrentIndex(idx)

        loader = LOADERS[instrument]
        # Include auxiliary channels when the loader supports it, so the
        # "Extra data" view and extra series are populated.
        kwargs = {}
        if "extra_data" in inspect.signature(loader).parameters:
            kwargs["extra_data"] = True
        try:
            obj = loader(path, **kwargs)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to load file",
                f"Could not load:\n{path}\n\nas '{instrument}'.\n\n"
                + traceback.format_exc(limit=2),
            )
            return None, None
        return obj, instrument

    def load_file(self, path: str, instrument: Optional[str] = None) -> None:
        """Load ``path`` as a new dataset, make it active, and (re)build tabs."""
        obj, instrument = self._load_obj(path, instrument)
        if obj is None:
            return

        ds = Dataset(obj=obj, source_path=path, instrument_key=instrument)
        self.project.add_dataset(ds)
        self.project.set_active(ds.id)

        self.reload_btn.setEnabled(True)
        self._set_adjust_enabled(True)
        self._sync_crop_fields()
        self._build_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    # -- dataset / sidebar management --------------------------------------
    def set_active_dataset(self, ds_id: int) -> None:
        """Switch which dataset the single-view tabs follow."""
        if ds_id == self.project.active_id or self.project.get(ds_id) is None:
            return
        self.project.set_active(ds_id)
        self.reload_btn.setEnabled(bool(self.source_path))
        self._sync_crop_fields()
        self._ensure_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def _remove_dataset(self, ds_id: int) -> None:
        self.project.remove_dataset(ds_id)
        if self.project.active is None:
            # Nothing left: detach the shared controls before clearing the tabs
            # (otherwise they would be destroyed with the old Time series tab).
            self.adjust_box.setParent(None)
            self.tabs.clear()
            self._tabs = []
            self._tab_sig = None
            self.reload_btn.setEnabled(False)
            self._set_adjust_enabled(False)
            self._set_2d_controls_enabled(False)
            self.info.setText("No file loaded. Use 'Open file…' to begin.")
        else:
            self._sync_crop_fields()
            self._build_tabs()
            self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def _rename_dataset(self, ds_id: int) -> None:
        ds = self.project.get(ds_id)
        if ds is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename dataset", "Dataset name:", text=ds.label
        )
        if ok and name.strip():
            ds.label = name.strip()
            self._refresh_sidebar()
            self._sync_header()

    def _add_derived_dataset(
        self, obj, instrument_key: str, label: str, remove_ids=()
    ) -> Dataset:
        """Add a combined/derived dataset, optionally replacing some sources."""
        for rid in remove_ids:
            d = self.project.get(rid)
            if d is not None:
                self.project.datasets.remove(d)
        ds = Dataset(
            obj=obj, source_path=None, instrument_key=instrument_key, label=label
        )
        self.project.datasets.append(ds)
        self.project._apply_activities(ds)  # project the shared tasks onto it
        self.project.active_id = ds.id
        # Derived datasets have no source file, so Reload does not apply.
        self.reload_btn.setEnabled(False)
        self._set_adjust_enabled(True)
        self._sync_crop_fields()
        self._build_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()
        return ds

    def _join_same_instrument(self, ds_id: int) -> None:
        """Concatenate all datasets sharing the selected one's instrument+serial."""
        ds = self.project.get(ds_id)
        if ds is None:
            return
        group = [
            d
            for d in self.project.datasets
            if d.instrument_key == ds.instrument_key
            and str(d.serial_number) == str(ds.serial_number)
        ]
        if len(group) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Join same instrument",
                "Need at least two datasets with the same instrument and serial "
                f"number to join.\n\nOnly one '{ds.instrument_key}' "
                f"(serial {ds.serial_number}) dataset is loaded.",
            )
            return
        names = "\n  • ".join(d.label for d in group)
        ans = QtWidgets.QMessageBox.question(
            self,
            "Join same instrument",
            f"Combine these {len(group)} '{ds.instrument_key}' datasets "
            f"(serial {ds.serial_number}) into one continuous dataset?\n\n"
            f"  • {names}\n\nThe originals will be replaced by the combined "
            "dataset.",
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        try:
            combined = combine_measurements([d.obj for d in group])
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Join failed", traceback.format_exc(limit=2)
            )
            return
        self._add_derived_dataset(
            combined,
            ds.instrument_key,
            f"{ds.instrument_key} (combined)",
            remove_ids=[d.id for d in group],
        )

    def _combine_ns_ops(self) -> None:
        """Combine a NanoScan + an OPS dataset into one merged distribution."""
        twod = [d for d in self.project.datasets if helpers.is_2d(d.obj)]
        if len(twod) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Combine NS + OPS",
                "Load at least two size-resolved (2D) datasets first "
                "(e.g. a NanoScan and an OPS).",
            )
            return
        dlg = _CombineNSOPSDialog(self, twod)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        ns_ds, ops_ds, match = dlg.result()
        if ns_ds is ops_ds:
            QtWidgets.QMessageBox.warning(
                self, "Combine NS + OPS", "Pick two different datasets."
            )
            return
        try:
            combined = Combine_NS_OPS(ns_ds.obj, ops_ds.obj, match=match)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Combine failed", traceback.format_exc(limit=2)
            )
            return
        # NS+OPS keeps the originals (they remain useful on their own).
        self._add_derived_dataset(combined, "NS_OPS", "NS + OPS (combined)")

    def _refresh_sidebar(self) -> None:
        self.sidebar.set_datasets(self.project.datasets, self.project.active_id)

    def _ensure_tabs(self) -> None:
        """Rebuild tabs only when the active dataset's shape (1D/2D) changes."""
        sig = "2d" if (self.obj is not None and helpers.is_2d(self.obj)) else "1d"
        if sig != self._tab_sig:
            self._build_tabs()

    # -- project save / load ----------------------------------------------
    def _new_project(self) -> None:
        """Discard the current project and start an empty one."""
        self.adjust_box.setParent(None)
        self.tabs.clear()
        self._tabs = []
        self._tab_sig = None
        self.project = Project()
        self._project_path = None
        self.reload_btn.setEnabled(False)
        self._set_adjust_enabled(False)
        self._set_2d_controls_enabled(False)
        self.info.setText("No file loaded. Use 'Open file…' to begin.")
        self._refresh_sidebar()
        self._update_title()

    def _open_project(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open project folder"
        )
        if not folder:
            return
        try:
            project, mode = load_project(folder)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Could not open project", traceback.format_exc(limit=2)
            )
            return
        self.project = project
        self._project_path = folder
        if mode in ("dark", "light") and mode != self._theme:
            self.set_theme(mode)
        self._apply_loaded_project()

    def _apply_loaded_project(self) -> None:
        """Rebuild the UI to reflect ``self.project`` (after open/load)."""
        has = self.project.active is not None
        self.reload_btn.setEnabled(has)
        self._set_adjust_enabled(has)
        if has:
            self._sync_crop_fields()
            self._build_tabs()
            self.refresh_all(reset_view=True)
        else:
            self.adjust_box.setParent(None)
            self.tabs.clear()
            self._tabs = []
            self._tab_sig = None
            self.info.setText("No file loaded. Use 'Open file…' to begin.")
        self._refresh_sidebar()
        self._update_title()

    def _save_project(self) -> None:
        if not self.project.datasets:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save", "Load at least one dataset first."
            )
            return
        if self._project_path:
            self._write_project(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self) -> None:
        if not self.project.datasets:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save", "Load at least one dataset first."
            )
            return
        # Step 1: choose the location (parent folder).
        location = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose where to create the project folder"
        )
        if not location:
            return
        # Step 2: specify the project name -> a new sub-folder is created.
        default = self.project.name if self.project.name != "Untitled project" else ""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Project name", "Project folder name:", text=default
        )
        if not ok or not name.strip():
            return
        safe = "".join(c for c in name.strip() if c not in '\\/:*?"<>|').strip()
        if not safe:
            QtWidgets.QMessageBox.warning(
                self, "Invalid name", "Please enter a valid folder name."
            )
            return
        target = os.path.join(location, safe)
        if os.path.isdir(target) and os.listdir(target):
            ans = QtWidgets.QMessageBox.question(
                self,
                "Folder exists",
                f"'{safe}' already exists and is not empty.\nOverwrite its "
                "project contents?",
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
        self.project.name = safe
        self._write_project(target)

    def _write_project(self, folder: str) -> None:
        try:
            save_project(self.project, folder, theme=self._theme)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", traceback.format_exc(limit=2)
            )
            return
        self._project_path = folder
        self.project.name = (
            os.path.basename(os.path.normpath(folder)) or self.project.name
        )
        self._update_title()
        self.statusBar().showMessage(f"Project saved to: {folder}", 5000)

    def _update_title(self) -> None:
        suffix = f" — {self._project_path}" if self._project_path else ""
        self.setWindowTitle(f"aerosoltools viewer{suffix}")

    # -- theme -------------------------------------------------------------
    def set_theme(self, mode: str) -> None:
        """Switch between the 'dark' and 'light' themes at runtime."""
        if mode not in ("dark", "light"):
            return
        self._theme = mode
        app = QtWidgets.QApplication.instance()
        if app is not None:
            theme.apply_qt_theme(app, mode)
        theme.apply_mpl_theme(mode)
        # Update existing figures' face colours, then redraw their contents so
        # the line colours pick up the new prop_cycle.
        for tab in self._tabs:
            fig = getattr(tab, "figure", None)
            if fig is None:
                continue
            fig.set_facecolor(theme.fig_facecolor())
            for ax in fig.axes:
                ax.set_facecolor(theme.axes_facecolor())
        if hasattr(self, "_topbar_shadow"):
            self._topbar_shadow.setColor(QtGui.QColor(*theme.shadow_rgba()))
        self.refresh_all(reset_view=False)
        if hasattr(self, "_theme_group"):  # keep the View-menu radio in sync
            for act in self._theme_group.actions():
                act.setChecked(act.text().lower() == mode)

    # -- time shift --------------------------------------------------------
    def _apply_timeshift(self) -> None:
        """Permanently shift the active dataset's time axis."""
        ds = self.project.active
        if ds is None:
            return
        val = float(self.shift_value.value())
        if val == 0.0:
            return
        kw = {"seconds": 0.0, "minutes": 0.0, "hours": 0.0}
        kw[self.shift_unit.currentText()] = val
        try:
            ds.obj.timeshift(**kw)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Time shift failed", traceback.format_exc(limit=1)
            )
            return
        # Shared tasks keep their absolute times: re-project them onto the
        # shifted axis so this dataset's masks/summaries reflect the new timing.
        self.project._apply_activities(ds)
        self._sync_crop_fields()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    # -- tab management ----------------------------------------------------
    def _build_tabs(self) -> None:
        # Detach the shared adjustments box before clearing the tabs, so
        # deleting the old Time series tab does not destroy it.
        self.adjust_box.setParent(None)

        self.tabs.clear()
        self._tabs = []

        raw = RawDataTab(self)
        ts = TimeSeriesTab(self)
        ts.attach_adjust_controls(self.adjust_box)
        self.tabs.addTab(raw, "Raw data")
        self.tabs.addTab(ts, "Time series")
        self._tabs += [raw, ts]

        if helpers.is_2d(self.obj):
            psd = PSDTab(self)
            heat = HeatmapTab(self)
            pm = PMBandsTab(self)
            self.tabs.addTab(psd, "PSD")
            self.tabs.addTab(heat, "2D heatmap")
            self.tabs.addTab(pm, "PM bands")
            self._tabs += [psd, heat, pm]

        # Summary is intentionally one of the last tabs, so it appears after the
        # raw/plot views rather than before anything has been marked.
        summ = SummaryTab(self)
        self.tabs.addTab(summ, "Summary")
        self._tabs.append(summ)

        # Overlay is a multi-dataset comparison tab (reads all datasets).
        overlay = OverlayTab(self)
        self.tabs.addTab(overlay, "Overlay")
        self._tabs.append(overlay)

        self._tab_sig = "2d" if helpers.is_2d(self.obj) else "1d"

    def refresh_all(self, reset_view: bool = False) -> None:
        """Update the info bar, dtype/density controls, and every tab.

        Args:
            reset_view: When True, plot tabs autoscale to the (possibly new)
                data range. When False, they preserve the current zoom/pan so
                that incremental actions (e.g. marking a task) don't snap the
                view back to the full range.
        """
        self._reset_view = reset_view
        self._sync_header()
        for tab in self._tabs:
            try:
                tab.refresh()
            except Exception:
                traceback.print_exc()

    def _sync_header(self) -> None:
        if self.obj is None:
            return
        is2d = helpers.is_2d(self.obj)
        self._set_2d_controls_enabled(is2d)

        dtype, unit = helpers.describe(self.obj)
        rows = self.obj.data.shape[0]
        ds = self.project.active
        n = len(self.project.datasets)
        pos = self.project.index_of(ds.id) + 1 if ds is not None else 0
        label = ds.label if ds is not None else ""
        self.info.setText(
            f"[{pos}/{n}] {label}  |  {self.source_instrument}  |  "
            f"{type(self.obj).__name__}  |  dtype = {dtype}  |  unit = {unit}  |  "
            f"{rows} time steps  |  file: {os.path.basename(self.source_path or '')}"
        )

        if is2d:
            base = dtype.split("/")[0]
            self.dtype_combo.blockSignals(True)
            di = self.dtype_combo.findText(base)
            if di >= 0:
                self.dtype_combo.setCurrentIndex(di)
            self.dtype_combo.blockSignals(False)

            self.density_spin.blockSignals(True)
            self.density_spin.setValue(float(self.obj.density))
            self.density_spin.blockSignals(False)

    # -- dtype / density handlers -----------------------------------------
    def _on_dtype_change(self) -> None:
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        target = self.dtype_combo.currentText()
        try:
            self.obj.dtype_converter(dtype=target)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Conversion failed", traceback.format_exc(limit=1)
            )
            return
        # Units changed: rescale the axes rather than keeping stale limits.
        self.refresh_all(reset_view=True)

    def _on_density_change(self) -> None:
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        try:
            self.obj.set_density(self.density_spin.value())
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "set_density failed", traceback.format_exc(limit=1)
            )
            return
        self.refresh_all(reset_view=True)

    # -- cropping ----------------------------------------------------------
    def _sync_crop_fields(self) -> None:
        """Set the crop pickers to the current data time range.

        The pickers are intentionally left *unconstrained* (no min/max range).
        Enforcing a range makes the fields awkward to edit, because typing an
        intermediate value outside the data span gets silently rejected.
        Out-of-range crop values are harmless (they simply keep all data), so
        validation happens on Apply instead.
        """
        if self.obj is None or len(self.obj.time) == 0:
            return
        tmin = pd.Timestamp(self.obj.time.min()).to_pydatetime()
        tmax = pd.Timestamp(self.obj.time.max()).to_pydatetime()
        for widget, value in ((self.crop_start, tmin), (self.crop_end, tmax)):
            widget.blockSignals(True)
            widget.setDateTime(QtCore.QDateTime(value))
            widget.blockSignals(False)

    def _do_crop(self, start, end) -> None:
        """Crop the working object to ``[start, end]`` and refresh."""
        if self.obj is None:
            return
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if end <= start:
            QtWidgets.QMessageBox.warning(
                self, "Invalid range", "The end time must be after the start time."
            )
            return
        try:
            self.obj.timecrop(start=start, end=end, focus=True)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Crop failed", traceback.format_exc(limit=1)
            )
            return
        if len(self.obj.time) == 0:
            QtWidgets.QMessageBox.warning(
                self, "Empty result", "Cropping removed all data; reload to recover."
            )
        self._sync_crop_fields()
        self.refresh_all(reset_view=True)

    def _apply_crop(self) -> None:
        self._do_crop(
            self.crop_start.dateTime().toPyDateTime(),
            self.crop_end.dateTime().toPyDateTime(),
        )

    def _crop_to_view(self) -> None:
        """Crop to the time window shown on the currently active time-based plot.

        Uses the active tab's x-limits if it exposes a time axis (Time series,
        2D heatmap, PM bands); otherwise falls back to the Time series tab.
        """
        import matplotlib.dates as mdates

        xlim = None
        active = self.tabs.currentWidget()
        if hasattr(active, "current_time_xlim"):
            xlim = active.current_time_xlim()
        if xlim is None:
            ts_tab = next((t for t in self._tabs if isinstance(t, TimeSeriesTab)), None)
            if ts_tab is not None:
                xlim = ts_tab.current_time_xlim()
        if xlim is None:
            QtWidgets.QMessageBox.information(
                self,
                "No time view",
                "Open a time-based plot (Time series, 2D heatmap, or PM bands) "
                "and zoom to the window you want before cropping to view.",
            )
            return

        start = pd.Timestamp(mdates.num2date(xlim[0])).tz_localize(None)
        end = pd.Timestamp(mdates.num2date(xlim[1])).tz_localize(None)
        self._do_crop(start, end)

    # -- smoothing / resampling -------------------------------------------
    def _apply_smoothing(self) -> None:
        if self.obj is None:
            return
        try:
            self.obj.timesmooth(
                window=int(self.smooth_window.value()),
                method=self.smooth_method.currentText(),
            )
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Smoothing failed", traceback.format_exc(limit=1)
            )
            return
        self.refresh_all(reset_view=True)

    def _apply_resampling(self) -> None:
        if self.obj is None:
            return
        freq = self.resample_freq.text().strip()
        if not freq:
            QtWidgets.QMessageBox.warning(
                self, "Resampling", "Enter a target time step, e.g. 30s, 1min, 5min."
            )
            return
        try:
            self.obj.timerebin(freq=freq, method=self.resample_method.currentText())
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "Resampling failed", traceback.format_exc(limit=1)
            )
            return
        self._sync_crop_fields()
        self.refresh_all(reset_view=True)
