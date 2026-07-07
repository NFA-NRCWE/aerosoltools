"""Main application window for the aerosoltools GUI."""

from __future__ import annotations

import inspect
import os
import traceback
from typing import List, Optional

from ..aerosol3d import Aerosol3d
from ..utility import combine_measurements, combine_size_ranges
from . import helpers, theme
from .adjustments import AdjustmentsBox
from .assets import icon_path
from .loaders import LOADERS, UnrecognizedInstrumentError, require_identified_instrument
from .project import Dataset, Project
from .projectio import load_project, save_project
from .qt import QtCore, QtGui, QtWidgets
from .sidebar import DatasetSidebar
from .tabs import (
    AeroOpticalTab,
    CorrelationTab,
    DecayTab,
    HeatmapTab,
    OverlayTab,
    PMBandsTab,
    PSDTab,
    RawDataTab,
    SummaryTab,
    TimeSeriesTab,
)
from .widgets import CombineInstrumentsDialog, KeyboardShortcutsDialog, SlackTabBar


class MainWindow(QtWidgets.QMainWindow):
    """Top-level window: a load bar, dtype/density controls, and data tabs."""

    def __init__(self, path: Optional[str] = None, instrument: Optional[str] = None):
        """Build the window and optionally load a file on startup.

        Args:
            path: Optional data file to open immediately.
            instrument: Optional loader name; guessed from the file name when None.
        """
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
        # Which axis of a correlated APS (Aerosol3d) the tabs currently show:
        # "aerodynamic" (default) or "optical".
        self._active_axis: str = "aerodynamic"
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
        """Active dataset's aerosol object for the current axis, or None.

        For a correlated APS (:class:`Aerosol3d`) the "Show axis" selector
        chooses whether the tabs see the aerodynamic (default) or the optical
        distribution; both are plain 2D objects, so every 2D tab works on
        either. Structural edits still go through :attr:`active_obj` (the
        parent), which keeps both axes in sync.
        """
        ds = self.project.active
        if ds is None:
            return None
        o = ds.obj
        axis = getattr(self, "_active_axis", "aerodynamic")
        if axis == "optical" and isinstance(o, Aerosol3d) and o.is_correlated:
            return o.axis_view("optical")
        return o

    @property
    def active_obj(self):
        """The active dataset's *parent* object (the aerodynamic axis for 3D).

        Structural operations (crop / rebin / smooth / activities) target this
        so a 3D dataset's two axes stay aligned via ``Aerosol3d``'s time-op
        overrides.
        """
        ds = self.project.active
        return ds.obj if ds is not None else None

    @property
    def source_path(self) -> Optional[str]:
        """Active dataset's source file path, or None."""
        ds = self.project.active
        return ds.source_path if ds is not None else None

    @property
    def source_instrument(self) -> Optional[str]:
        """Active dataset's instrument key, or None."""
        ds = self.project.active
        return ds.instrument_key if ds is not None else None

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        """Assemble the menu, sidebar, top bar, tabs and status bar."""
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
        self.adjust_box = AdjustmentsBox(self)

        # Tabs sit directly below the top bar.
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabBar(SlackTabBar())
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
        self.info = QtWidgets.QLabel("No data loaded. Use 'Import data…' to begin.")
        self.statusBar().addWidget(self.info)

        self._set_2d_controls_enabled(False)
        self.adjust_box.set_enabled(False)

    def _build_top_bar(self) -> QtWidgets.QWidget:
        """Build the raised top control bar (open / instrument / dtype / density)."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("TopBar")
        bar = QtWidgets.QHBoxLayout(frame)
        bar.setContentsMargins(14, 10, 14, 10)
        bar.setSpacing(8)

        open_btn = QtWidgets.QPushButton("Import data…")
        open_btn.setObjectName("primary")
        open_btn.setToolTip(
            "Import one or more instrument data files as datasets (Ctrl+O)."
        )
        open_btn.clicked.connect(self._open_dialog)
        bar.addWidget(open_btn)

        bar.addWidget(QtWidgets.QLabel("Instrument:"))
        self.instrument_combo = QtWidgets.QComboBox()
        self.instrument_combo.addItems(list(LOADERS.keys()))
        self.instrument_combo.setToolTip(
            "Loader to use for files whose instrument cannot be guessed from the "
            "file name."
        )
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
        self.dtype_combo.setToolTip(
            "Distribution basis for size-resolved data: number (dN), mass (dM), "
            "surface (dS) or volume (dV)."
        )
        self.dtype_combo.currentIndexChanged.connect(self._on_dtype_change)
        bar.addWidget(self.dtype_combo)

        self.density_label = QtWidgets.QLabel("density (g/cm³):")
        bar.addWidget(self.density_label)
        self.density_spin = QtWidgets.QDoubleSpinBox()
        self.density_spin.setRange(0.1, 25.0)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setValue(1.0)
        self.density_spin.setToolTip(
            "Particle density (g/cm³) used when converting to mass-based metrics."
        )
        self.density_spin.editingFinished.connect(self._on_density_change)
        bar.addWidget(self.density_spin)

        # Aerodynamic / optical axis selector — only meaningful for a correlated
        # APS (Aerosol3d); hidden otherwise.
        self.axis_label = QtWidgets.QLabel("show axis:")
        bar.addWidget(self.axis_label)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem("Aerodynamic", "aerodynamic")
        self.axis_combo.addItem("Optical", "optical")
        self.axis_combo.setToolTip(
            "For a correlated APS dataset, choose which size axis the tabs show "
            "and analyse — aerodynamic or optical. Both behave as normal 2D "
            "size distributions (heatmap, PSD, decay fit, …)."
        )
        self.axis_combo.currentIndexChanged.connect(self._on_axis_change)
        bar.addWidget(self.axis_combo)

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
        """Build the top menu bar (File / View / Help) with shortcuts + tooltips.

        Every shortcut-bearing action is created through :meth:`_menu_action`,
        which records the binding in ``self._shortcut_help`` so the Help →
        Keyboard shortcuts dialog stays in sync with the real menu.
        """
        # (keys, description) pairs collected as shortcuts are assigned, used to
        # populate the Keyboard-shortcuts help dialog.
        self._shortcut_help: List[tuple] = []
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.setToolTipsVisible(True)
        self._menu_action(
            file_menu,
            "New project",
            self._new_project,
            "Ctrl+N",
            "Discard the current project and start an empty one.",
        )
        self._menu_action(
            file_menu,
            "Open project…",
            self._open_project,
            "Ctrl+Shift+O",
            "Open a previously saved project folder.",
        )
        file_menu.addSeparator()
        self._menu_action(
            file_menu,
            "Save project",
            self._save_project,
            "Ctrl+S",
            "Save the project to its folder (prompts for one the first time).",
        )
        self._menu_action(
            file_menu,
            "Save project as…",
            self._save_project_as,
            "Ctrl+Shift+S",
            "Save the project to a new folder.",
        )
        file_menu.addSeparator()
        self._menu_action(
            file_menu,
            "Import data…",
            self._open_dialog,
            "Ctrl+O",
            "Import one or more instrument data files as datasets.",
        )
        file_menu.addSeparator()
        self._menu_action(
            file_menu, "Exit", self.close, "Ctrl+Q", "Quit the application."
        )

        view_menu = mb.addMenu("&View")
        view_menu.setToolTipsVisible(True)
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
        self._dock_action.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        self._dock_action.setToolTip("Show or hide the datasets sidebar.")
        self._dock_action.triggered.connect(lambda c: self.datasets_dock.setVisible(c))
        view_menu.addAction(self._dock_action)
        self._shortcut_help.append(("Ctrl+D", "Show / hide the datasets panel"))

        help_menu = mb.addMenu("&Help")
        help_menu.setToolTipsVisible(True)
        self._menu_action(
            help_menu,
            "Keyboard shortcuts",
            self._show_shortcuts,
            "F1",
            "List the available keyboard shortcuts.",
        )
        help_menu.addAction("About", self._about)

    def _menu_action(self, menu, label, handler, shortcut=None, tip=None):
        """Add an action to ``menu``, wiring an optional shortcut and tooltip.

        Args:
            menu: The :class:`QtWidgets.QMenu` to add the action to.
            label: Action text.
            handler: Slot called when the action triggers.
            shortcut: Optional shortcut string (e.g. ``"Ctrl+O"``). When given it
                is also recorded in ``self._shortcut_help`` for the help dialog.
            tip: Optional tooltip / status-tip text.

        Returns:
            QtWidgets.QAction: The created action.
        """
        act = menu.addAction(label, handler)
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
            self._shortcut_help.append((shortcut, label.rstrip("…")))
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        return act

    def _show_shortcuts(self) -> None:
        """Open the read-only keyboard-shortcuts reference dialog."""
        KeyboardShortcutsDialog(self, self._shortcut_help).exec_()

    def _about(self) -> None:
        """Show the small 'About' message box."""
        QtWidgets.QMessageBox.about(
            self,
            "About aerosoltools viewer",
            "aerosoltools viewer\n\nInteractive viewer for aerosol instrument "
            "data.\nLoad multiple datasets, mark shared tasks, and explore.",
        )

    def _set_2d_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable the dtype + density controls (size-resolved data only)."""
        for w in (
            self.dtype_label,
            self.dtype_combo,
            self.density_label,
            self.density_spin,
        ):
            w.setEnabled(enabled)

    # -- loading -----------------------------------------------------------
    def _open_dialog(self) -> None:
        """Prompt for one or more data files and import them as datasets.

        Uses a multi-select file dialog so several files can be imported in one
        step; each file's instrument is guessed individually from its name.
        """
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import aerosol data file(s)",
            "",
            "Data files (*.txt *.csv *.dat *.xlsx *.xls);;All files (*)",
        )
        if paths:
            self.load_files(paths)

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
        self.adjust_box.sync_crop_fields()
        self._build_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def _load_obj(self, path: str, instrument: Optional[str] = None):
        """Resolve a loader and load ``path``; return ``(obj, instrument)``.

        Returns ``(None, None)`` (after showing a message) if the instrument is
        unknown or the file fails to load.
        """
        if instrument is None:
            try:
                instrument = require_identified_instrument(path)
            except UnrecognizedInstrumentError as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Unknown instrument",
                    str(e),
                )
                return None, None

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

    def _ingest_file(self, path: str, instrument: Optional[str] = None):
        """Load one file into a new dataset *without* touching the UI.

        Args:
            path: File to load.
            instrument: Loader name, or None to guess from the file name.

        Returns:
            Dataset | None: The added dataset, or None if loading failed.
        """
        obj, instrument = self._load_obj(path, instrument)
        if obj is None:
            return None
        ds = Dataset(obj=obj, source_path=path, instrument_key=instrument)
        self.project.add_dataset(ds)
        return ds

    def _finalize_after_load(self) -> None:
        """Refresh window state after one or more datasets were added."""
        self.reload_btn.setEnabled(bool(self.source_path))
        self.adjust_box.set_enabled(True)
        self.adjust_box.sync_crop_fields()
        self._build_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def load_file(self, path: str, instrument: Optional[str] = None) -> None:
        """Load ``path`` as a new dataset, make it active, and (re)build tabs."""
        ds = self._ingest_file(path, instrument)
        if ds is None:
            return
        self.project.set_active(ds.id)
        self._finalize_after_load()

    def load_files(self, paths, instrument: Optional[str] = None) -> None:
        """Import several files at once, refreshing the UI only once at the end.

        Args:
            paths: Iterable of file paths to import.
            instrument: Optional loader name to force for *every* file. When None
                (the usual case), each file's instrument is guessed from its name,
                falling back to the instrument selector.
        """
        last = None
        for path in paths:
            # If an instrument was explicitly provided, force that loader.
            # Otherwise, let _load_obj identify the file by:
            # content sniffer -> filename convention -> user-facing error.
            ds = self._ingest_file(path, instrument)
            if ds is not None:
                last = ds
        if last is None:  # every file failed to load
            return
        self.project.set_active(last.id)
        self._finalize_after_load()

    # -- dataset / sidebar management --------------------------------------
    def set_active_dataset(self, ds_id: int) -> None:
        """Switch which dataset the single-view tabs follow."""
        if ds_id == self.project.active_id or self.project.get(ds_id) is None:
            return
        self.project.set_active(ds_id)
        self.reload_btn.setEnabled(bool(self.source_path))
        self.adjust_box.sync_crop_fields()
        self._ensure_tabs()
        self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def _remove_dataset(self, ds_id: int) -> None:
        """Remove a dataset, tearing the tabs down if it was the last one."""
        self.project.remove_dataset(ds_id)
        if self.project.active is None:
            # Nothing left: detach the shared controls before clearing the tabs
            # (otherwise they would be destroyed with the old Time series tab).
            self.adjust_box.setParent(None)
            self.tabs.clear()
            self._tabs = []
            self._tab_sig = None
            self.reload_btn.setEnabled(False)
            self.adjust_box.set_enabled(False)
            self._set_2d_controls_enabled(False)
            self.info.setText("No data loaded. Use 'Import data…' to begin.")
        else:
            self.adjust_box.sync_crop_fields()
            self._build_tabs()
            self.refresh_all(reset_view=True)
        self._refresh_sidebar()

    def _rename_dataset(self, ds_id: int) -> None:
        """Prompt for and apply a new label for a dataset."""
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
        self, obj, instrument_key: str, label: str, remove_ids=(), source_files=()
    ) -> Dataset:
        """Add a combined/derived dataset, optionally replacing some sources.

        Args:
            obj: The derived aerosol object.
            instrument_key: Instrument key for the result.
            label: Display label.
            remove_ids: Dataset ids to remove (e.g. the originals a join replaces).
            source_files: Raw files behind the result (its constituents' files),
                recorded so 'Save project' still archives them even though the
                derived dataset itself has no single source file.

        Returns:
            Dataset: The newly added dataset.
        """
        for rid in remove_ids:
            d = self.project.get(rid)
            if d is not None:
                self.project.datasets.remove(d)
        ds = Dataset(
            obj=obj, source_path=None, instrument_key=instrument_key, label=label
        )
        # De-duplicate while preserving order so the raw archive has each file once.
        ds.contributing_files = list(dict.fromkeys(source_files))
        self.project.datasets.append(ds)
        self.project._apply_activities(ds)  # project the shared tasks onto it
        self.project.active_id = ds.id
        # Derived datasets have no source file, so Reload does not apply.
        self.reload_btn.setEnabled(False)
        self.adjust_box.set_enabled(True)
        self.adjust_box.sync_crop_fields()
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
        # Carry every constituent's raw files onto the combined dataset so the
        # originals are still archived on save even though they're removed here.
        raw_files = [f for d in group for f in d.contributing_files]
        self._add_derived_dataset(
            combined,
            ds.instrument_key,
            f"{ds.instrument_key} (combined)",
            remove_ids=[d.id for d in group],
            source_files=raw_files,
        )

    def _combine_ns_ops(self) -> None:
        """Stitch two range-extending size instruments at a chosen crossover."""
        twod = [d for d in self.project.datasets if helpers.is_2d(d.obj)]
        if len(twod) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Combine size ranges",
                "Load at least two size-resolved (2D) datasets first "
                "(e.g. a NanoScan/FMPS and an OPS/APS).",
            )
            return
        dlg = CombineInstrumentsDialog(self, twod)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        ds_a, ds_b, crossover, match = dlg.result()
        if ds_a is ds_b:
            QtWidgets.QMessageBox.warning(
                self, "Combine size ranges", "Pick two different datasets."
            )
            return
        try:
            combined = combine_size_ranges(
                ds_a.obj, ds_b.obj, crossover=crossover, match=match
            )
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Combine failed", traceback.format_exc(limit=2)
            )
            return
        # The originals remain useful on their own; still record their raw files
        # on the combined dataset for completeness.
        self._add_derived_dataset(
            combined,
            "Combined",
            f"{ds_a.instrument} + {ds_b.instrument} (combined)",
            source_files=ds_a.contributing_files + ds_b.contributing_files,
        )

    def _refresh_sidebar(self) -> None:
        """Mirror the current datasets and active id into the sidebar."""
        self.sidebar.set_datasets(self.project.datasets, self.project.active_id)

    def _shape_sig(self) -> str:
        """Tab-set signature for the active dataset ("1d" / "2d" / "3d")."""
        if isinstance(self.active_obj, Aerosol3d) and self.active_obj.is_correlated:
            return "3d"
        return "2d" if (self.obj is not None and helpers.is_2d(self.obj)) else "1d"

    def _ensure_tabs(self) -> None:
        """Rebuild tabs only when the active dataset's shape (1D/2D/3D) changes."""
        if self._shape_sig() != self._tab_sig:
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
        self.adjust_box.set_enabled(False)
        self._set_2d_controls_enabled(False)
        self.info.setText("No data loaded. Use 'Import data…' to begin.")
        self._refresh_sidebar()
        self._update_title()

    def _open_project(self) -> None:
        """Prompt for a saved project folder and load it."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project folder")
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
        self.adjust_box.set_enabled(has)
        if has:
            self.adjust_box.sync_crop_fields()
            self._build_tabs()
            self.refresh_all(reset_view=True)
        else:
            self.adjust_box.setParent(None)
            self.tabs.clear()
            self._tabs = []
            self._tab_sig = None
            self.info.setText("No data loaded. Use 'Import data…' to begin.")
        self._refresh_sidebar()
        self._update_title()

    def _save_project(self) -> None:
        """Save to the project's folder, or prompt for one if it was never saved."""
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
        """Prompt for a parent folder + name and save the project there."""
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
        """Write the project to ``folder`` and update the title and status bar."""
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
        """Set the window title to reflect the saved-project path."""
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

    # -- tab management ----------------------------------------------------
    def _build_tabs(self) -> None:
        """(Re)create the tab set appropriate to the active dataset's shape.

        The PSD and Summary tabs read the *whole project* (one or many datasets),
        so they replace the former single-view PSD/Summary tabs and are always
        shown. The 2D heatmap and PM-bands tabs stay single-view (they follow the
        active dataset) and so are only built for a size-resolved active dataset.
        """
        # Detach the shared adjustments box before clearing the tabs, so
        # deleting the old Time series tab does not destroy it.
        self.adjust_box.setParent(None)

        self.tabs.clear()
        self._tabs = []

        raw = RawDataTab(self)
        ts = TimeSeriesTab(self)
        ts.attach_adjust_controls(self.adjust_box)
        decay = DecayTab(self)
        self.tabs.addTab(raw, "Raw data")
        self.tabs.addTab(ts, "Time series")
        self.tabs.addTab(decay, "Decay / Source")
        self._tabs += [raw, ts, decay]

        # Single-view 2D plots that follow the active dataset.
        if helpers.is_2d(self.obj):
            heat = HeatmapTab(self)
            pm = PMBandsTab(self)
            self.tabs.addTab(heat, "2D heatmap")
            self.tabs.addTab(pm, "PM bands")
            self._tabs += [heat, pm]

        # Correlated APS (Aerosol3d): the aerodynamic↔optical comparison pane.
        if isinstance(self.active_obj, Aerosol3d) and self.active_obj.is_correlated:
            aero_opt = AeroOpticalTab(self)
            self.tabs.addTab(aero_opt, "Aero ↔ Optical")
            self._tabs.append(aero_opt)

        # Project-level tabs (work for a single dataset or compare several):
        # PSD + Summary subsume the old single-view tabs; Overlay + Correlation
        # are multi-dataset comparisons. All are always shown.
        psd = PSDTab(self)
        summ = SummaryTab(self)
        overlay = OverlayTab(self)
        correlation = CorrelationTab(self)
        self.tabs.addTab(psd, "PSD")
        self.tabs.addTab(summ, "Summary")
        self.tabs.addTab(overlay, "Overlay")
        self.tabs.addTab(correlation, "Correlation")
        self._tabs += [psd, summ, overlay, correlation]

        self._tab_sig = self._shape_sig()

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
        """Refresh the status line and dtype/density controls for the active dataset."""
        if self.obj is None:
            return
        is2d = helpers.is_2d(self.obj)
        self._set_2d_controls_enabled(is2d)

        # The axis selector only applies to a correlated APS (both size axes).
        is_3d = isinstance(self.active_obj, Aerosol3d) and self.active_obj.is_correlated
        self.axis_label.setVisible(is_3d)
        self.axis_combo.setVisible(is_3d)
        if not is_3d and self._active_axis != "aerodynamic":
            self._active_axis = "aerodynamic"
        self.axis_combo.blockSignals(True)
        self.axis_combo.setCurrentIndex(1 if self._active_axis == "optical" else 0)
        self.axis_combo.blockSignals(False)

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
            base = helpers.base_dtype(dtype)
            self.dtype_combo.blockSignals(True)
            di = self.dtype_combo.findText(base)
            if di >= 0:
                self.dtype_combo.setCurrentIndex(di)
            self.dtype_combo.blockSignals(False)

            self.density_spin.blockSignals(True)
            self.density_spin.setValue(float(self.obj.density))
            self.density_spin.blockSignals(False)

    def _on_axis_change(self) -> None:
        """Switch the tabs between the aerodynamic and optical axis (3D data)."""
        self._active_axis = self.axis_combo.currentData() or "aerodynamic"
        self.refresh_all(reset_view=True)

    # -- dtype / density handlers -----------------------------------------
    def _on_dtype_change(self) -> None:
        """Convert the active 2D dataset to the selected distribution basis."""
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

    @staticmethod
    def _is_elpi(obj) -> bool:
        """True if ``obj`` is an ELPI dataset (whose sizes are density-dependent)."""
        return str(getattr(obj, "metadata", {}).get("instrument", "")).upper() == "ELPI"

    def _on_density_change(self) -> None:
        """Apply the selected density to the active dataset and recalc ELPI data.

        The active 2D dataset always takes the new density. In addition, the
        project is scanned for ELPI datasets: because the ELPI reports
        density-dependent particle sizes, each is recalculated (diameters, and
        number for raw-current files) so it never shows fictitious diameters.
        """
        if self.obj is None or not helpers.is_2d(self.obj):
            return
        new_density = self.density_spin.value()

        # Targets: the active dataset plus every ELPI dataset in the project.
        targets = []
        active = self.project.active
        if active is not None:
            targets.append(active)
        for ds in self.project.datasets:
            if ds not in targets and self._is_elpi(ds.obj):
                targets.append(ds)

        n_elpi = 0
        try:
            for ds in targets:
                if not helpers.is_2d(ds.obj):
                    continue
                ds.obj.set_density(new_density)
                if self._is_elpi(ds.obj):
                    n_elpi += 1
        except Exception:
            QtWidgets.QMessageBox.warning(
                self, "set_density failed", traceback.format_exc(limit=1)
            )
            return
        self.refresh_all(reset_view=True)
        if n_elpi:
            self._warn_elpi_recalc(n_elpi)

    def _warn_elpi_recalc(self, n: int) -> None:
        """Non-blocking 'speech bubble' noting ELPI data were recalculated."""
        pos = self.density_spin.mapToGlobal(
            QtCore.QPoint(0, self.density_spin.height())
        )
        word = "dataset" if n == 1 else "datasets"
        QtWidgets.QToolTip.showText(
            pos,
            f"ELPI sizes are density-dependent — recalculated {n} ELPI {word} for "
            "the new density (particle diameters, and number for raw-current "
            "files).",
            self.density_spin,
        )
