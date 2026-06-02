"""Main application window for the aerosoltools GUI."""

from __future__ import annotations

import inspect
import os
import traceback
from typing import List, Optional

import pandas as pd

from . import helpers
from .assets import icon_path
from .loaders import LOADERS, guess_instrument
from .qt import QtCore, QtGui, QtWidgets
from .tabs import (
    HeatmapTab,
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


class MainWindow(QtWidgets.QMainWindow):
    """Top-level window: a load bar, dtype/density controls, and data tabs."""

    def __init__(self, path: Optional[str] = None, instrument: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("aerosoltools viewer")
        self.resize(1200, 800)
        _icon = icon_path()
        if _icon:
            self.setWindowIcon(QtGui.QIcon(_icon))

        self.obj = None
        self.source_path: Optional[str] = None
        self.source_instrument: Optional[str] = None
        self._tabs: List = []
        # When True, plot tabs autoscale on the next refresh; when False they
        # preserve the user's current zoom/pan (e.g. after marking a task).
        self._reset_view: bool = True

        self._build_ui()

        if path:
            self.load_file(path, instrument)

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        bar = QtWidgets.QHBoxLayout()

        open_btn = QtWidgets.QPushButton("Open file…")
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
        layout.addLayout(bar)

        self.info = QtWidgets.QLabel("No file loaded. Use 'Open file…' to begin.")
        self.info.setStyleSheet("color: #5b6573; padding: 2px;")
        layout.addWidget(self.info)

        layout.addWidget(self._build_crop_bar())
        layout.addWidget(self._build_processing_bar())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabBar(_SlackTabBar())
        self.tabs.tabBar().setExpanding(False)
        self.tabs.setElideMode(QtCore.Qt.ElideNone)
        layout.addWidget(self.tabs, stretch=1)

        self._set_2d_controls_enabled(False)
        self._set_crop_enabled(False)
        self._set_processing_enabled(False)

    def _build_processing_bar(self) -> QtWidgets.QWidget:
        """Build smoothing and resampling (time-resolution) controls."""
        box = QtWidgets.QGroupBox("Smoothing && resampling")
        row = QtWidgets.QHBoxLayout(box)

        methods = ["mean", "median", "min", "max", "sum"]

        # --- smoothing (rolling window over samples) ---
        row.addWidget(QtWidgets.QLabel("Smooth window:"))
        self.smooth_window = QtWidgets.QSpinBox()
        self.smooth_window.setRange(2, 999)
        self.smooth_window.setValue(5)
        self.smooth_window.setToolTip("Rolling window size, in number of samples")
        row.addWidget(self.smooth_window)

        self.smooth_method = QtWidgets.QComboBox()
        self.smooth_method.addItems(methods)
        row.addWidget(self.smooth_method)

        self.smooth_btn = QtWidgets.QPushButton("Apply smoothing")
        self.smooth_btn.clicked.connect(self._apply_smoothing)
        row.addWidget(self.smooth_btn)

        row.addSpacing(24)

        # --- resampling (rebin to a coarser time step) ---
        row.addWidget(QtWidgets.QLabel("Resample to:"))
        self.resample_freq = QtWidgets.QLineEdit("1min")
        self.resample_freq.setFixedWidth(80)
        self.resample_freq.setToolTip(
            "Target time step as a pandas offset, e.g. 30s, 1min, 5min, 1H"
        )
        row.addWidget(self.resample_freq)

        self.resample_method = QtWidgets.QComboBox()
        self.resample_method.addItems(methods)
        row.addWidget(self.resample_method)

        self.resample_btn = QtWidgets.QPushButton("Apply resampling")
        self.resample_btn.clicked.connect(self._apply_resampling)
        row.addWidget(self.resample_btn)

        row.addStretch(1)
        hint = QtWidgets.QLabel("(use Reload to undo)")
        hint.setStyleSheet("color: #5b6573;")
        row.addWidget(hint)
        return box

    def _set_processing_enabled(self, enabled: bool) -> None:
        for w in (
            self.smooth_window,
            self.smooth_method,
            self.smooth_btn,
            self.resample_freq,
            self.resample_method,
            self.resample_btn,
        ):
            w.setEnabled(enabled)

    def _build_crop_bar(self) -> QtWidgets.QWidget:
        """Build the time-cropping controls (start/end pickers, apply, trim)."""
        box = QtWidgets.QGroupBox("Crop time range")
        row = QtWidgets.QHBoxLayout(box)

        fmt = "yyyy-MM-dd HH:mm:ss"
        row.addWidget(QtWidgets.QLabel("From:"))
        self.crop_start = QtWidgets.QDateTimeEdit()
        self.crop_start.setDisplayFormat(fmt)
        self.crop_start.setCalendarPopup(True)
        row.addWidget(self.crop_start)

        row.addWidget(QtWidgets.QLabel("To:"))
        self.crop_end = QtWidgets.QDateTimeEdit()
        self.crop_end.setDisplayFormat(fmt)
        self.crop_end.setCalendarPopup(True)
        row.addWidget(self.crop_end)

        self.crop_btn = QtWidgets.QPushButton("Apply crop")
        self.crop_btn.clicked.connect(self._apply_crop)
        row.addWidget(self.crop_btn)

        self.crop_view_btn = QtWidgets.QPushButton("Crop to current view")
        self.crop_view_btn.setToolTip(
            "Crop to the time window currently shown on the active plot "
            "(Time series, 2D heatmap, or PM bands)"
        )
        self.crop_view_btn.clicked.connect(self._crop_to_view)
        row.addWidget(self.crop_view_btn)

        row.addStretch(1)
        return box

    def _set_crop_enabled(self, enabled: bool) -> None:
        for w in (
            self.crop_start,
            self.crop_end,
            self.crop_btn,
            self.crop_view_btn,
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
        if self.source_path:
            self.load_file(self.source_path, self.source_instrument)

    def load_file(self, path: str, instrument: Optional[str] = None) -> None:
        """Load ``path`` using the named instrument loader and (re)build tabs."""
        if instrument is None:
            instrument = (
                guess_instrument(os.path.basename(path))
                or self.instrument_combo.currentText()
            )
        if instrument not in LOADERS:
            QtWidgets.QMessageBox.warning(
                self, "Unknown instrument", f"No loader registered for '{instrument}'."
            )
            return

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
            return

        self.obj = obj
        self.source_path = path
        self.source_instrument = instrument
        self.reload_btn.setEnabled(True)
        self._set_crop_enabled(True)
        self._set_processing_enabled(True)
        self._sync_crop_fields()
        self._build_tabs()
        self.refresh_all(reset_view=True)

    # -- tab management ----------------------------------------------------
    def _build_tabs(self) -> None:
        self.tabs.clear()
        self._tabs = []

        raw = RawDataTab(self)
        summ = SummaryTab(self)
        ts = TimeSeriesTab(self)
        self.tabs.addTab(raw, "Raw data")
        self.tabs.addTab(summ, "Summary")
        self.tabs.addTab(ts, "Time series")
        self._tabs += [raw, summ, ts]

        if helpers.is_2d(self.obj):
            psd = PSDTab(self)
            heat = HeatmapTab(self)
            pm = PMBandsTab(self)
            self.tabs.addTab(psd, "PSD")
            self.tabs.addTab(heat, "2D heatmap")
            self.tabs.addTab(pm, "PM bands")
            self._tabs += [psd, heat, pm]

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
        self.info.setText(
            f"{self.source_instrument}  |  {type(self.obj).__name__}  |  "
            f"dtype = {dtype}  |  unit = {unit}  |  {rows} time steps  |  "
            f"file: {os.path.basename(self.source_path or '')}"
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
