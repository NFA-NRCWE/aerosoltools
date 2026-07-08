"""The "Data adjustments" control box: crop / resample / smooth / time-shift.

The box is owned by :class:`~aerosoltools.gui.main_window.MainWindow` but embedded
into the Time series tab, so adjustments happen where the data is visible. It is a
self-contained widget that holds a back-reference to the window and acts on the
window's *active* dataset (``main.obj``), asking the window to refresh afterwards.

Keeping it here (rather than inline on ``MainWindow``) isolates all data-mutation
controls behind a small API: :meth:`set_enabled` and :meth:`sync_crop_fields`.
"""

from __future__ import annotations

import traceback

import matplotlib.dates as mdates
import pandas as pd

from .qt import QtCore, QtWidgets
from .tabs import TimeSeriesTab


class AdjustmentsBox(QtWidgets.QGroupBox):
    """Compact one-column box of data-adjustment operations bound to ``main``.

    Each operation gets its own full-width row (crop, resample, smooth, time
    shift) so action buttons always have room for their full label.
    """

    _METHODS = ["mean", "median", "min", "max", "sum"]

    def __init__(self, main):
        """Build the crop/resample/smooth/time-shift rows.

        Args:
            main: The owning :class:`MainWindow`; handlers act on its active dataset.
        """
        super().__init__("Data adjustments")
        self.main = main

        grid = QtWidgets.QVBoxLayout(self)
        grid.setSpacing(6)

        def _row(label: str) -> QtWidgets.QHBoxLayout:
            """Add a labelled full-width row to the box and return its layout."""
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
        self.resample_method.addItems(self._METHODS)
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
        self.smooth_method.addItems(self._METHODS)
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

        # --- extract / split ----------------------------------------------
        # The Time series tab's "Extract range" toggle is hosted here (a more
        # logical home than the activities panel): toggle it, then drag a window
        # on the plot to split the dataset or copy the window out. The button is
        # owned by the current Time series tab and (re)attached on each rebuild.
        ext = _row("Extract")
        self._extract_row = ext
        self._extract_btn = None
        ext_hint = QtWidgets.QLabel("toggle, then drag a window on the plot")
        ext_hint.setStyleSheet("color: palette(mid);")
        ext.addWidget(ext_hint)
        ext.addStretch(1)

        hint = QtWidgets.QLabel("Tip: use Reload to undo resample / smooth / shift.")
        hint.setStyleSheet("color: palette(mid);")
        grid.addWidget(hint)

        # Hug the content width so the box stays compact and its right-aligned
        # action buttons sit close to the settings (instead of far to the right),
        # leaving more horizontal room for the activities panel.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred
        )

    # -- public API --------------------------------------------------------
    def attach_extract_button(self, btn) -> None:
        """Embed the Time series tab's Extract-range toggle in the Extract row.

        The button is created and owned by the current Time series tab (which is
        rebuilt with the tab set), so any previously hosted button is removed
        first and the new one inserted just after the row's label.
        """
        old = self._extract_btn
        if old is not None and old is not btn:
            self._extract_row.removeWidget(old)
            old.setParent(None)
        self._extract_btn = btn
        self._extract_row.insertWidget(1, btn)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable every control (used when no dataset is loaded)."""
        for w in (
            self.crop_start,
            self.crop_end,
            self.crop_btn,
            self.crop_view_btn,
            self.resample_freq,
            self.resample_method,
            self.resample_btn,
            self.smooth_window,
            self.smooth_method,
            self.smooth_btn,
            self.shift_value,
            self.shift_unit,
            self.shift_btn,
        ):
            w.setEnabled(enabled)

    def sync_crop_fields(self) -> None:
        """Set the crop pickers to the active dataset's time range.

        The pickers are intentionally left *unconstrained* (no min/max range).
        Enforcing a range makes the fields awkward to edit, because typing an
        intermediate value outside the data span gets silently rejected.
        Out-of-range crop values are harmless (they simply keep all data), so
        validation happens on Apply instead.
        """
        obj = self.main.obj
        if obj is None or len(obj.time) == 0:
            return
        tmin = pd.Timestamp(obj.time.min()).to_pydatetime()
        tmax = pd.Timestamp(obj.time.max()).to_pydatetime()
        for widget, value in ((self.crop_start, tmin), (self.crop_end, tmax)):
            widget.blockSignals(True)
            widget.setDateTime(QtCore.QDateTime(value))
            widget.blockSignals(False)

    # -- handlers ----------------------------------------------------------
    def _do_crop(self, start, end) -> None:
        """Crop the active object to ``[start, end]`` and refresh."""
        # Target the parent object so a correlated APS keeps both axes aligned.
        obj = self.main.active_obj
        if obj is None:
            return
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if end <= start:
            QtWidgets.QMessageBox.warning(
                self.main, "Invalid range", "The end time must be after the start time."
            )
            return
        try:
            obj.timecrop(start=start, end=end, focus=True)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self.main, "Crop failed", traceback.format_exc(limit=1)
            )
            return
        if len(obj.time) == 0:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Empty result",
                "Cropping removed all data; reload to recover.",
            )
        self.sync_crop_fields()
        self.main.refresh_all(reset_view=True)

    def _apply_crop(self) -> None:
        """Crop to the time window in the from/to pickers."""
        self._do_crop(
            self.crop_start.dateTime().toPyDateTime(),
            self.crop_end.dateTime().toPyDateTime(),
        )

    def _crop_to_view(self) -> None:
        """Crop to the time window shown on the currently active time-based plot.

        Uses the active tab's x-limits if it exposes a time axis (Time series,
        2D heatmap, PM bands); otherwise falls back to the Time series tab.
        """
        xlim = None
        active = self.main.tabs.currentWidget()
        if hasattr(active, "current_time_xlim"):
            xlim = active.current_time_xlim()
        if xlim is None:
            ts_tab = next(
                (t for t in self.main._tabs if isinstance(t, TimeSeriesTab)), None
            )
            if ts_tab is not None:
                xlim = ts_tab.current_time_xlim()
        if xlim is None:
            QtWidgets.QMessageBox.information(
                self.main,
                "No time view",
                "Open a time-based plot (Time series, 2D heatmap, or PM bands) "
                "and zoom to the window you want before cropping to view.",
            )
            return

        start = pd.Timestamp(mdates.num2date(xlim[0])).tz_localize(None)
        end = pd.Timestamp(mdates.num2date(xlim[1])).tz_localize(None)
        self._do_crop(start, end)

    def _apply_smoothing(self) -> None:
        """Apply a rolling smooth to the active dataset (in place)."""
        obj = self.main.active_obj
        if obj is None:
            return
        try:
            obj.timesmooth(
                window=int(self.smooth_window.value()),
                method=self.smooth_method.currentText(),
            )
        except Exception:
            QtWidgets.QMessageBox.warning(
                self.main, "Smoothing failed", traceback.format_exc(limit=1)
            )
            return
        self.main.refresh_all(reset_view=True)

    def _apply_resampling(self) -> None:
        """Resample the active dataset to the chosen time step (in place)."""
        obj = self.main.active_obj
        if obj is None:
            return
        freq = self.resample_freq.text().strip()
        if not freq:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Resampling",
                "Enter a target time step, e.g. 30s, 1min, 5min.",
            )
            return
        try:
            obj.timerebin(freq=freq, method=self.resample_method.currentText())
        except Exception:
            QtWidgets.QMessageBox.warning(
                self.main, "Resampling failed", traceback.format_exc(limit=1)
            )
            return
        self.sync_crop_fields()
        self.main.refresh_all(reset_view=True)

    def _apply_timeshift(self) -> None:
        """Permanently shift the active dataset's time axis."""
        ds = self.main.project.active
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
                self.main, "Time shift failed", traceback.format_exc(limit=1)
            )
            return
        # Shared tasks keep their absolute times: re-project them onto the
        # shifted axis so this dataset's masks/summaries reflect the new timing.
        self.main.project._apply_activities(ds)
        self.sync_crop_fields()
        self.main.refresh_all(reset_view=True)
        self.main._refresh_sidebar()
