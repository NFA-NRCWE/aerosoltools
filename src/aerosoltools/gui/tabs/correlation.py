"""Correlation / Bland-Altman two-dataset comparison tab."""

from __future__ import annotations

import contextlib
import io

import matplotlib.colors as mcolors

from ...intercomparison import bland_altman_analysis, plot_correlation
from ..qt import QtWidgets
from ..view import theme
from ._base import _PlotTab


class CorrelationTab(_PlotTab):
    """Correlate (or Bland–Altman compare) one parameter between two datasets.

    A two-dataset comparison: pick an **X** and a **Y** dataset and a parameter
    they share, and the tab draws the library's own
    :func:`~aerosoltools.intercomparison.plot_correlation` (scatter + 1:1 +
    regression + R²) or :func:`~aerosoltools.intercomparison.bland_altman_analysis`
    on the embedded axis. Time alignment between the two instruments is delegated
    to the core via ``match`` ("exact"/"nearest"/"rebin") + ``tolerance``. The
    plot is built on **Compute** (alignment can be costly), so a plain ``refresh``
    only re-syncs the dataset/parameter selectors and leaves the plot in place.
    """

    export_tag = "correlation"

    def __init__(self, main):
        """Build the analysis selectors, alignment options and plot."""
        super().__init__(main, nrows=1)

        self.analysis = QtWidgets.QComboBox()
        self.analysis.addItems(["Correlation", "Bland–Altman"])
        self.analysis.currentIndexChanged.connect(self._on_analysis_change)
        self.controls.addWidget(QtWidgets.QLabel("Analysis:"))
        self.controls.addWidget(self.analysis)

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.currentIndexChanged.connect(self._on_pair_change)
        self.controls.addWidget(QtWidgets.QLabel("X:"))
        self.controls.addWidget(self.x_combo)

        self.y_combo = QtWidgets.QComboBox()
        self.y_combo.currentIndexChanged.connect(self._on_pair_change)
        self.controls.addWidget(QtWidgets.QLabel("Y:"))
        self.controls.addWidget(self.y_combo)

        self.param = QtWidgets.QComboBox()
        self.param.setMinimumWidth(90)
        self.controls.addWidget(QtWidgets.QLabel("Parameter:"))
        self.controls.addWidget(self.param)

        # Restrict the correlation to one activity (e.g. a marked side-by-side
        # region). "All data" uses the full overlapping record.
        self.activity = QtWidgets.QComboBox()
        self.activity.setMinimumWidth(90)
        self.activity.setToolTip(
            "Correlate only the timestamps inside this activity — mark a "
            "side-by-side region in the Time series tab, then pick it here."
        )
        self.controls.addWidget(QtWidgets.QLabel("Activity:"))
        self.controls.addWidget(self.activity)
        self.controls.addStretch(1)
        self.controls.addWidget(self.save_btn)

        # -- side panel: time alignment + analysis options + Compute -------
        self.match = QtWidgets.QComboBox()
        self.match.addItems(["exact", "nearest", "rebin"])
        self.match.currentTextChanged.connect(self._on_match_change)
        self.tolerance = QtWidgets.QLineEdit("30s")
        self.tolerance.setToolTip("Max timestamp separation for 'nearest' match.")
        self.rebin_freq = QtWidgets.QLineEdit()
        self.rebin_freq.setPlaceholderText("auto")
        self.rebin_freq.setToolTip(
            "Common time step for 'rebin', e.g. 1min (blank = auto)."
        )
        self.rebin_method = QtWidgets.QComboBox()
        self.rebin_method.addItems(["mean", "median", "min", "max", "sum"])

        align = QtWidgets.QGroupBox("Time alignment")
        align_col = QtWidgets.QVBoxLayout(align)
        align_col.addWidget(self._field_row("Match:", self.match))
        self._tol_row = self._field_row("Tolerance:", self.tolerance)
        self._freq_row = self._field_row("Rebin to:", self.rebin_freq)
        self._method_row = self._field_row("Aggregation:", self.rebin_method)
        align_col.addWidget(self._tol_row)
        align_col.addWidget(self._freq_row)
        align_col.addWidget(self._method_row)

        # Correlation-only options.
        self.intercept = QtWidgets.QCheckBox("Fit intercept (y = A·x + B)")
        self.intercept.setChecked(True)
        self.uniform = QtWidgets.QCheckBox("Uniform axis scaling")
        self.uniform.setChecked(True)
        self.robust = QtWidgets.QCheckBox("Robust fit (Theil–Sen)")
        self.robust.setToolTip(
            "Use the outlier-resistant Theil–Sen estimator instead of "
            "least-squares (drops the confidence band)."
        )
        self.corr_box = QtWidgets.QGroupBox("Regression")
        corr_col = QtWidgets.QVBoxLayout(self.corr_box)
        corr_col.addWidget(self.intercept)
        corr_col.addWidget(self.uniform)
        corr_col.addWidget(self.robust)

        # Bland–Altman-only options.
        self.ba_method = QtWidgets.QComboBox()
        self.ba_method.addItem("Bland–Altman (difference)", "BA")
        self.ba_method.addItem("Giavarina (% of mean)", "Gi")
        self.ba_method.addItem("Euser (log)", "Eu")
        self.ba_conf = QtWidgets.QDoubleSpinBox()
        self.ba_conf.setRange(0.50, 0.999)
        self.ba_conf.setSingleStep(0.01)
        self.ba_conf.setDecimals(3)
        self.ba_conf.setValue(0.95)
        self.ba_box = QtWidgets.QGroupBox("Bland–Altman")
        ba_col = QtWidgets.QVBoxLayout(self.ba_box)
        ba_col.addWidget(self._field_row("Method:", self.ba_method))
        ba_col.addWidget(self._field_row("Confidence:", self.ba_conf))

        self.compute_btn = QtWidgets.QPushButton("Compute")
        self.compute_btn.setObjectName("primary")
        self.compute_btn.setToolTip(
            "Align the two datasets and draw the correlation / Bland–Altman plot."
        )
        self.compute_btn.clicked.connect(self._draw)

        self.calibrate_btn = QtWidgets.QPushButton("Calibrate…")
        self.calibrate_btn.setToolTip(
            "Calibrate one instrument to match the other, using this comparison "
            "(total concentration or bin-by-bin). Affects only that one dataset."
        )
        self.calibrate_btn.clicked.connect(self._open_calibration)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(align)
        side.addWidget(self.corr_box)
        side.addWidget(self.ba_box)
        side.addWidget(self.compute_btn)
        side.addWidget(self.calibrate_btn)
        hint = QtWidgets.QLabel(
            "Pick two datasets and a shared parameter, then click Compute. Use "
            "'nearest'/'rebin' match when the two instruments log on different "
            "time stamps."
        )
        hint.setWordWrap(True)
        side.addWidget(hint)
        side.addStretch(1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)

        # Plot on the left of a resizable divider, side panel on the right. The
        # side panel here is wider than the other tabs' (time-alignment +
        # regression + Bland–Altman groups), so start the split narrower and let
        # the canvas shrink, keeping the whole pane inside the window (no
        # horizontal scroll) at the default size.
        self._split_with_side(side_w, sizes=(560, 320))
        self.canvas.setMinimumWidth(360)

        self.ax = self.figure.add_subplot(111)
        self._has_drawn = False
        self._on_analysis_change()
        self._on_match_change()

    # -- small helpers -----------------------------------------------------
    @staticmethod
    def _field_row(label_text: str, widget) -> QtWidgets.QWidget:
        """Wrap ``label + widget`` in a row so both can be hidden together."""
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel(label_text))
        h.addWidget(widget, stretch=1)
        return row

    def _dataset(self, ds_id):
        """Return the dataset with id ``ds_id`` from the project."""
        return self.main.project.get(ds_id)

    # -- visibility toggles ------------------------------------------------
    def _on_analysis_change(self, *_a) -> None:
        """Show the option box matching the selected analysis type."""
        is_corr = self.analysis.currentText() == "Correlation"
        self.corr_box.setVisible(is_corr)
        self.ba_box.setVisible(not is_corr)

    def _on_match_change(self, *_a) -> None:
        """Show only the alignment fields relevant to the current match mode."""
        mode = self.match.currentText()
        self._tol_row.setVisible(mode == "nearest")
        self._freq_row.setVisible(mode == "rebin")
        self._method_row.setVisible(mode == "rebin")

    # -- selector sync -----------------------------------------------------
    def _sync_pair(self) -> None:
        """Repopulate the X/Y dataset combos, preserving selections."""
        datasets = self.main.project.datasets
        for combo, default_idx in ((self.x_combo, 0), (self.y_combo, 1)):
            combo.blockSignals(True)
            prev = combo.currentData()
            combo.clear()
            for ds in datasets:
                combo.addItem(ds.label, ds.id)
            idx = combo.findData(prev)
            if idx < 0:
                idx = min(default_idx, combo.count() - 1)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _common_parameters(self) -> list:
        """Numeric column names present in *both* selected datasets."""
        xds = self._dataset(self.x_combo.currentData())
        yds = self._dataset(self.y_combo.currentData())
        if xds is None or yds is None:
            return []

        def numeric_cols(obj) -> set:
            """Return the object's non-boolean column names (data + extra)."""
            cols = set(obj.data.select_dtypes(exclude="bool").columns)
            extra = getattr(obj, "extra_data", None)
            if extra is not None and not extra.empty:
                cols |= set(extra.select_dtypes(exclude="bool").columns)
            return cols - set(getattr(obj, "activities", []))

        common = numeric_cols(xds.obj) & numeric_cols(yds.obj)
        ordered = ["Total_conc"] if "Total_conc" in common else []
        ordered += sorted(c for c in common if c != "Total_conc")
        return ordered

    def _sync_params(self) -> None:
        """Repopulate the shared-parameter combo from the chosen X/Y pair."""
        self.param.blockSignals(True)
        prev = self.param.currentText()
        self.param.clear()
        self.param.addItems(self._common_parameters())
        idx = self.param.findText(prev)
        if idx >= 0:
            self.param.setCurrentIndex(idx)
        self.param.blockSignals(False)

    def _sync_activities(self) -> None:
        """Repopulate the activity combo from the project's tasks."""
        self.activity.blockSignals(True)
        prev = self.activity.currentText()
        self.activity.clear()
        self.activity.addItems(["All data"] + self.main.project.user_activities())
        idx = self.activity.findText(prev)
        self.activity.setCurrentIndex(idx if idx >= 0 else 0)
        self.activity.blockSignals(False)

    def _on_pair_change(self, *_a) -> None:
        """Re-sync the parameter list when X or Y changes."""
        self._sync_params()

    def _align_kwargs(self) -> dict:
        """Time-alignment settings shared by the plot and the calibration.

        Mirrors how :meth:`_draw_on` aligns the two datasets (match mode,
        tolerance, activity restriction and the rebin options) so a calibration
        is fitted on exactly the points the correlation plot shows.
        """
        mode = self.match.currentText()
        activity = self.activity.currentText() or "All data"
        kwargs = dict(
            match=mode,
            tolerance=self.tolerance.text().strip() or "30s",
            activity=None if activity == "All data" else activity,
        )
        if mode == "rebin":
            kwargs["rebin_freq"] = self.rebin_freq.text().strip() or None
            kwargs["rebin_method"] = self.rebin_method.currentText()
        return kwargs

    def _open_calibration(self) -> None:
        """Open the per-instrument calibration dialog for the current X/Y pair."""
        from ..calibration import CalibrationDialog

        xds = self._dataset(self.x_combo.currentData())
        yds = self._dataset(self.y_combo.currentData())
        if xds is None or yds is None:
            QtWidgets.QMessageBox.information(
                self, "Calibrate", "Load at least two datasets first."
            )
            return
        if xds is yds:
            QtWidgets.QMessageBox.information(
                self, "Calibrate", "Pick two different datasets for X and Y."
            )
            return
        CalibrationDialog(self, xds, yds, self._align_kwargs()).exec_()

    def refresh(self) -> None:
        """Re-sync selectors only; the plot is (re)built on Compute."""
        self._sync_pair()
        self._sync_params()
        self._sync_activities()

    # -- rendering ---------------------------------------------------------
    def _draw_on(self, ax) -> None:
        """Draw the correlation / Bland-Altman plot for the pair onto ``ax``."""
        xds = self._dataset(self.x_combo.currentData())
        yds = self._dataset(self.y_combo.currentData())
        if xds is None or yds is None:
            raise ValueError("Load at least two datasets to compare.")
        if xds is yds:
            raise ValueError("Pick two different datasets for X and Y.")
        if not self.param.count():
            raise ValueError("The two datasets share no common parameter to correlate.")

        parameter = self.param.currentText().strip() or None
        kwargs = dict(parameter=parameter, **self._align_kwargs())

        # The core draws here but also prints a summary line containing a "µ"
        # glyph, which crashes on a non-UTF-8 console — swallow that stdout (the
        # plot is the output the GUI shows), as the summary tabs do.
        with contextlib.redirect_stdout(io.StringIO()):
            if self.analysis.currentText() == "Correlation":
                plot_correlation(
                    xds.obj,
                    yds.obj,
                    ax_in=ax,
                    intercept=self.intercept.isChecked(),
                    uniform_scaling=self.uniform.isChecked(),
                    outlier_influence=not self.robust.isChecked(),
                    **kwargs,
                )
            else:
                bland_altman_analysis(
                    xds.obj,
                    yds.obj,
                    ax_in=ax,
                    method=self.ba_method.currentData(),
                    C=float(self.ba_conf.value()),
                    **kwargs,
                )

    def _draw(self) -> None:
        """Compute and draw onto the embedded axis, reporting errors inline."""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            self._draw_on(ax)
            # The core draws the scatter / reference lines in black, which is
            # invisible on the dark theme, so brighten them when the dark theme
            # is active. Saved figures capture the on-screen plot, so for a
            # black-on-white publication figure, export from the light theme.
            if theme.is_dark():
                self._brighten_for_dark(ax)
            self.ax = ax
            self._has_drawn = True
            self._sync_toolbar_home()
            self.canvas.draw_idle()
        except Exception as exc:
            self._show_message(f"Could not compute: {exc}")

    @staticmethod
    def _brighten_for_dark(ax) -> None:
        """Recolour near-black core-drawn artists so they read on a dark axis.

        The Bland–Altman scatter (``c="k"``) becomes a bright accent colour, and
        black reference lines (the 1:1 line, the zero line) become a light grey.
        Coloured artists (the regression line, the correlation scatter, the grey
        ±LOA lines) are left untouched.
        """

        def _is_black(color) -> bool:
            r, g, b = mcolors.to_rgb(color)
            return max(r, g, b) < 0.25

        accent = theme.mpl_cycle()[0]
        for coll in ax.collections:  # scatter PathCollections
            fc = coll.get_facecolor()
            if len(fc) and _is_black(fc[0]):
                coll.set_color(accent)
                coll.set_alpha(0.7)
        for line in ax.get_lines():
            if _is_black(line.get_color()):
                line.set_color("#9bb0c9")
