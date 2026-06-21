"""Per-activity summary tab (single dataset)."""

from __future__ import annotations

import contextlib
import io
import os

import pandas as pd

from .. import helpers
from ..models import PandasTableModel
from ..qt import QtWidgets
from ._base import _export_table, _tune_table


class SummaryTab(QtWidgets.QWidget):
    """Per-activity tabular summaries (activity stats and exposure metrics)."""

    def __init__(self, main):
        """Build the summary-type, metric and exposure-limit controls and table."""
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Activity summary", "Exposure summary"])
        self.kind.currentIndexChanged.connect(self._on_kind_change)
        bar.addWidget(QtWidgets.QLabel("Type:"))
        bar.addWidget(self.kind)

        # Metric selector: a kind drop-down, plus a cut-off drop-down that is
        # only shown for size-selective fractions (PM/PN/PS/PV). This replaces
        # the old free-text field, which let users type invalid metric strings.
        self.metric_label = QtWidgets.QLabel("Metric:")
        bar.addWidget(self.metric_label)
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        bar.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)  # allow custom cut-offs too
        self.metric_cut.setFixedWidth(80)
        bar.addWidget(self.metric_cut)
        self.metric_cut_label = QtWidgets.QLabel("µm")
        bar.addWidget(self.metric_cut_label)
        self._synced_obj = None

        self.compute = QtWidgets.QPushButton("Compute")
        self.compute.setObjectName("primary")
        self.compute.setToolTip("Compute the summary for the current activities.")
        self.compute.clicked.connect(self.refresh)
        bar.addWidget(self.compute)
        bar.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Export to Excel…")
        self.export_btn.setToolTip("Save the table to an .xlsx or .csv file.")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        layout.addLayout(bar)

        # Second row: exposure-limit parameters (only shown for exposure).
        self.exp_bar = QtWidgets.QHBoxLayout()
        self.short_limit = self._add_field("STEL (short-term limit):", "1.0", width=80)
        self.short_window = self._add_field("over", "15min", width=70)
        self.long_limit = self._add_field("OEL (8h limit):", "1.0", width=80)
        self.twa_window = self._add_field("TWA window", "8h", width=70)
        self.exp_bar.addStretch(1)
        layout.addLayout(self.exp_bar)

        self.model = PandasTableModel()
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        _tune_table(self.view)
        layout.addWidget(self.view)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._on_kind_change()

    def _add_field(self, label: str, default: str, width: int) -> QtWidgets.QLineEdit:
        """Add a labelled line-edit to the exposure-parameter row."""
        lbl = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(default)
        edit.setFixedWidth(width)
        self.exp_bar.addWidget(lbl)
        self.exp_bar.addWidget(edit)
        # Stash the label so visibility can be toggled together with the field.
        edit._label = lbl  # type: ignore[attr-defined]
        return edit

    @property
    def obj(self):
        """Active aerosol object (proxied from the main window)."""
        return self.main.obj

    _FRACTION_KINDS = ("PM", "PN", "PS", "PV")

    def _limit_fields(self):
        """The four exposure-limit line-edits."""
        return (self.short_limit, self.short_window, self.long_limit, self.twa_window)

    def _exposure_widgets(self):
        """All widgets (and their labels) shown only in Exposure mode."""
        widgets = [
            self.metric_label,
            self.metric_kind,
            self.metric_cut,
            self.metric_cut_label,
        ]
        for field in self._limit_fields():
            widgets.append(field)
            label = getattr(field, "_label", None)
            if label is not None:
                widgets.append(label)
        return widgets

    def _ensure_metric_options(self) -> None:
        """Populate the metric drop-downs for the loaded object (once per file)."""
        if self.obj is None or self._synced_obj is self.obj:
            return
        self._synced_obj = self.obj
        self.metric_kind.blockSignals(True)
        self.metric_cut.blockSignals(True)
        self.metric_kind.clear()
        self.metric_cut.clear()
        if helpers.is_2d(self.obj):
            self.metric_kind.addItems(["PNC", "MASS", "PM", "PN", "PS", "PV"])
            self.metric_cut.addItems(
                ["0.1", "0.25", "0.5", "1", "2.5", "4", "4.2", "10"]
            )
            self.metric_kind.setCurrentText("PM")
            self.metric_cut.setCurrentText("4.2")
        else:
            # 1D / AerosolAlt: offer PNC plus the available numeric channels.
            names = ["PNC"]
            for _label, kind, name in helpers.plottable_columns(self.obj):
                if kind in ("data", "extra") and name not in names:
                    names.append(name)
            self.metric_kind.addItems(names)
        self.metric_kind.blockSignals(False)
        self.metric_cut.blockSignals(False)
        self._on_metric_kind_change()

    def _on_metric_kind_change(self, *_args) -> None:
        """Show the cut-off field only for the size-selective fractions."""
        exposure = self.kind.currentText() == "Exposure summary"
        is_fraction = self.metric_kind.currentText().strip() in self._FRACTION_KINDS
        show_cut = exposure and is_fraction
        self.metric_cut.setVisible(show_cut)
        self.metric_cut_label.setVisible(show_cut)

    def _build_metric(self) -> str:
        """Assemble the metric string (kind plus cut-off where relevant)."""
        kind = self.metric_kind.currentText().strip()
        if kind in self._FRACTION_KINDS:
            return f"{kind}{self.metric_cut.currentText().strip()}"
        return kind

    def _on_kind_change(self) -> None:
        """Toggle the exposure-only widgets for the chosen summary type."""
        exposure = self.kind.currentText() == "Exposure summary"
        if exposure:
            self._ensure_metric_options()
        for widget in self._exposure_widgets():
            widget.setVisible(exposure)
        if exposure:
            self._on_metric_kind_change()  # refine cut-off visibility

    @staticmethod
    def _to_float(text: str, default: float) -> float:
        """Parse ``text`` as a float, returning ``default`` on failure."""
        try:
            return float(text.strip())
        except (ValueError, AttributeError):
            return default

    def refresh(self) -> None:
        """Compute the activity or exposure summary for the active dataset."""
        if self.obj is None:
            return
        # Keep the metric drop-downs in sync with the loaded object.
        if self._synced_obj is not self.obj:
            self._synced_obj = None
            self._ensure_metric_options()
        self.status.setText("")
        try:
            # The core summarize_* methods print their table to stdout; swallow
            # it so the GUI is the single source of truth and a non-UTF-8 console
            # cannot crash the method on the µg/m³ / cm⁻³ glyphs mid-print.
            with contextlib.redirect_stdout(io.StringIO()):
                if self.kind.currentText() == "Exposure summary":
                    df = self.obj.summarize_exposure(
                        metric=self._build_metric(),
                        short_limit=self._to_float(self.short_limit.text(), 1.0),
                        long_limit=self._to_float(self.long_limit.text(), 1.0),
                        short_window=self.short_window.text().strip() or "15min",
                        twa_window=self.twa_window.text().strip() or "8h",
                    )
                else:
                    df = self.obj.summarize_activities()
            if df is None or df.empty:
                self.model.set_dataframe(pd.DataFrame())
                self.status.setText("No data to summarize for the current activities.")
            else:
                self.model.set_dataframe(df)
        except Exception as exc:  # surface errors instead of crashing the GUI
            self.model.set_dataframe(pd.DataFrame())
            self.status.setText(f"Could not compute summary: {exc}")

    def _export(self) -> None:
        """Save the summary table to an .xlsx or .csv file."""
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        base = os.path.splitext(os.path.basename(self.main.source_path or "summary"))[0]
        _export_table(self, df, f"{base}_{kind}_summary", with_index=False)
