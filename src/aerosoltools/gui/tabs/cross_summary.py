"""Cross-instrument summary table comparison tab."""

from __future__ import annotations

import contextlib
import io

import pandas as pd

from .. import helpers
from ..models import PandasTableModel
from ..qt import QtCore, QtWidgets
from ._base import _export_table, _tune_table


class CrossSummaryTab(QtWidgets.QWidget):
    """One combined activity/exposure summary table across several datasets.

    The single-dataset :class:`SummaryTab` runs ``summarize_activities`` /
    ``summarize_exposure`` on the active object. This comparison tab runs the
    same core methods on **each ticked dataset**, prepends ``Dataset`` /
    ``Instrument`` columns, and concatenates the per-dataset tables into one
    (columns align by name, so metrics that only some instruments report simply
    leave blanks for the others). It is compute-on-demand (a ``Compute`` button)
    rather than recomputing on every refresh, since exposure stats over many
    datasets can be costly.
    """

    _FRACTION_KINDS = ("PM", "PN", "PS", "PV")

    def __init__(self, main):
        """Build the dataset checklist, summary controls and table."""
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QHBoxLayout()
        layout.addLayout(body, stretch=1)

        # -- left: datasets to include ------------------------------------
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.setMaximumWidth(240)
        self.ds_list.itemChanged.connect(self._on_ds_changed)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to include:"))
        side.addWidget(self.ds_list, stretch=1)
        body.addLayout(side)

        # -- right: controls + table --------------------------------------
        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, stretch=1)

        bar = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Activity summary", "Exposure summary"])
        self.kind.currentIndexChanged.connect(self._on_kind_change)
        bar.addWidget(QtWidgets.QLabel("Type:"))
        bar.addWidget(self.kind)

        # Metric selector (exposure only): a kind drop-down plus a cut-off
        # drop-down shown only for the size-selective fractions (PM/PN/PS/PV).
        self.metric_label = QtWidgets.QLabel("Metric:")
        bar.addWidget(self.metric_label)
        self.metric_kind = QtWidgets.QComboBox()
        self.metric_kind.currentTextChanged.connect(self._on_metric_kind_change)
        bar.addWidget(self.metric_kind)
        self.metric_cut = QtWidgets.QComboBox()
        self.metric_cut.setEditable(True)
        self.metric_cut.setFixedWidth(80)
        bar.addWidget(self.metric_cut)
        self.metric_cut_label = QtWidgets.QLabel("µm")
        bar.addWidget(self.metric_cut_label)

        self.compute = QtWidgets.QPushButton("Compute")
        self.compute.setObjectName("primary")
        self.compute.setToolTip(
            "Build the combined summary table for the ticked datasets."
        )
        self.compute.clicked.connect(self._compute)
        bar.addWidget(self.compute)
        bar.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Export to Excel…")
        self.export_btn.setToolTip("Save the combined table to an .xlsx or .csv file.")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        right.addLayout(bar)

        # Second row: exposure-limit parameters (only shown for exposure).
        self.exp_bar = QtWidgets.QHBoxLayout()
        self.short_limit = self._add_field("STEL (short-term limit):", "1.0", width=80)
        self.short_window = self._add_field("over", "15min", width=70)
        self.long_limit = self._add_field("OEL (8h limit):", "1.0", width=80)
        self.twa_window = self._add_field("TWA window", "8h", width=70)
        self.exp_bar.addStretch(1)
        right.addLayout(self.exp_bar)

        self.model = PandasTableModel()
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        _tune_table(self.view)
        right.addWidget(self.view, stretch=1)

        self.status = QtWidgets.QLabel(
            "Tick datasets and click Compute to build the combined summary."
        )
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        self._on_kind_change()

    # -- small helpers -----------------------------------------------------
    def _add_field(self, label: str, default: str, width: int) -> QtWidgets.QLineEdit:
        """Add a labelled line-edit to the exposure-parameter row and return it."""
        lbl = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(default)
        edit.setFixedWidth(width)
        self.exp_bar.addWidget(lbl)
        self.exp_bar.addWidget(edit)
        edit._label = lbl  # type: ignore[attr-defined]
        return edit

    def _selected_datasets(self) -> list:
        """Datasets currently ticked for inclusion."""
        return [d for d in self.main.project.datasets if d.summary_on]

    def _limit_fields(self):
        """The four exposure-limit line-edits."""
        return (self.short_limit, self.short_window, self.long_limit, self.twa_window)

    def _exposure_widgets(self):
        """Widgets (and their labels) shown only in Exposure mode."""
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

    # -- metric options ----------------------------------------------------
    def _ensure_metric_options(self) -> None:
        """Populate the metric drop-downs from the union of selected datasets."""
        selected = self._selected_datasets()
        any_2d = any(helpers.is_2d(d.obj) for d in selected)
        cur_kind = self.metric_kind.currentText()
        cur_cut = self.metric_cut.currentText()

        self.metric_kind.blockSignals(True)
        self.metric_cut.blockSignals(True)
        self.metric_kind.clear()
        self.metric_cut.clear()

        kinds = ["PNC"]
        if any_2d:
            kinds += ["MASS", "PM", "PN", "PS", "PV"]
        for d in selected:  # add any extra numeric channels present
            for _label, kind, name in helpers.plottable_columns(d.obj):
                if kind in ("data", "extra") and name not in kinds:
                    kinds.append(name)
        self.metric_kind.addItems(kinds)
        self.metric_cut.addItems(["0.1", "0.25", "0.5", "1", "2.5", "4", "4.2", "10"])

        # Preserve the user's selection across rebuilds where still valid.
        ki = self.metric_kind.findText(cur_kind)
        self.metric_kind.setCurrentText(
            cur_kind if ki >= 0 else ("PM" if any_2d else "PNC")
        )
        self.metric_cut.setCurrentText(cur_cut or "4.2")
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
            self._on_metric_kind_change()

    @staticmethod
    def _to_float(text: str, default: float) -> float:
        """Parse ``text`` as a float, returning ``default`` on failure."""
        try:
            return float(text.strip())
        except (ValueError, AttributeError):
            return default

    # -- dataset list sync -------------------------------------------------
    def _sync_datasets(self) -> None:
        """Rebuild the dataset checklist from the project."""
        self._building = True
        self.ds_list.blockSignals(True)
        self.ds_list.clear()
        for ds in self.main.project.datasets:
            item = QtWidgets.QListWidgetItem(f"{ds.label}  ({ds.instrument})")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if ds.summary_on else QtCore.Qt.Unchecked
            )
            item.setData(QtCore.Qt.UserRole, ds.id)
            self.ds_list.addItem(item)
        self.ds_list.blockSignals(False)
        self._building = False

    def _on_ds_changed(self, item) -> None:
        """Persist a dataset's include flag and refresh the metric options."""
        if self._building:
            return
        ds = self.main.project.get(item.data(QtCore.Qt.UserRole))
        if ds is not None:
            ds.summary_on = item.checkState() == QtCore.Qt.Checked
        # The available metrics may change with the selection (e.g. a 2D
        # dataset added/removed), so keep the metric drop-downs in step.
        if self.kind.currentText() == "Exposure summary":
            self._ensure_metric_options()

    def refresh(self) -> None:
        """Re-sync the dataset list; the table itself is built on Compute."""
        self._sync_datasets()
        if self.kind.currentText() == "Exposure summary":
            self._ensure_metric_options()

    # -- compute -----------------------------------------------------------
    def _compute(self) -> None:
        """Run the summary per ticked dataset and concatenate the tables."""
        datasets = self._selected_datasets()
        if not datasets:
            self.model.set_dataframe(pd.DataFrame())
            self.status.setText("Tick at least one dataset to include.")
            return
        exposure = self.kind.currentText() == "Exposure summary"
        metric = self._build_metric() if exposure else None

        frames: list[pd.DataFrame] = []
        skipped: list[str] = []
        # The core summarize_* methods print their result table to stdout; with
        # many datasets that is just noise (the user reads the GUI table), and on
        # a non-UTF-8 console the unit glyphs (µg/m³, cm⁻³) can even raise an
        # encoding error mid-method. Swallow that console output during compute.
        with contextlib.redirect_stdout(io.StringIO()):
            for ds in datasets:
                try:
                    if exposure:
                        df = ds.obj.summarize_exposure(
                            metric=metric,
                            short_limit=self._to_float(self.short_limit.text(), 1.0),
                            long_limit=self._to_float(self.long_limit.text(), 1.0),
                            short_window=self.short_window.text().strip() or "15min",
                            twa_window=self.twa_window.text().strip() or "8h",
                        )
                    else:
                        df = ds.obj.summarize_activities()
                except Exception as exc:  # e.g. a PM metric on a 1D instrument
                    skipped.append(f"{ds.label} ({exc})")
                    continue
                if df is None or df.empty:
                    continue
                df = df.copy()
                df.insert(0, "Instrument", ds.instrument)
                df.insert(0, "Dataset", ds.label)
                frames.append(df)

        if not frames:
            self.model.set_dataframe(pd.DataFrame())
            msg = "No data to summarize for the current selection."
            if skipped:
                msg += "  Skipped — " + "; ".join(skipped)
            self.status.setText(msg)
            return

        combined = pd.concat(frames, ignore_index=True, sort=False)
        self.model.set_dataframe(combined)
        note = f"{len(frames)} dataset(s) combined, {len(combined)} rows."
        if skipped:
            note += "  Skipped — " + "; ".join(skipped)
        self.status.setText(note)

    def _export(self) -> None:
        """Save the combined table to an .xlsx or .csv file."""
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        _export_table(self, df, f"cross_instrument_{kind}_summary", with_index=False)
