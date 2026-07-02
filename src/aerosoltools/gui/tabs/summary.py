"""Per-activity summary table tab (activity stats and exposure metrics).

Works for one *or* several datasets: it runs the core ``summarize_activities`` /
``summarize_exposure`` on each ticked dataset, prepends ``Dataset`` /
``Instrument`` columns and concatenates the results, so a single instrument
behaves like the old single-view Summary tab while multiple instruments are
combined into one cross-instrument table.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from ..._core import _stats
from .. import helpers
from ..models import PandasTableModel
from ..qt import QtCore, QtWidgets
from ._base import _export_table, _tune_table


class SummaryTab(QtWidgets.QWidget):
    """One activity/exposure summary table across the ticked datasets.

    Runs the core ``summarize_activities`` / ``summarize_exposure`` on **each
    ticked dataset**, prepends ``Dataset`` / ``Instrument`` columns, and
    concatenates the per-dataset tables into one (columns align by name, so
    metrics that only some instruments report simply leave blanks for the
    others). With a single dataset ticked it is just that instrument's summary.
    It is compute-on-demand (a ``Compute`` button) rather than recomputing on
    every refresh, since exposure stats over many datasets can be costly.
    """

    _FRACTION_KINDS = ("PM", "PN", "PS", "PV")

    def __init__(self, main):
        """Build the dataset checklist, summary controls and table."""
        super().__init__()
        self.main = main

        layout = QtWidgets.QVBoxLayout(self)
        # A draggable divider between the dataset checklist and the table, so the
        # list pane can be widened when labels are long (like the datasets dock).
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        # -- left: datasets to include ------------------------------------
        self._building = False
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.itemChanged.connect(self._on_ds_changed)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Datasets to include:"))
        side.addWidget(self.ds_list, stretch=1)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(side)
        splitter.addWidget(left_widget)

        # -- right: controls + table --------------------------------------
        right = QtWidgets.QVBoxLayout()
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 900])

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
        self.metric_kind.setToolTip(
            "Quantity to summarise exposure for: total number (PNC), total mass "
            "(MASS), a size-selective fraction (PM/PN/PS/PV at the chosen cut-off), "
            "or any extra channel the datasets carry."
        )
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

        # Second row (activity-only): which metrics and stats to report,
        # instead of being locked to a fixed PM1/2.5/4/10 mean+std set.
        self.act_bar = QtWidgets.QHBoxLayout()
        self.act_metrics_label = QtWidgets.QLabel("Metrics:")
        self.act_bar.addWidget(self.act_metrics_label)
        self.act_metrics = QtWidgets.QLineEdit()
        self.act_metrics.setPlaceholderText(
            "PNC, PM1, PM2.5, PM4, PM10, MASS, MODE, MEDIAN, GMD (blank = default)"
        )
        self.act_metrics.setToolTip(
            "Comma-separated metrics to summarize, e.g. 'PNC, PM2.5, PM7'. Any "
            "cut-off works for PM/PN/PS/PV, not just 1/2.5/4/10. Leave blank to "
            "use each dataset's own default set."
        )
        self.act_bar.addWidget(self.act_metrics, stretch=1)
        self.act_stats_label = QtWidgets.QLabel("Stats:")
        self.act_bar.addWidget(self.act_stats_label)
        self.act_stat_boxes: dict[str, QtWidgets.QCheckBox] = {}
        for stat, checked in (
            ("mean", True),
            ("std", True),
            ("min", False),
            ("max", False),
            ("median", False),
        ):
            box = QtWidgets.QCheckBox(stat.capitalize())
            box.setChecked(checked)
            self.act_stat_boxes[stat] = box
            self.act_bar.addWidget(box)
        right.addLayout(self.act_bar)

        # Third row: exposure-limit parameters (only shown for exposure).
        self.exp_bar = QtWidgets.QHBoxLayout()
        self.short_limit = self._add_field(
            "STEL (short-term limit):", "1.0", width=80,
            tip="Short-term exposure limit. The highest short-window average is "
            "compared against this value (same unit as the chosen metric).",
        )
        self.short_window = self._add_field(
            "over", "15min", width=70,
            tip="Averaging window for the short-term (STEL) check, as a pandas "
            "offset, e.g. 15min.",
        )
        self.long_limit = self._add_field(
            "OEL (8h limit):", "1.0", width=80,
            tip="Occupational exposure limit. The time-weighted average is "
            "compared against this value (same unit as the chosen metric).",
        )
        self.twa_window = self._add_field(
            "TWA window", "8h", width=70,
            tip="Averaging window for the time-weighted average (TWA), e.g. 8h.",
        )
        self.exp_bar.addStretch(1)
        right.addLayout(self.exp_bar)

        # Editing any limit/metric makes the shown (cached) table no longer match
        # the inputs, so re-evaluate staleness as the user types.
        for field in self._limit_fields():
            field.editingFinished.connect(self._recheck_stale)
        self.metric_kind.currentTextChanged.connect(self._recheck_stale)
        self.metric_cut.currentTextChanged.connect(self._recheck_stale)
        self.act_metrics.editingFinished.connect(self._recheck_stale)
        for box in self.act_stat_boxes.values():
            box.stateChanged.connect(self._recheck_stale)

        # Stale banner: shown when the displayed values were computed from inputs
        # (tasks, data, or settings) that have since changed.
        self.stale_banner = QtWidgets.QLabel(
            "⚠ These values may be out of date — tasks, data, or settings changed "
            "since they were computed. Click Compute to refresh."
        )
        self.stale_banner.setWordWrap(True)
        self.stale_banner.setStyleSheet(
            "background:#7a4a00; color:#ffe8c2; border:1px solid #b3791f;"
            "border-radius:6px; padding:6px;"
        )
        self.stale_banner.setVisible(False)
        right.addWidget(self.stale_banner)

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
        # Tracks the project a kind/params restore was last done for, so a
        # project load restores the saved kind exactly once.
        self._restored_proj_id = None
        self._on_kind_change()

    # -- small helpers -----------------------------------------------------
    def _add_field(
        self, label: str, default: str, width: int, tip: str | None = None
    ) -> QtWidgets.QLineEdit:
        """Add a labelled line-edit to the exposure-parameter row and return it.

        Args:
            label: Caption shown to the left of the field.
            default: Initial text.
            width: Fixed field width in pixels.
            tip: Optional tooltip applied to both the label and the field.
        """
        lbl = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(default)
        edit.setFixedWidth(width)
        if tip:
            lbl.setToolTip(tip)
            edit.setToolTip(tip)
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

    def _activity_widgets(self):
        """Widgets shown only in Activity summary mode."""
        return [
            self.act_metrics_label,
            self.act_metrics,
            self.act_stats_label,
            *self.act_stat_boxes.values(),
        ]

    def _selected_stats(self) -> list[str]:
        """Ticked stat names, in a fixed order; falls back to ["mean"]."""
        order = ("mean", "std", "min", "max", "median")
        stats = [s for s in order if self.act_stat_boxes[s].isChecked()]
        return stats or ["mean"]

    def _parsed_metrics(self) -> list[str] | None:
        """Metrics from the comma-separated field, or None to use the default."""
        text = self.act_metrics.text().strip()
        if not text:
            return None
        return [m.strip() for m in text.split(",") if m.strip()]

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

    @staticmethod
    def _clarify_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Append " mean" to activity-summary value columns that lack it.

        ``summarize_activities`` reports the "mean" stat as an unlabeled
        column (e.g. a "PNC [cm⁻³]" column next to "PNC [cm⁻³] std"), for
        backward compatibility with callers that already expect that name.
        In the GUI table that reads as ambiguous — is it the mean, the min,
        the max? Renaming here (rather than in the core function) keeps
        ``summarize_activities``'s return value unchanged for other callers;
        this only affects what the GUI displays/exports.
        """
        skip = {"Segment", "Duration (HH:MM)"}
        known_suffixes = tuple(f" {stat}" for stat in _stats.VALID_STATS)
        rename = {
            col: f"{col} mean"
            for col in df.columns
            if col not in skip and not col.endswith(known_suffixes)
        }
        return df.rename(columns=rename) if rename else df

    def _apply_kind_visibility(self) -> None:
        """Show only the widgets relevant to the current summary kind."""
        exposure = self.kind.currentText() == "Exposure summary"
        for widget in self._exposure_widgets():
            widget.setVisible(exposure)
        for widget in self._activity_widgets():
            widget.setVisible(not exposure)
        if exposure:
            self._on_metric_kind_change()

    def _on_kind_change(self) -> None:
        """Rebuild options for the chosen kind, then show its cached table.

        Restores that kind's saved inputs and table (so switching back to a
        previously-computed kind shows it without recomputing) and re-checks
        whether the shown values are now stale.
        """
        kind = self.kind.currentText()
        exposure = kind == "Exposure summary"
        if exposure:
            self._ensure_metric_options()
        self._apply_kind_visibility()
        self._restore_params_from_cache(kind)
        self._show_cache()

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
        # A different dataset selection means the shown table no longer matches.
        self._recheck_stale()

    def refresh(self) -> None:
        """Re-sync the dataset list and show any cached summary for this kind.

        On the first refresh after a project load, restore the kind that was
        last computed (and its saved inputs); thereafter just re-display the
        cached table and re-evaluate staleness.
        """
        self._sync_datasets()
        proj = self.main.project
        if self._restored_proj_id != id(proj):
            self._restored_proj_id = id(proj)
            active = (proj.summary_state or {}).get("active_kind")
            if active and self.kind.findText(active) >= 0 and active != self.kind.currentText():
                # Switching kind triggers _on_kind_change, which restores that
                # kind's params + table and checks staleness.
                self.kind.setCurrentText(active)
                return
        if self.kind.currentText() == "Exposure summary":
            self._ensure_metric_options()
        self._show_cache()

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
        act_metrics = self._parsed_metrics()
        act_stats = self._selected_stats()

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
                        kwargs = {"stats": act_stats}
                        if act_metrics is not None:
                            kwargs["metrics"] = act_metrics
                        df = ds.obj.summarize_activities(**kwargs)
                        df = self._clarify_activity_columns(df)
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
            # Nothing to show: drop any stale cache for this kind.
            self._cache().pop(self.kind.currentText(), None)
            self._set_stale(False)
            msg = "No data to summarize for the current selection."
            if skipped:
                msg += "  Skipped — " + "; ".join(skipped)
            self.status.setText(msg)
            return

        combined = pd.concat(frames, ignore_index=True, sort=False)
        self.model.set_dataframe(combined)
        # Persist the result + inputs on the project so reopening shows it
        # directly, and record the input signature for staleness detection.
        self._store_cache(combined)
        note = f"{len(frames)} dataset(s) combined, {len(combined)} rows."
        if skipped:
            note += "  Skipped — " + "; ".join(skipped)
        self.status.setText(note)

    # -- persistence / staleness ------------------------------------------
    def _cache(self) -> dict:
        """The project's per-kind summary cache (created on first use)."""
        state = self.main.project.summary_state
        if "cache" not in state:
            state["cache"] = {}
        return state["cache"]

    def _set_stale(self, stale: bool) -> None:
        """Show/hide the 'values out of date' banner."""
        self.stale_banner.setVisible(bool(stale))

    def _recheck_stale(self, *_a) -> None:
        """Re-flag the shown table stale if the inputs no longer match it."""
        entry = self._cache().get(self.kind.currentText())
        if entry is not None:
            self._set_stale(self._current_signature() != entry.get("signature"))

    def _fingerprint(self, ds) -> dict:
        """A cheap, stable summary of a dataset's state for staleness checks.

        Captures size/shape, time span, dtype, density and a numeric checksum of
        the total concentration, so cropping, rebinning, a density change or a
        calibration all change the fingerprint and mark the summary stale.
        """
        obj = ds.obj
        span = ds.time_span()
        try:
            checksum = round(
                float(np.nansum(np.asarray(obj.total_concentration, dtype=float))), 3
            )
        except Exception:
            checksum = None
        return {
            "label": ds.label,
            "instrument": ds.instrument,
            "n": ds.n_points(),
            "start": span[0].isoformat() if span else None,
            "end": span[1].isoformat() if span else None,
            "dtype": str(getattr(obj, "dtype", "")),
            "density": round(float(getattr(obj, "density", 0) or 0), 6),
            "checksum": checksum,
        }

    def _exposure_params(self) -> dict:
        """Current exposure-limit inputs (stored so they restore on reload)."""
        return {
            "metric": self._build_metric(),
            "metric_kind": self.metric_kind.currentText(),
            "metric_cut": self.metric_cut.currentText(),
            "short_limit": self.short_limit.text().strip(),
            "short_window": self.short_window.text().strip(),
            "long_limit": self.long_limit.text().strip(),
            "twa_window": self.twa_window.text().strip(),
        }

    def _activity_params(self) -> dict:
        """Current activity-summary inputs (stored so they restore on reload)."""
        return {
            "metrics_text": self.act_metrics.text().strip(),
            "stats": self._selected_stats(),
        }

    def _current_signature(self) -> dict:
        """Signature of everything that affects the result, for staleness checks."""
        proj = self.main.project
        kind = self.kind.currentText()
        sig = {
            "kind": kind,
            "datasets": [self._fingerprint(d) for d in self._selected_datasets()],
            "activities": {
                name: [
                    [pd.Timestamp(s).isoformat(), pd.Timestamp(e).isoformat()]
                    for s, e in periods
                ]
                for name, periods in sorted(proj.activities.items())
            },
        }
        if kind == "Exposure summary":
            p = self._exposure_params()
            sig["params"] = {
                k: p[k]
                for k in (
                    "metric",
                    "short_limit",
                    "short_window",
                    "long_limit",
                    "twa_window",
                )
            }
        elif kind == "Activity summary":
            sig["params"] = self._activity_params()
        return sig

    def _table_payload(self, df: pd.DataFrame):
        """Convert a table to JSON-safe ``(columns, records)`` for the project file."""
        cols = [str(c) for c in df.columns]
        records = []
        for row in df.itertuples(index=False, name=None):
            rec = []
            for v in row:
                if isinstance(v, (np.integer,)):
                    rec.append(int(v))
                elif isinstance(v, (np.bool_, bool)):
                    rec.append(bool(v))
                elif isinstance(v, (float, np.floating)):
                    f = float(v)
                    rec.append(f if np.isfinite(f) else None)
                elif v is None or isinstance(v, (int, str)):
                    rec.append(v)
                else:  # timestamps and the like
                    rec.append(str(v))
            records.append(rec)
        return cols, records

    def _store_cache(self, df: pd.DataFrame) -> None:
        """Persist the computed table + inputs + signature on the project."""
        kind = self.kind.currentText()
        if kind == "Exposure summary":
            params = self._exposure_params()
        elif kind == "Activity summary":
            params = self._activity_params()
        else:
            params = {}
        cols, records = self._table_payload(df)
        self._cache()[kind] = {
            "signature": self._current_signature(),
            "params": params,
            "columns": cols,
            "records": records,
        }
        self.main.project.summary_state["active_kind"] = kind
        self._set_stale(False)

    def _show_cache(self) -> None:
        """Display the cached table for the current kind and re-check staleness."""
        kind = self.kind.currentText()
        entry = self._cache().get(kind)
        if not entry:
            self.model.set_dataframe(pd.DataFrame())
            self._set_stale(False)
            self.status.setText(
                "Tick datasets and click Compute to build the combined summary."
            )
            return
        df = pd.DataFrame.from_records(
            entry.get("records", []), columns=entry.get("columns", [])
        )
        self.model.set_dataframe(df)
        self.status.setText(f"Stored summary — {len(df)} row(s). Recompute to refresh.")
        self._set_stale(self._current_signature() != entry.get("signature"))

    def _restore_params_from_cache(self, kind: str) -> None:
        """Restore the saved inputs for ``kind`` into the fields."""
        entry = self._cache().get(kind)
        params = entry.get("params") if entry else None
        if not params:
            return
        if kind == "Activity summary":
            self.act_metrics.blockSignals(True)
            self.act_metrics.setText(str(params.get("metrics_text") or ""))
            self.act_metrics.blockSignals(False)
            wanted = set(params.get("stats") or ["mean", "std"])
            for stat, box in self.act_stat_boxes.items():
                box.blockSignals(True)
                box.setChecked(stat in wanted)
                box.blockSignals(False)
            return
        for widget, value in (
            (self.short_limit, params.get("short_limit")),
            (self.short_window, params.get("short_window")),
            (self.long_limit, params.get("long_limit")),
            (self.twa_window, params.get("twa_window")),
        ):
            if value is not None:
                widget.blockSignals(True)
                widget.setText(str(value))
                widget.blockSignals(False)
        mk = params.get("metric_kind")
        if mk:
            self.metric_kind.blockSignals(True)
            idx = self.metric_kind.findText(mk)
            if idx >= 0:
                self.metric_kind.setCurrentIndex(idx)
            self.metric_kind.blockSignals(False)
        mc = params.get("metric_cut")
        if mc is not None:
            self.metric_cut.blockSignals(True)
            self.metric_cut.setCurrentText(str(mc))
            self.metric_cut.blockSignals(False)
        self._on_metric_kind_change()

    def _export(self) -> None:
        """Save the combined table to an .xlsx or .csv file."""
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        _export_table(self, df, f"{kind}_summary", with_index=False)
