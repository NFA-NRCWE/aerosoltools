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
import re

import numpy as np
import pandas as pd

from ..._core import _stats
from ..._core.metrics import canonical_unit, convert_value, unit_key
from ..metric_picker import MetricPickerDialog, default_keys, metric_catalog
from ..models import PandasTableModel
from ..qt import QtCore, QtWidgets
from ..summary_cache import SummaryCacheEntry
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

        # Metric selection: a picker dialog (grouped by instrument, primary
        # metrics pre-checked) replaces the old free-text / kind+cut controls, so
        # only metrics an instrument actually measures can be chosen and the same
        # metric at different unit scales merges into one column.
        self.metric_btn = QtWidgets.QPushButton("Choose metrics…")
        self.metric_btn.setToolTip(
            "Pick which metrics to summarise, grouped by instrument. Each metric "
            "is computed only for datasets that provide it; comparable quantities "
            "at different unit scales (e.g. ng/m³ and µg/m³) merge into one column."
        )
        self.metric_btn.clicked.connect(self._open_metric_picker)
        bar.addWidget(self.metric_btn)
        self.metric_summary = QtWidgets.QLabel("")
        self.metric_summary.setStyleSheet("color: palette(mid);")
        bar.addWidget(self.metric_summary, stretch=1)
        # Selected metric keys, kept per kind so each summary remembers its own.
        self._metric_keys_by_kind: dict[str, list[str]] = {
            "Activity summary": [],
            "Exposure summary": [],
        }

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

        # Second row (activity-only): which per-activity statistics to report.
        # (Which metrics is chosen via the shared "Choose metrics…" picker above.)
        self.act_bar = QtWidgets.QHBoxLayout()
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
            "STEL (short-term limit):",
            "1.0",
            width=80,
            tip="Short-term exposure limit. The highest short-window average is "
            "compared against this value (same unit as the chosen metric).",
        )
        self.short_window = self._add_field(
            "over",
            "15min",
            width=70,
            tip="Averaging window for the short-term (STEL) check, as a pandas "
            "offset, e.g. 15min.",
        )
        self.long_limit = self._add_field(
            "OEL (8h limit):",
            "1.0",
            width=80,
            tip="Occupational exposure limit. The time-weighted average is "
            "compared against this value (same unit as the chosen metric).",
        )
        self.twa_window = self._add_field(
            "TWA window",
            "8h",
            width=70,
            tip="Averaging window for the time-weighted average (TWA), e.g. 8h.",
        )
        self.exp_bar.addStretch(1)
        right.addLayout(self.exp_bar)

        # Editing any limit makes the shown (cached) table no longer match the
        # inputs, so re-evaluate staleness as the user types.
        for field in self._limit_fields():
            field.editingFinished.connect(self._recheck_stale)
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
        widgets = []
        for field in self._limit_fields():
            widgets.append(field)
            label = getattr(field, "_label", None)
            if label is not None:
                widgets.append(label)
        return widgets

    def _activity_widgets(self):
        """Widgets shown only in Activity summary mode."""
        return [self.act_stats_label, *self.act_stat_boxes.values()]

    def _selected_stats(self) -> list[str]:
        """Ticked stat names, in a fixed order; falls back to ["mean"]."""
        order = ("mean", "std", "min", "max", "median")
        stats = [s for s in order if self.act_stat_boxes[s].isChecked()]
        return stats or ["mean"]

    # -- metric selection --------------------------------------------------
    def _current_metric_keys(self) -> list[str]:
        """Selected metric keys for the current kind (instrument defaults if none)."""
        keys = self._metric_keys_by_kind.get(self.kind.currentText()) or []
        return list(keys) if keys else default_keys(self._selected_datasets())

    def _open_metric_picker(self) -> None:
        """Open the grouped metric picker and store the chosen keys."""
        datasets = self._selected_datasets()
        if not datasets:
            QtWidgets.QMessageBox.information(
                self, "Choose metrics", "Tick at least one dataset first."
            )
            return
        single = self.kind.currentText() == "Exposure summary"
        dlg = MetricPickerDialog(
            datasets, self._current_metric_keys(), single=single, parent=self
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        keys = dlg.selected_keys()
        if single:
            keys = keys[:1]
        self._metric_keys_by_kind[self.kind.currentText()] = keys
        self._update_metric_summary()
        self._recheck_stale()

    def _update_metric_summary(self) -> None:
        """Refresh the label describing the current metric selection."""
        chosen = self._metric_keys_by_kind.get(self.kind.currentText()) or []
        keys = self._current_metric_keys()
        prefix = "" if chosen else "default — "
        self.metric_summary.setText(
            "Metrics: " + (prefix + ", ".join(keys) if keys else "none available")
        )

    def _canonical_units(self, datasets) -> dict:
        """Map each metric key → the unit its merged column should use.

        Uses the **first-seen native unit** for a metric, so a single-scale
        metric keeps its natural unit (e.g. black carbon stays ng/m³ rather than
        being force-converted to the dimension's canonical µg/m³); other
        instruments reporting that same metric at a different scale are converted
        to this unit so they still share one column.
        """
        canon: dict = {}
        for _instr, specs in metric_catalog(datasets):
            for m in specs:
                canon.setdefault(m.key, m.unit)
        return canon

    #: Parse a "key [unit] stat" activity-summary value column.
    _COL_RE = re.compile(
        r"^(?P<key>.+) \[(?P<unit>.+)\] (?P<stat>mean|std|min|max|median)$"
    )

    def _unify_activity_units(self, df: pd.DataFrame, canon: dict) -> pd.DataFrame:
        """Convert each metric column to its canonical unit and relabel it.

        So the *same* metric reported by different instruments at different unit
        scales (e.g. black carbon in ng/m³ and µg/m³) lands in one column.
        """
        keep = {"Segment", "Duration (HH:MM)"}
        rename: dict = {}
        for col in df.columns:
            if col in keep:
                continue
            m = self._COL_RE.match(str(col))
            if not m:
                continue
            key, unit, stat = m["key"], m["unit"], m["stat"]
            target = canon.get(key) or canonical_unit(unit)
            if unit_key(unit) != unit_key(target):
                df[col] = convert_value(
                    pd.to_numeric(df[col], errors="coerce"), unit, target
                )
            rename[col] = f"{key} [{target}] {stat}"
        return df.rename(columns=rename) if rename else df

    @staticmethod
    def _unify_exposure_units(
        df: pd.DataFrame, native: str, target: str
    ) -> pd.DataFrame:
        """Convert an exposure table's metric-unit columns to the canonical unit."""
        if not native or not target or unit_key(native) == unit_key(target):
            return df
        scale = convert_value(1.0, native, target)
        nk = unit_key(native)
        rename: dict = {}
        for col in df.columns:
            m = re.search(r"\[(.+)\]$", str(col))
            if m and unit_key(m.group(1)) == nk:
                df[col] = pd.to_numeric(df[col], errors="coerce") * scale
                rename[col] = str(col)[: m.start(1)] + target + "]"
        return df.rename(columns=rename) if rename else df

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
        self._update_metric_summary()

    def _on_kind_change(self) -> None:
        """Rebuild options for the chosen kind, then show its cached table.

        Restores that kind's saved inputs and table (so switching back to a
        previously-computed kind shows it without recomputing) and re-checks
        whether the shown values are now stale.
        """
        kind = self.kind.currentText()
        self._apply_kind_visibility()
        self._restore_params_from_cache(kind)
        self._update_metric_summary()
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
        # The available metrics may change with the selection (e.g. a dataset
        # added/removed), so refresh the metric-summary label.
        self._update_metric_summary()
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
            if (
                active
                and self.kind.findText(active) >= 0
                and active != self.kind.currentText()
            ):
                # Switching kind triggers _on_kind_change, which restores that
                # kind's params + table and checks staleness.
                self.kind.setCurrentText(active)
                return
        self._update_metric_summary()
        self._show_cache()

    # -- compute -----------------------------------------------------------
    def _compute(self) -> None:
        """Run the summary per ticked dataset and concatenate the tables.

        Each selected metric is computed only for datasets that actually provide
        it (via ``available_metrics``); a metric is then converted to its
        canonical unit so the *same* quantity reported at different unit scales
        merges into one column, and datasets that don't provide it leave blanks.
        """
        datasets = self._selected_datasets()
        if not datasets:
            self.model.set_dataframe(pd.DataFrame())
            self.status.setText("Tick at least one dataset to include.")
            return
        exposure = self.kind.currentText() == "Exposure summary"
        keys = self._current_metric_keys()
        if exposure:
            keys = keys[:1]  # exposure summarises one metric at a time
        act_stats = self._selected_stats()
        canon = self._canonical_units(datasets)

        frames: list[pd.DataFrame] = []
        skipped: list[str] = []
        # The core summarize_* methods print their result table to stdout; with
        # many datasets that is just noise (the user reads the GUI table), and on
        # a non-UTF-8 console the unit glyphs (µg/m³, cm⁻³) can even raise an
        # encoding error mid-method. Swallow that console output during compute.
        with contextlib.redirect_stdout(io.StringIO()):
            for ds in datasets:
                obj = ds.obj
                try:
                    ds_specs = obj.available_metrics()
                except Exception:
                    ds_specs = []
                native = {m.key: m.unit for m in ds_specs}
                applicable = [k for k in keys if k in native]
                if not applicable:
                    continue  # this dataset provides none of the chosen metrics
                try:
                    if exposure:
                        key = applicable[0]
                        df = obj.summarize_exposure(
                            metric=key,
                            short_limit=self._to_float(self.short_limit.text(), 1.0),
                            long_limit=self._to_float(self.long_limit.text(), 1.0),
                            short_window=self.short_window.text().strip() or "15min",
                            twa_window=self.twa_window.text().strip() or "8h",
                        )
                        df = self._unify_exposure_units(
                            df, native.get(key), canon.get(key)
                        )
                    else:
                        df = obj.summarize_activities(
                            metrics=applicable, stats=act_stats
                        )
                        df = self._clarify_activity_columns(df)
                        df = self._unify_activity_units(df, canon)
                except Exception as exc:
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
            self._set_stale(entry.is_stale(self._current_signature()))

    def _fingerprint(self, ds) -> dict:
        """A cheap, stable summary of a dataset's state for staleness checks.

        Captures size/shape, time span, dtype, density and a numeric checksum of
        the primary series, so cropping, rebinning, a density change or a
        calibration all change the fingerprint and mark the summary stale.
        """
        obj = ds.obj
        span = ds.time_span()
        try:
            checksum = round(float(np.nansum(np.asarray(obj._primary, dtype=float))), 3)
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
            "metric_keys": list(
                self._metric_keys_by_kind.get("Exposure summary") or []
            ),
            "short_limit": self.short_limit.text().strip(),
            "short_window": self.short_window.text().strip(),
            "long_limit": self.long_limit.text().strip(),
            "twa_window": self.twa_window.text().strip(),
        }

    def _activity_params(self) -> dict:
        """Current activity-summary inputs (stored so they restore on reload)."""
        return {
            "metric_keys": list(
                self._metric_keys_by_kind.get("Activity summary") or []
            ),
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
            # Rescoping a task changes which datasets contribute its rows, so the
            # scope is part of what makes a cached summary stale.
            "activity_scopes": {
                name: (
                    None if (ids := proj.activity_scope(name)) is None else sorted(ids)
                )
                for name in sorted(proj.activities)
            },
        }
        if kind == "Exposure summary":
            p = self._exposure_params()
            sig["params"] = {
                k: p[k]
                for k in (
                    "metric_keys",
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
        self._cache()[kind] = SummaryCacheEntry(
            signature=self._current_signature(),
            params=params,
            columns=cols,
            records=records,
        )
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
        df = entry.dataframe()
        self.model.set_dataframe(df)
        self.status.setText(f"Stored summary — {len(df)} row(s). Recompute to refresh.")
        self._set_stale(entry.is_stale(self._current_signature()))

    def _restore_params_from_cache(self, kind: str) -> None:
        """Restore the saved inputs for ``kind`` into the fields."""
        entry = self._cache().get(kind)
        params = entry.params if entry else None
        if not params:
            return
        # Restore the saved metric selection for this kind.
        self._metric_keys_by_kind[kind] = list(params.get("metric_keys") or [])
        if kind == "Activity summary":
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

    def _export(self) -> None:
        """Save the combined table to an .xlsx or .csv file."""
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        _export_table(self, df, f"{kind}_summary", with_index=False)
