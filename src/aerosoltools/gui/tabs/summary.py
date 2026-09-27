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
from ...exposure_limits import applicable_limit, basis_note
from ..logic import exposure_limits
from ..qt import QtCore, QtWidgets
from ..state.summary_cache import SummaryCacheEntry
from ..view.exposure_limit_picker import (
    ExposureLimitCombo,
    format_value,
    pick_record,
    record_candidates,
    record_limit,
)
from ..view.metric_picker import MetricPickerDialog, default_keys, metric_catalog
from ..view.models import PandasTableModel
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

        # Substance row (exposure only): picking what is being measured from the
        # Danish limit-value order fills the STEL/OEL fields below — converted to
        # the metric's unit — and records the order's number and date with the
        # result, so it is clear which limits a summary was compared against.
        self.oel_bar = QtWidgets.QHBoxLayout()
        self.oel_label = QtWidgets.QLabel("Substance:")
        self.oel_combo = ExposureLimitCombo()
        self.oel_combo.set_limits(exposure_limits.active_list())
        self.oel_combo.picked.connect(self._on_oel_picked)
        self.oel_source = QtWidgets.QLabel("")
        self.oel_source.setStyleSheet("color: palette(mid);")
        tip = (
            "The substance being measured. Its 8-hour and short-term limits from "
            "the Danish limit-value order fill the OEL and STEL fields, converted "
            "to the chosen metric's unit. Typing a limit by hand clears the pick."
        )
        self.oel_label.setToolTip(tip)
        self.oel_bar.addWidget(self.oel_label)
        self.oel_bar.addWidget(self.oel_combo)
        self.oel_bar.addWidget(self.oel_source, stretch=1)
        right.addLayout(self.oel_bar)
        # The picked substance (a view.exposure_limit_picker.pick_record), or None
        # when the limits were typed.
        self._oel_pick: dict | None = None
        self._sync_oel_source()

        # Third row: exposure-limit parameters (only shown for exposure).
        self.exp_bar = QtWidgets.QHBoxLayout()
        self.short_limit = self._add_field(
            "STEL (short-term limit):",
            "1.0",
            width=80,
            tip="Short-term exposure limit. The highest short-window average is "
            "compared against this value (in the unit shown next to it).",
            unit=True,
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
            "compared against this value (in the unit shown next to it).",
            unit=True,
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
        # A limit typed by hand no longer matches the picked substance.
        for field in (self.short_limit, self.long_limit):
            field.textEdited.connect(self._clear_oel_pick)
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
        self,
        label: str,
        default: str,
        width: int,
        tip: str | None = None,
        unit: bool = False,
    ) -> QtWidgets.QLineEdit:
        """Add a labelled line-edit to the exposure-parameter row and return it.

        Args:
            label: Caption shown to the left of the field.
            default: Initial text.
            width: Fixed field width in pixels.
            tip: Optional tooltip applied to both the label and the field.
            unit: Add a label after the field showing the metric's unit (kept
                current by :meth:`_sync_limit_units`).
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
        edit._unit = None  # type: ignore[attr-defined]
        if unit:
            edit._unit = QtWidgets.QLabel("")  # type: ignore[attr-defined]
            self.exp_bar.addWidget(edit._unit)
        return edit

    def _selected_datasets(self) -> list:
        """Datasets currently ticked for inclusion."""
        return [d for d in self.main.project.datasets if d.summary_on]

    def _limit_fields(self):
        """The four exposure-limit line-edits."""
        return (self.short_limit, self.short_window, self.long_limit, self.twa_window)

    def _exposure_widgets(self):
        """Widgets (and their labels) shown only in Exposure mode."""
        widgets = [self.oel_label, self.oel_combo, self.oel_source]
        for field in self._limit_fields():
            widgets.append(field)
            for extra in (
                getattr(field, "_label", None),
                getattr(field, "_unit", None),
            ):
                if extra is not None:
                    widgets.append(extra)
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
        if single:
            self._apply_oel_pick()  # re-express the limits in the new metric's unit
        self._recheck_stale()

    def _update_metric_summary(self) -> None:
        """Refresh the label describing the current metric selection."""
        chosen = self._metric_keys_by_kind.get(self.kind.currentText()) or []
        keys = self._current_metric_keys()
        prefix = "" if chosen else "default — "
        self.metric_summary.setText(
            "Metrics: " + (prefix + ", ".join(keys) if keys else "none available")
        )
        self._sync_limit_units()

    # -- exposure limits (substance pick) -----------------------------------
    def _exposure_unit(self) -> str | None:
        """Unit of the exposure metric's column — the unit the limit fields use."""
        datasets = self._selected_datasets()
        keys = self._current_metric_keys()[:1]
        if not datasets or not keys:
            return None
        return self._canonical_units(datasets).get(keys[0])

    def _sync_limit_units(self) -> None:
        """Show the exposure metric's unit next to the STEL/OEL fields."""
        unit = self._exposure_unit() or ""
        for field in (self.short_limit, self.long_limit):
            if getattr(field, "_unit", None) is not None:
                field._unit.setText(unit)

    def _sync_oel_source(self) -> None:
        """Describe the picked substance's source (or the list offered)."""
        pick = self._oel_pick
        if pick is None:
            source = self.oel_combo.limits()
            self.oel_source.setText(
                f"Limits from {source.source.label}" if source else ""
            )
            self.oel_source.setToolTip("")
            return
        variants = record_candidates(pick)
        fractions = " / ".join(dict.fromkeys(v.fraction for v in variants))
        self.oel_source.setText(f"from {pick.get('source')} · {fractions}")
        self.oel_source.setToolTip(
            f"{pick.get('eli')}\nLimits exist for the {fractions} fraction(s); the "
            "one that fits the exposure metric is used, and none when no fraction "
            "fits it."
        )

    def _on_oel_picked(self) -> None:
        """A substance (or "none") was chosen: record it and fill the limits."""
        limit = self.oel_combo.current_limit()
        limits = self.oel_combo.limits()
        self._oel_pick = pick_record(limit, limits) if limit is not None else None
        self._apply_oel_pick()
        self._recheck_stale()

    def _clear_oel_pick(self, *_args) -> None:
        """A limit was typed by hand: it no longer belongs to the picked substance."""
        if self._oel_pick is not None:
            self._oel_pick = None
            self.oel_combo.set_current_name(None)
            self._sync_oel_source()

    def _metric_dataset(self):
        """``(dataset, key)``: the first ticked dataset providing the exposure metric."""
        keys = self._current_metric_keys()[:1]
        if not keys:
            return None, None
        for ds in self._selected_datasets():
            try:
                specs = ds.obj.available_metrics()
            except Exception:
                continue
            if any(m.key == keys[0] for m in specs):
                return ds, keys[0]
        return None, keys[0]

    def _resolve_pick(self, obj, key: str):
        """The picked substance's fraction variant that fits metric ``key`` of ``obj``.

        Returns:
            ``(limit, check)`` from ``exposure_limits.applicable_limit`` — e.g.
            "Kvarts, total" for a Total channel although "Kvarts, respirabel"
            was picked, or the pick with the reason no variant fits.
        """
        cut = exposure_limits.series_size_cut(obj, key)
        return applicable_limit(record_candidates(self._oel_pick), key, cut)

    def _set_limit_fields(self, twa: float | None, stel: float | None) -> None:
        """Show limits in the OEL/STEL fields (blank for none)."""
        self.long_limit.setText(format_value(twa))
        self.short_limit.setText(format_value(stel))

    def _apply_oel_pick(self) -> None:
        """Fill the STEL/OEL fields with the limits that apply to the metric.

        The fraction variant of the picked substance that fits the exposure
        metric is used (the total-dust limit for a Total channel, the respirable
        one for PM4), converted to the metric's unit. When none fits — or the
        metric is not a mass concentration — the fields are cleared and the
        status line says why; the summary then lists no limit for that metric.
        """
        self._sync_oel_source()
        self._sync_limit_units()
        pick = self._oel_pick
        if pick is None:
            return
        picked = record_limit(pick)
        ds, key = self._metric_dataset()
        unit = self._exposure_unit()
        if ds is None:
            self.status.setText(
                f"{picked.name}: tick a dataset that provides the exposure metric "
                "to fill in its limits."
            )
            return
        if not picked.is_comparable_to(unit):
            self._set_limit_fields(None, None)
            self.status.setText(
                f"{picked.name}: its limits are in {picked.unit}, but '{key}' is in "
                f"{unit or 'no known unit'}. Choose a mass metric with “Choose "
                "metrics…”."
            )
            return
        limit, check = self._resolve_pick(ds.obj, key)
        if not check.applies:
            self._set_limit_fields(None, None)
            self.status.setText(
                f"No limit for {picked.base_name} applies to {key}: {check.message} "
                "The summary lists no limit for it."
            )
            return
        self._set_limit_fields(limit.twa_in(unit), limit.stel_in(unit))
        self.short_window.setText(f"{limit.stel_minutes}min")
        parts = [
            f"Limits for {limit.name} ({pick.get('source')}) filled in {unit} "
            f"for {key}."
        ]
        if limit.name != picked.name:
            parts.append(f"Its {limit.fraction} variant is the one that fits {key}.")
        if limit.stel_derived:
            parts.append(f"STEL = 2 × OEL ({limit.stel_rule}).")
        if limit.twa is None:
            parts.append("The order gives no 8-hour limit.")
        if limit.stel is None:
            parts.append("The order gives no short-term limit.")
        if limit.ceiling:
            parts.append("The short-term value is a ceiling: compare it with Max.")
        parts.extend(n for n in (check.message, basis_note(limit)) if n)
        self.status.setText(" ".join(parts))

    def _pick_limits_for(self, obj, key: str, unit: str | None) -> dict:
        """The picked substance's limits for one dataset's metric, in its unit.

        Returns:
            ``{"limit", "twa", "stel", "applies"}`` — ``twa``/``stel`` are
            ``None`` when that limit does not exist or does not apply, and
            ``applies`` is the text for the table's "Limit applies" column.
        """
        picked = record_limit(self._oel_pick)
        if not unit or not picked.is_comparable_to(unit):
            return {
                "limit": picked,
                "twa": None,
                "stel": None,
                "applies": f"no – {key} is not a mass concentration",
            }
        limit, check = self._resolve_pick(obj, key)
        if not check.applies:
            return {
                "limit": limit,
                "twa": None,
                "stel": None,
                "applies": f"no – {check.message}",
            }
        notes = [n for n in (check.message, basis_note(limit)) if n]
        if limit.ceiling:
            notes.append("the short-term value is a ceiling: compare it with Max")
        return {
            "limit": limit,
            "twa": limit.twa_in(unit),
            "stel": limit.stel_in(unit),
            "applies": "yes" + (" – " + " ".join(notes) if notes else ""),
        }

    def _tag_limits(self, df: pd.DataFrame, info: dict) -> pd.DataFrame:
        """Blank limit columns that do not apply and record which limit was used."""
        limit = info["limit"]
        for col in df.columns:
            name = str(col)
            if info["twa"] is None and name.startswith("Exposure limit"):
                df[col] = np.nan
            stel_gone = info["stel"] is None or (
                limit.ceiling and not name.startswith("STEL [")
            )
            if stel_gone and name.startswith("STEL") and "window" not in name:
                df[col] = np.nan
        df["Substance"] = limit.name
        df["Limit source"] = self._oel_pick.get("source")
        df["Limit applies"] = info["applies"]
        return df

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
        # ...and with it the unit the limit fields are in.
        if self.kind.currentText() == "Exposure summary":
            self._apply_oel_pick()
        # A different dataset selection means the shown table no longer matches.
        self._recheck_stale()

    def refresh(self) -> None:
        """Re-sync the dataset list and show any cached summary for this kind.

        On the first refresh after a project load, restore the kind that was
        last computed (and its saved inputs); thereafter just re-display the
        cached table and re-evaluate staleness.
        """
        self._sync_datasets()
        # Offer the newest limit list (a newer order may have been fetched).
        latest = exposure_limits.active_list()
        if self.oel_combo.limits() is not latest:
            self.oel_combo.set_limits(latest)
            self.oel_combo.set_current_name(
                self._oel_pick["substance"] if self._oel_pick else None
            )
        self._sync_oel_source()
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
        not_applied: list[str] = []  # datasets the picked substance's limits miss
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
                        info = None
                        if self._oel_pick is not None:
                            # A picked substance: the limits that apply to this
                            # dataset's metric, in its own unit (or none).
                            info = self._pick_limits_for(obj, key, native.get(key))
                            short = info["stel"] if info["stel"] is not None else 1.0
                            long_ = info["twa"] if info["twa"] is not None else 1.0
                        else:
                            # The limit fields are in the merged column's unit;
                            # the core compares in this dataset's own unit.
                            short = self._to_float(self.short_limit.text(), 1.0)
                            long_ = self._to_float(self.long_limit.text(), 1.0)
                            if canon.get(key) and native.get(key):
                                short = convert_value(short, canon[key], native[key])
                                long_ = convert_value(long_, canon[key], native[key])
                        df = obj.summarize_exposure(
                            metric=key,
                            short_limit=short,
                            long_limit=long_,
                            short_window=self.short_window.text().strip() or "15min",
                            twa_window=self.twa_window.text().strip() or "8h",
                        )
                        if info is not None:
                            df = self._tag_limits(df.copy(), info)
                            if info["twa"] is None and info["stel"] is None:
                                not_applied.append(ds.label)
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
        if not_applied:
            note += (
                "  No exposure limit applies to the metric of: "
                + ", ".join(not_applied)
                + " (see the 'Limit applies' column)."
            )
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
            "oel": dict(self._oel_pick) if self._oel_pick else None,
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
            # Only when picked, so summaries cached before substance picks
            # existed are not all flagged stale.
            if p["oel"]:
                sig["params"]["oel"] = [
                    p["oel"].get("substance"),
                    p["oel"].get("source"),
                ]
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
        oel = params.get("oel")
        self._oel_pick = dict(oel) if isinstance(oel, dict) else None
        self.oel_combo.set_current_name(
            self._oel_pick["substance"] if self._oel_pick else None
        )
        self._sync_oel_source()

    def _export(self) -> None:
        """Save the combined table to an .xlsx or .csv file."""
        df = self.model.dataframe
        kind = (
            "exposure" if self.kind.currentText() == "Exposure summary" else "activity"
        )
        _export_table(self, df, f"{kind}_summary", with_index=False)
