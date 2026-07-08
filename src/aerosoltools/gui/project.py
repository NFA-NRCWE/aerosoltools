"""Project / dataset container for the multi-dataset GUI.

A :class:`Project` holds an ordered collection of :class:`Dataset` objects (the
files the user has loaded) plus a single *active* dataset that the single-view
tabs follow. User-defined activities/tasks live on the **project** (not on any
one dataset): they are stored once here and projected onto their in-scope
datasets' time axes. Each activity has a *scope* — a set of dataset ids, or
None for "all datasets" — so a task can apply to one instrument, a chosen few,
or every dataset, rather than always being forced onto all of them.

This module is intentionally Qt-free so the data model can be reasoned about and
tested independently of the widgets.
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from . import helpers

# Process-wide counter giving every dataset a stable, unique id.
_ID = itertools.count(1)

Period = Tuple[pd.Timestamp, pd.Timestamp]

# Fixed, theme-independent palette (Matplotlib "tab10") used to give every
# dataset a stable colour that follows it across every plot (Overlay, PSD, …),
# so the same dataset is always the same colour regardless of how many are
# shown or in what order. Users can override a dataset's colour (see the
# sidebar); the assignment here just picks a sensible, distinct default.
DEFAULT_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class Dataset:
    """One loaded file (or a derived/combined result) inside a project."""

    def __init__(
        self,
        obj,
        source_path: Optional[str],
        instrument_key: str,
        label: Optional[str] = None,
    ):
        """Create a dataset wrapper.

        Args:
            obj: The loaded aerosol object.
            source_path: Original file path, or None for a derived dataset.
            instrument_key: Loader / instrument name.
            label: Display name; defaults to the file stem or ``Dataset <id>``.
        """
        self.id: int = next(_ID)
        self.obj = obj
        self.source_path = source_path
        self.instrument_key = instrument_key
        if label:
            self.label = label
        elif source_path:
            self.label = os.path.splitext(os.path.basename(source_path))[0]
        else:
            self.label = f"Dataset {self.id}"
        # Raw files behind this dataset, used so 'Save project' archives every
        # source involved in the analysis. A plain loaded dataset is backed by
        # its own file; a derived/combined dataset is backed by the union of its
        # constituents' raw files (set via _add_derived_dataset).
        self.contributing_files: List[str] = [source_path] if source_path else []
        # Comparison-tab state (kept on the dataset so it survives tab
        # rebuilds): a *view-only* time shift for manual peak alignment in the
        # Overlay tab, and whether the dataset is included in the Overlay /
        # Combined PSD comparisons.
        self.view_shift: pd.Timedelta = pd.Timedelta(0)
        self.overlay_on: bool = True
        self.psd_on: bool = True
        self.summary_on: bool = True
        # Stable plot colour for this dataset, followed across every plot. None
        # until assigned by the owning Project (or the user); see DEFAULT_PALETTE.
        self.color: Optional[str] = None
        # Reversible calibration state (see gui/calibration.py). ``calibration``
        # is the applied spec (basis, factors, per-bin R²) or None; ``_cal_baseline``
        # is a snapshot of the uncalibrated object taken when calibration was first
        # applied, so the correction can be toggled on/off or reset from any pane.
        self.calibration: Optional[dict] = None
        self.calibration_enabled: bool = False
        self._cal_baseline = None
        # Lognormal PSD fits, keyed by activity name (e.g. "All data", "Task 1").
        # Each value is ``{"modes": [{mu, sigma, peak, bound}, ...],
        # "optimized": bool}``. Stored per dataset because a fit describes that
        # instrument's PSD for one activity; persisted in the project file and
        # dropped when the activity is edited or removed (see Project).
        self.psd_fits: Dict[str, dict] = {}

    @property
    def instrument(self) -> str:
        """Instrument name (the loader key, falling back to the object's own)."""
        return self.instrument_key or getattr(self.obj, "instrument", "Unknown")

    @property
    def serial_number(self) -> str:
        """Instrument serial number, or a placeholder when unavailable."""
        return getattr(self.obj, "serial_number", "Unknown serial number")

    def time_span(self) -> Optional[Period]:
        """Return ``(start, end)`` timestamps, or ``None`` for empty data."""
        t = getattr(self.obj, "time", None)
        if t is None or len(t) == 0:
            return None
        return pd.Timestamp(t.min()), pd.Timestamp(t.max())

    def n_points(self) -> int:
        """Number of time steps in the dataset (0 if empty)."""
        return int(self.obj.data.shape[0]) if self.obj is not None else 0


class Project:
    """An ordered set of datasets with a shared, project-level task registry."""

    def __init__(self, name: str = "Untitled project"):
        """Create an empty project with the given name."""
        self.name = name
        self.datasets: List[Dataset] = []
        self.active_id: Optional[int] = None
        # name -> list of (start, end). "All data" is intentionally NOT stored
        # here; each dataset manages its own "All data" span over its own range.
        self.activities: Dict[str, List[Period]] = {}
        # name -> the set of dataset ids the activity applies to, or None meaning
        # "all datasets" (the shared default and the back-compat state for
        # projects saved before scoping existed). A scoped activity is only
        # projected onto its listed datasets, so marking a task on one instrument
        # need not force it onto instruments that have no data in that span.
        self.activity_scopes: Dict[str, Optional[Set[int]]] = {}
        # Cached Summary-tab results so reopening a project shows the computed
        # values (and the STEL/OEL/window inputs) directly, without recomputing.
        # ``cache`` is keyed by summary kind; each entry holds the table plus an
        # input ``signature`` used to flag the values stale when tasks/data/
        # settings change. See gui/tabs/summary.py and projectio.py.
        self.summary_state: Dict = {"active_kind": None, "cache": {}}
        # Concentration-threshold (e.g. OEL) overlays, keyed by a plot tab's
        # ``export_tag`` ("timeseries"/"overlay"/"heatmap"). Each value is a
        # ``{"on": bool, "value": str, "label": str}`` state. Held on the project
        # (not the tab) so the line survives tab rebuilds and is saved with the
        # project. See gui/widgets.ThresholdControls and projectio.py.
        self.plot_thresholds: Dict[str, dict] = {}

    # -- dataset access ----------------------------------------------------
    @property
    def active(self) -> Optional[Dataset]:
        """The active :class:`Dataset`, or None."""
        return self.get(self.active_id)

    def get(self, ds_id: Optional[int]) -> Optional[Dataset]:
        """Return the dataset with id ``ds_id``, or None."""
        if ds_id is None:
            return None
        return next((d for d in self.datasets if d.id == ds_id), None)

    def index_of(self, ds_id: int) -> int:
        """Return the position of ``ds_id`` in the dataset list, or -1."""
        for i, d in enumerate(self.datasets):
            if d.id == ds_id:
                return i
        return -1

    def add_dataset(self, ds: Dataset) -> Dataset:
        """Append a dataset, projecting the shared activities onto it."""
        self.datasets.append(ds)
        self._apply_activities(ds)
        self.assign_color(ds)
        if self.active_id is None:
            self.active_id = ds.id
        return ds

    def assign_color(self, ds: Dataset) -> None:
        """Give ``ds`` a stable default colour if it has none.

        Picks the first palette colour not already used by another dataset, so
        every dataset starts out visually distinct; falls back to cycling the
        palette once every colour is in use. A colour the user set is kept.
        """
        if ds.color:
            return
        used = {d.color for d in self.datasets if d is not ds and d.color}
        for c in DEFAULT_PALETTE:
            if c not in used:
                ds.color = c
                return
        ds.color = DEFAULT_PALETTE[(len(self.datasets) - 1) % len(DEFAULT_PALETTE)]

    def remove_dataset(self, ds_id: int) -> None:
        """Remove a dataset, dropping it from any activity scopes and reactivating."""
        ds = self.get(ds_id)
        if ds is None:
            return
        self.datasets.remove(ds)
        # Drop the removed dataset from every explicit activity scope so a stale
        # id can never keep an activity "assigned" to a dataset that is gone.
        for ids in self.activity_scopes.values():
            if ids is not None:
                ids.discard(ds_id)
        if self.active_id == ds_id:
            self.active_id = self.datasets[0].id if self.datasets else None

    def set_active(self, ds_id: int) -> None:
        """Make ``ds_id`` the active dataset if it exists."""
        if self.get(ds_id) is not None:
            self.active_id = ds_id

    # -- project-level activities (optionally scoped to some datasets) ------
    def user_activities(self) -> List[str]:
        """Names of the user tasks (excludes per-dataset 'All data')."""
        return list(self.activities.keys())

    def activity_scope(self, name: str) -> Optional[Set[int]]:
        """The dataset ids a task applies to, or None for 'all datasets'."""
        return self.activity_scopes.get(name)

    def applies_to(self, name: str, ds_id: int) -> bool:
        """Whether task ``name`` applies to dataset ``ds_id`` (None scope = all)."""
        ids = self.activity_scopes.get(name)
        return ids is None or ds_id in ids

    def _project_activity(self, ds: Dataset, name: str, periods) -> None:
        """Apply (in scope) or remove (out of scope) one task on one dataset."""
        if self.applies_to(name, ds.id):
            helpers.set_activity_periods(ds.obj, name, periods)
        else:
            helpers.delete_activity(ds.obj, name)

    def _apply_activities(self, ds: Dataset) -> None:
        """(Re)apply every in-scope activity onto a single dataset's time axis."""
        for name, periods in self.activities.items():
            self._project_activity(ds, name, periods)

    def reapply_all(self) -> None:
        """Re-project the whole registry onto every dataset (honouring scope)."""
        for ds in self.datasets:
            self._apply_activities(ds)

    def add_activity(
        self, name: str, start, end, scope: Optional[Set[int]] = None
    ) -> None:
        """Append one occurrence to a task and sync it onto its datasets.

        ``scope`` sets the datasets a *new* task applies to (a set of dataset
        ids, or None for all datasets); it is ignored for a task that already
        exists (its current scope is kept, the occurrence is just added).
        """
        if name not in self.activities:
            self.activity_scopes[name] = None if scope is None else set(scope)
        periods = list(self.activities.get(name, []))
        periods.append((pd.Timestamp(start), pd.Timestamp(end)))
        self.set_activity_periods(name, periods)

    def set_activity_periods(self, name: str, periods) -> None:
        """Replace a task's full period list and sync it onto its datasets.

        Editing a task changes which samples it covers, so any stored PSD fit
        for it is now stale and is dropped (the user re-fits the new data).
        """
        norm: List[Period] = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in periods]
        self.activities[name] = norm
        self.activity_scopes.setdefault(name, None)
        for ds in self.datasets:
            self._project_activity(ds, name, norm)
            ds.psd_fits.pop(name, None)

    def set_activity_scope(self, name: str, scope: Optional[Set[int]]) -> None:
        """Set which datasets a task applies to and re-project it accordingly.

        Args:
            name: Task to rescope.
            scope: Set of dataset ids the task applies to, or None for all
                datasets. Datasets dropped from the scope lose the task's mask;
                datasets added to it gain it. A rescoped dataset's stale PSD fit
                for the task is dropped.
        """
        if name not in self.activities:
            return
        self.activity_scopes[name] = None if scope is None else set(scope)
        periods = self.activities[name]
        for ds in self.datasets:
            self._project_activity(ds, name, periods)
            ds.psd_fits.pop(name, None)

    def delete_activity(self, name: str) -> None:
        """Remove a task from the registry, every dataset, and its PSD fits."""
        self.activities.pop(name, None)
        self.activity_scopes.pop(name, None)
        for ds in self.datasets:
            helpers.delete_activity(ds.obj, name)
            ds.psd_fits.pop(name, None)

    def rename_activity(self, old_name: str, new_name: str) -> None:
        """Rename a shared task across the registry, every dataset, and PSD fits.

        Args:
            old_name: Current task name.
            new_name: New task name; must not already be in use.

        Raises:
            ValueError: If ``old_name`` is unknown or ``new_name`` collides
                with an existing task.
        """
        if old_name == new_name:
            return
        if old_name not in self.activities:
            raise ValueError(f"No task named {old_name!r} to rename.")
        if new_name in self.activities:
            raise ValueError(f"A task named {new_name!r} already exists.")
        self.activities[new_name] = self.activities.pop(old_name)
        if old_name in self.activity_scopes:
            self.activity_scopes[new_name] = self.activity_scopes.pop(old_name)
        for ds in self.datasets:
            helpers.rename_activity(ds.obj, old_name, new_name)
            if old_name in ds.psd_fits:
                ds.psd_fits[new_name] = ds.psd_fits.pop(old_name)
