"""Project / dataset container for the multi-dataset GUI.

A :class:`Project` holds an ordered collection of :class:`Dataset` objects (the
files the user has loaded) plus a single *active* dataset that the single-view
tabs follow. User-defined activities/tasks live on the **project** (not on any
one dataset): they are stored once here and projected onto every dataset's time
axis, so marking a task on one instrument applies it to all of them.

This module is intentionally Qt-free so the data model can be reasoned about and
tested independently of the widgets.
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import helpers

# Process-wide counter giving every dataset a stable, unique id.
_ID = itertools.count(1)

Period = Tuple[pd.Timestamp, pd.Timestamp]


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
        if self.active_id is None:
            self.active_id = ds.id
        return ds

    def remove_dataset(self, ds_id: int) -> None:
        """Remove a dataset, reassigning the active id when needed."""
        ds = self.get(ds_id)
        if ds is None:
            return
        self.datasets.remove(ds)
        if self.active_id == ds_id:
            self.active_id = self.datasets[0].id if self.datasets else None

    def set_active(self, ds_id: int) -> None:
        """Make ``ds_id`` the active dataset if it exists."""
        if self.get(ds_id) is not None:
            self.active_id = ds_id

    # -- shared (project-level) activities --------------------------------
    def user_activities(self) -> List[str]:
        """Names of the shared user tasks (excludes per-dataset 'All data')."""
        return list(self.activities.keys())

    def _apply_activities(self, ds: Dataset) -> None:
        """(Re)apply every shared activity onto a single dataset's time axis."""
        for name, periods in self.activities.items():
            helpers.set_activity_periods(ds.obj, name, periods)

    def reapply_all(self) -> None:
        """Re-project the whole shared registry onto every dataset."""
        for ds in self.datasets:
            self._apply_activities(ds)

    def add_activity(self, name: str, start, end) -> None:
        """Append one occurrence to a task and sync it across all datasets."""
        periods = list(self.activities.get(name, []))
        periods.append((pd.Timestamp(start), pd.Timestamp(end)))
        self.set_activity_periods(name, periods)

    def set_activity_periods(self, name: str, periods) -> None:
        """Replace a task's full period list and sync it across all datasets.

        Editing a task changes which samples it covers, so any stored PSD fit
        for it is now stale and is dropped (the user re-fits the new data).
        """
        norm: List[Period] = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in periods]
        self.activities[name] = norm
        for ds in self.datasets:
            helpers.set_activity_periods(ds.obj, name, norm)
            ds.psd_fits.pop(name, None)

    def delete_activity(self, name: str) -> None:
        """Remove a task from the registry, every dataset, and its PSD fits."""
        self.activities.pop(name, None)
        for ds in self.datasets:
            helpers.delete_activity(ds.obj, name)
            ds.psd_fits.pop(name, None)
