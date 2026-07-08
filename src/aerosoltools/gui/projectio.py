"""Save / load a :class:`~aerosoltools.gui.project.Project` to a folder.

A saved project is a **self-contained, movable folder** so it can be copied or
moved without breaking::

    <project folder>/
        project.json        manifest (datasets, shared activities, theme, …)
        raw_data/           a copy of every dataset's original source file
        datasets/           the live aerosol objects, pickled (preserves all
                            applied processing: crop/smooth/rebin/shift/dtype)

The pickled objects preserve the exact processed state; the ``raw_data`` copies
make the folder portable and are also used as the (in-folder) reload source, so
paths are reconstructed relative to the folder's *current* location on load.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from typing import Optional

import pandas as pd

from .project import Dataset, Project

MANIFEST = "project.json"
RAW_DIR = "raw_data"
DS_DIR = "datasets"
FORMAT = "aerosoltools-project"
VERSION = 1


def _clean_psd_fits(psd_fits: dict) -> dict:
    """Coerce stored lognormal fits to plain JSON types for serialization."""
    out: dict = {}
    for activity, rec in (psd_fits or {}).items():
        modes = []
        for m in rec.get("modes", []):
            modes.append(
                {
                    "mu": float(m["mu"]),
                    "sigma": float(m["sigma"]),
                    "peak": float(m["peak"]),
                    "bound": bool(m.get("bound", False)),
                }
            )
        if modes:
            out[activity] = {"modes": modes, "optimized": bool(rec.get("optimized"))}
    return out


def save_project(project: Project, folder: str, theme: str = "dark") -> None:
    """Write ``project`` into ``folder`` (created if needed).

    Every raw file behind any dataset — including the constituent files of a
    combined/derived dataset — is copied into ``raw_data/`` (de-duplicated), so
    the saved folder always contains the full set of sources used in the
    analysis and the work can be reassessed or recreated from it.
    """
    os.makedirs(folder, exist_ok=True)
    raw_dir = os.path.join(folder, RAW_DIR)
    ds_dir = os.path.join(folder, DS_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ds_dir, exist_ok=True)

    # Copy each unique raw file once; map absolute path -> its relative copy.
    copied: dict[str, str] = {}

    def _copy_raw(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        abspath = os.path.abspath(path)
        if abspath in copied:
            return copied[abspath]
        if not os.path.exists(abspath):
            return None
        rel = f"{RAW_DIR}/{len(copied)}_{os.path.basename(abspath)}"
        shutil.copy2(abspath, os.path.join(folder, rel))
        copied[abspath] = rel
        return rel

    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "name": project.name,
        "theme": theme,
        "active_index": (
            project.index_of(project.active_id) if project.active_id is not None else -1
        ),
        "activities": {
            name: [
                [pd.Timestamp(s).isoformat(), pd.Timestamp(e).isoformat()]
                for s, e in periods
            ]
            for name, periods in project.activities.items()
        },
        # Which datasets each activity applies to, stored by dataset *index*
        # (ids are per-session), or null for "all datasets". Older projects
        # have no entry and load as all-datasets (the pre-scoping behaviour).
        "activity_scopes": {
            name: (
                None
                if (ids := project.activity_scopes.get(name)) is None
                else sorted(
                    project.index_of(i) for i in ids if project.index_of(i) >= 0
                )
            )
            for name in project.activities
        },
        # Cached Summary-tab results (table + inputs + staleness signature).
        # Already built from JSON-safe primitives by the Summary tab.
        "summary_state": getattr(
            project, "summary_state", {"active_kind": None, "cache": {}}
        ),
        # Per-plot concentration-threshold (OEL) overlays, keyed by tab tag.
        "plot_thresholds": getattr(project, "plot_thresholds", {}),
        "datasets": [],
    }

    for i, ds in enumerate(project.datasets):
        raw_rel = _copy_raw(ds.source_path)
        # Archive every contributing raw file (the constituents for a combined
        # dataset; just the source file for a plain one).
        contributing = []
        for f in ds.contributing_files:
            rel = _copy_raw(f)
            if rel and rel not in contributing:
                contributing.append(rel)

        pkl_rel = f"{DS_DIR}/ds_{i}.pkl"
        with open(os.path.join(folder, pkl_rel), "wb") as fh:
            pickle.dump(ds.obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # A calibrated dataset also archives its uncalibrated baseline, so the
        # correction can still be toggled off / reset after a save+reload. The
        # spec + on/off flag go in the manifest; the pickled ``obj`` above already
        # reflects the current (on/off) state.
        cal_rel = None
        if ds._cal_baseline is not None:
            cal_rel = f"{DS_DIR}/ds_{i}_baseline.pkl"
            with open(os.path.join(folder, cal_rel), "wb") as fh:
                pickle.dump(ds._cal_baseline, fh, protocol=pickle.HIGHEST_PROTOCOL)

        manifest["datasets"].append(
            {
                "label": ds.label,
                "instrument_key": ds.instrument_key,
                "raw": raw_rel,
                "contributing": contributing,
                "pickle": pkl_rel,
                "color": ds.color,
                "calibration": ds.calibration,
                "calibration_enabled": ds.calibration_enabled,
                "calibration_baseline": cal_rel,
                "psd_fits": _clean_psd_fits(ds.psd_fits),
            }
        )

    with open(os.path.join(folder, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_project(folder: str) -> tuple[Project, str]:
    """Load a project from ``folder``; return ``(project, theme)``.

    Raises:
        FileNotFoundError: if the folder has no ``project.json``.
        ValueError: if the manifest is not an aerosoltools project.
    """
    manifest_path = os.path.join(folder, MANIFEST)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No '{MANIFEST}' found in:\n{folder}\n\n"
            "Pick a folder that was created with 'Save project'."
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("format") != FORMAT:
        raise ValueError("This folder is not an aerosoltools project.")

    project = Project(name=manifest.get("name", "Untitled project"))
    project.activities = {
        name: [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in periods]
        for name, periods in manifest.get("activities", {}).items()
    }
    # Restore cached summaries (older projects simply have none).
    state = manifest.get("summary_state")
    if isinstance(state, dict) and isinstance(state.get("cache"), dict):
        project.summary_state = state
    # Restore per-plot threshold overlays (older projects simply have none).
    thresholds = manifest.get("plot_thresholds")
    if isinstance(thresholds, dict):
        project.plot_thresholds = thresholds

    for entry in manifest.get("datasets", []):
        pkl_path = os.path.join(folder, entry["pickle"])
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
        raw_rel = entry.get("raw")
        # Reconstruct the source path from the folder's *current* location so
        # the project stays valid after being moved.
        source_path = os.path.join(folder, raw_rel) if raw_rel else None
        ds = Dataset(
            obj=obj,
            source_path=source_path,
            instrument_key=entry.get("instrument_key", ""),
            label=entry.get("label"),
        )
        # Repoint the archived contributing raw files into this folder (so a
        # re-save preserves them). Older projects without the field keep the
        # source-based default set in Dataset.__init__.
        contributing = entry.get("contributing")
        if contributing:
            ds.contributing_files = [os.path.join(folder, r) for r in contributing]
        ds.color = entry.get("color")  # None for pre-colour projects
        ds.psd_fits = _clean_psd_fits(entry.get("psd_fits", {}))
        # Restore the calibration spec/state and its uncalibrated baseline (the
        # pickled obj above already reflects the on/off state). Older projects
        # simply have none.
        ds.calibration = entry.get("calibration")
        ds.calibration_enabled = bool(entry.get("calibration_enabled"))
        cal_rel = entry.get("calibration_baseline")
        if cal_rel and os.path.exists(os.path.join(folder, cal_rel)):
            with open(os.path.join(folder, cal_rel), "rb") as fh:
                ds._cal_baseline = pickle.load(fh)
        project.datasets.append(ds)
        project.assign_color(ds)  # give older/uncoloured datasets a stable colour

    # Restore each activity's dataset scope (saved by index). A missing entry or
    # a null value means "all datasets" (also the pre-scoping default), so older
    # projects keep behaving as before.
    scopes = manifest.get("activity_scopes", {})
    for name in project.activities:
        raw = scopes.get(name)
        if raw is None:
            project.activity_scopes[name] = None
        else:
            ids = {
                project.datasets[i].id
                for i in raw
                if isinstance(i, int) and 0 <= i < len(project.datasets)
            }
            project.activity_scopes[name] = ids

    # Restore the active dataset, then re-project the activities so every
    # object's masks are guaranteed consistent with the saved registry + scopes.
    idx = manifest.get("active_index", -1)
    if 0 <= idx < len(project.datasets):
        project.active_id = project.datasets[idx].id
    elif project.datasets:
        project.active_id = project.datasets[0].id
    project.reapply_all()

    return project, manifest.get("theme", "dark")
