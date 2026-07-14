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

from .fit_specs import DecayFitSpec, PsdFitSpec
from .project import Dataset, Project
from .summary_cache import SummaryCacheEntry

MANIFEST = "project.json"
RAW_DIR = "raw_data"
DS_DIR = "datasets"
FORMAT = "aerosoltools-project"
#: Current on-disk schema version. Bump this whenever the manifest layout
#: changes in a way older loaders can't read as-is, and add a step to
#: ``_MIGRATIONS`` (keyed by the *source* version) that upgrades a manifest of
#: the previous version to the new one.
VERSION = 2


def _migrate_v1_to_v2(manifest: dict) -> dict:
    """v1 → v2: no field changes (v2 formalizes the versioned/typed fit specs).

    v1 and v2 manifests are byte-compatible — v2 only marks that PSD/decay fits
    are written by the typed specs — so older files load unchanged. This step
    exists to establish the migration seam and keep the version monotonic.
    """
    return manifest


#: Ordered upgrade steps keyed by the manifest version they upgrade *from*. Each
#: takes a manifest of that version and returns one a single version newer.
_MIGRATIONS = {1: _migrate_v1_to_v2}


def _migrate_manifest(manifest: dict) -> dict:
    """Upgrade a loaded manifest to the current :data:`VERSION` in place.

    Applies the ordered :data:`_MIGRATIONS` steps until the manifest reaches the
    current version. A manifest with no ``version`` is treated as v1 (the first
    schema, before the field existed).

    Raises:
        ValueError: If the manifest was written by a newer, forward-incompatible
            build of aerosoltools (its version exceeds :data:`VERSION`).
    """
    version = manifest.get("version", 1)
    if not isinstance(version, int) or version < 1:
        version = 1
    if version > VERSION:
        raise ValueError(
            "This project was saved by a newer version of aerosoltools "
            f"(project schema v{version}, this build reads up to v{VERSION}).\n"
            "Update aerosoltools to open it."
        )
    while version < VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:  # no path forward; stop rather than loop
            break
        manifest = step(manifest)
        version += 1
    manifest["version"] = version
    return manifest


def _dump_psd_fits(psd_fits: dict) -> dict:
    """Serialize the ``{activity: PsdFitSpec}`` map, dropping empty fits."""
    out: dict = {}
    for activity, spec in (psd_fits or {}).items():
        payload = spec.to_dict()
        if payload["modes"]:
            out[activity] = payload
    return out


def _load_psd_fits(stored: dict) -> dict:
    """Rebuild the ``{activity: PsdFitSpec}`` map from JSON, dropping empty fits."""
    out: dict = {}
    for activity, payload in (stored or {}).items():
        spec = PsdFitSpec.from_dict(payload)
        if spec.modes:
            out[activity] = spec
    return out


def _dump_decay_fits(decay_fits: list) -> list:
    """Serialize the Decay tab's live specs to JSON via DecayFitSpec."""
    out: list = []
    for rec in decay_fits or []:
        spec = DecayFitSpec.from_live(rec)
        if spec is not None:
            out.append(spec.to_dict())
    return out


def _load_decay_fits(stored: list) -> list:
    """Rebuild the Decay tab's live specs from JSON via DecayFitSpec."""
    out: list = []
    for rec in stored or []:
        spec = DecayFitSpec.from_dict(rec)
        if spec is not None:
            out.append(spec.to_live())
    return out


def _summary_state_payload(project: Project) -> dict:
    """JSON-safe summary-tab state: active kind + typed cache entries as dicts."""
    state = getattr(project, "summary_state", None) or {}
    cache = state.get("cache") or {}
    return {
        "active_kind": state.get("active_kind"),
        "cache": {
            kind: (entry.to_dict() if isinstance(entry, SummaryCacheEntry) else entry)
            for kind, entry in cache.items()
        },
    }


def _restore_summary_state(state) -> dict:
    """Rebuild the in-memory summary state (cache entries as SummaryCacheEntry)."""
    if not isinstance(state, dict):
        return {"active_kind": None, "cache": {}}
    cache = state.get("cache")
    entries = {}
    if isinstance(cache, dict):
        for kind, entry in cache.items():
            entries[kind] = SummaryCacheEntry.from_dict(entry)
    return {"active_kind": state.get("active_kind"), "cache": entries}


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
        # Cached Summary-tab results (table + inputs + staleness signature),
        # serialized from the typed SummaryCacheEntry objects.
        "summary_state": _summary_state_payload(project),
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
                "psd_fits": _dump_psd_fits(ds.psd_fits),
                "decay_fits": _dump_decay_fits(ds.decay_fits),
                "activity_colors": dict(ds.activity_colors),
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
    # Upgrade older manifests (and reject forward-incompatible newer ones) so the
    # restore code below always sees the current schema.
    manifest = _migrate_manifest(manifest)

    project = Project(name=manifest.get("name", "Untitled project"))
    project.activities = {
        name: [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in periods]
        for name, periods in manifest.get("activities", {}).items()
    }
    # Restore cached summaries as typed entries (older projects simply have none).
    state = manifest.get("summary_state")
    if isinstance(state, dict) and isinstance(state.get("cache"), dict):
        project.summary_state = _restore_summary_state(state)
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
        ds.psd_fits = _load_psd_fits(entry.get("psd_fits", {}))
        ds.decay_fits = _load_decay_fits(entry.get("decay_fits", []))
        ds.activity_colors = dict(entry.get("activity_colors", {}) or {})
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
