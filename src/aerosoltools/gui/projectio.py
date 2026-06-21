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


def _safe_name(idx: int, path: Optional[str]) -> str:
    """Build a unique, index-prefixed file name for a copied raw file."""
    base = os.path.basename(path) if path else f"dataset_{idx}.dat"
    return f"{idx}_{base}"


def save_project(project: Project, folder: str, theme: str = "dark") -> None:
    """Write ``project`` into ``folder`` (created if needed)."""
    os.makedirs(folder, exist_ok=True)
    raw_dir = os.path.join(folder, RAW_DIR)
    ds_dir = os.path.join(folder, DS_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ds_dir, exist_ok=True)

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
        "datasets": [],
    }

    for i, ds in enumerate(project.datasets):
        raw_rel = None
        if ds.source_path and os.path.exists(ds.source_path):
            raw_rel = f"{RAW_DIR}/{_safe_name(i, ds.source_path)}"
            shutil.copy2(ds.source_path, os.path.join(folder, raw_rel))

        pkl_rel = f"{DS_DIR}/ds_{i}.pkl"
        with open(os.path.join(folder, pkl_rel), "wb") as fh:
            pickle.dump(ds.obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

        manifest["datasets"].append(
            {
                "label": ds.label,
                "instrument_key": ds.instrument_key,
                "raw": raw_rel,
                "pickle": pkl_rel,
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
        project.datasets.append(ds)

    # Restore the active dataset, then re-project the shared activities so every
    # object's masks are guaranteed consistent with the saved registry.
    idx = manifest.get("active_index", -1)
    if 0 <= idx < len(project.datasets):
        project.active_id = project.datasets[idx].id
    elif project.datasets:
        project.active_id = project.datasets[0].id
    project.reapply_all()

    return project, manifest.get("theme", "dark")
