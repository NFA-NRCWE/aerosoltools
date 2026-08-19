# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this project is

`aerosoltools` is a Python library for **loading, processing, analyzing, and
plotting data from aerosol instruments** (CPC, ELPI, SMPS, OPS, NanoScan,
Partector, DiSCmini, Aethalometer, DustTrak, etc.), developed at NFA/NRCWE. It
provides consistent data structures for time-resolved and size-resolved particle
measurements, activity/task segmentation, exposure assessment (8 h TWA, STEL,
peaks), and an optional **PyQt5 desktop GUI** for a point-and-click workflow.

Data classes are organised by **shape**, composed from topic mixins in `_core/`.
Particle instruments: `Aerosol1D` (1D time series — single- *or* multi-channel),
`Aerosol2D` (size-resolved), `Aerosol3d` (dual-distribution, APS), plus thin
subclasses where an instrument has its own physics/accessors — `DiSCmini`,
`DustTrak`, `ELPI`. Non-particle instruments (no `total_concentration`; they
override it to raise and expose domain accessors via `_core/nonparticle`):
`Gas1D`, `Aethalometer`, `Environmental1D`, `Partector`. All are exported from
the top-level package. (The former `AerosolAlt` catch-all has been removed —
multi-channel instruments now use `Aerosol1D` or a dedicated subclass; a generic
per-channel `unit`/`dtype` dict + the internal `_primary` hook make multi-channel
work on the base class.)

## Language, tooling & main packages

- **Language:** Python, `requires-python >= 3.10` (targets 3.10–3.12).
- **Core dependencies:** `pandas`, `numpy`, `matplotlib`, `scipy`, `tabulate`,
  `openpyxl`, `tqdm`, `typing_extensions`.
- **GUI extra (`pip install aerosoltools[gui]`):** `PyQt5` (with the Matplotlib
  QtAgg backend).
- **Dev tooling:** `pytest` (suite in `tests/`), `ruff` and `black` — both at
  **line-length 88**, target `py310`. Ruff lint selects `E,F,W,I` (ignores
  `E501`). Keep changed code ruff- and black-clean.
- Build backend: setuptools + setuptools-scm (version from git tags). Docs:
  Sphinx (deployed to GitHub Pages).

## Package layout

```
src/aerosoltools/
  __init__.py        Public API: classes, loaders, intercomparison workflows.
  aerosol1d.py       Aerosol1D  — thin public facade; data model + properties.
  aerosol2d.py       Aerosol2D  — size-resolved facade (extends Aerosol1D).
  aerosol3d.py       Aerosol3d  — dual-distribution (APS) facade (extends 2D).
  discmini.py / dusttrak.py / elpi.py   thin particle-instrument subclasses.
  gas1d.py / aethalometer.py / environmental.py / partector.py
                     non-particle facades (compose _core/nonparticle mixin).
  _core/             Internal topic mixins composed by the facades (NOT public API):
                       activities, time_ops, statistics, statistics2d, fractions,
                       size_distribution, corrections, fitting, plotting, plotting2d,
                       alt, + shared helpers _shading (activity shading) / _labels.
  loaders/           One module per instrument + registry.py (dispatch / auto-detect)
                       + __init__; load_<instr>_file functions. Shared infra lives in
                       loaders/support/ (parsing.py — the delimiter/encoding sniffer +
                       folder batch-loader, formerly Common.py; exceptions.py — the
                       LoaderError hierarchy).
  intercomparison/   Public multi-dataset workflows: combination.py
                       (combine_measurements, combine_size_ranges), correlation.py
                       (plot_correlation, bland_altman_analysis, fit_data),
                       calibration.py (CalibrationModel, fit_calibration,
                       apply_calibration, calibrate_against_reference), and private
                       _alignment.py (shared time-alignment). Replaced the old
                       vague `utility/` package.
  gui/               PyQt5 desktop app (see below).
tests/               pytest suite + tests/data/ sample instrument files.
docs/                Sphinx documentation sources.
```

**Core architecture:** `Aerosol1D/2D/Alt` are deliberately thin *facades*. The
heavy behaviour lives in topic **mixins** under `_core/`, composed via multiple
inheritance, so each file stays readable while the public API is unchanged.
Every method is still reachable on the facade class. Watch the MRO when adding
methods, and prefer adding to the relevant mixin over the facade.

**GUI architecture (`src/aerosoltools/gui/`):** grouped into concern-based
subpackages (plus `qt.py` — the single PyQt5/backend binding point — and the
`__init__.py`/`__main__.py`/`shortcut.py` entry-point modules at the top;
`shortcut.py` is the `aerosoltools-gui-shortcut` console script and stays at the
top level so its pip-generated launcher path doesn't break on a reorg):
- `app/` — application shell / lifecycle: `main_window.py` (`MainWindow`) and
  `sidebar.py`.
- `state/` — the Qt-free data model + persistence: `project.py` (`Project`:
  datasets + active id + shared activity registry), `projectio.py` (save/load to
  a movable self-contained folder, schema-versioned with a migration seam), and
  the typed caches `summary_cache.py` + `fit_specs.py` (`PsdFitSpec`/
  `DecayFitSpec`).
- `view/` — appearance: `theme.py` (dark/light + `export_rc`), `widgets.py`,
  `models.py` (pandas Qt table model), `metric_picker.py`, and `assets.py`
  (+ the bundled `aerosoltools.ico`).
- `logic/` — Qt-light domain workflows: `helpers.py`, `calibration.py`,
  `adjustments.py`, `loaders.py` (GUI file-open shim over the loader registry).
- `tabs/` — one module per tab, with `_base.py` providing the `_PlotTab` base
  (embedded Matplotlib figure, the publication export pipeline, table helpers).

## Branch & git workflow

- **Work on the `GUI_test` branch** (checked out in the main working directory).
  This branch contains the GUI and is ahead of `main`; `origin/main` does **not**
  contain the GUI. Note: Claude Code worktrees branch from `main`, so a worktree
  will *not* have `gui/` — edit in the main `GUI_test` checkout.
- **Commit after each change** (a coherent unit of work), with a clear message.
- **Push to GitHub only when explicitly told** (`git push origin GUI_test`).
- **Never merge.** Do not merge `GUI_test` into `main`, and do not open/merge
  PRs — the maintainer handles all merges manually.

## Implementation rules

- **Backward compatibility (core):** the `aerosoltools` core API (loaders,
  `Aerosol1D/2D/3d`, `plot_*`, `plot_correlation`, `combine_size_ranges`,
  `fit_psd`, `summarize_*`, …) is used by **other people**. Change it
  **additively** — new optional kwargs / new functions — and do not break
  existing signatures or behaviour. (The 2026-07 restructure was an authorized
  exception: it removed the `utility/` package and the PascalCase function
  spellings and replaced `data.calibrate(Variables=)` with
  `data.apply_calibration(model)`. Those breaks were explicitly sanctioned;
  the additive default still holds for everything else.)
- **The GUI may change freely** (single user), as long as it doesn't break the
  core contract it depends on.
- **Renaming / deprecating:** when an old name or convention must eventually
  change, keep the old one working and emit a `DeprecationWarning` pointing to
  the new name, rather than removing it outright. (Exception: if the maintainer
  explicitly says back-compat isn't needed for a specific, rarely-used name,
  rename it directly.)
- **Naming convention:** public functions use PEP 8 `snake_case`
  (`load_elpi_file`, `plot_correlation`, `fit_calibration`); classes use
  `PascalCase` (`Aerosol2D`, `CalibrationModel`, `DiSCmini`). The legacy
  PascalCase *function* spellings have been removed. Use lowercase parameter
  names. (Some older mixins still expose a `PascalCase`/`snake_case` method pair,
  e.g. `Mark_threshold`/`mark_threshold`; leave those dual names in place unless
  told otherwise.)
- **Don't reinvent shared logic:** reuse the existing helpers (e.g. activity
  shading via `_core/_shading`, dtype label via `_core/_labels.base_dtype`,
  cross-dataset alignment via `intercomparison/_alignment`, the lognormal model
  via `aerosoltools.lognormal_modes`) instead of duplicating across modules.
- **Docstrings & types:** keep the Google-style docstrings (`Args:`/`Returns:`)
  and type annotations that the codebase uses; add/update them when you touch a
  function.

## Testing & verification

- Run the suite before committing: `python -m pytest tests/ -q` (18 tests).
- **GUI smoke tests are headless** — set `QT_QPA_PLATFORM=offscreen`. Build a
  `MainWindow`, `load_file(...)` a sample from `tests/data/`, and assert. Example:
  `QT_QPA_PLATFORM=offscreen python -m aerosoltools.gui tests/data/Sample_OPS2.txt -i OPS`.
- **Windows console encoding:** the shell is often cp1252 and chokes on unit
  glyphs (`µg/m³`, `cm⁻³`). When a script prints such text, run it with
  `PYTHONIOENCODING=utf-8 python -X utf8 ...`. And when calling core methods that
  `print()` their result tables (`summarize_*`) from the GUI, wrap them in
  `contextlib.redirect_stdout(io.StringIO())`.

## Other observations

- **Plot exports** save the *live* on-screen figure (a detached pickled copy),
  recoloured to a light print palette and with text/lines enlarged to
  publication sizes; the on-screen figure is never modified. The export profile
  lives in `theme.export_rc()`; the resize/recolour passes are in
  `gui/tabs/_base.py` (`_lighten_for_export`, `_enlarge_for_export`).
- **Activities/tasks are project-level** and stored in **absolute time**, then
  projected onto every dataset. Per-object mutation goes through
  `gui/helpers.set_activity_periods` / `delete_activity`; the `Project` owns the
  registry and loops over datasets.
- PyQt5 specifics: `QAction`/`QActionGroup` come from `QtWidgets` (not `QtGui`),
  and a bare Qt widget is always truthy — call `.isChecked()` on toggles.

## Memory

Claude keeps project notes under
`.../memory/` (indexed by `MEMORY.md`): see `gui-location`,
`gui-backward-compat`, `core-mixin-architecture`,
`gui-multidataset-roadmap`, and the dated GUI fix/housecleaning batches. These
are point-in-time notes — verify against current code before relying on them.
