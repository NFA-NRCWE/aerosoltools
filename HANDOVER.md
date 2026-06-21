# aerosoltools GUI — Handover

_Last updated: 2026-06-19_

This document lets the author (or another Claude agent) pick up the GUI work
cold. Read it top-to-bottom once; it explains where the code lives, what is
done, the design decisions, and exactly what to do next.

---

## 0. Where things live (important!)

- The GUI is the package **`src/aerosoltools/gui/`**.
- It is committed **only on the `GUI_test` branch**, checked out in the **main
  working directory** `C:\Users\28098\OneDrive\Dokumenter\Github\aerosoltools`.
  `origin/main` does **not** contain the GUI.
- Claude Code worktrees are created from `main`, so a worktree will **not** have
  the `gui/` folder. **Edit the GUI in the main checkout (the `GUI_test`
  working tree).** Changes there are currently uncommitted — commit when ready.
- **Backward-compat rule:** the GUI may change freely (single user). The core
  `aerosoltools` API (loaders, `Aerosol1D/2D/Alt`, `plot_*`, `Combine_NS_OPS`,
  `Plot_correlation`, …) is used by **other people** — change it only
  additively (new optional kwargs / new functions), never break signatures.

## 1. Run & test

```bash
# from the main checkout
python -m aerosoltools.gui                       # empty window
python -m aerosoltools.gui tests/data/Sample_OPS2.txt -i OPS

# headless smoke test (no display needed)
QT_QPA_PLATFORM=offscreen python your_script.py  # build MainWindow, load_file, assert
```
`MainWindow().grab().save("x.png")` renders a screenshot offscreen. Note: under
the offscreen platform **text glyphs don't rasterize** (blank labels/buttons) —
colours/layout are accurate, text is fine on a real display.

Requires `PyQt5` (extra: `pip install aerosoltools[gui]`).

## 2. Architecture

| File | Responsibility |
|------|----------------|
| `__init__.py` | `launch()` / `main()` entry points; applies theme then builds `MainWindow`. |
| `qt.py` | Single place that binds PyQt5 + the Matplotlib QtAgg backend. |
| `theme.py` | **Light & dark** themes: palettes, one QSS `string.Template`, `apply_qt_theme(app, mode)`, `apply_mpl_theme(mode)`, runtime accessors (`is_dark()`, `mpl_cycle()`, `fig_facecolor()`…), and the light **`export_rc()`** used for saved figures. |
| `project.py` | **`Project`** (datasets + active id + shared activity registry) and **`Dataset`** (obj + label + source_path + instrument_key). Qt-free. |
| `projectio.py` | `save_project(project, folder, theme)` / `load_project(folder) -> (project, theme)`. Self-contained movable folder. |
| `sidebar.py` | `DatasetSidebar` widget (list + add/remove/rename); emits Qt signals. |
| `main_window.py` | `MainWindow`: menu bar, top bar card, datasets dock, tabs, status bar; loading, active-dataset switching, crop/smooth/resample/**time-shift**, theme switching, project save/load. |
| `tabs.py` | `RawDataTab`, `SummaryTab`, `TimeSeriesTab` (marks tasks, hosts the embedded crop/processing/shift boxes), `PSDTab`, `HeatmapTab`, `PMBandsTab`, `_PlotTab` base. |
| `models.py` | `PandasTableModel` for table views. |
| `loaders.py` | Instrument → loader registry + filename guesser. |
| `helpers.py` | dtype/unit resolution, plottable columns, activity shading, `delete_activity`. |
| `assets.py`, `shortcut.py` | App icon path; Windows shortcut creator. |

### Key data-flow facts
- `MainWindow.obj / source_path / source_instrument` are **properties** that
  resolve to `self.project.active`. The single-view tabs read `self.main.obj`
  unchanged — switching the active dataset just re-points them.
- Tabs rebuild **only when the active dataset's shape changes** (`_tab_sig`
  "1d"/"2d"). `TimeSeriesTab` re-syncs its series list on object change
  (`_cols_obj`).
- A single **`self.adjust_box`** ("Data adjustments": crop / resample / smooth /
  time-shift, one operation per row) is built once and **owned by
  `MainWindow`**, but **embedded into the Time series tab**. Because tabs are
  destroyed on rebuild, it is **detached (`setParent(None)`) before
  `tabs.clear()`** and re-attached via
  `TimeSeriesTab.attach_adjust_controls(box)`. Keep this invariant if you touch
  tab building. Enable/disable all of it with `_set_adjust_enabled(bool)`.
- The central content is wrapped in a `QScrollArea` (and so is the sidebar), so
  panes get a scrollbar when too small instead of a hard minimum — this is what
  lets the datasets dock splitter be dragged. Don't re-introduce large
  `setMinimumWidth`/`setMinimumSize` on those.
- **"Mark activities"** is a checkable `QPushButton#toggle` in the Time series
  side panel (not a checkbox); it reads `.isChecked()`, flips its label to
  "Marking", and drives the `SpanSelector`. Reminder: a bare Qt widget is always
  truthy — always call `.isChecked()` (this bug had hidden the PM "Cumulative"
  toggle).
- **Activities/tasks are project-level** (`Project.activities`), projected onto
  every dataset via `mark_activities(mode="replace")`. The Time series mark /
  edit / delete actions call `main.project.add_activity /
  set_activity_periods / delete_activity`. Stored in **absolute time**.

## 3. Decisions already made (don't relitigate)
1. Tasks shared across datasets, **absolute-time**. Switching the active
   dataset shares tasks; time-shifting a dataset does **not** move task times —
   it changes only that dataset's masks/summaries (we re-project after a shift).
2. Time-shift: support **both** a view-only shift (for the future overlay tab)
   **and** a permanent one. The permanent one is **done** ("Apply time shift"
   in the Time series tab). View-only belongs to the overlay tab (step 5).
3. Save project = a **movable, self-contained folder**: `raw_data/` (copied
   sources) + `datasets/*.pkl` (pickled objects, preserves processing) +
   `project.json` (manifest: labels, instrument keys, relative raw paths,
   shared activities, theme, active index). **Done.**
4. Sidebar = detachable, default left dock. **Done.**
5. Theme = dark by default (matches `aerosoltools.ico`); light theme also
   previews export appearance. Toggle in **View → Theme**. **Done.**

## 4. Roadmap status

Done: **theme polish + dark/light toggle**, **menu bar**, **Project + sidebar
(step 1)**, **project-level shared activities (step 2)**, **permanent
time-shift**, **save/load project**, **join same-instrument (step 3)**,
**combine NS+OPS (step 4)**.

- **Step 3 (done):** new additive core util
  `aerosoltools.combine_measurements([obj, ...])` in `utility.py` (checks same
  class + serial, identical `bin_edges` for 2D, concatenates `data`+`extra_data`
  in time, sorts, dedups, unions activity periods). GUI: sidebar **"Join same
  instrument"** → `MainWindow._join_same_instrument` groups by
  instrument_key+serial, combines, replaces the originals.
- **Step 4 (done):** sidebar **"Combine NS + OPS…"** → `_CombineNSOPSDialog`
  (pick NS/OPS dataset + match) → core `Combine_NS_OPS`, adds the result while
  keeping the originals.
- Both go through `MainWindow._add_derived_dataset(obj, instrument_key, label,
  remove_ids=())`, which makes a `Dataset` with `source_path=None` (no Reload;
  `set_active_dataset`/`load_file`/`_add_derived_dataset` keep the Reload button
  in sync with `bool(source_path)`). Derived datasets still save fine (pickled,
  no `raw_data` copy).

- **Step 5 (done):** `OverlayTab` in `tabs.py` (a multi-dataset comparison tab,
  reads `project.datasets`). Per-dataset **view-only** shift + include flag are
  stored on the `Dataset` (`view_shift`, `overlay_on`) so they survive tab
  rebuilds; the plot shifts each series' index by `view_shift` (data untouched).
  "Apply shifts permanently" calls `obj.timeshift` + re-projects tasks + resets
  `view_shift`. Metric = "Total concentration" or any column present in a
  dataset. Added in `_build_tabs` after Summary (always shown).

Remaining (build in this order; reassess with the user after each):

6. **Combined PSD** comparison tab: overlay `obj.plot_psd(ax=shared)` per
   (dataset × task). `plot_psd` already accepts `ax`.
7. **Cross-instrument summary**: run `summarize_activities/exposure` per
   dataset, prepend a Dataset/Instrument column, concat into one table.
8. **Correlation / Bland-Altman** tab: core `Plot_correlation(X, Y,
   parameter=…, match=…, tolerance=…)` and `bland_altman_analysis(...)` exist.
   Time alignment is handled by `match` ("exact"/"nearest"/"rebin") + tolerance.

### How to add a comparison tab (steps 5–8)
- Comparison tabs read a **set** of datasets, not the active one. Give them a
  multi-select of `project.datasets` (checkbox list) and a Compute/Refresh.
- They are NOT shape-gated; add them once in `_build_tabs` after Summary, or
  introduce a second `QTabWidget`/section. Decide with the user whether
  comparison tabs always show or only when ≥2 datasets exist.
- Reuse `_PlotTab` for the embedded figure + export. For dark-mode legibility,
  recolour core-drawn artists with `theme.mpl_cycle()` on the screen path only
  (see `PSDTab._brighten_for_dark`), never in `_render_export`.

## 5. Gotchas / conventions
- **Detach `self.adjust_box`** (`setParent(None)`) before `tabs.clear()`
  anywhere you tear tabs down (see `_build_tabs`, `_remove_dataset`,
  `_new_project`, `_apply_loaded_project`).
- **Primary buttons**: set `objectName("primary")` to get the accent-gradient
  style; everything else is the subtle secondary style.
- **Exports stay light** regardless of UI theme (`theme.export_rc()` in
  `_PlotTab.save_figure`). The light UI theme is the on-screen preview of that.
- Pickled projects depend on the `aerosoltools` classes being importable; the
  `raw_data/` copies are the portability + Reload fallback. If you later add a
  pickle-failure fallback, re-load from `raw_data/` via the `LOADERS` registry
  and re-apply `project.activities`.
- PyQt5 specifics: `QtWidgets.QAction` / `QActionGroup` (not `QtGui`).

## 6. Memory pointers (Claude)
Project memory files: `gui-location`, `gui-backward-compat`,
`gui-multidataset-roadmap` (under the project's `memory/`). Update
`gui-multidataset-roadmap` as steps complete.
