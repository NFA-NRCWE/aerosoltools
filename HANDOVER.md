# aerosoltools GUI — Handover

_Last updated: 2026-06-21_

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
| `main_window.py` | `MainWindow`: menu bar, top bar card, datasets dock, tabs, status bar; loading, active-dataset switching, theme switching, project save/load, derived-dataset ops. Delegates data adjustments to `AdjustmentsBox`. |
| `adjustments.py` | **`AdjustmentsBox(QGroupBox)`** — the "Data adjustments" controls (crop / resample / smooth / **time-shift**) + handlers. Owned by `MainWindow`, embedded in the Time series tab. Public API: `set_enabled(bool)`, `sync_crop_fields()`. |
| `widgets.py` | Small presentation-only helpers: `SlackTabBar` (tab-label sizing), `CombineNSOPSDialog` (NS+OPS picker). |
| `tabs/` | **Package**, one module per tab (re-exported from `tabs/__init__.py`). `_base.py` = `_PlotTab` base + `_export_table`/`_tune_table`/`_active_color_cycle`. `raw`, `timeseries` (+`ActivityEditorDialog`), `heatmap`, `pmbands` (single-view, follow the active dataset). `psd` (`PSDTab`), `summary` (`SummaryTab`), `overlay`, `correlation` read the **whole project** (work for one *or* many datasets). |
| `theme.py` | **Light & dark** themes: palettes, one QSS `string.Template`, `apply_qt_theme(app, mode)`, `apply_mpl_theme(mode)`, runtime accessors (`is_dark()`, `mpl_cycle()`, `fig_facecolor()`…), and the light **`export_rc()`** used for saved figures. |
| `models.py` | `PandasTableModel` for table views. |
| `loaders.py` | Instrument → loader registry + filename guesser. |
| `helpers.py` | dtype/unit resolution, plottable columns, activity shading, `delete_activity`. |
| `assets.py`, `shortcut.py` | App icon path; Windows shortcut creator. |

> Note: `theme.py` is moved up in the table for grouping; the actual import order is unchanged. Run `python -m ruff check src/aerosoltools/gui/` after editing — the package is ruff-clean (E/F/W/I, line-length 88).

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
**combine NS+OPS (step 4)**, **overlay tab (step 5)**, **combined PSD (step 6)**,
**cross-instrument summary (step 7)**, **correlation / Bland–Altman (step 8)**.
**The original 8-step multi-dataset roadmap is now complete.**

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

- **Step 6 (done):** `CombinedPSDTab` in `tabs.py` (multi-dataset comparison,
  reads the project's **2D** datasets only). Side panel = a checkable dataset
  list (include flag persisted on `Dataset.psd_on`, like `overlay_on`) + a
  multi-select activity list (`["All data"] + project.user_activities()`). It
  draws one curve per (dataset × activity) by calling the core
  `obj.plot_psd(activities=[act], normalize=…, ax=shared)`, then **relabels +
  recolours the artists it added** (track `len(ax.lines)`/`len(ax.collections)`
  before/after each call; the suffix is the new artists). Labels collapse to
  just the dataset (when one activity) or just the activity (when one dataset),
  else "dataset – activity". Empty (dataset × task) combos add no line and are
  skipped. Colours come from `_active_color_cycle()` = the **live**
  `rcParams['axes.prop_cycle']`, so they're dark on screen and light under
  `export_rc()` automatically — no `_brighten_for_dark` needed. Controls:
  Normalize, Log Y, "±σ band" (off by default — bands overlap when comparing
  many curves). Added in `_build_tabs` right after Overlay (always shown).

- **Step 7 (done):** `CrossSummaryTab` in `tabs.py` (table, **not** a
  `_PlotTab`; mirrors `SummaryTab`'s controls but operates on a dataset set).
  Side: a checkable dataset list (include flag `Dataset.summary_on`). Type =
  Activity / Exposure summary (same metric + cut-off + STEL/OEL controls as
  `SummaryTab`); the exposure metric drop-down is the **union** across the
  ticked datasets. **Compute-on-demand** (a `Compute` button) rather than
  recomputing on every `refresh` — exposure over many datasets is costly, and
  `refresh()` only re-syncs the dataset list. Per dataset it calls the core
  `summarize_activities()` / `summarize_exposure(metric=…)`, `insert`s
  `Dataset`/`Instrument` columns, and `pd.concat(sort=False)` — columns align by
  name, so metrics only some instruments report leave blanks for the rest. A
  metric invalid for a dataset (e.g. `PM4.2` on a CPC) is caught per-dataset and
  reported in the status line ("Skipped — …"), not fatal. The whole compute loop
  runs under `contextlib.redirect_stdout(io.StringIO())` because the core
  `summarize_*` methods `print()` their table (noise across many datasets, and a
  non-UTF-8 console chokes on the µ/³ glyphs mid-method). Export to Excel/CSV via
  the shared `_export_table`. Added in `_build_tabs` after Combined PSD.

- **Step 8 (done):** `CorrelationTab` in `tabs.py` (a `_PlotTab`; a **two**-dataset
  comparison — **X** + **Y** combos, not a checklist). Draws the core
  `Plot_correlation(X, Y, ax_in=…)` (scatter + 1:1 + regression + R²) or
  `bland_altman_analysis(X, Y, ax_in=…, method=BA/Gi/Eu, C=…)` on the embedded
  axis. The **Parameter** combo is the intersection of numeric column names
  common to both objects (`Total_conc` first if present). Time alignment is
  delegated to the core via `match` (exact/nearest/rebin) + `tolerance` /
  `rebin_freq` / `rebin_method`; correlation also exposes intercept /
  uniform-scaling / robust(Theil–Sen, = `outlier_influence=False`). Side-panel
  rows show/hide by match mode and analysis type. **Compute-on-demand** (button):
  `refresh()` only re-syncs the X/Y/parameter selectors and leaves the existing
  plot in place — alignment can be costly and `refresh_all` fires often. Errors
  (no time overlap, same dataset picked, no shared parameter) are caught and
  shown via `_show_message`. Added in `_build_tabs` after Cross summary.
  - **Activity-scoped (2026-06-21):** an **Activity** combo restricts the
    correlation/Bland–Altman to one marked region (e.g. a side-by-side window).
    This is an **additive core change**: `Plot_correlation` and
    `bland_altman_analysis` gained an `activity: str | None = None` kwarg, threaded
    into `_align_series`, which filters the aligned points to the activity's
    absolute-time `(start, end)` periods (so **multiple occurrences** work, and it's
    uniform across exact/nearest/rebin). `None`/`"All data"` = full record. Helper:
    `utility._activity_period_mask(index, X, Y, activity)`.

### Two follow-on fixes made alongside steps 6–8 (2026-06-21)
- **`SummaryTab` hardened**: its `summarize_*` calls now also run under
  `contextlib.redirect_stdout(io.StringIO())` — same reason as the cross-summary
  tab (the core `print()` of the result table can raise `UnicodeEncodeError` on a
  non-UTF-8 Windows console and abort a 2D activity summary).
- **`theme.export_rc()` legend fix**: it now pins `legend.facecolor="white"` +
  `legend.edgecolor="#cccccc"`. Previously those keys leaked from the dark screen
  theme (rc_context only overrides listed keys), so **every** tab's export drew a
  dark legend box with unreadable dark text when saved from the dark theme.

### GUI refactor (2026-06-21) — file de-bloat + layering
The GUI was split so no file is oversized (was: `tabs.py` 2169, `main_window.py`
1106). Behaviour is unchanged; verified by the offscreen smoke tests + `pytest`
(18 passed) + `ruff` (clean).
- **`tabs.py` → `tabs/` package**: one module per tab + `_base.py` (the
  `_PlotTab` base and `_export_table`/`_tune_table`/`_active_color_cycle`
  helpers). `tabs/__init__.py` re-exports every public tab, so existing
  `from .tabs import SummaryTab` imports are unchanged. Inside `tabs/*`, imports
  gained a dot: core is `from ...utility import …`, sibling gui modules are
  `from .. import helpers, theme` / `from ..qt import …`, base is `from ._base import …`.
- **`AdjustmentsBox` extracted** to `adjustments.py` from `MainWindow`
  (`_build_adjust_box` + all crop/smooth/resample/time-shift handlers +
  `_set_adjust_enabled`/`_sync_crop_fields`). It's a `QGroupBox` subclass holding
  a `main` back-reference; handlers act on `main.obj` / `main.project` and call
  `main.refresh_all(...)`. MainWindow now talks to it through two methods:
  `set_enabled()` and `sync_crop_fields()`. The detach/re-attach invariant is
  unchanged (it's still `self.adjust_box`, a widget, embedded via
  `TimeSeriesTab.attach_adjust_controls`).
- **`SlackTabBar` + `CombineNSOPSDialog`** moved to `widgets.py` (dropped the
  leading underscore now that they're a shared module's API).
- **Mechanics:** the tabs split used `ast.get_source_segment` to move classes
  verbatim, then `ruff check --fix` (`F401`/`I`) to prune/sort imports per module.
  Re-running that recipe is the way to split further.

### NOT refactored (needs a separate, explicit decision)
The **core** modules are still large: `aerosol2d.py` (~3074), `aerosol1d.py`
(~1789), `utility.py` (~1408). They are **shared by other users** and the API must
stay backward-compatible, so they were left alone. Splitting a class across files
is un-Pythonic; the realistic options are method-group **mixins** or a subpackage
with re-exports from `aerosoltools/__init__.py`. Either is higher-risk — do it only
with the user's sign-off and a full `pytest` + downstream-import check.

### Usability + documentation pass (2026-06-21)
- **Multi-file import:** the open dialog now uses `getOpenFileNames`. `_open_dialog`
  → `MainWindow.load_files(paths)`, which loops the new `_ingest_file(path,
  instrument)` (load → `Dataset` → `project.add_dataset`, no UI work), then calls
  `_finalize_after_load()` **once**. `load_file` is now a thin single-file wrapper
  over the same two helpers. Each file's instrument is guessed individually.
- **Menu bar:** File → **Import data…** (Ctrl+O) does the multi-file import.
  Shortcuts: New Ctrl+N, Open project Ctrl+Shift+O, Save Ctrl+S, Save as
  Ctrl+Shift+S, Exit Ctrl+Q, Datasets panel Ctrl+D, Help → **Keyboard shortcuts**
  F1. (Theme is set from the View → Theme submenu only.) All shortcut-bearing
  items are created via
  `MainWindow._menu_action(menu, label, handler, shortcut, tip)`, which records
  `(keys, label)` in `self._shortcut_help` — the single source the
  `KeyboardShortcutsDialog` (in `widgets.py`) renders. `setToolTipsVisible(True)`
  is set so menu tooltips show.
- **Tooltips:** added on the top-bar controls, sidebar buttons, the shared
  `_PlotTab` Save button, and the comparison tabs' Compute/Export buttons.
- **Docstrings:** every GUI function/method/class now has a docstring (Google
  style — `Args:`/`Returns:` where useful — matching the core's convention). The
  pass was applied with an `ast`-driven inserter (find each def's first body
  statement, insert a docstring at its indent). Whole GUI is `ruff`- and
  `black`-clean (line-length 88) and `pytest` stays green (18).

### Bug-fix round (2026-06-21)
1. **Overlay view no longer resets on shift.** `OverlayTab._draw(preserve=False)`;
   `_on_shift`/`_on_item_changed` pass `preserve=True` (capture+restore
   xlim/ylim around the redraw). Log/normalize toggles still reset (wrapped in
   `lambda: self._draw()` so the signal's int isn't read as `preserve`). Also
   fixed a latent refactor bug here: `_apply_shifts` called the removed
   `main._sync_crop_fields()` → now `main.adjust_box.sync_crop_fields()`.
2/3. **PSD and Summary tabs merged.** The single-view `PSDTab`/`SummaryTab` were
   deleted; the former `CombinedPSDTab`/`CrossSummaryTab` were **renamed**
   `PSDTab`/`SummaryTab` and their files moved to `tabs/psd.py` / `tabs/summary.py`.
   They already handle a single dataset, so one instrument behaves like the old
   tabs. Labels are now just "PSD"/"Summary".
4. **Time series / Overlay / 2D heatmap / PM bands stay separate** (unchanged) —
   heatmap + PM bands remain single-view (built only for a 2D active dataset).
5. **One theme control.** Removed the standalone "Toggle theme" (Ctrl+T) action +
   `_toggle_theme`; the View → Theme submenu (which shows the active mode) is the
   single control.
6. **Bland–Altman markers legible on dark.** The core draws the scatter `c="k"`
   and a black zero line. `CorrelationTab._brighten_for_dark(ax)` recolours
   near-black artists (scatter → accent, ref lines → light grey) on the **screen
   path only** (exports stay light). While here, the BA draw is wrapped in
   `redirect_stdout` — the core ends with `print("… µ …")`, which crashed on a
   non-UTF-8 console (same class as the summarize fix).
7. **Combined datasets keep their raw files.** `Dataset.contributing_files` lists
   the raw files behind a dataset (its own file, or the union of a combine's
   constituents — set by `_add_derived_dataset(..., source_files=…)`, fed by
   `_join_same_instrument`/`_combine_ns_ops`). `save_project` copies **every**
   unique contributing file into `raw_data/` (de-duped) and records per-dataset
   `"contributing"` rels in the manifest; `load_project` repoints them into the
   folder. So a join (which drops the originals) still archives them.

### How to add a comparison tab (the pattern used by steps 5–8)
- Comparison tabs read a **set** of datasets, not the active one. Give them a
  multi-select of `project.datasets` (checkbox list) and a Compute/Refresh.
- They are NOT shape-gated; add them once in `_build_tabs` after Summary, or
  introduce a second `QTabWidget`/section. Decide with the user whether
  comparison tabs always show or only when ≥2 datasets exist.
- Reuse `_PlotTab` for the embedded figure + export. For dark-mode legibility,
  recolour core-drawn artists with `theme.mpl_cycle()` on the screen path only
  (see `PSDTab._brighten_for_dark`), never in `_render_export`.

## 5. Gotchas / conventions
- **Resizable side panels:** a tab's right-hand panel goes in a `QSplitter`, not
  a fixed-width column. `_PlotTab._split_with_side(side_widget, sizes=(…))` pulls
  the controls/toolbar/canvas into a left pane and adds a draggable divider; it
  leaves `self._left_col` pointing at the left pane's layout (Time series inserts
  the adjust box there). `SummaryTab` (a plain `QWidget`) builds its own splitter.
  Don't put `setMaximumWidth` on side panels — it re-freezes the width.
- **Detach `self.adjust_box`** (`setParent(None)`) before `tabs.clear()`
  anywhere you tear tabs down (see `_build_tabs`, `_remove_dataset`,
  `_new_project`, `_apply_loaded_project`).
- **Primary buttons**: set `objectName("primary")` to get the accent-gradient
  style; everything else is the subtle secondary style.
- **Exports stay light** regardless of UI theme (`theme.export_rc()` in
  `_PlotTab.save_figure`). The light UI theme is the on-screen preview of that.
  `rc_context(export_rc())` only overrides the keys it lists — anything omitted
  leaks from the dark screen profile (this bit the legend box; now pinned light).
- **Core methods that `print()`** (the `summarize_*` table dumps via `tabulate`)
  must be wrapped in `contextlib.redirect_stdout(io.StringIO())` when called from
  the GUI: it silences the redundant console spam and stops a non-UTF-8 console
  from raising `UnicodeEncodeError` on the µ/³ glyphs mid-method (see
  `SummaryTab.refresh` and `CrossSummaryTab._compute`).
- Pickled projects depend on the `aerosoltools` classes being importable; the
  `raw_data/` copies are the portability + Reload fallback. If you later add a
  pickle-failure fallback, re-load from `raw_data/` via the `LOADERS` registry
  and re-apply `project.activities`.
- PyQt5 specifics: `QtWidgets.QAction` / `QActionGroup` (not `QtGui`).

## 6. Memory pointers (Claude)
Project memory files: `gui-location`, `gui-backward-compat`,
`gui-multidataset-roadmap` (under the project's `memory/`). Update
`gui-multidataset-roadmap` as steps complete.
