# Documentation pipeline overhaul — plan

**Branch:** `GUI_test` (all edits + commits here; nothing merges to `main`).
**Status:** Phase 1 **done**, plus the Phase 2 notebook rewrite (commit
`208e79e`, 2026-08-18). Remaining: new example notebooks and the GUI docs
section — both deferred by the maintainer until the pipeline is reviewed.
**Date drafted:** 2026-07-17.

> **Corrections to the findings below, from the 2026-08-18 implementation pass:**
> - It was **all 7** notebooks that called the removed PascalCase API, not 5.
> - Notebooks were **not** executed at build time. `nbsphinx_execute` was unset,
>   so nbsphinx's `"auto"` default skipped every notebook that had stored
>   outputs — which was all of them. That is why the build passed while
>   publishing output from deleted functions.
> - Three breaks were **not** simple renames: `Combine_NS_OPS` →
>   `combine_size_ranges`, `summarize_exposure(activity=)` → `activities=`
>   (now a sequence), and `plot_correlation(column=)` → `parameter=`.
> - `.nojekyll` was lost on `Dev-gh-pages` by the manual "New documentation
>   page" commit; the automated deploy had written it correctly.
> - Two problems the plan missed: ~40 AutoAPI warnings from parsing the GUI
>   (fixed with `autoapi_ignore`), and 36 duplicate-object-description warnings
>   where Napoleon's `Attributes:` handling collided with AutoAPI (fixed with
>   `napoleon_use_ivar`).
> - The docs workflow cannot use plain `-W`: Sphinx logs unreachable-intersphinx
>   warnings without a type, so they are filtered explicitly in the workflow.

> **Deviation from decision 1:** the maintainer chose **two symmetric build
> workflows** (staging + live) over build-then-promote. Implemented as one
> reusable `_docs-build.yml` called by both, so the two cannot drift. Note the
> consequence: the live site is a *rebuild*, not a byte-copy of what was
> reviewed. Run both from the same commit.

This file is a durable handoff of the agreed plan for reworking how the
`aerosoltools` docs are built and deployed. Read the "Findings" and "Decisions"
sections first — they record *why* each choice was made.

---

## Goal

Move from a single manual `workflow_dispatch` docs build to a **staged,
controlled pipeline**: build docs to a staging branch, review locally, publish
to PyPI, then promote staging → live. Also modernise config, fix the stale
example notebooks, add a GUI documentation section, and let Sphinx own the API
tree instead of a hand-maintained schema.

---

## Key findings from investigation (verify before relying on them)

- **Docs deploy is fully manual today.** `.github/workflows/deploy-docs.yml`
  triggers only on `workflow_dispatch`. Nothing rebuilds docs on push/merge.
- **PyPI publish triggers on git tag `v*`** (`publish-to-pypi.yml`), via
  setuptools-scm. You publish by pushing a `vX.Y.Z` tag.
- **Both `gh-pages` (live) and `Dev-gh-pages` (staging) branches exist** on
  `origin`, each already containing a built HTML site. Live URL:
  `https://nfa-nrcwe.github.io/aerosoltools/`.
- **GitHub Pages serves exactly one branch per repo** → can't host `gh-pages`
  and `Dev-gh-pages` publicly at the same time. Staging review = check out
  `Dev-gh-pages` locally and open `index.html`.
- **`pyproject.toml` already has a `[docs]` extra** but it's incomplete: missing
  `nbsphinx`, `ipython`, `ipykernel`, `linkify-it-py` (all required by
  `conf.py`), and carries unused `sphinx-rtd-theme`. `requirements.txt` currently
  fills the gap.
- **`conf.py` uses nbsphinx** → notebooks are *executed* at build time.
- **5 of 7 example notebooks call the removed PascalCase API** (`Load_ELPI_file`,
  `Combine_NS_OPS`, etc.) → build would fail once executed.
- **Notebook data paths use Windows backslashes**
  (`"..\\..\\tests\\data\\Sample_ELPI.txt"`) → **break on the Ubuntu CI runner**.
  Must become forward slashes / `pathlib`. Data resolves from `docs/examples/`
  to repo-root `tests/data/`, which exists and is checked out in CI.
- **`docs/api/` is committed** (not gitignored) — the "manual schema". Contains
  stale `aerosolalt/` and `utility/` stubs (both packages removed in the
  restructure).
- **`docs/index.md` is stale**: documents `AerosolAlt`, `utility`, and PascalCase
  functions; its bottom `toctree` links to removed `aerosolalt`/`utility` pages.
- **`.nojekyll` gotcha:** GitHub Pages runs Jekyll by default, which ignores
  folders starting with `_` (`_static`, `_images`, `_sources`). `gh-pages` has a
  `.nojekyll`; `Dev-gh-pages` does **not** → staging could render with broken
  CSS/images. Both deploys must always write `.nojekyll`. `gh-pages` also has a
  leftover `.buildinfo.bak` to clean up.
- **`intersphinx` is enabled but unconfigured** (`sphinx.ext.intersphinx` loaded,
  no `intersphinx_mapping`) → currently a no-op.

---

## Decisions (agreed with maintainer)

1. **Pipeline:** keep PyPI publish separate. Flow = build to `Dev-gh-pages` →
   review locally → push tag (PyPI) → promote `Dev-gh-pages` → `gh-pages`.
   Manual but staged/controlled.
2. **Dependencies:** delete `requirements.txt`; install via `pip install -e .[docs]`
   with only the packages the build actually needs.
3. **Staging:** review by checking out `Dev-gh-pages` locally and opening
   `index.html` (no second public Pages site).
4. **Notebooks:** rewrite all to the current `snake_case` API, execute them, make
   the build **fail on any notebook error**. Explore the API surface first and
   propose *major* missing examples (loaders are interchangeable — don't document
   every one).
5. **GUI docs:** add a dedicated hand-written section with **screenshots** + short
   per-pane descriptions. Do **not** un-exclude `.gui` from AutoAPI. GIFs/videos
   are the maintainer's job, later.
6. **API tree:** let Sphinx/AutoAPI own it (`autoapi_add_toctree_entry = True`),
   then tune if unhappy.
7. **New points:** (1) new staged checkout/build + separate deploy action;
   (2) CI fails on broken docs — never deploy faulty docs; (3) configure
   intersphinx; (4) fix `.nojekyll`/cruft so staging renders identically to live.

---

## The backbone — three workflows

| Workflow | Trigger | Does |
|---|---|---|
| `publish-to-pypi.yml` *(unchanged)* | push tag `v*` | Build + upload to PyPI. |
| `build-staging-docs.yml` *(new, replaces old `deploy-docs.yml`)* | manual `workflow_dispatch` | `pip install -e .[docs]` + `apt-get pandoc` → `sphinx-build -b html docs build/` (executes notebooks, **fails on error**) → publish `build/` to **`Dev-gh-pages`** with `.nojekyll`. |
| `deploy-docs.yml` *(new "Go live")* | manual `workflow_dispatch` | Checkout `Dev-gh-pages`, **copy as-is** to `gh-pages` (no rebuild) → what you reviewed is exactly what ships. |

**Release flow:** ① run *Build-staging* → ② `git fetch && git checkout Dev-gh-pages`,
open `index.html`, accept/reject → ③ push `vX.Y.Z` tag (PyPI) → ④ run *Go-live*.

Also add a **PR check**: build-only (no deploy) so broken docs are caught before
release, not during it.

---

## Phase 1 — pipeline & config (mechanical, low-risk) — DO FIRST

- [ ] **`pyproject.toml`**: add `nbsphinx`, `ipython`, `ipykernel`,
      `linkify-it-py` to `[project.optional-dependencies].docs`; remove
      `sphinx-rtd-theme`.
- [ ] **Delete `requirements.txt`.**
- [ ] **`conf.py`**:
  - `autoapi_add_toctree_entry = True`, `autoapi_keep_files = False`.
  - Add `intersphinx_mapping` (python, numpy, pandas, matplotlib, scipy).
  - `nbsphinx_execute = "always"`, `nbsphinx_allow_errors = False`.
- [ ] **`.gitignore`** `docs/api/`; delete committed stubs (removes stale
      `aerosolalt`/`utility` pages).
- [ ] **Rewrite `docs/index.md`** bottom `toctree`: curated intro + AutoAPI's
      auto-generated reference (review result together, then tune).
- [ ] **Workflows**: replace `deploy-docs.yml` with `build-staging-docs.yml`
      (build → `Dev-gh-pages`) + new `deploy-docs.yml` (promote
      `Dev-gh-pages` → `gh-pages`); update `actions/checkout@v3` → v4; ensure
      `.nojekyll` written on both; add build-only PR check.
- [ ] Clean `.buildinfo.bak` off `gh-pages`.

**Phase 1 end state:** a clean build you can run to `Dev-gh-pages`. Then status
check with maintainer.

**Phase 1 outcome (2026-08-18):** done. Local build is warning-clean, all 7
notebooks execute, and the site was built and committed to local branch
`_docs_staging`. The push to `Dev-gh-pages` is **blocked**: the `gh` PAT in use
(`AndersBros`) can read the repo but is denied write (403), so the maintainer
must push it. Pushing `GUI_test` additionally needs the token's *Workflows*
permission, because the commit adds files under `.github/workflows/`.

---

## Phase 2 — content (after Phase 1 verified)

### Point 4 — notebooks
1. **Explore pass** (subagents): inventory the public API on `GUI_test`; map vs.
   what the 7 notebooks demonstrate; produce a gap list of *major* missing
   features (candidates: Gas1D/Aethalometer/Partector non-particle classes,
   Aerosol3d/APS, calibration, decay/source fitting).
2. Rewrite each notebook: PascalCase → snake_case, **forward-slash data paths**,
   real `tests/data/` samples.
3. Execute locally on Windows (`PYTHONIOENCODING=utf-8 python -X utf8 ...`) until
   all run clean.
4. **Propose new-notebook shortlist for maintainer sign-off** before writing —
   don't balloon the example set unilaterally.
5. Fix stale prose in `docs/index.md` (AerosolAlt/utility/PascalCase).

### Point 5 — GUI section
- New `docs/gui/` with `index` + per-tab pages (Overview, Install `[gui]`,
  Launch, then Time series, Overlay, Heatmap, PM/PSD, Decay/Source, Correlation,
  Calibration, …). Each: screenshot + short capability description. One `toctree`
  entry links it in.
- **Screenshots:** attempt automated headless capture
  (`QT_QPA_PLATFORM=offscreen`, load a sample dataset, `widget.grab().save()`
  per tab). ⚠️ May not be pixel-perfect — first pass; maintainer can replace with
  polished shots/GIFs later.

---

## Open items / confirmations

- Docs build depends on `tests/data/` being present at build time (it is —
  full repo checkout). Confirmed acceptable.
- New example notebooks: maintainer approves shortlist before they're written.

---

## Guardrails (from CLAUDE.md)

- Work on `GUI_test`; commit after each coherent unit; **push only when told**;
  **never merge** to `main` or open/merge PRs.
- Core API stays backward-compatible; GUI may change freely.
- Run `python -m pytest tests/ -q` before committing code changes.
- Windows console is cp1252 — use `PYTHONIOENCODING=utf-8 python -X utf8` for
  scripts printing unit glyphs.
