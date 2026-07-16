# Design note — Central derived-cache + generation invalidation (GUI)

> Plan for review (2026-07-16). Supersedes handoff **Task 2**. Scope: make the
> GUI snappy by not recomputing basis conversions it already has, via one
> centralized cache with generation-based invalidation. **No threading, no
> scheduler, no disk persistence** — those were considered and rejected as
> disproportionate for this GUI (≈40 derived objects, single user).
> _Working note; not committed unless you decide to._

---

## 1. Goal & non-goals

**Goal.** The display tabs deep-copy + `dtype_converter` the active object on
*every* redraw. Cache the converted result so a redraw with unchanged inputs is
free, and switching panes reuses work already done. Correctness first: a missed
invalidation must cause a *recompute* (slow), never a *stale plot* (wrong).

**Non-goals (explicitly out of scope).**
- Background/worker-thread computation and any priority scheduler.
- A four-state freshness model (up-to-date / stale / imminent / background).
- Persisting derived caches to disk.
- Rewriting the existing working caches (`summary_cache`, `fit_specs`,
  `Aerosol2D` bin-property cache) — only *accommodate* them for a later,
  optional migration.

---

## 2. Principles

1. **Generation-counter invalidation, not fine-grained clearing.** Each dataset
   owns a monotonic `generation: int`. Any mutation bumps it. Cache entries are
   tagged with the generation they were built from; an entry whose tag ≠ the
   dataset's current generation is simply ignored (and recomputed lazily).
   *Invalidation is O(1) and stale-proof — the worst case of a missed bump is a
   needless recompute, never wrong data.*
2. **One small façade, reusing the existing layers.** A single Qt-free
   `DerivedCache` class lives in `gui/state/` (like `Project`), owned by the
   `Project`. It is the only door the view layer uses to obtain converted data;
   the only door the logic layer uses to invalidate is `Dataset.touch()`. No tab
   computes conversions itself; no layer pokes another's privates.
3. **Qt-free + headless-testable**, consistent with the rest of `state/`.
4. **Lazy, synchronous compute.** Compute on first request, cache until the
   generation bumps. The active pane computes only what it shows; pane-switches
   compute on demand and cache thereafter. This alone delivers the snappiness.

---

## 3. Data model

```
# gui/state/project.py  — Dataset gains one field + one method
class Dataset:
    ...
    self.generation: int = 0          # bumped on every in-place/​swap mutation

    def touch(self) -> None:
        """Mark this dataset's data changed; invalidates its derived cache."""
        self.generation += 1
```

```
# gui/state/derived_cache.py  (new, Qt-free) — owned by Project
class DerivedCache:
    """Memoises expensive per-dataset derived objects, keyed by dataset
    generation so a stale entry is impossible to hand out."""

    def converted(self, dataset, basis: str) -> tuple[Aerosol2D, str]:
        """Return (converted_view, unit) for `dataset.obj` on `basis`.
        `basis == "dN"` returns the object itself (no copy). The returned
        object is CACHE-OWNED and must be treated READ-ONLY (see §4)."""

    def invalidate(self, dataset) -> None: ...   # optional explicit drop
    def clear(self) -> None: ...                 # e.g. on project load/close
```

- Key: `(id(dataset), dataset.generation, basis)`. Old-generation entries are
  never returned and can be lazily evicted.
- `Project` constructs one `DerivedCache` and exposes it (e.g.
  `project.derived`), so tabs call `self.main.project.derived.converted(ds, basis)`.

---

## 4. The one real decision — the read-only contract

Today `helpers.converted_copy` **returns a fresh copy on purpose**, because some
callers *mutate* it. A shared cache changes that. Audit of every conversion site:

| Site | File:line | Uses | Reads or mutates the converted object? |
|---|---|---|---|
| Overlay | `tabs/overlay.py:396` (`_converted` per-draw memo) | `converted_copy` | **Reader** — reads `.total_concentration` |
| Time series | `tabs/timeseries.py:635` | `converted_copy` | **Reader** — reads `.total_concentration` |
| Raw export | `tabs/raw.py:107` | `copy_self`+`dtype_converter` | **Reader** — reads `.data` (on-demand export) |
| Heatmap | `tabs/heatmap.py:125` | `copy_self`+`dtype_converter` | **Mutator** — also `normalize_logdp()` |
| PM bands | `tabs/pmbands.py:105` | `converted_copy` | **Mutator** — `pm_calc()` adds Pₓ to `.extra_data` |

**Proposed contract: the cache hands out a shared, READ-ONLY converted object.**

- **Readers** (Overlay, Time series, Raw) use it directly → they save **both**
  the deep-copy and the `dtype_converter` compute. These are the redraw-heavy
  panes (overlay pan/zoom, timeseries), so this is where the win lands.
- **Mutators** (Heatmap normalize, PM bands Pₓ) take the cached converted object
  and make **one** copy before mutating. They still save the `dtype_converter`
  compute (the per-bin conversion) but pay a deep-copy for their in-place step.
  Both are on-demand panes (user changes a cut-off/toggle), not continuous
  redraws, so the residual copy is acceptable.

Rationale: the deep-copy is the expensive half for readers, and readers dominate
the hot paths. Trying to preserve today's "you may mutate what I return"
everywhere (option B: copy-on-return) would keep the deep-copy cost we're trying
to remove. So: **shared read-only, mutators copy explicitly.**

> Enforcement: document the contract in `DerivedCache.converted`'s docstring, and
> have the two mutator sites call an explicit `.copy_self()` on the returned
> object. (We rely on convention, not a runtime freeze — matching how the rest of
> the codebase treats "views".)

---

## 5. Public API surface (the two guarded doors)

- **Invalidate (logic → state):** `dataset.touch()` — the *only* way data-change
  is signalled. Never set `dataset.generation` directly.
- **Read (view → state):** `project.derived.converted(dataset, basis)` — the
  *only* way tabs obtain a basis conversion. Tabs stop calling
  `helpers.converted_copy` / `copy_self` + `dtype_converter` directly.

`helpers.converted_copy` stays as the underlying compute (the cache calls it),
but tabs no longer call it — keeping one implementation, one door.

---

## 6. `touch()` call sites (audited — all funnel through logic/main_window)

| Mutation | Where | Action |
|---|---|---|
| Crop | `logic/adjustments.py` `_apply_crop` (246) / `_crop_to_view` (253) | `project.active.touch()` |
| Smoothing | `_apply_smoothing` (282) | `touch()` |
| Resampling / rebin | `_apply_resampling` (299) | `touch()` |
| Time shift | `_apply_timeshift` (322) | `touch()` |
| Calibration apply/reset | `logic/calibration.py` (73–105, reassigns `ds.obj`) | `touch()` after swap |
| Density change | `app/main_window.py:1128` (`ds.obj.set_density`) | `touch()` |
| New dataset / split / copy-window | dataset created fresh | none (generation starts at 0) |

All of these already end by calling `main.refresh_all(...)`, so the redraw that
follows a `touch()` will naturally recompute through the cache. (We bump at the
mutation, **not** inside `refresh_all`, so view-only refreshes — toggles, zoom —
keep their cache.)

---

## 7. Migration order (small, reviewable steps)

1. **Add `generation` + `touch()`** to `Dataset`; wire the ~6 `touch()` sites in
   §6. No behaviour change yet (nothing reads generation). Commit.
2. **Add `DerivedCache`** (Qt-free) + a headless test: same dataset+basis twice
   returns the *same* object; after `touch()` a fresh one; `dN` returns the
   object itself. Commit.
3. **Route the readers** (Overlay, Time series) through `project.derived.converted`.
   Commit. Verify redraw output unchanged.
4. **Route the mutators** (Heatmap, PM bands) — fetch cached converted, then
   `copy_self()` before `normalize_logdp` / `pm_calc`. Commit.
5. **(Optional) Raw export** through the cache — low value (on-demand), do last.
6. **(Later, optional) Fold in other caches.** The `(id, generation, kind,
   params)` key can also host summaries/fits/bin-props — migrate *only* if it
   clearly simplifies; do not rewrite working, tested caches for elegance.

---

## 8. Why this is correct (invalidation argument)

- A cache entry is returned **iff** its generation tag equals the dataset's
  current generation. Any mutation bumps the generation *before* the next
  `refresh_all`, so the next read misses and recomputes.
- A **missed** `touch()` (bug) → the read hits an entry whose generation still
  matches → returns *slightly stale-but-recomputed-next-real-touch* … no: it
  would return stale. **So the safety net is the test in step 4** plus the small,
  audited, centralized site list (§6). The residual risk is far lower than
  fine-grained per-path clearing because there is exactly **one** invalidation
  primitive (`touch`) and it's called at a handful of known chokepoints.
- Calibration swaps `ds.obj` entirely; `touch()` after the swap covers it. (We do
  **not** key on `id(obj)` — id reuse after GC could alias.)

---

## Policy 1 — Persistent derived columns (PNC / MASS / Pₓ): keep, but invalidate

**Decided.** The store-in-`extra_data`-and-reuse-if-present memo stays (less
compute; and scripting users can read the columns directly). The missing piece
is invalidation, and after the units work `extra_data` now holds **two kinds** of
column that must be treated differently:

- **Loaded / real data** — `Temperature`, `RH`, `Pressure`, … (per-timestamp
  measurements; *cannot* be recomputed).
- **Derived / memoised** — `PNC`, `MASS`, `PM2.5`, `PN10`, … (recomputable).

Rules:
1. **A `self._derived_columns: set[str]` registry** — `PM_calc` /
   `_get_metric_series` add to it whenever they cache a result, so the object
   knows which `extra_data` columns are derived.
2. **Invalidation lives in the core** mutation methods, so the API and the GUI
   share **one** mechanism — the GUI adds no policy. Each data-changing mutation
   calls `self._drop_derived()`:
   - **Loaded** columns are carried along (crop subsets them, rebin rebins them) —
     never dropped.
   - **Derived** columns are dropped; the existing lazy *compute-if-absent* path
     refills them on next access. (No new recompute machinery — we just delete
     stale entries so the current path re-runs.)
3. **Uniform rule, no per-mutation cleverness:** crop / rebin / smooth /
   `set_density` (ELPI diameters) / calibration **all** drop derived. Recompute is
   cheap; robustness beats the micro-optimisation of trying to keep some.

   | mutation | PNC | MASS | Pₓ | why |
   |---|---|---|---|---|
   | crop | ok | ok | ok | per-timestamp subset — still drop for uniformity |
   | time rebin | ~ | ~ | ~ | linear metrics happen to commute; don't rely on it |
   | smoothing | stale | stale | stale | values change |
   | density (ELPI) | ok | **stale** | **stale** | diameters change (mass ∝ d³, cut-offs shift) |
   | calibration | stale | stale | stale | data scaled |
   | dtype (view) | ok | ok | ok | metrics are basis-independent |

4. **Picker hygiene:** the GUI excludes derived-metric columns from the "extra
   housekeeping" group — they are metrics, not loaded channels. Fixes the
   confusing case where `PNC` appears as a pickable "extra" only *after* the user
   visited Summary/Decay (which triggered the caching).
5. **This is a core correctness fix — the bug is confirmed.** There is currently
   *no* such invalidation, so API users are already exposed to stale derived
   columns. Measured (2026-07-16):
   - ELPI `set_density(2.0)` then re-request `MASS`: cache returns **31900.8**
     (the ρ=1 value) vs the correct **37947.3** — a ~19 % error, silently reused.
   - OPS `timesmooth(11)` then `MASS`/`PM2.5`: cached vs fresh differ (small,
     edge-effect sized) — stale but minor.
   The change is additive/back-compat (columns still appear; they just refresh
   instead of going stale).

## Policy 2 — Normalization & view-basis: transient display transforms, never stored

**Decided.** Correctness here is *already sound* — no bug to fix:
`normalize_logdp`/`unnormalize_logdp` are **idempotent**, guarded by the dtype
string (`normalize_logdp` no-ops when `"/dlogDp"` is already in `dtype`,
`size_distribution.py:690`), and `plot_psd(normalize=…)` respects the same flag.
A user cannot unknowingly double-normalize.

The only inconsistency is **mechanism**, so this policy is tidy-up:

> Display transforms — view-basis (`dN→dM…`) and dlogDp normalization — are
> applied **transiently via plot kwargs** (`dtype=` / `normalize=`) on the core
> plot methods. GUI panes pass flags; they never mutate stored data for display.

Action: give the heatmap's core plot path a `normalize=` kwarg (mirroring
`plot_psd`, which already has `normalize: bool = True`), so the heatmap tab drops
its hand-rolled `copy_self + dtype_converter + normalize_logdp` and delegates like
the PSD panes. The GUI `DerivedCache` (§3) then caches converted **data** only for
panes that need the *values* (timeseries `.total_concentration`, pmbands Pₓ) — not
for plotting, which this kwarg handles.

## Policy 3 — Drop the `PNC` column; one canonical user-facing name per quantity

**Decided.**

- **Remove the derived `PNC` column entirely.** `_get_metric_series("PNC")` — both
  the 1-D base and the 2-D `fractions` override — resolves to
  `self.total_concentration`, i.e. the maintained `Total_conc` column
  (`dtype_converter` always recomputes it from the base number distribution,
  `size_distribution.py:245`, so it is always the number total). No bin-recompute,
  no cached column. This takes PNC out of the derived/invalidation surface
  altogether — **only `MASS`/`Pₓ` remain genuinely derived**.
- **Keep `"PNC"` as the internal metric *key*** (it's the cross-instrument merge
  identifier that lets `summarize_*` combine a CPC, an OPS and an SMPS into one
  number-concentration column) — but never surface that token to users.
- **One canonical user-facing name per quantity, everywhere** — functions,
  dataset accessors, and every GUI pane must agree. Today one quantity wears
  three names (`PNC` / "Number concentration" / "Total concentration (dN)"); that
  ends. **DECIDED canonical names** (we drop the `"Total concentration (dX)"`
  notation entirely — it read oddly when the same quantity can be a *measured*
  mass from a DustTrak rather than a derived `dM`):
  - `dN` → **"Number concentration"**
  - `dM` → **"Mass concentration"**
  - `dS` → **"Surface area concentration"**
  - `dV` → **"Volume concentration"**
  - plus **"PM1" / "PM2.5" / "PM4" / "PM10"**, instrument-named channels (Cl₂, BC,
    LDSA…), and the ambient metrics (Phase 4).

  The picker also excludes derived-metric columns from the "extra housekeeping"
  group (see Policy 1.4).
- Minor back-compat: a script reading `obj.extra_data["PNC"]` loses that column;
  `obj.total_concentration` returns the identical series, and the PNC column was
  never an advertised feature (only the Pₓ columns are).

### Policy 3b — Mass / surface / volume / PM: one entry per quantity, provenance annotated

**Decided.** Mass has the same duplication *and* a provenance twist:

- For a size-resolved instrument, `MASS` ≡ "Total concentration (dM)" — the *same*
  derived computation (`_convert_array(..., "dM", self.density)` then sum,
  `fractions.py:395`). Likewise `dS`→surface, `dV`→volume. Unlike PNC, these have
  **no maintained column**, so they stay genuinely derived (cached + invalidated
  per Policy 1).
- **Derived vs measured are comparable, not separate metrics.** An OPS-derived
  PM2.5 (from number × density × sphere assumption) and a DustTrak *measured*
  PM2.5 (optical, factory-calibrated) are the **same quantity obtained two ways**
  and are fine to overlay together. So they share **one** picker entry.

Rules:
1. **One canonical picker entry per physical quantity**, named by the quantity —
   *Total/Number concentration, Mass concentration, Surface concentration, Volume
   concentration, PM1/2.5/4/10* — plus instrument-named channels (Cl₂, BC, LDSA…)
   and the ambient metrics (Phase 4). **Remove the basis-suffixed duplicates**
   ("Total concentration (dM/dS/dV)" and the separate "MASS" entry) — they collapse
   into the named quantities.
2. **Per-dataset resolution** (like the Phase-4 canonical metrics): each dataset
   supplies the quantity as either a **measured/raw column** (kept, never
   invalidated) or a **derived value** (computed on demand, cached, invalidated).
   The `self._derived_columns` registry (Policy 1) is what tells them apart — so it
   does double duty: invalidation **and** provenance.
3. **Provenance is a per-series annotation, never a separate picker entry.** A
   derived series is labelled e.g. `"OPS — Mass (derived, ρ=1.0 g/cm³)"`; a
   measured one `"DustTrak — Mass (measured)"`. The user picks **"Mass
   concentration"** once; the legend/label carries the how. (DustTrak's optical
   calibration implies an effective density/response, not a simple ρ — label it
   "measured"; note the assumption only if the instrument reports it.)
4. **No triple options.** The picker must never show "Total concentration (dM)"
   *and* "Mass (derived, ρ=…)" *and* "Mass (measured)" as three choices — that's
   the confusion this policy exists to kill. One "Mass concentration" entry;
   provenance in the series label.

Net effect on the four requirements you set:
| Requirement | Mechanism |
|---|---|
| clear staleness/invalidation | Policy 1 (`_drop_derived` on mutation) |
| clear naming, no mismatched metrics | Policies 3 + 3b (one canonical name per quantity) |
| derived vs raw distinction (don't invalidate raw) | `self._derived_columns` registry (Policy 1) |
| don't show duplicate plottable metrics | Policy 3b.1/3b.4 (one entry per quantity; remove basis-suffixed duplicates) |

## 9. Testing

- `DerivedCache` unit test (headless, Qt-free): identity/reuse, post-`touch`
  refresh, `dN` identity, unit correctness.
- One GUI headless test (offscreen) per hot pane: a redraw after a mutation
  (`crop`/`set_density`) reflects the change — guards against a missed `touch()`.
- Existing suite must stay green; converted output byte-compared before/after on
  a sample (overlay/timeseries/heatmap draw the same figure data).

---

## 10. Decisions — ALL RESOLVED (2026-07-16)

No loose ends; the plan is code-ready.

1. **Read-only contract (§4)** — ✅ **Yes.** The cache returns a shared,
   read-only converted object; the two mutators (heatmap normalize, pmbands Pₓ)
   `copy_self()` before mutating.
2. **Where the cache hangs** — ✅ **Project-owned single façade.** `project.derived`
   (one `DerivedCache`), one `clear()` on project load/close. Not per-Dataset.
3. **Eviction** — ✅ **Trivial, no LRU.** Reuse if the entry matches the current
   generation; a generation bump supersedes/drops old entries; recompute on
   demand. No size cap (space is `datasets × 4 bases`, ~40 small objects).
4. **`touch()` wiring** — ✅ **Full-scale now.** Wire every §6 mutation site in
   one go; no half-invalidated states.
5. **Number metric naming** — ✅ Drop `PNC` and the `"Total concentration (dX)"`
   notation from user-facing surfaces; keep `"PNC"` internal only. Canonical
   family (Policy 3): **Number / Mass / Surface area / Volume concentration**,
   plus PM1/2.5/4/10, named channels, and the ambient metrics.
6. **Mass metric (Policy 3b)** — ✅ One **"Mass concentration"** entry; derived and
   measured are comparable and share it; provenance (measured vs `derived,
   ρ=… g/cm³`) is a per-series label, never a separate picker option;
   basis-suffixed duplicates removed.

### Confirmed before coding
The staleness bug is **verified** (see Policy 1.5): ELPI `set_density(2.0)` leaves
`MASS` ~19 % stale; OPS `timesmooth` leaves `MASS`/`PM2.5` stale (small). Policy 1
is a correctness fix, not a hypothetical.
