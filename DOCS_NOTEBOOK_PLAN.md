# Example-notebook restructure — plan

**Branch:** `GUI_test`. **Status:** awaiting maintainer sign-off. **Drafted:** 2026-08-18.

Companion to `DOCS_PIPELINE_PLAN.md`, which covers how the docs are built and
deployed. This file covers *what the examples contain*.

---

## Why

The seven existing notebooks each mix several unrelated topics and repeat the
same material — a time-series plot appears in almost all of them — while whole
subsystems (decay fitting, calibration, the non-particle classes, APS) appear
nowhere. The fix is one clear theme per notebook.

Because notebooks are now executed on every docs build, an example that calls a
renamed API fails the build instead of silently publishing stale output. Keeping
them small and single-purpose also keeps that failure easy to localise.

---

## The rule that keeps them clean

Every notebook may **use** plots and activities freely; only **#05 explains the
plotting API** and only **#03 explains activities**. Elsewhere they appear in one
line, uncommented, with a cross-reference. Notebook 02 needs a plot to show what
cropping did — that is fine, it just will not explain `y_3d`.

Each notebook opens with a one-line statement of its theme and a link to the
notebooks either side of it.

---

## The set — 15 notebooks

Files are renamed to `NN-topic.ipynb` (clean URLs; the current names produce
`3%20-%20Defining%20time%20segments%2C...`). The `toctree` is grouped into the
four sections below.

### Basics

| File | Theme | Principal API |
|---|---|---|
| `01-loading-data.ipynb` | Loading one file, a folder, or letting the package decide; what a loaded object contains | `load_*_file`, `load_file`, `detect_instrument`, `INSTRUMENT_LOADERS`, `load_data_from_folder`, `.data`, `.extra_data`, `.metadata`, `.original_data`, `.column_units`, `.measurement`, `.instrument`, `.serial_number`, `copy_self` |
| `02-time-adjustments.ipynb` | Getting two instruments onto a common, regular time base | `timecrop`, `timeshift`, `timerebin`, `timesmooth` |
| `03-activities.ipynb` | Marking what happened when, and getting the data back out | `mark_activities`, `mark_threshold`, `peak_finder`, `rename_activity`, `.activities`, `.activity_periods`, `get_activity_data`, `get_activity_extra_data` |
| `04-statistics-and-exposure.ipynb` | Task statistics and occupational exposure assessment | `available_metrics`, `summarize`, `summarize_activities`, `summarize_exposure` (PNC/MASS/PM/PN/PS/PV, band-limited, TWA, STEL, background as value or activity, peaks, limits, export) |
| `05-plotting.ipynb` | The plotting API itself — axes, limits, log scales, shading, dtype | `plot_total_conc`, `plot_timeseries`, `plot_psd`, `plot_PM_timeseries` |

### Transformations

| File | Theme | Principal API |
|---|---|---|
| `06-dtypes-density-corrections.ipynb` | Converting what the numbers *mean* before analysing them | `dtype_converter`, `set_density`, `.density`, `dtype_of`, `unit_of`, `normalize_logdp`, `unnormalize_logdp`, `rebin_bin_edges`, `pm_calc`, `correct_diffusion_losses`, `.bin_mids`, `.bin_edges`, `.size_data` |
| `07-combining-datasets.ipynb` | Joining runs and joining size ranges, and how both differ from folder loading | `combine_measurements`, `combine_size_ranges` |

### Analysis

| File | Theme | Principal API |
|---|---|---|
| `08-psd-fitting.ipynb` | Describing a size distribution as lognormal modes | `fit_psd`, `PSDFitResult` (`.modes`, `.errors`, `.evaluate`), `lognormal_modes` |
| `09-decay-and-source.ipynb` | Emission + decay peaks: source strength, air exchange, wall loss | `fit_decay`, `DecayResult` |
| `10a-correlation-and-agreement.ipynb` | Do two instruments agree, and by how much? | `plot_correlation`, `bland_altman_analysis`, `fit_data` |
| `10b-calibration.ipynb` | Turning a measured disagreement into a correction and applying it | `fit_calibration`, `CalibrationModel`, `apply_calibration`, `calibrate_against_reference` |

### Instruments

| File | Theme | Principal API |
|---|---|---|
| `11-simple-instruments.ipynb` | Single-value instruments that behave like 1D aerosol data | `Partector` (`ldsa`, `flow`, `tem_samples`), `DiSCmini` (`size`, `ldsa`), `Gas1D` (Ranger and Tiger), `Environmental1D` basics (Fourtec: temperature, RH); the no-`total_concentration` contract |
| `12-aethalometer.ipynb` | Wavelength-resolved black carbon | 5 BCc channels + `fossil_bcc`, `biomass_bcc`, `aae`; per-channel unit/dtype dicts |
| `13-aps.ipynb` | Dual aerodynamic/optical distributions | `.aerodynamic`, `.optical`, `.is_correlated`, `.correlation`, `as_2d`, `axis_view`, `correlation_cube`, `plot_aero_vs_optical`, `plot_aero_optical_3d` |
| `14-weather-station.ipynb` | Multi-channel weather + gas, and source direction | `Environmental1D` full channel set (`wind_speed`, `wind_direction`, `CO`, `CO2`, `NO2`, `pressure`), `wind_rose` |
| `15-acsm.ipynb` | Chemical speciation of the aerosol mass | `ACSM_simple` (`org`, `sulfate`, `nitrate`, `ammonia`, `chlorine`) |

---

## Decisions taken

- **#04 added.** Statistics/exposure had no home in the first draft; folding it
  into #03 would have recreated the sprawl being removed.
- **#10 split** into agreement (10a) and calibration (10b).
- **#14 split out** from #11: `Environmental1D` grew `wind_speed`,
  `wind_direction`, `CO`, `CO2` and `NO2`, and `wind_rose` is a real analysis
  feature, so the weather station outgrew a shared notebook. The *simple*
  Fourtec temperature/RH case stays in #11.
- **Tiger goes in #11**, being a simple 1D gas record.
- **#15 ACSM kept separate.** Assessed as requested: it is only five species
  accessors today, but it is a chemically distinct measurement whose natural
  companions (mass closure against PM, source apportionment) do not belong in a
  notebook about Partector and DiSCmini. It stays short until it grows.

---

## Blocked / open

- **`ZZ_am-sensor_7.csv` is not a DustTrak file.** Its columns (`bin0`-`bin23`,
  `MTof_bin*`, `period`, `flowrate`, `pm1/pm2.5/pm10`, `checksum`) are the
  Alphasense OPC-N3 signature; the DustTrak loader expects a 35-row metadata
  header and `PM1 [mg/m3]`-style columns. It fails both loaders, and
  `detect_instrument` already routes it to OPC-N3, where parsing then fails —
  so it is an OPC-N3 *variant* the current loader cannot read. Needs a decision:
  extend `load_opcn3_file`, or supply a genuine DustTrak export. Until then
  **DustTrak stays undocumented** beyond the API reference.
- **`Environmental1D` accessors do not match the DevLabs loader's columns.**
  `load_devlabs_file` emits `Temp` and `W_direction`, while the class requires
  `Temperature` and `W_direc`, so `.temperature` and `.wind_direction` both
  raise on a DevLabs file; the loader's own `unit`/`dtype` dicts are keyed a
  third way again. Not fixed here: it needs a naming decision from the author,
  and there is no DevLabs sample to verify against. **#14 is blocked on this.**
- No DevLabs sample file exists in `tests/data/`.

---

## Sequencing

1. #01-#03 — establish the house style; everything later cross-references them.
2. #05, #04 — plotting then statistics, both building on activities.
3. #06, #07 — transformations.
4. #08-#10b — analysis.
5. #11-#13, #15 — instruments (data already available).
6. #14 — last, once the `Environmental1D` naming is resolved.

Build time grows with notebook count: seven currently take ~80 s, and fifteen
(several fitting curves, one building an APS correlation cube) will likely run
3-5 minutes locally and in CI. Sample data will be cropped where it does not
weaken the example.
