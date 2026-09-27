# Summary

Turns marked activities into numbers: mean concentration per task, and exposure
metrics against occupational limits. This is usually the tab you came for — the
one whose table ends up in the report.

```{figure} ../_static/gui/tab-summary.png
:alt: The Summary tab showing a per-activity table with duration, mean and standard deviation for each dataset
:width: 100%

One row per dataset × activity, with the statistics you asked for. Several
instruments are combined into a single table.
```

**Available for:** any project. It works with one dataset or many — tick several
and they are combined into one table with `Dataset` and `Instrument` columns, so
you can compare instruments task by task.

Mark your activities on the [Time series](time-series.md) tab first; with none
marked you only get an **All data** row.

## Datasets to include

Tick the datasets to summarise. Nothing is computed until you press **Compute**,
because exposure statistics over several long datasets can take a moment.

## Type

**Activity summary** — descriptive statistics per activity: duration plus
whichever of **Mean**, **Std**, **Min**, **Max** and **Median** you tick.

**Exposure summary** — the occupational-hygiene view: the time-weighted average
over the TWA window, the highest short-term average, peaks, and how each compares
against the limits you set. It reveals the extra fields:

| Field | Meaning |
| --- | --- |
| **STEL (short-term limit)** | Short-term exposure limit. The highest short-window average is compared against it. |
| **over** | Averaging window for the STEL check, as a pandas offset — `15min` by default. |
| **OEL (8h limit)** | Occupational exposure limit. The time-weighted average is compared against it. |
| **TWA window** | Averaging window for the time-weighted average — `8h` by default. |

Both limits are in the **unit shown next to them** — the unit of the metric you
chose, so if you are summarising a mass concentration in µg/m³, enter the limit
in µg/m³. When several datasets report that metric at different scales, the
limit is converted to each dataset's own unit before comparing.

### Substance

Instead of typing the limits, pick the substance being measured. The dropdown
lists every dust limit in the Danish limit-value order (*Bekendtgørelse om
grænseværdier for stoffer og materialer*, Bilag 2, Afsnit B) with its name, CAS
number, 8-hour and short-term limits in mg/m³, and the order's remarks
(**E** EU limit exists · **L** ceiling value · **H** skin uptake · **K**
carcinogenic). Hover a row for the full explanation.

Picking one fills **OEL** and **STEL**, converted to the metric's unit
(0.1 mg/m³ becomes 100 µg/m³). A mass-based limit cannot be compared with a
number concentration, so if the chosen metric is not a mass the fields are left
alone and the status line says why — choose a mass metric first. Typing a limit
by hand clears the pick.

Keep in mind:

- **Short-term values marked (2×)** are not printed in the order: where it
  refers to *§ 3, stk. 2* the short-term limit is twice the 8-hour limit.
- **Fraction matters.** Most limits name a fraction (respirable, inhalable);
  one that names none applies to total dust. The fraction is shown after the
  pick — compare against a measurement of that fraction.
- **Fibres are not listed**: their limits are counts per cm³, which these
  instruments do not measure. Afsnit A (gases, vapours and some metal dusts and
  fumes) and Afsnit C (welding) are not included.

The computed table gains **Substance** and **Limit source** columns (e.g.
`BEK nr 613 af 29/06/2026`), so an exported table always says which order it
was compared against. The pick is saved with the project.

Tools → *Occupational exposure limits…* shows the whole list and its source.
**Check for a newer order** asks Retsinformation whether the order is still in
force and, if it was replaced, offers the new one; you can also fetch an order
by number. A fetched list is shown against the current one before you accept
it, then used for new picks — picks already saved in a project keep their
values.

## Choose metrics…

Picks which quantities to summarise, grouped by instrument. Two things make this
work across a mixed project:

- Each metric is computed **only for the datasets that provide it**, so adding a
  gas monitor to a project full of particle counters does not fill the table with
  blanks.
- Comparable quantities recorded at different unit scales — ng/m³ and µg/m³, say
  — are merged into one column.

## Compute, and the staleness warning

**Compute** builds the table. The result is cached and saved with the project, so
reopening it later shows what you computed without recalculating.

Because it is cached, it can go out of date. If you edit activities, change the
data, or alter any of the limits, an amber banner appears:

> ⚠ These values may be out of date — tasks, data, or settings changed since they
> were computed. Click Compute to refresh.

Take it seriously — the numbers shown were computed from the *previous* inputs.

## Export to Excel…

Writes the combined table to `.xlsx` or `.csv`, exactly as displayed.

## Under the hood

```python
import aerosoltools as at

data = at.load_ops_file("measurement.csv")

# Activity summary
data.summarize_activities(stats=["mean", "std"])

# Exposure summary
data.summarize_exposure(
    metric="PM4.2",
    long_limit=1.0,       # OEL (8h limit)
    twa_window="8h",      # TWA window
    short_limit=1.0,      # STEL
    short_window="15min", # over
)

# Substance: limits from the Danish limit-value order (bundled, offline)
limits = at.load_exposure_limits()
quartz = limits["Kvarts, respirabel"]
limits.source.label                  # 'BEK nr 613 af 29/06/2026'
data.summarize_exposure(
    metric="PM4.2",
    long_limit=quartz.twa_in("µg/m³"),   # 100.0
    short_limit=quartz.stel_in("µg/m³"), # 200.0
)
```

To check for, and fetch, a newer order from a script or the command line:

```python
from aerosoltools.exposure_limits import fetch_exposure_limits, find_current_order

current = find_current_order()          # follows replaced orders to the one in force
newer = fetch_exposure_limits(current.current.eli)
newer.to_json("limits.json")            # at.load_exposure_limits("limits.json")
```

```bash
python -m aerosoltools.exposure_limits check
python -m aerosoltools.exposure_limits fetch --latest -o limits.json
```

See [4 — Statistics and exposure](../examples/04-statistics-and-exposure.ipynb).
