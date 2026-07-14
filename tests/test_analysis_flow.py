import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerosoltools import Aerosol1D
from aerosoltools._core import decay as _decay
from aerosoltools.loaders import load_cpc_file, load_elpi_file


def test_full_elpi_pipeline_with_plotting():
    """End-to-end ELPI pipeline + basic checks on core 2D methods."""
    test_file = os.path.join(os.path.dirname(__file__), "data", "Sample_ELPI.txt")
    data = load_elpi_file(test_file)

    activity_periods = {
        "Background": [("2023/09/07 09:06:50", "2023/09/07 09:07:50")],
        "Emission": [("2023/09/07 09:07:55", "2023/09/07 09:08:30")],
        "Decay": [("2023/09/07 09:09:00", "2023/09/07 09:10:50")],
    }

    data.mark_activities(activity_periods)

    # --- conversion to volume concentration -----------------------------------
    data._convert_to_volume_concentration()
    # Check dtype or column name contains 'dV'
    assert "dV" in str(data.dtype), f"Expected 'dV' in dtype, got: {data.dtype}"

    # Check unit
    assert getattr(data, "unit", None) == "nm³/cm³", f"Unexpected unit: {data.unit}"

    # --- plotting PSD and timeseries -----------------------------------------
    fig, axs = plt.subplots(ncols=2, nrows=2)
    data.plot_psd(["Decay", "Emission"], ax=axs[0, 0], normalize=False)
    data.plot_psd(["All data"], ax=axs[0, 0])
    data.plot_psd(normalize=True, ax=axs[1, 0])

    # --- normalize by dlogDp --------------------------------------------------
    data.normalize_logdp()
    # Check dtype for "dlogDp"
    assert "dlogDp" in str(data.dtype), f"Expected 'dlogDp' in dtype, got: {data.dtype}"

    data.plot_timeseries(
        y_3d=(1, 0), mark_activities=True, ax1=axs[0, 1], ax2=axs[1, 1]
    )

    # --- summarize_activities (new version) -----------------------------------
    summary_table = data.summarize_activities()

    # Must be a DataFrame
    assert isinstance(summary_table, pd.DataFrame), "summary_table is not a DataFrame"

    # Must have a 'Segment' column
    assert (
        "Segment" in summary_table.columns
    ), "'Segment' column missing from summary_table"

    # Must include expected segment names
    expected_segments = {"All data", "Background", "Emission", "Decay"}
    found_segments = set(summary_table["Segment"])
    missing = expected_segments - found_segments
    assert not missing, f"Missing expected segments: {missing}"

    # Duration column (2D version uses only HH:MM)
    assert "Duration (HH:MM)" in summary_table.columns

    # There should be at least one metric column with units in the header
    assert any(
        "PNC" in col for col in summary_table.columns
    ), "Expected at least one PNC metric column in summarize_activities output"


def test_pm_calc_creates_and_reuses_px_columns():
    """Check that PM_calc adds P{x} columns once and reuses them on second call."""
    test_file = os.path.join(os.path.dirname(__file__), "data", "Sample_ELPI.txt")
    data = load_elpi_file(test_file)

    old_cols = set(data.extra_data.columns)

    # First call: cumulative PM4.2
    data.PM_calc(dtype="dM", PM=4.2)
    new_cols_after_first = set(data.extra_data.columns) - old_cols
    assert len(new_cols_after_first) == 1, "PM_calc should add exactly one new column"
    pm42_label = new_cols_after_first.pop()
    series_first = data.extra_data[pm42_label].copy()

    # Second call with the same parameters should not create another column
    data.PM_calc(dtype="dM", PM=4.2)
    new_cols_after_second = set(data.extra_data.columns) - old_cols
    assert (
        len(new_cols_after_second) == 1
    ), "PM_calc should reuse existing P{x} column, not add duplicates"
    series_second = data.extra_data[pm42_label]
    pd.testing.assert_series_equal(series_first, series_second)

    # Band-limited call: e.g. 1–4.2 µm
    cols_before_band = set(data.extra_data.columns)
    data.PM_calc(dtype="dM", PM=4.2, lower_lim=1.0)
    new_band_cols = set(data.extra_data.columns) - cols_before_band
    assert (
        len(new_band_cols) == 1
    ), "Band-limited PM_calc should add exactly one new band column"
    band_label = new_band_cols.pop()
    band_series = data.extra_data[band_label]
    assert (
        not band_series.isna().all()
    ), "Band-limited P{x} series should not be all NaN"


def test_aerosol2d_summarize_exposure_pm42_with_background_activity():
    """Test summarize_exposure on Aerosol2D with PM4.2 and background as activity."""
    test_file = os.path.join(os.path.dirname(__file__), "data", "Sample_ELPI.txt")
    data = load_elpi_file(test_file)

    activity_periods = {
        "Background": [("2023/09/07 09:06:50", "2023/09/07 09:07:50")],
        "Emission": [("2023/09/07 09:07:55", "2023/09/07 09:08:30")],
        "Decay": [("2023/09/07 09:09:00", "2023/09/07 09:10:50")],
    }
    data.mark_activities(activity_periods)

    # New API: multi-activity summary; restrict to "Emission" only
    result = data.summarize_exposure(
        metric="PM4.2",
        background="Background",
        exposure_hours=8.0,
        short_limit=1.0,
        long_limit=1.0,
        short_window="15min",
        twa_window="8h",
        activities=["Emission"],
    )

    # One-row result for the single activity
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    row = result.iloc[0]
    cols = list(result.columns)

    # Basic identity checks
    assert row["Segment"] == "Emission"
    assert row["Metric"] == "PM4.2"

    # Required structural columns / patterns (unit-agnostic)
    assert "Segment" in cols
    assert "Metric" in cols

    # Duration column (string "HH:MM")
    assert any("Duration" in c for c in cols)

    # Max and percentile columns with units in the header
    assert any(c.startswith("Max [") for c in cols)
    assert any("95th percentile" in c for c in cols)
    assert any("5th percentile" in c for c in cols)

    # Peak info
    assert any("Peaks" in c for c in cols)

    # STEL / short-term limit info
    assert any(c.startswith("STEL [") for c in cols)
    assert any("STEL window" in c for c in cols)
    assert any("STEL exceedance" in c for c in cols)
    assert any("STEL episodes" in c for c in cols)

    # Long-term limit info
    assert any("Exposure limit [" in c for c in cols)
    assert any("Exposure limit exceedance" in c for c in cols)

    # TWA info
    assert any("TWA concentration" in c for c in cols)
    assert any("TWA window" in c for c in cols)

    # --- simple numeric sanity checks -----------------------------------------
    # TWA concentration should be non-negative
    twa_conc_col = next(c for c in cols if c.startswith("TWA concentration"))
    assert row[twa_conc_col] >= 0.0

    # STEL exceedance min should be non-negative
    stel_exceed_col = next(c for c in cols if "STEL exceedance" in c)
    assert row[stel_exceed_col] >= 0.0

    # Episodes count should be an integer-like non-negative number
    stel_episodes_col = next(c for c in cols if "STEL episodes" in c)
    assert row[stel_episodes_col] >= 0


def test_aerosol1d_summarize_and_exposure_pnc():
    """Test summarize_activities and summarize_exposure on a 1D CPC dataset."""
    test_file = os.path.join(os.path.dirname(__file__), "data", "Sample_CPC_Direct.txt")
    data = load_cpc_file(test_file)

    # Simple artificial activity: first half vs second half
    times = data.time
    midpoint = times[int(len(times) / 2)]
    activity_periods = {
        "First_half": [(times.min(), midpoint)],
        "Second_half": [(midpoint, times.max())],
    }
    data.mark_activities(activity_periods)

    # --- summarize_activities (1D version) ------------------------------------
    summary = data.summarize_activities()

    assert isinstance(summary, pd.DataFrame)
    assert "Segment" in summary.columns

    # Duration columns – 1D implementation may provide minutes and/or HH:MM
    assert "Duration (HH:MM)" in summary.columns
    has_duration_min = "Duration (min)" in summary.columns

    # There should be a row for "All data" and for both halves
    segments = set(summary["Segment"])
    for seg in ["All data", "First_half", "Second_half"]:
        assert seg in segments, f"Missing segment '{seg}' in summarize_activities."

    # Durations should be positive / non-zero for all non-empty segments
    if has_duration_min:
        # Preferred: explicit minutes column
        assert (summary["Duration (min)"] > 0).all()
    else:
        # Fallback: use HH:MM string representation – nothing should be "00:00"
        assert not (summary["Duration (HH:MM)"] == "00:00").any()

    # --- summarize_exposure on PNC (1D new-style API) ------------------------
    exp = data.summarize_exposure(
        metric="PNC",
        background=None,
        exposure_hours=8.0,
        short_limit=1.0,
        long_limit=1.0,
        short_window="15min",
        twa_window="8h",
        activities=["All data"],
    )

    assert isinstance(exp, pd.DataFrame)
    assert len(exp) == 1
    row = exp.iloc[0]
    cols = list(exp.columns)

    # Identity checks
    assert row["Segment"] == "All data"
    assert row["Metric"] == "PNC"

    # Structural checks (unit-agnostic, no assumptions about old column names)
    assert "Segment" in cols
    assert "Metric" in cols
    assert any("Duration" in c for c in cols)

    # Max + percentile columns
    assert any(c.startswith("Max [") for c in cols)
    assert any("95th percentile" in c for c in cols)
    assert any("75th percentile" in c for c in cols)
    assert any("50th percentile" in c for c in cols)
    assert any("25th percentile" in c for c in cols)
    assert any("5th percentile" in c for c in cols)

    # Peaks, STEL, long-term limit, TWA
    assert any("Peaks" in c for c in cols)
    assert any(c.startswith("STEL [") for c in cols)
    assert any("STEL window" in c for c in cols)
    assert any("STEL exceedance" in c for c in cols)
    assert any("STEL episodes" in c for c in cols)
    assert any("Exposure limit [" in c for c in cols)
    assert any("Exposure limit exceedance" in c for c in cols)
    assert any("TWA concentration" in c for c in cols)
    assert any("TWA window" in c for c in cols)

    # --- numeric sanity checks ------------------------------------------------
    # Duration should not be zero-ish
    duration_col = next(c for c in cols if c.startswith("Duration"))
    # If it's HH:MM string, at least ensure it's not "00:00"
    if isinstance(row[duration_col], str):
        assert row[duration_col] != "00:00"

    # TWA concentration non-negative
    twa_conc_col = next(c for c in cols if c.startswith("TWA concentration"))
    assert row[twa_conc_col] >= 0.0

    # STEL exceedance minutes non-negative
    stel_exceed_col = next(c for c in cols if "STEL exceedance" in c)
    assert row[stel_exceed_col] >= 0.0

    # Peaks count non-negative
    peaks_col = next(c for c in cols if "Peaks" in c)
    assert row[peaks_col] >= 0


def _make_peak_dataset(model, popt, seed=0, n=200, step_s=10):
    """Build a 1D dataset from a known emission+decay model, plus noise."""
    times = pd.date_range("2024-01-01 00:00", periods=n, freq=f"{step_s}s")
    t = np.arange(n) * float(step_s)
    y = _decay.decay_curve(model, t, popt)
    y = y + np.random.default_rng(seed).normal(0, 0.01 * float(np.max(y)), n)
    df = pd.DataFrame({"time": times, "Total_conc": y})
    obj = Aerosol1D(df)
    obj._meta["unit"] = "cm⁻³"
    obj.mark_activities({"Peak": [(times[0], times[-1])]})
    return obj


def test_fit_decay_recovers_first_order_and_source_strength():
    """A first-order emission+decay peak fit recovers k, E and source strength."""
    # Truth: k = 1/300 /s (=12 /h), E = 50 cm-3/s, background 100, emit 100..500 s.
    obj = _make_peak_dataset("first_order", [1 / 300.0, 100.0, 50.0, 100.0, 400.0])
    res = obj.fit_decay("Peak", model="first_order", volume=20.0, air_exchange_rate=2.0)

    from collections.abc import Mapping

    from aerosoltools import DecayResult

    # fit_decay returns a typed DecayResult that is *also* a read-only mapping,
    # so attribute and item access agree and legacy `res[...]` keeps working.
    assert isinstance(res, DecayResult)
    assert isinstance(res, Mapping)
    assert res.r_squared == res["r_squared"]
    assert res.model == res["model"]
    # Optional keys appear only when applicable (here: volume + air-exchange given).
    assert "source_strength" in res and res.source_strength == res["source_strength"]
    assert res.to_dict()["model"] == "first_order"

    assert res["model"] == "first_order"
    assert res["r_squared"] > 0.99
    # Loss rate ~ 12 /h (1/300 s^-1 * 3600).
    assert abs(res["loss_rate_per_hour"] - 12.0) < 1.0
    # Emission rate ~ 50 cm-3/s.
    assert abs(res["emission_rate"] - 50.0) < 5.0
    # Wall loss = loss - known ACH.
    assert res["wall_loss_rate_per_hour"] == res["loss_rate_per_hour"] - 2.0
    # Source strength = E * 1e6 (cm3/m3) * volume; unit is a per-second rate.
    assert res["source_strength"] > 0
    assert res["source_strength_unit"].endswith("/s")
    assert res["half_life_hours"] > 0


def test_fit_decay_auto_selects_and_supports_tuple_window():
    """Auto mode picks a good model and a (start, end) window works."""
    obj = _make_peak_dataset(
        "first_order", [1 / 250.0, 80.0, 40.0, 120.0, 350.0], seed=1
    )
    span = (obj.time.min(), obj.time.max())
    res = obj.fit_decay(span, model="auto")
    assert res["model"] in {"first_order", "second_order", "combined"}
    assert res["r_squared"] > 0.98
    # Model params are echoed with a P0/E and the timing fields.
    for key in ("P0", "E", "t0", "tp"):
        assert key in res["params"]
    # No volume given -> source-strength fields are omitted from the mapping view
    # (the attribute is still present and None), matching the historical dict.
    assert "source_strength" not in res
    assert res.source_strength is None


def test_fit_decay_second_order_recovers_rate():
    """A second-order (coagulation) peak fit recovers the C rate constant."""
    obj = _make_peak_dataset("second_order", [2e-5, 100.0, 50.0, 100.0, 400.0], seed=2)
    res = obj.fit_decay("Peak", model="second_order")
    assert res["r_squared"] > 0.99
    assert abs(res["second_order_rate"] - 2e-5) < 1e-5
    # A pure second-order model reports no first-order loss term.
    assert "loss_rate_per_hour" not in res
