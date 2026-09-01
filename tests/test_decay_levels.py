"""Separate start and background levels in the decay fit (GitHub #30).

The fit used to take the level the window opened at *as* the background, so a
window that opened while an earlier event was still decaying got a background
that was much too high -- and, to make the decay flatten out there, a loss rate
that was much too fast. Background and start level are now fitted separately.

The synthetic peaks here are built from the closed-form solution directly rather
than through :func:`decay_curve`, so the tests check the fit against an
independent ground truth rather than against the module's own model code.
"""

import numpy as np
import pandas as pd
import pytest

from aerosoltools import Aerosol1D
from aerosoltools._core import decay as _decay

TRUE_BG = 100.0
TRUE_K = 1 / 400.0  # 9 /h
TRUE_E = 2.0
T0, TP, N = 300.0, 300.0, 1200


def _first_order_peak(initial_excess, noise_frac=0.01, seed=1):
    """A first-order emission+decay peak starting ``initial_excess`` above bg."""
    t = np.arange(N, dtype=float)
    s = t - T0
    xss = TRUE_E / TRUE_K
    relax = np.exp(-TRUE_K * np.clip(s, 0, None))
    emit = xss * (1 - relax) + initial_excess * relax
    relax_tp = np.exp(-TRUE_K * TP)
    xmax = xss * (1 - relax_tp) + initial_excess * relax_tp
    dec = xmax * np.exp(-TRUE_K * np.clip(s - TP, 0, None))
    x = np.where(s < 0, initial_excess, np.where(s < TP, emit, dec))

    y = TRUE_BG + x
    y = y + np.random.default_rng(seed).normal(0, noise_frac * y.max(), N)
    times = pd.date_range("2026-01-01", periods=N, freq="1s")
    obj = Aerosol1D(pd.DataFrame({"Datetime": times, "Total_conc": y}))
    obj._meta["unit"] = "cm-3"
    return obj, y


def _fit(obj, **kw):
    return obj.fit_decay((obj.time[0], obj.time[-1]), model="first_order", **kw)


@pytest.mark.parametrize("initial_excess", [0.0, 20.0, 50.0, 100.0, 150.0, 300.0])
def test_background_recovered_however_elevated_the_start(initial_excess):
    """The fitted background is the asymptote, not wherever the window opened."""
    obj, y = _first_order_peak(initial_excess)
    res = _fit(obj)

    assert res.background == pytest.approx(TRUE_BG, rel=0.10)
    # The start level tracks the data, and sits above the background.
    assert res.start_concentration == pytest.approx(float(y[0]), rel=0.15)
    assert res.start_concentration >= res.background
    assert res.initial_excess == pytest.approx(initial_excess, abs=0.15 * TRUE_BG + 15)


@pytest.mark.parametrize("initial_excess", [0.0, 50.0, 150.0, 300.0])
def test_loss_rate_not_inflated_by_an_elevated_start(initial_excess):
    """The old behaviour doubled k for an elevated start; it must not now."""
    obj, _ = _first_order_peak(initial_excess)
    res = _fit(obj)
    assert res.params["k"] == pytest.approx(TRUE_K, rel=0.10)


def test_levels_are_self_consistent():
    obj, _ = _first_order_peak(150.0)
    res = _fit(obj)
    assert res.start_concentration == pytest.approx(
        res.background + res.initial_excess, rel=1e-9
    )
    assert res.peak_concentration == pytest.approx(
        res.background + res.peak_excess, rel=1e-9
    )


def test_the_curve_starts_at_the_start_level_and_relaxes_to_the_background():
    obj, _ = _first_order_peak(150.0)
    res = _fit(obj)

    at_start = _decay.decay_curve(res.model, [0.0], res.model_popt)[0]
    assert at_start == pytest.approx(res.start_concentration, rel=1e-9)

    # Far past the peak the curve has relaxed towards the background.
    far = _decay.decay_curve(res.model, [res.peak_time_s + 10 * 3600], res.model_popt)[
        0
    ]
    assert far == pytest.approx(res.background, rel=1e-3)


# -- manual values are guesses, not constraints ------------------------------


def test_a_guessed_background_is_refined_not_pinned():
    obj, _ = _first_order_peak(150.0)
    guessed = _fit(obj, background=40.0)
    assert guessed.background != pytest.approx(40.0, rel=1e-3)
    assert guessed.background == pytest.approx(TRUE_BG, rel=0.10)


def test_a_guessed_peak_is_refined_not_pinned():
    obj, y = _first_order_peak(150.0)
    honest = _fit(obj)
    guessed = _fit(obj, peak_concentration=float(y.max()) * 0.75)
    # The bad guess does not survive as the reported peak.
    assert guessed.peak_concentration > float(y.max()) * 0.80
    assert guessed.peak_concentration == pytest.approx(
        honest.peak_concentration, rel=0.20
    )


def test_a_guessed_start_level_is_accepted_as_a_seed():
    obj, y = _first_order_peak(150.0)
    res = _fit(obj, start_concentration=float(y[0]))
    assert res.start_concentration == pytest.approx(float(y[0]), rel=0.15)


def test_optimize_false_stops_at_the_guess():
    """The GUI's live preview must show exactly what was typed/dragged."""
    obj, _ = _first_order_peak(150.0)
    res = _fit(obj, background=42.0, peak_concentration=900.0, optimize=False)
    assert res.background == pytest.approx(42.0, rel=1e-6)
    assert res.peak_concentration == pytest.approx(900.0, rel=1e-6)


def test_errors_are_reported_for_the_fitted_levels():
    obj, _ = _first_order_peak(150.0)
    res = _fit(obj)
    for key in ("P0", "xi", "xmax"):
        assert key in res.errors
        assert np.isfinite(res.errors[key])


# -- model preference --------------------------------------------------------


def test_zeroth_order_is_disfavoured_relative_to_first_order():
    assert _decay._MODEL_PENALTY["zeroth_order"] > _decay._MODEL_PENALTY["first_order"]
    assert _decay._MODEL_PENALTY["zeroth_order"] >= 1.4


@pytest.mark.parametrize("initial_excess", [0.0, 150.0])
def test_auto_selects_first_order_for_a_first_order_peak(initial_excess):
    """An elevated start used to push 'auto' onto the zeroth-order model."""
    obj, _ = _first_order_peak(initial_excess)
    res = obj.fit_decay((obj.time[0], obj.time[-1]), model="auto")
    assert res.model != "zeroth_order"


# -- back-compat -------------------------------------------------------------


@pytest.mark.parametrize(
    "model,loss",
    [
        ("zeroth_order", [0.5]),
        ("first_order", [1 / 400.0]),
        ("second_order", [2e-5]),
        ("combined", [1 / 600.0, 1e-5]),
    ],
)
def test_legacy_popt_without_the_initial_excess_still_evaluates(model, loss):
    """Saved fits predate the trailing ``xi`` parameter; they must still draw."""
    t = np.array([0.0, 450.0, 900.0])
    legacy = _decay.decay_curve(model, t, [*loss, 100.0, 2.0, 300.0, 300.0])
    explicit = _decay.decay_curve(model, t, [*loss, 100.0, 2.0, 300.0, 300.0, 0.0])
    assert np.allclose(legacy, explicit)
    # xi = 0 means the curve opens at the background, as it always did.
    assert legacy[0] == pytest.approx(100.0)
