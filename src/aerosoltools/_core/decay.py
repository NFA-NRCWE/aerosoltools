"""Emission + decay peak fitting (source strength and loss kinetics).

Models a concentration peak in a well-mixed single-zone (box) chamber, where a
source is switched on, the concentration rises to a peak, the source is switched
off, and the concentration decays back towards background. The same three
mass-balance models can be fitted, differing in how particles are lost:

* **First order** ``dP/dt = E - k*P``
  Losses proportional to concentration — air exchange (ventilation) plus wall
  deposition and other first-order sinks. ``k`` is in 1/s.

* **Second order** ``dP/dt = E - C*P**2``
  Losses proportional to concentration squared — coagulation. ``C`` is in
  (concentration·s)⁻¹.

* **Combined** ``dP/dt = E - (K*P + C*P**2)``
  Both a first-order loss ``K`` and a second-order (coagulation) loss ``C``.

In every model ``E`` is the volumetric emission rate (concentration per second)
while the source is on, ``P0`` the background concentration, and the source runs
from ``t0`` for a duration ``tp`` (so the peak occurs at ``t0 + tp``). During the
source-on phase the closed-form solution of the ODE is used; afterwards the
matching source-off decay solution continues from the peak.

Given the chamber volume the emission rate becomes a **source strength**
(``E * volume`` → particles or µg per second). Given an independently known air
exchange rate the first-order loss splits into ventilation + a **wall-loss**
estimate (``k - air_exchange_rate``).

This is a corrected/robustified port of the ``Peak_fitter`` family of functions
(``EXP_FUNC`` / ``LIN_FUNC`` / ``THREE_FUNC``) from the NFA modelling library:
time is measured from the start of the fitted window (rather than from midnight,
which broke across day boundaries) and the ``arctanh`` / ``sqrt(E/C)``
singularities are guarded so the optimiser stays numerically stable.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Value substituted for any non-finite model output, so curve_fit never sees a
# NaN/inf (which would abort the fit) but is still strongly steered away from
# the offending parameter region.
_BIG = 1e20


def _numerator_and_cm3_factor(unit: str) -> tuple[str, float]:
    """Split a concentration unit into its numerator and a per-m³ factor.

    Aerosoltools concentration units are either explicitly per m³ (e.g.
    "µg/m³") or implicitly per cm³ (e.g. "cm⁻³", "nm²/cm³"). This returns the
    numerator quantity (what is being counted) and the multiplicative factor
    that converts the per-cm³ forms to a per-m³ basis (1 m³ = 1e6 cm³), so a
    source strength can be reported as "<numerator>/s".
    """
    if "/" in unit:
        numerator, denom = unit.split("/", 1)
        return numerator, (1e6 if "cm" in denom else 1.0)
    if "cm" in unit:
        return "count", 1e6
    return "count", 1.0


def _sanitize(p: np.ndarray) -> np.ndarray:
    """Replace non-finite model values with a large finite penalty value."""
    return np.nan_to_num(p, nan=_BIG, posinf=_BIG, neginf=_BIG)


# -- the three emission + decay models --------------------------------------
# Each returns the concentration over time ``t`` (seconds from the window start)
# for a source that is on from ``t0`` for a duration ``tp``. ``abs`` mirrors the
# original library and keeps the models well-defined if the optimiser probes a
# slightly negative parameter; the bounds keep every parameter non-negative.


def _first_order(t, k, P0, E, t0, tp):
    """First-order model: ``dP/dt = E - k*P``.

    Source-on: ``P = E/k + (P0 - E/k)*exp(-k*(t-t0))``. Source-off:
    exponential decay from the peak. Reduces to a pure exponential decay when
    ``E -> 0``.
    """
    k = abs(float(k))
    P0 = abs(float(P0))
    E = abs(float(E))
    t0 = abs(float(t0))
    tp = abs(float(tp))
    t = np.asarray(t, dtype=float)
    tau = t - t0
    Pss = E / k if k > 0 else 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        emit = Pss + (P0 - Pss) * np.exp(-k * np.clip(tau, 0, None))
        Pmax = Pss + (P0 - Pss) * np.exp(-k * tp)
        post = Pmax * np.exp(-k * np.clip(tau - tp, 0, None))
        p = np.where(tau < 0, P0, np.where(tau < tp, emit, post))
    return _sanitize(p)


def _second_order(t, C, P0, E, t0, tp):
    """Second-order (coagulation) model: ``dP/dt = E - C*P**2``.

    Source-on: relaxation towards the steady state ``sqrt(E/C)`` via ``tanh``.
    Source-off: ``P = 1/(C*(t - t_peak) + 1/Pmax)``.
    """
    C = abs(float(C))
    P0 = abs(float(P0))
    E = abs(float(E))
    t0 = abs(float(t0))
    tp = abs(float(tp))
    t = np.asarray(t, dtype=float)
    tau = t - t0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Pss = np.sqrt(E / C) if (C > 0 and E > 0) else 0.0
        if Pss > 0:
            a = np.sqrt(E * C)  # = C * Pss
            b = np.arctanh(np.clip(P0 / Pss, -0.999999, 0.999999))
            emit = Pss * np.tanh(a * np.clip(tau, 0, None) + b)
            Pmax = Pss * np.tanh(a * tp + b)
        else:  # no source: pure second-order decay from P0
            inv_P0 = 1.0 / P0 if P0 > 0 else np.inf
            emit = 1.0 / (C * np.clip(tau, 0, None) + inv_P0)
            Pmax = 1.0 / (C * tp + inv_P0) if np.isfinite(inv_P0) else 0.0
        inv_max = 1.0 / Pmax if Pmax > 0 else np.inf
        post = 1.0 / (C * np.clip(tau - tp, 0, None) + inv_max)
        p = np.where(tau < 0, P0, np.where(tau < tp, emit, post))
    return _sanitize(p)


def _combined(t, K, C, P0, E, t0, tp):
    """Combined first + second order model: ``dP/dt = E - (K*P + C*P**2)``.

    A Riccati equation; source-on relaxes towards the positive root of
    ``C*P**2 + K*P - E = 0`` via ``tanh``, and source-off follows the matching
    combined-decay solution.
    """
    K = abs(float(K))
    C = abs(float(C))
    P0 = abs(float(P0))
    E = abs(float(E))
    t0 = abs(float(t0))
    tp = abs(float(tp))
    t = np.asarray(t, dtype=float)
    tau = t - t0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Det = np.sqrt(K * K + 4 * C * E)
        if Det > 0 and C > 0:
            b = np.arctanh(np.clip((2 * C * P0 + K) / Det, -0.999999, 0.999999))
            emit = (Det * np.tanh(0.5 * Det * np.clip(tau, 0, None) + b) - K) / (2 * C)
            Pmax = (Det * np.tanh(0.5 * Det * tp + b) - K) / (2 * C)
        else:
            emit = np.full_like(tau, P0)
            Pmax = P0
        if Pmax > 0 and K > 0 and C > 0:
            ratio = (Pmax * C + K) / (Pmax * K)
            post = 1.0 / (ratio * np.exp(K * np.clip(tau - tp, 0, None)) - C / K)
        else:
            post = np.full_like(tau, Pmax)
        p = np.where(tau < 0, P0, np.where(tau < tp, emit, post))
    return _sanitize(p)


#: model name -> (function, ordered parameter names). ``P0, E, t0, tp`` are
#: shared; the leading entries are the model-specific rate constant(s).
_MODELS = {
    "first_order": (_first_order, ["k", "P0", "E", "t0", "tp"]),
    "second_order": (_second_order, ["C", "P0", "E", "t0", "tp"]),
    "combined": (_combined, ["K", "C", "P0", "E", "t0", "tp"]),
}

#: Accepted aliases mapping onto the canonical model names.
_MODEL_ALIASES = {
    "first_order": "first_order",
    "first": "first_order",
    "1": "first_order",
    "exp": "first_order",
    "exponential": "first_order",
    "second_order": "second_order",
    "second": "second_order",
    "2": "second_order",
    "lin": "second_order",
    "coagulation": "second_order",
    "combined": "combined",
    "third": "combined",
    "3": "combined",
    "three": "combined",
}

#: Complexity penalties for auto-selection: a more complex model is only chosen
#: when it improves the (1 - R²) misfit by more than this factor, mirroring the
#: original ``Determine_fit`` preference for simpler models.
_MODEL_PENALTY = {"first_order": 1.0, "second_order": 1.10, "combined": 1.25}


def decay_curve(model: str, t, popt) -> np.ndarray:
    """Evaluate a fitted model over ``t`` (seconds from the window start).

    Args:
        model: Canonical model name (see :data:`_MODELS`).
        t: Times in seconds from the fit window's start.
        popt: Fitted parameters in the model's own order.

    Returns:
        numpy.ndarray: Modelled concentration at each time.
    """
    func, _ = _MODELS[model]
    return func(np.asarray(t, dtype=float), *popt)


def _r_squared(y: np.ndarray, fit: np.ndarray) -> float:
    """Coefficient of determination of ``fit`` against ``y``."""
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


class DecayFitMixin:
    """Fit an emission + decay peak to estimate source strength and losses."""

    def fit_decay(
        self,
        period: Union[str, tuple],
        metric: str = "PNC",
        model: str = "auto",
        volume: Optional[float] = None,
        air_exchange_rate: Optional[float] = None,
    ) -> dict:
        """Description:
            Fit a single-zone emission + decay peak to one time window of a
            concentration metric, estimating the emission/source strength and
            the loss kinetics (first order, second order, or both).

        Args:
            period (str | tuple): Either an activity name (a boolean column in
                :attr:`data`, e.g. a "Peak" or "Emission" task) or a
                ``(start, end)`` pair (anything :func:`pandas.Timestamp`
                accepts) selecting the window to fit. The window should span
                the background, the rise, the peak and (ideally) the decay.
            metric (str): Metric to fit, as resolved by the class's metric
                lookup (for example "PNC", "MASS", "PM2.5"). Default "PNC".
            model (str): Which loss model to fit: ``"first_order"`` (air
                exchange + deposition), ``"second_order"`` (coagulation),
                ``"combined"`` (both), or ``"auto"`` (fit all and pick the best
                with a bias towards the simpler model). Aliases such as
                "first"/"exp", "second"/"lin", "combined"/"three" are accepted.
                Default "auto".
            volume (float | None): Chamber/room volume in m³. When given, the
                fitted emission rate is reported as a source strength
                (``E * volume``) and a total emitted amount over the source-on
                period.
            air_exchange_rate (float | None): An independently known air
                exchange rate in 1/hour. When given and the model has a
                first-order loss term, the wall-loss/deposition rate is
                estimated as ``(first-order loss) - air_exchange_rate``.

        Returns:
            dict: With keys including:

                * "model": The fitted (or auto-selected) model name.
                * "unit", "metric", "r_squared", "n_points".
                * "params", "errors": Fitted parameters and their standard
                  errors, keyed by name (``k``/``K``/``C``, ``P0``, ``E``,
                  ``t0``, ``tp``).
                * "background": Fitted background concentration ``P0``.
                * "peak_concentration": Modelled peak concentration.
                * "emission_rate", "emission_rate_unit": ``E`` and its unit.
                * "loss_rate_per_hour": First-order loss rate in 1/h (models
                  with a first-order term only).
                * "half_life_hours": ``ln(2)/loss_rate`` (first-order term).
                * "second_order_rate": ``C`` (second-order / combined models).
                * "wall_loss_rate_per_hour": Present when ``air_exchange_rate``
                  is given and a first-order term exists.
                * "source_strength", "source_strength_unit",
                  "total_emitted", "total_emitted_unit": Present when
                  ``volume`` is given.
                * "emission_start_s", "emission_duration_s", "peak_time_s":
                  Timing relative to the window start.
                * "peak_time": Absolute timestamp of the peak.
                * "window_start", "model_popt": For reconstructing the fitted
                  curve (see :func:`decay_curve`).

        Raises:
            ValueError: If ``period`` is neither a known activity nor a valid
                ``(start, end)`` pair, if the window has too few finite
                samples, if ``model`` is unrecognised, or if no model could be
                fitted.

        Examples:
            Fit a chamber emission peak and read the source strength::

                res = data.fit_decay("Emission", metric="PNC", volume=20.0)
                print(res["source_strength"], res["source_strength_unit"])

            Force a first-order fit and split out the wall losses::

                res = data.fit_decay(
                    "Decay", model="first_order", air_exchange_rate=1.5
                )
                print(res["wall_loss_rate_per_hour"])
        """
        key = str(model).strip().lower()
        if key != "auto" and key not in _MODEL_ALIASES:
            raise ValueError(
                f"Unknown model {model!r}. Use 'auto', 'first_order', "
                "'second_order' or 'combined'."
            )

        series, unit = self._get_metric_series(metric)
        times, values = self._decay_window(period, series)

        finite = np.isfinite(values)
        if finite.sum() < 8:
            raise ValueError(
                "Need at least 8 finite samples in the chosen window to fit an "
                "emission + decay peak."
            )
        times = times[finite]
        values = values[finite].astype(float)
        t = (times - times[0]).total_seconds().to_numpy()

        guesses, bounds = self._decay_initial_guesses(t, values)

        # Fit the requested model, or every model when auto-selecting.
        wanted = list(_MODELS) if key == "auto" else [_MODEL_ALIASES[key]]
        fits = {}
        for name in wanted:
            outcome = self._fit_single_model(name, t, values, guesses, bounds)
            if outcome is not None:
                fits[name] = outcome
        if not fits:
            raise ValueError(
                "No emission + decay model could be fitted to this window; try "
                "a wider window that includes the rise and decay of the peak."
            )

        chosen = min(
            fits,
            key=lambda n: (1.0 - fits[n][2]) * _MODEL_PENALTY[n],
        )
        popt, perr, r2 = fits[chosen]
        return self._decay_result(
            chosen,
            popt,
            perr,
            r2,
            unit,
            metric,
            times[0],
            int(finite.sum()),
            volume,
            air_exchange_rate,
        )

    # -- helpers -----------------------------------------------------------
    def _decay_window(self, period, series):
        """Return the (times, values) inside ``period`` for a metric series."""
        if isinstance(period, str):
            if period not in self.activities:
                raise ValueError(f"Activity '{period}' not found.")
            mask = self.data[period].astype(bool)
        elif isinstance(period, tuple) and len(period) == 2:
            start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
            mask = (self.time >= start) & (self.time <= end)
        else:
            raise ValueError("period must be an activity name or a (start, end) tuple.")
        return self.time[mask], np.asarray(series.loc[mask], dtype=float)

    @staticmethod
    def _decay_initial_guesses(t: np.ndarray, y: np.ndarray):
        """Build per-model initial guesses and bounds seeded from the data."""
        window = float(t[-1]) or 1.0
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        dt = dt if dt > 0 else 1.0

        peak_idx = int(np.argmax(y))
        t_peak = float(t[peak_idx])
        pre = y[: peak_idx + 1]
        premin_idx = int(np.argmin(pre)) if peak_idx > 0 else 0
        t0_guess = float(t[premin_idx])
        P0_guess = max(float(y[premin_idx]), 0.0)
        Pmax_obs = float(y[peak_idx])

        tp_guess = max(t_peak - t0_guess, dt)
        E_guess = max((Pmax_obs - P0_guess) / max(tp_guess, dt), 1e-9)

        # First-order loss rate guessed from the post-peak log-linear slope.
        k_guess = 1.0 / 3600.0
        post_t = t[peak_idx:] - t_peak
        post_y = y[peak_idx:]
        good = post_y > 0
        if good.sum() >= 3:
            try:
                slope = np.polyfit(post_t[good], np.log(post_y[good]), 1)[0]
                if slope < 0:
                    k_guess = min(max(-slope, 1e-6), 1.0)
            except Exception:
                pass
        C_guess = k_guess / max(Pmax_obs, 1e-9)

        # Shared bounds for [P0, E, t0, tp].
        t0_hi = max(t_peak, dt)
        shared_lo = [0.0, 0.0, 0.0, dt]
        shared_hi = [Pmax_obs * 2 + 1.0, np.inf, t0_hi, window * 1.5 + dt]
        shared_p0 = [
            min(P0_guess, Pmax_obs * 2),
            E_guess,
            min(t0_guess, t0_hi),
            min(max(tp_guess, dt), window * 1.5),
        ]
        # A small positive emission floor keeps the tanh models away from the
        # E -> 0 singularity; first order has no such issue.
        e_floor = max(E_guess * 0.02, 1e-9)

        guesses = {
            "first_order": [k_guess, *shared_p0],
            "second_order": [C_guess, *shared_p0],
            "combined": [k_guess, C_guess, *shared_p0],
        }
        bounds = {
            "first_order": (
                [1e-8, *shared_lo],
                [np.inf, *shared_hi],
            ),
            "second_order": (
                [1e-20, shared_lo[0], e_floor, shared_lo[2], shared_lo[3]],
                [np.inf, *shared_hi],
            ),
            "combined": (
                [1e-8, 1e-20, shared_lo[0], e_floor, shared_lo[2], shared_lo[3]],
                [np.inf, np.inf, *shared_hi],
            ),
        }
        return guesses, bounds

    @staticmethod
    def _fit_single_model(name, t, y, guesses, bounds):
        """Fit one model; return ``(popt, perr, r2)`` or None on failure."""
        func, _ = _MODELS[name]
        try:
            popt, pcov = curve_fit(
                func, t, y, p0=guesses[name], bounds=bounds[name], maxfev=20000
            )
        except Exception:
            return None
        fit = func(t, *popt)
        r2 = _r_squared(y, fit)
        if not np.isfinite(r2):
            return None
        with np.errstate(invalid="ignore"):
            perr = np.sqrt(np.diag(pcov))
        perr = np.nan_to_num(perr, nan=0.0, posinf=0.0, neginf=0.0)
        return popt, perr, r2

    def _decay_result(
        self, model, popt, perr, r2, unit, metric, t_start, n, volume, ach
    ) -> dict:
        """Assemble the public result dict from a fitted model."""
        names = _MODELS[model][1]
        params = {nm: float(abs(v)) for nm, v in zip(names, popt)}
        errors = {nm: float(v) for nm, v in zip(names, perr)}

        t0 = params["t0"]
        tp = params["tp"]
        E = params["E"]
        P0 = params["P0"]
        peak_conc = float(decay_curve(model, np.array([t0 + tp]), popt)[0])

        result = {
            "model": model,
            "unit": unit,
            "metric": metric,
            "r_squared": r2,
            "n_points": n,
            "params": params,
            "errors": errors,
            "background": P0,
            "peak_concentration": peak_conc,
            "emission_rate": E,
            "emission_rate_unit": f"{unit}/s",
            "emission_start_s": t0,
            "emission_duration_s": tp,
            "peak_time_s": t0 + tp,
            "peak_time": t_start + pd.to_timedelta(t0 + tp, unit="s"),
            "window_start": t_start,
            "model_popt": [float(v) for v in popt],
        }

        # First-order loss term (k for first order, K for combined).
        linear_rate = params.get("k", params.get("K"))
        if linear_rate is not None:
            loss_per_hour = linear_rate * 3600.0
            result["loss_rate_per_hour"] = loss_per_hour
            result["half_life_hours"] = (
                np.log(2) / loss_per_hour if loss_per_hour > 0 else float("nan")
            )
            if ach is not None:
                result["wall_loss_rate_per_hour"] = loss_per_hour - ach
        if "C" in params:
            result["second_order_rate"] = params["C"]

        # Source strength from the emission rate and the chamber volume.
        if volume is not None:
            numerator, factor = _numerator_and_cm3_factor(unit)
            strength = E * factor * volume
            result["source_strength"] = strength
            result["source_strength_unit"] = f"{numerator}/s"
            result["total_emitted"] = strength * tp
            result["total_emitted_unit"] = numerator

        return result
