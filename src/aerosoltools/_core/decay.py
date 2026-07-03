"""Emission + decay peak fitting (source strength and loss kinetics).

Models a concentration peak in a well-mixed single-zone (box) chamber, where a
source is switched on, the concentration rises from the background ``P0`` to a
peak, the source is switched off, and the concentration decays back **towards
that same background**. The models describe the *excess* over background,
``X = P - P0``, so the curve starts at ``P0``, rises during emission, and
relaxes to ``P0`` again — matching what a real decay measurement looks like.

Four loss models are offered, differing in how the excess is removed:

* **Zeroth order** ``dX/dt = E - a``
  A constant removal rate ``a`` (concentration/s), independent of how much is in
  the air — the excess decays *linearly*. Empirical; useful when the decay looks
  straight rather than exponential.

* **First order** ``dX/dt = E - k*X``
  Loss proportional to concentration — air exchange (ventilation/dilution) plus
  wall deposition and other first-order sinks. ``k`` is in 1/s; the excess
  decays exponentially. This is the workhorse indoor-aerosol model.

* **Second order** ``dX/dt = E - C*X**2``
  Loss proportional to concentration squared — coagulation. ``C`` is in
  (concentration·s)⁻¹.

* **Combined** ``dX/dt = E - (K*X + C*X**2)``
  Both a first-order loss ``K`` (air exchange + deposition) and a second-order
  (coagulation) loss ``C`` together.

In every model ``E`` is the volumetric emission rate (concentration per second)
while the source is on, ``P0`` the background, and the source runs from ``t0``
for a duration ``tp`` (so the peak is at ``t0 + tp``).

**Two-stage fit.** Rather than fitting the whole rise+peak+decay at once — which
lets the many decay points outvote the few rise points and pulls the modelled
peak *below* the data — the fit is done in two stages:

1. *Decay stage.* The loss model is fitted to the **post-peak** points only,
   giving the loss kinetics and, by extrapolating back to the peak time, the
   peak excess ``Xmax`` and the background ``P0``. Because it uses only the
   monotone decay, it is robust and it anchors the peak to the data instead of
   averaging it away.
2. *Emission stage.* With the loss kinetics fixed, the emission rate ``E`` is
   back-solved from the anchored peak and the emission duration ``tp``.

Given the chamber volume the emission rate becomes a **source strength**
(``E * volume`` → particles or µg per second). Given an independently known air
exchange rate the first-order loss splits into ventilation + a **wall-loss**
estimate (``k - air_exchange_rate``).

This is a corrected/robustified port of the ``Peak_fitter`` family of functions
(``EXP_FUNC`` / ``LIN_FUNC`` / ``THREE_FUNC``) from the NFA modelling library:
time is measured from the start of the fitted window (rather than from midnight,
which broke across day boundaries), the models decay back to background instead
of to zero, and the fit is staged so the peak height is respected.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

# Value substituted for any non-finite model output, so curve_fit never sees a
# NaN/inf (which would abort the fit) but is still strongly steered away from
# the offending parameter region.
_BIG = 1e20

# Upper clamp for exponent arguments so exp() never overflows to inf.
_EXP_CLAMP = 700.0


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


def _pos(x) -> float:
    """Absolute value of a scalar parameter (the fit bounds keep them ≥ 0)."""
    return abs(float(x))


# -- full emission + decay curves (excess above background, back to background)
# Each returns the concentration over time ``t`` (seconds from the window start)
# for a source on from ``t0`` for a duration ``tp``. Signatures are
# ``(t, <loss params>, P0, E, t0, tp)`` so a fitted parameter vector round-trips
# through :func:`decay_curve` for plotting.


def _zeroth_order(t, a, P0, E, t0, tp):
    """Zeroth-order model ``dX/dt = E - a``: linear rise and linear decay."""
    a, P0, E, t0, tp = map(_pos, (a, P0, E, t0, tp))
    s = np.asarray(t, dtype=float) - t0
    rise = (E - a) * np.clip(s, 0.0, tp)
    xmax = (E - a) * tp
    dec = xmax - a * np.clip(s - tp, 0.0, None)
    x = np.where(s < 0, 0.0, np.where(s < tp, rise, dec))
    return _sanitize(P0 + np.clip(x, 0.0, None))


def _first_order(t, k, P0, E, t0, tp):
    """First-order model ``dX/dt = E - k*X``: exponential rise and decay."""
    k, P0, E, t0, tp = map(_pos, (k, P0, E, t0, tp))
    s = np.asarray(t, dtype=float) - t0
    with np.errstate(over="ignore", invalid="ignore"):
        xss = E / k if k > 0 else 0.0
        emit = xss * (1.0 - np.exp(-k * np.clip(s, 0.0, None)))
        xmax = xss * (1.0 - np.exp(-k * tp))
        dec = xmax * np.exp(-k * np.clip(s - tp, 0.0, None))
        x = np.where(s < 0, 0.0, np.where(s < tp, emit, dec))
    return _sanitize(P0 + x)


def _second_order(t, C, P0, E, t0, tp):
    """Second-order (coagulation) model ``dX/dt = E - C*X**2``."""
    C, P0, E, t0, tp = map(_pos, (C, P0, E, t0, tp))
    s = np.asarray(t, dtype=float) - t0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if C > 0 and E > 0:
            root = np.sqrt(E / C)
            rate = np.sqrt(E * C)
            emit = root * np.tanh(rate * np.clip(s, 0.0, None))
            xmax = float(root * np.tanh(rate * tp))
        else:
            emit = np.zeros_like(s)
            xmax = 0.0
        dec = xmax / (1.0 + C * xmax * np.clip(s - tp, 0.0, None))
        x = np.where(s < 0, 0.0, np.where(s < tp, emit, dec))
    return _sanitize(P0 + x)


def _combined(t, K, C, P0, E, t0, tp):
    """Combined first + second order model ``dX/dt = E - (K*X + C*X**2)``."""
    K, C, P0, E, t0, tp = map(_pos, (K, C, P0, E, t0, tp))
    s = np.asarray(t, dtype=float) - t0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if C > 0 and E > 0:
            det = np.sqrt(K * K + 4.0 * C * E)
            r1 = (-K + det) / (2.0 * C)
            r2 = (-K - det) / (2.0 * C)

            def _emit(se):
                e = np.exp(-det * np.clip(se, 0.0, None))
                return r1 * (1.0 - e) / (1.0 - (r1 / r2) * e)

            emit = _emit(s)
            xmax = float(_emit(np.array([tp]))[0])
        elif K > 0:  # C -> 0 reduces to first order
            xss = E / K
            emit = xss * (1.0 - np.exp(-K * np.clip(s, 0.0, None)))
            xmax = float(xss * (1.0 - np.exp(-K * tp)))
        else:
            emit = np.zeros_like(s)
            xmax = 0.0
        dec = _combined_decay(np.clip(s - tp, 0.0, None), K, C, xmax)
        x = np.where(s < 0, 0.0, np.where(s < tp, emit, dec))
    return _sanitize(P0 + x)


def _combined_decay(sd, K, C, xmax):
    """Excess during a combined-loss decay from ``xmax`` (K first, C second)."""
    if xmax <= 0:
        return np.zeros_like(sd)
    if K > 0:
        growth = np.exp(np.clip(K * sd, 0.0, _EXP_CLAMP))
        return K * xmax / ((K + C * xmax) * growth - C * xmax)
    if C > 0:  # pure second order
        return xmax / (1.0 + C * xmax * sd)
    return np.full_like(sd, xmax)


# -- decay-only excess curves (post-peak), fitted in stage 1 ----------------
# ``(td, <loss params>, xmax)`` returning the excess above background ``td``
# seconds after the peak; the background P0 is measured, not fitted.


def _excess_zeroth(td, a, xmax):
    return np.clip(_pos(xmax) - _pos(a) * np.clip(td, 0.0, None), 0.0, None)


def _excess_first(td, k, xmax):
    return _pos(xmax) * np.exp(-_pos(k) * np.clip(td, 0.0, None))


def _excess_second(td, C, xmax):
    C, xmax = _pos(C), _pos(xmax)
    return xmax / (1.0 + C * xmax * np.clip(td, 0.0, None))


def _excess_combined(td, K, C, xmax):
    return _combined_decay(np.clip(td, 0.0, None), _pos(K), _pos(C), _pos(xmax))


#: model name -> excess decay kernel used in the stage-1 (post-peak) fit.
_DECAY_EXCESS = {
    "zeroth_order": _excess_zeroth,
    "first_order": _excess_first,
    "second_order": _excess_second,
    "combined": _excess_combined,
}


#: model name -> metadata. ``func`` draws the full emission+decay curve;
#: ``params`` are its ordered names; ``n_loss`` is how many leading parameters
#: are loss-rate constants (the rest are ``P0, E, t0, tp``).
_MODELS = {
    "zeroth_order": {
        "func": _zeroth_order,
        "params": ["a", "P0", "E", "t0", "tp"],
        "n_loss": 1,
    },
    "first_order": {
        "func": _first_order,
        "params": ["k", "P0", "E", "t0", "tp"],
        "n_loss": 1,
    },
    "second_order": {
        "func": _second_order,
        "params": ["C", "P0", "E", "t0", "tp"],
        "n_loss": 1,
    },
    "combined": {
        "func": _combined,
        "params": ["K", "C", "P0", "E", "t0", "tp"],
        "n_loss": 2,
    },
}

#: Accepted aliases mapping onto the canonical model names.
_MODEL_ALIASES = {
    "zeroth_order": "zeroth_order",
    "zeroth": "zeroth_order",
    "zero": "zeroth_order",
    "0": "zeroth_order",
    "0th": "zeroth_order",
    "constant": "zeroth_order",
    "first_order": "first_order",
    "first": "first_order",
    "1": "first_order",
    "1st": "first_order",
    "exp": "first_order",
    "exponential": "first_order",
    "second_order": "second_order",
    "second": "second_order",
    "2": "second_order",
    "2nd": "second_order",
    "coagulation": "second_order",
    "combined": "combined",
    "combi": "combined",
    "both": "combined",
    "third": "combined",
    "3": "combined",
}

#: Complexity penalties for auto-selection: a more complex/less-common model is
#: only chosen when it improves the (1 - R²) misfit by more than this factor.
_MODEL_PENALTY = {
    "zeroth_order": 1.10,
    "first_order": 1.0,
    "second_order": 1.10,
    "combined": 1.25,
}


def decay_curve(model: str, t, popt) -> np.ndarray:
    """Evaluate a fitted model over ``t`` (seconds from the window start).

    Args:
        model: Canonical model name (see :data:`_MODELS`).
        t: Times in seconds from the fit window's start.
        popt: Fitted parameters in the model's own order (``model_popt``).

    Returns:
        numpy.ndarray: Modelled concentration at each time.
    """
    return _MODELS[model]["func"](np.asarray(t, dtype=float), *popt)


def _r_squared(y: np.ndarray, fit: np.ndarray) -> float:
    """Coefficient of determination of ``fit`` against ``y``."""
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _emission_peak(model: str, E: float, loss: list, tp: float) -> float:
    """Excess reached at the end of the source-on phase for a trial ``E``."""
    popt = [*loss, 0.0, E, 0.0, tp]  # P0=0, t0=0 -> curve is the excess itself
    return float(_MODELS[model]["func"](np.array([tp]), *popt)[0])


def _invert_emission(model: str, loss: list, xmax: float, tp: float) -> float:
    """Back-solve the emission rate ``E`` from the anchored peak excess.

    The emission-phase peak excess is a monotone increasing function of ``E``
    (more source → higher peak), so it inverts cleanly. Closed forms are used
    where they exist; otherwise a bracketed root find.
    """
    if tp <= 0 or xmax <= 0:
        return 0.0
    if model == "zeroth_order":
        return xmax / tp + loss[0]
    if model == "first_order":
        k = loss[0]
        denom = 1.0 - np.exp(-k * tp)
        return k * xmax / denom if denom > 1e-12 else xmax / tp
    try:
        return float(
            brentq(
                lambda E: _emission_peak(model, E, loss, tp) - xmax,
                1e-15,
                1e18,
                maxiter=200,
            )
        )
    except (ValueError, RuntimeError):
        return xmax / tp


class DecayFitMixin:
    """Fit an emission + decay peak to estimate source strength and losses."""

    def fit_decay(
        self,
        period: Union[str, tuple],
        metric: str = "PNC",
        model: str = "auto",
        volume: Optional[float] = None,
        air_exchange_rate: Optional[float] = None,
        emission_start=None,
        peak_time=None,
    ) -> dict:
        """Description:
            Fit a single-zone emission + decay peak to one time window of a
            concentration metric, estimating the emission/source strength and
            the loss kinetics (zeroth, first, second order, or combined).

            The loss kinetics come from a robust fit of the **decay** (post-peak)
            part alone; the peak is anchored to the data and the emission rate is
            back-solved from it, so the fit does not undershoot the peak.

        Args:
            period (str | tuple): Either an activity name (a boolean column in
                :attr:`data`) or a ``(start, end)`` pair (anything
                :func:`pandas.Timestamp` accepts) selecting the window to fit.
                The window should span the background, the rise, the peak and
                the decay.
            metric (str): Metric to fit, as resolved by the class's metric
                lookup (for example "PNC", "MASS", "PM2.5"). Default "PNC".
            model (str): Which loss model to fit: ``"zeroth_order"`` (constant
                removal), ``"first_order"`` (air exchange + deposition),
                ``"second_order"`` (coagulation), ``"combined"`` (first +
                second), or ``"auto"`` (fit all and pick the best, favouring the
                simpler). Aliases such as "0"/"constant", "first"/"exp",
                "second"/"coagulation", "combined"/"both" are accepted.
            volume (float | None): Chamber/room volume in m³. When given, the
                fitted emission rate is reported as a source strength
                (``E * volume``) and a total emitted amount.
            air_exchange_rate (float | None): An independently known air
                exchange rate in 1/hour. When given and the model has a
                first-order loss term, the wall-loss/deposition rate is
                estimated as ``(first-order loss) - air_exchange_rate``.
            emission_start: Optional explicit start of the source-on phase
                (anything :func:`pandas.Timestamp` accepts, within the window).
                When ``None`` it is detected from the rise. Only affects the
                emission duration ``tp`` and hence the emission rate/source
                strength, not the loss kinetics.
            peak_time: Optional explicit peak time (source-off) splitting the
                emission from the decay. When ``None`` it is detected as the
                (smoothed) maximum.

        Returns:
            dict: With keys including "model", "unit", "metric", "r_squared"
            (whole window), "decay_r_squared" (post-peak fit), "n_points",
            "params", "errors", "background", "peak_concentration",
            "peak_excess", "emission_rate"/"emission_rate_unit",
            "loss_rate_per_hour"/"half_life_hours" (first-order term),
            "zeroth_order_rate" (zeroth order), "second_order_rate"
            (second/combined), "wall_loss_rate_per_hour" (with
            ``air_exchange_rate``), "source_strength"/"total_emitted" (with
            ``volume``), the timing ("emission_start_s", "emission_duration_s",
            "peak_time_s", "peak_time") and, for redrawing, "window_start" and
            "model_popt" (see :func:`decay_curve`).

        Raises:
            ValueError: If ``period`` is neither a known activity nor a valid
                ``(start, end)`` pair, if the window has too few finite
                samples, if ``model`` is unrecognised, or if no model could be
                fitted.

        Examples:
            Fit a chamber emission peak and read the source strength::

                res = data.fit_decay("Emission", metric="PNC", volume=20.0)
                print(res["source_strength"], res["source_strength_unit"])
        """
        key = str(model).strip().lower()
        if key != "auto" and key not in _MODEL_ALIASES:
            raise ValueError(
                f"Unknown model {model!r}. Use 'auto', 'zeroth_order', "
                "'first_order', 'second_order' or 'combined'."
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

        peak_idx, t0_idx = self._decay_split(
            t, values, times, emission_start, peak_time
        )

        wanted = list(_MODELS) if key == "auto" else [_MODEL_ALIASES[key]]
        fits = {}
        for name in wanted:
            outcome = self._fit_two_stage(name, t, values, peak_idx, t0_idx)
            if outcome is not None:
                fits[name] = outcome
        if not fits:
            raise ValueError(
                "No emission + decay model could be fitted to this window; try "
                "a wider window that includes the rise and decay of the peak."
            )

        chosen = min(
            fits,
            key=lambda n: (1.0 - fits[n]["decay_r2"]) * _MODEL_PENALTY[n],
        )
        return self._decay_result(
            chosen,
            fits[chosen],
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
    def _smooth(y: np.ndarray) -> np.ndarray:
        """Light 3-point moving average for robust peak/rise detection."""
        if y.size < 3:
            return y
        kern = np.ones(3) / 3.0
        return np.convolve(y, kern, mode="same")

    def _decay_split(self, t, y, times, emission_start, peak_time):
        """Return ``(peak_idx, t0_idx)`` splitting emission from decay.

        The peak (source-off) and emission start are detected from the smoothed
        rise unless the caller passes explicit timestamps.
        """
        sm = self._smooth(y)
        if peak_time is not None:
            peak_idx = int(np.argmin(np.abs(t - self._to_seconds(peak_time, times))))
        else:
            peak_idx = int(np.argmax(sm))
            # A genuine peak sits at the maximum, but a *saturated* emission
            # plateaus at steady state before the source turns off, so the max
            # is random within a flat top. Detect that (the near-max band extends
            # well before the argmax) and take the source-off at the plateau's
            # end instead.
            base = float(np.median(sm[: max(3, len(sm) // 10)]))
            noise = 1.4826 * np.median(np.abs(np.diff(y))) / np.sqrt(2.0)
            band = max(2.0 * noise, 0.01 * (float(sm.max()) - base))
            near = sm >= float(sm.max()) - band
            left = peak_idx
            while left > 0 and near[left - 1]:
                left -= 1
            if peak_idx - left >= 3:  # flat top -> use the end of the plateau
                right = peak_idx
                while right < len(sm) - 1 and near[right + 1]:
                    right += 1
                peak_idx = right
        peak_idx = min(max(peak_idx, 1), len(t) - 2)

        if emission_start is not None:
            t0_idx = int(np.argmin(np.abs(t - self._to_seconds(emission_start, times))))
            t0_idx = min(t0_idx, peak_idx)
        else:
            # Emission start = the foot of the rise: searching back from the peak,
            # the last sample still below 10 % of the peak excess. The background
            # is the median of the first quarter of the pre-peak segment (robust,
            # and free of the smoothing's zero-padded edge dip).
            pre = sm[: peak_idx + 1]
            base = float(np.median(y[: max(3, peak_idx // 4)]))
            thr = base + 0.10 * (float(sm[peak_idx]) - base)
            below = np.where(pre <= thr)[0]
            t0_idx = int(below[-1]) if below.size else 0
            t0_idx = max(min(t0_idx, peak_idx - 1), 0) if peak_idx > 0 else 0
        return peak_idx, t0_idx

    @staticmethod
    def _to_seconds(when, times) -> float:
        """Seconds from the window start for a timestamp/second offset."""
        if isinstance(when, (int, float)):
            return float(when)
        return float((pd.Timestamp(when) - times[0]).total_seconds())

    @staticmethod
    def _background(y: np.ndarray, t0_idx: int, peak_idx: int) -> float:
        """Estimate the background from the pre-emission baseline.

        Uses the median of the samples before the source turns on. When too few
        such samples exist (the window starts on the rise), falls back to a low
        percentile of the whole pre-peak segment. Held fixed during the fit.
        """
        pre = y[: t0_idx + 1]
        if pre.size >= 3:
            bg = float(np.median(pre))
        else:
            bg = float(np.percentile(y[: peak_idx + 1], 10))
        return max(bg, 0.0)

    def _fit_two_stage(self, name, t, y, peak_idx, t0_idx):
        """Fit one model in two stages; return an outcome dict or None."""
        info = _MODELS[name]
        t_peak = float(t[peak_idx])
        td = t[peak_idx:] - t_peak
        yd = y[peak_idx:]
        if td.size < 4:
            return None

        # Background and peak height are both *measured* and held fixed. The
        # decay starts at the peak, so its excess at t=0 is the observed peak
        # excess; fitting it as a free parameter is degenerate (it lets the
        # flexible combined model overshoot the true peak) and would reintroduce
        # the "misses the peak" problem. Only the loss constant(s) are fitted, on
        # the excess above background. Both P0 and the peak use a local 3-point
        # median so a single noisy sample cannot set them.
        P0 = self._background(y, t0_idx, peak_idx)
        peak_val = float(np.median(y[max(0, peak_idx - 1) : peak_idx + 2]))
        xmax = max(peak_val - P0, 1e-9)
        yd_ex = yd - P0

        rate = 1.0 / 1800.0
        good = yd_ex > 0
        if good.sum() >= 3:
            try:
                slope = np.polyfit(td[good], np.log(yd_ex[good]), 1)[0]
                if slope < 0:
                    rate = min(max(-slope, 1e-6), 1.0)
            except (ValueError, np.linalg.LinAlgError):
                pass

        loss0 = self._decay_loss_seed(name, rate, xmax)
        kernel = _DECAY_EXCESS[name]

        def excess(td_, *loss_params):
            return kernel(td_, *loss_params, xmax)

        lo = [1e-30] * info["n_loss"]
        hi = [np.inf] * info["n_loss"]
        try:
            popt_d, pcov_d = curve_fit(
                excess, td, yd_ex, p0=loss0, bounds=(lo, hi), maxfev=20000
            )
        except (RuntimeError, ValueError):
            return None
        loss = list(popt_d)
        decay_r2 = _r_squared(yd, excess(td, *popt_d) + P0)
        if not np.isfinite(decay_r2):
            return None

        # Stage 2: emission rate from the anchored peak and duration.
        tp = max(t_peak - float(t[t0_idx]), float(np.median(np.diff(t)) or 1.0))
        E = _invert_emission(name, loss, xmax, tp)

        popt = [*loss, P0, E, float(t[t0_idx]), tp]
        fit = info["func"](t, *popt)
        r2 = _r_squared(y, fit)

        with np.errstate(invalid="ignore"):
            perr_d = np.sqrt(np.diag(pcov_d))
        loss_err = list(np.nan_to_num(perr_d[: info["n_loss"]], nan=0.0))
        return {
            "popt": popt,
            "loss": loss,
            "loss_err": loss_err,
            "xmax": xmax,
            "P0": P0,
            "E": E,
            "t0": float(t[t0_idx]),
            "tp": tp,
            "r2": r2,
            "decay_r2": decay_r2,
        }

    @staticmethod
    def _decay_loss_seed(name, rate, xmax):
        """Initial loss-parameter guess(es) for the stage-1 decay fit."""
        if name == "zeroth_order":
            return [rate * xmax]  # constant rate ≈ k·Xmax
        if name == "first_order":
            return [rate]
        if name == "second_order":
            return [rate / max(xmax, 1e-9)]
        return [rate, rate / max(xmax, 1e-9)]  # combined

    def _decay_result(self, model, fit, unit, metric, t_start, n, volume, ach) -> dict:
        """Assemble the public result dict from a two-stage fit outcome."""
        info = _MODELS[model]
        names = info["params"]
        popt = fit["popt"]
        params = {nm: float(abs(v)) for nm, v in zip(names, popt)}
        errors = {nm: 0.0 for nm in names}
        for nm, v in zip(names[: info["n_loss"]], fit["loss_err"]):
            errors[nm] = float(v)

        P0 = fit["P0"]
        E = fit["E"]
        t0 = fit["t0"]
        tp = fit["tp"]
        peak_conc = P0 + fit["xmax"]

        result = {
            "model": model,
            "unit": unit,
            "metric": metric,
            "r_squared": fit["r2"],
            "decay_r_squared": fit["decay_r2"],
            "n_points": n,
            "params": params,
            "errors": errors,
            "background": P0,
            "peak_concentration": peak_conc,
            "peak_excess": fit["xmax"],
            "emission_rate": E,
            "emission_rate_unit": f"{unit}/s",
            "emission_start_s": t0,
            "emission_duration_s": tp,
            "peak_time_s": t0 + tp,
            "peak_time": t_start + pd.to_timedelta(t0 + tp, unit="s"),
            "window_start": t_start,
            "model_popt": [float(v) for v in popt],
        }

        # Loss-term reporting depends on the model order.
        if model == "zeroth_order":
            result["zeroth_order_rate"] = params["a"]
            result["zeroth_order_rate_unit"] = f"{unit}/s"
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
