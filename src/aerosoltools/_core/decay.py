"""Emission + decay peak fitting (source strength and loss kinetics).

Models a concentration peak in a well-mixed single-zone (box) chamber, where a
source is switched on, the concentration rises, the source is switched off, and
the concentration decays back towards the background. The models describe the
*excess* over background, ``X = P - P0``.

**The start and the background are two independent levels.** Taking the level a
window opens at for the background — as this module originally did — forces the
modelled decay to flatten out wherever the measurement happened to begin, which
biases the loss rate, the peak excess and every quantity derived from them. The
curve therefore starts at ``P0 + xi``, where the *signed* initial excess ``xi``
is fitted alongside the background ``P0``:

* ``xi > 0`` — the window opened **above** the background, e.g. while an earlier
  event was still decaying.
* ``xi = 0`` — the classic peak that starts and ends at the same level.
* ``xi < 0`` — the window opened **below** the level the decay settles at, e.g.
  because another activity or process raised the baseline during or after the
  emission, so the concentration never returns to where it started.

Neither level constrains the other; only the concentration itself has a floor,
which keeps ``P0 + xi >= 0``.

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
while the source is on, ``P0`` the background, ``xi`` the excess already present
when the source came on, and the source runs from ``t0`` for a duration ``tp``
(so the peak is at ``t0 + tp``).

**Two-stage fit.** Rather than fitting the whole rise+peak+decay at once — which
lets the many decay points outvote the few rise points and pulls the modelled
peak *below* the data — the fit is done in two stages:

1. *Decay stage.* The loss model is fitted to the **post-peak** points only,
   giving the loss kinetics, the peak excess ``Xmax`` and the background ``P0``.
   Because it uses only the monotone decay, it is robust and it anchors the peak
   to the data instead of averaging it away.
2. *Emission stage.* With the loss kinetics and background fixed, the initial
   excess ``xi`` is fitted over the pre-peak points and the emission rate ``E``
   back-solved from the anchored peak and the emission duration ``tp``.

Every measured or user-supplied level (background, peak, start, decay rate) is a
**seed** for these stages, never a hard constraint — the optimiser is free to
move it. ``optimize=False`` stops at the seed, which is what the GUI previews
while a guess is being dragged.

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

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

# Keys that only appear for some models / when a chamber volume is given, so the
# dict view (and the persisted/legacy shape) omits them when they are not set.
_DECAY_OPTIONAL = (
    "zeroth_order_rate",
    "zeroth_order_rate_unit",
    "loss_rate_per_hour",
    "half_life_hours",
    "wall_loss_rate_per_hour",
    "second_order_rate",
    "source_strength",
    "source_strength_unit",
    "total_emitted",
    "total_emitted_unit",
)


@dataclass
class DecayResult(Mapping):
    """A fitted emission+decay peak: parameters, diagnostics and reporting.

    Returned by :meth:`Aerosol1D.fit_decay`. It is a typed record — read fields
    as attributes (``result.r_squared``, ``result.source_strength``) — but it is
    **also a read-only mapping**, so existing ``result["r_squared"]`` /
    ``result.get(...)`` access keeps working unchanged. Model- or volume-specific
    fields (see :data:`_DECAY_OPTIONAL`) are ``None`` when not applicable and are
    then omitted from the mapping view / :meth:`to_dict`, matching the historical
    dict exactly.

    Reconstruct the modelled curve from :attr:`model` and :attr:`model_popt` via
    :func:`decay_curve`.

    Attributes:
        model: Loss model name (``"zeroth_order"``/``"first_order"``/
            ``"second_order"``/``"combined"``).
        unit: Concentration unit of the fitted series.
        metric: Metric that was fitted (e.g. ``"PNC"``).
        r_squared: R² of the full rise+peak+decay curve.
        decay_r_squared: R² of the post-peak decay stage alone.
        n_points: Number of samples in the fitted window.
        params: Fitted model parameters ``{name: value}``.
        errors: 1σ uncertainties for the fitted loss parameters ``{name: value}``.
        background: The fitted **asymptote** ``P0`` -- the true background the
            decay relaxes to. Not necessarily the level the window started at.
        start_concentration: Level the series sat at when the source came on.
            Equals ``background + initial_excess``. Independent of
            ``background``: above it when an earlier event was still decaying,
            below it when a later activity raised the baseline.
        initial_excess: Signed difference between the start level and
            ``background`` (``xi``). Positive when the window opened above the
            background, negative when the concentration never came back down to
            where it started, zero for a peak that starts at background.
        peak_concentration: Modelled peak concentration (``background`` + excess).
        peak_excess: Peak excess over background (``Xmax``).
        emission_rate: Volumetric emission rate ``E`` (concentration/s).
        decay_rate / decay_rate_per_hour: First-order-equivalent loss rate.
        half_life_hours: Hours for the peak excess to halve. Defined for every
            loss model, not just the exponential one.
        emission_start_s / emission_duration_s / peak_time_s: Timing in seconds
            from the window start.
        peak_time / window_start: Absolute timestamps.
        model_popt: Raw optimised parameters for :func:`decay_curve`.
        source_strength / total_emitted: Present when a chamber ``volume`` was
            given; ``wall_loss_rate_per_hour`` present when an air-exchange rate
            was given.
    """

    model: str
    unit: str
    metric: str
    r_squared: float
    decay_r_squared: float
    n_points: int
    params: dict
    errors: dict
    background: float
    start_concentration: float
    initial_excess: float
    peak_concentration: float
    peak_excess: float
    emission_rate: float
    emission_rate_unit: str
    decay_rate: float
    decay_rate_per_hour: float
    emission_start_s: float
    emission_duration_s: float
    peak_time_s: float
    peak_time: pd.Timestamp
    window_start: pd.Timestamp
    model_popt: list
    # Optional (model-order / volume / air-exchange dependent).
    zeroth_order_rate: Optional[float] = None
    zeroth_order_rate_unit: Optional[str] = None
    loss_rate_per_hour: Optional[float] = None
    half_life_hours: Optional[float] = None
    wall_loss_rate_per_hour: Optional[float] = None
    second_order_rate: Optional[float] = None
    source_strength: Optional[float] = None
    source_strength_unit: Optional[str] = None
    total_emitted: Optional[float] = None
    total_emitted_unit: Optional[str] = None

    def _present(self) -> dict:
        """The fields that are set — optionals appear only when not ``None``."""
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _DECAY_OPTIONAL and value is None:
                continue
            out[f.name] = value
        return out

    # -- read-only Mapping interface (keeps dict-style access working) --------
    def __getitem__(self, key: str) -> Any:
        present = self._present()
        if key not in present:
            raise KeyError(key)
        return present[key]

    def __iter__(self):
        return iter(self._present())

    def __len__(self) -> int:
        return len(self._present())

    def to_dict(self) -> dict:
        """Plain dict of the set fields (the historical result shape)."""
        return dict(self._present())

    @classmethod
    def from_dict(cls, data: Mapping) -> "DecayResult":
        """Build a result from a mapping, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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


def _signed(x) -> float:
    """Pass a scalar parameter through with its sign intact.

    Used for the initial excess ``xi`` alone. Every other model parameter is a
    magnitude, but ``xi`` is a *difference* between two independent levels — the
    concentration when the source came on and the one the decay settles at — and
    either can be the larger. A peak whose baseline is raised by some later
    activity never returns to its pre-emission level, and is described by
    ``xi < 0``.
    """
    return float(x)


# -- full emission + decay curves (excess above background, back to background)
# Each returns the concentration over time ``t`` (seconds from the window start)
# for a source on from ``t0`` for a duration ``tp``. Signatures are
# ``(t, <loss params>, P0, E, t0, tp)`` so a fitted parameter vector round-trips
# through :func:`decay_curve` for plotting.


def _zeroth_order(t, a, P0, E, t0, tp, xi=0.0):
    """Zeroth-order model ``dX/dt = E - a``: linear rise and linear decay."""
    a, P0, E, t0, tp = map(_pos, (a, P0, E, t0, tp))
    xi = _signed(xi)
    s = np.asarray(t, dtype=float) - t0
    rise = xi + (E - a) * np.clip(s, 0.0, tp)
    xmax = xi + (E - a) * tp
    # A linear decay reaches the background in finite time and stops there, so
    # the *decay* excess is floored at zero. The pre-emission level is not: it
    # may legitimately sit below the level the decay settles at.
    dec = np.clip(xmax - a * np.clip(s - tp, 0.0, None), 0.0, None)
    x = np.where(s < 0, xi, np.where(s < tp, rise, dec))
    return _sanitize(np.clip(P0 + x, 0.0, None))


def _first_order(t, k, P0, E, t0, tp, xi=0.0):
    """First-order model ``dX/dt = E - k*X``: exponential rise and decay."""
    k, P0, E, t0, tp = map(_pos, (k, P0, E, t0, tp))
    xi = _signed(xi)
    s = np.asarray(t, dtype=float) - t0
    with np.errstate(over="ignore", invalid="ignore"):
        xss = E / k if k > 0 else 0.0
        # X(s) relaxes from the initial excess xi towards the steady state xss.
        # xi < 0 (a start below the background) needs no special case: the same
        # exponential simply relaxes upward instead of downward.
        relax = np.exp(-k * np.clip(s, 0.0, None))
        emit = xss * (1.0 - relax) + xi * relax
        relax_tp = np.exp(-k * tp)
        xmax = xss * (1.0 - relax_tp) + xi * relax_tp
        dec = xmax * np.exp(-k * np.clip(s - tp, 0.0, None))
        x = np.where(s < 0, xi, np.where(s < tp, emit, dec))
    return _sanitize(np.clip(P0 + x, 0.0, None))


def _second_order(t, C, P0, E, t0, tp, xi=0.0):
    """Second-order (coagulation) model ``dX/dt = E - C*X**2``."""
    C, P0, E, t0, tp = map(_pos, (C, P0, E, t0, tp))
    xi = _signed(xi)
    s = np.asarray(t, dtype=float) - t0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if C > 0 and E > 0:
            root = np.sqrt(E / C)  # steady-state excess
            rate = np.sqrt(E * C)
            # X(0) = xi shifts the rising tanh by artanh(xi / root); a negative
            # shift starts it below the background. artanh diverges at +-1, so
            # clamp just inside: |xi| at or beyond the steady state cannot be
            # reached by a rising tanh.
            ratio = np.clip(xi / root, -1.0 + 1e-12, 1.0 - 1e-12) if root > 0 else 0.0
            shift = float(np.arctanh(ratio))
            emit = root * np.tanh(rate * np.clip(s, 0.0, None) + shift)
            xmax = float(root * np.tanh(rate * tp + shift))
        else:
            emit = np.full_like(s, xi)
            xmax = xi
        if xmax > 0:
            dec = xmax / (1.0 + C * xmax * np.clip(s - tp, 0.0, None))
        else:
            # dX/dt = -C X**2 drives a non-positive excess away from zero rather
            # than towards it, so there is no decay to draw; hold it instead of
            # letting the hyperbola run through its pole.
            dec = np.full_like(s, xmax)
        x = np.where(s < 0, xi, np.where(s < tp, emit, dec))
    return _sanitize(np.clip(P0 + x, 0.0, None))


def _combined(t, K, C, P0, E, t0, tp, xi=0.0):
    """Combined first + second order model ``dX/dt = E - (K*X + C*X**2)``."""
    K, C, P0, E, t0, tp = map(_pos, (K, C, P0, E, t0, tp))
    xi = _signed(xi)
    s = np.asarray(t, dtype=float) - t0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if C > 0 and E > 0:
            det = np.sqrt(K * K + 4.0 * C * E)
            r1 = (-K + det) / (2.0 * C)  # positive root = steady-state excess
            r2 = (-K - det) / (2.0 * C)  # negative root
            # Riccati solution through X(0) = xi:
            #   (X - r1) / (X - r2) = A e^{-det s},  A = (xi - r1) / (xi - r2)
            # A = r1 / r2 recovers the xi = 0 form. Starting below r2 puts the
            # solution on the runaway branch (it diverges through the pole
            # rather than rising to r1), so hold the start just inside it.
            xi_eff = max(xi, r2 * (1.0 - 1e-9) if r2 < 0 else xi)
            denom0 = xi_eff - r2
            a_coef = (xi_eff - r1) / denom0 if denom0 != 0 else 0.0

            def _emit(se):
                e = a_coef * np.exp(-det * np.clip(se, 0.0, None))
                return (r1 - r2 * e) / (1.0 - e)

            emit = _emit(s)
            xmax = float(_emit(np.array([tp]))[0])
        elif K > 0:  # C -> 0 reduces to first order
            xss = E / K
            relax = np.exp(-K * np.clip(s, 0.0, None))
            emit = xss * (1.0 - relax) + xi * relax
            relax_tp = np.exp(-K * tp)
            xmax = float(xss * (1.0 - relax_tp) + xi * relax_tp)
        else:
            emit = np.full_like(s, xi)
            xmax = xi
        dec = _combined_decay(np.clip(s - tp, 0.0, None), K, C, xmax)
        x = np.where(s < 0, xi, np.where(s < tp, emit, dec))
    return _sanitize(np.clip(P0 + x, 0.0, None))


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
#: are loss-rate constants (the rest are ``P0, E, t0, tp, xi``).
_MODELS = {
    "zeroth_order": {
        "func": _zeroth_order,
        "params": ["a", "P0", "E", "t0", "tp", "xi"],
        "n_loss": 1,
    },
    "first_order": {
        "func": _first_order,
        "params": ["k", "P0", "E", "t0", "tp", "xi"],
        "n_loss": 1,
    },
    "second_order": {
        "func": _second_order,
        "params": ["C", "P0", "E", "t0", "tp", "xi"],
        "n_loss": 1,
    },
    "combined": {
        "func": _combined,
        "params": ["K", "C", "P0", "E", "t0", "tp", "xi"],
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

#: Complexity/plausibility penalties for auto-selection: a model is only chosen
#: over first order when it improves the (1 - R²) misfit by more than this
#: factor. First order -- air exchange plus deposition -- is the workhorse
#: indoor-aerosol loss path and so carries no penalty. Zeroth order (a constant
#: removal rate, independent of how much is in the air) is rarely the real
#: mechanism, so it has to fit clearly better before it wins, not merely tie.
_MODEL_PENALTY = {
    "zeroth_order": 1.40,
    "first_order": 1.0,
    "second_order": 1.10,
    "combined": 1.25,
}

#: How far above the observed peak excess a *freed* peak may go. The true peak
#: can sit above the highest sample when the averaging interval straddles it,
#: but it must not run away -- the second-order and combined models would
#: otherwise raise the peak clear of the data to buy a better decay fit.
_PEAK_HEADROOM = 1.25

#: Freedom ladder for one model's fit. Each rung adds a degree of freedom:
#:
#: * ``"anchored"`` -- the classic peak: the level the window opens at *is* the
#:   background, and the peak sits where the data put it. Only the loss
#:   constant(s) are fitted. This is what the module did before.
#: * ``"split"``    -- background and start level fitted separately, so a window
#:   that opens on the tail of an earlier event is not mistaken for background.
#: * ``"free_peak"`` -- as ``"split"``, and the peak excess is fitted too.
#:
#: All three are fitted and the penalised whole-window misfit picks one:
#: ``(1 - R²) * penalty``, the same idiom :data:`_MODEL_PENALTY` uses across
#: models.
#:
#: ``"split"`` is the *unpenalised* rung -- separating the two levels is the
#: behaviour this module is supposed to have, so it wins near-ties. Measured
#: against synthetic peaks with a known background, forcing start == background
#: costs as little as 1.0-1.4x in misfit for a moderately elevated start while
#: getting the loss rate 24-50 % wrong, so the raw misfit must not be allowed to
#: choose it. ``"anchored"`` therefore carries a small penalty of its own, and
#: ``"free_peak"`` a large one: freeing the peak as well is genuinely degenerate
#: with the loss constant for slowly-decaying (second-order, combined) tails,
#: which will raise the peak clear of the data to buy a better decay fit.
_LEVEL_PENALTY = {
    "anchored": 1.05,
    "split": 1.0,
    "free_peak": 1.6,
}

#: A fitted background at or below this fraction of the lowest observed decay
#: sample is treated as *unidentified* rather than measured: it means the
#: optimiser slid the asymptote onto the bottom of its allowed range, which a
#: slow (hyperbolic) tail lets it do almost for free while the initial excess
#: absorbs the difference. Such a candidate is discarded whenever another one
#: survives.
_COLLAPSED_BACKGROUND_FRACTION = 0.02


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


def _emission_peak(
    model: str, E: float, loss: list, tp: float, xi: float = 0.0
) -> float:
    """Excess reached at the end of the source-on phase for a trial ``E``."""
    # P0 = 0, t0 = 0 -> the curve *is* the excess above background.
    popt = [*loss, 0.0, E, 0.0, tp, xi]
    return float(_MODELS[model]["func"](np.array([tp]), *popt)[0])


def _invert_emission(
    model: str, loss: list, xmax: float, tp: float, xi: float = 0.0
) -> float:
    """Back-solve the emission rate ``E`` from the anchored peak excess.

    The emission-phase peak excess is a monotone increasing function of ``E``
    (more source → higher peak), so it inverts cleanly. Closed forms are used
    where they exist; otherwise a bracketed root find.

    Args:
        model: Canonical model name.
        loss: The model's fitted loss constant(s).
        xmax: Peak excess above the background asymptote.
        tp: Emission duration (s).
        xi: Excess already present when the source turned on. Positive when the
            window opened above the background (the source only has to supply
            the difference), negative when it opened below and the source has
            further to climb.
    """
    if tp <= 0 or xmax <= 0:
        return 0.0
    xi = float(xi)
    if xmax <= xi:
        # The peak is no higher than the level the window started at, so the
        # rise carries no information about a source.
        return 0.0
    if model == "zeroth_order":
        return (xmax - xi) / tp + loss[0]
    if model == "first_order":
        k = loss[0]
        relax = np.exp(-k * tp)
        denom = 1.0 - relax
        return k * (xmax - xi * relax) / denom if denom > 1e-12 else (xmax - xi) / tp
    try:
        return float(
            brentq(
                lambda E: _emission_peak(model, E, loss, tp, xi) - xmax,
                1e-15,
                1e18,
                maxiter=200,
            )
        )
    except (ValueError, RuntimeError):
        return (xmax - xi) / tp


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
        background: Optional[float] = None,
        start_concentration: Optional[float] = None,
        peak_concentration: Optional[float] = None,
        decay_rate: Optional[float] = None,
        optimize: bool = True,
        snap_to_samples: bool = True,
        series: Optional[pd.Series] = None,
        unit: Optional[str] = None,
    ) -> dict:
        """Fit a single-zone emission + decay peak to one time window of a
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
            background (float | None): **Initial guess** for the background
                ``P0`` -- the asymptote the decay relaxes to. The fit is free to
                move it. When ``None`` the seed comes from the data and is
                bounded above by the lowest post-peak sample, which a decay
                approaching its asymptote from above cannot pass below; an
                explicit value is taken as given and widens that bound instead
                of being clipped to it. Pass ``optimize=False`` to stop at the
                guess, which then round-trips exactly.
            start_concentration (float | None): **Initial guess** for the level
                the series sat at when the source came on. Independent of
                ``background`` and free to sit on either side of it: above when
                an earlier event was still decaying, below when a later activity
                raises the baseline so the concentration never returns to its
                pre-emission level. The signed difference is fitted as the
                initial excess. When ``None`` the seed is the pre-emission
                baseline median.
            peak_concentration (float | None): **Initial guess** for the peak
                concentration (so the peak excess seed is
                ``peak_concentration - background``). The fit may raise the peak
                by up to 50 % above the guess -- the real peak can sit above the
                highest sample when the averaging interval straddles it -- but
                no further, so a flexible model cannot run away from the data.
            decay_rate (float | None): Seed for the loss kinetics, given as a
                first-order-equivalent rate in **1/s** (the ``decay_rate`` key a
                previous fit returns). With ``optimize=True`` it just seeds the
                optimiser; with ``optimize=False`` it *is* the loss rate, so the
                returned curve is a pure manual guess. When ``None`` the rate is
                estimated from the decay slope.
            optimize (bool): When True (default) the loss kinetics are optimised
                against the post-peak data. When False the fit is skipped and the
                model is evaluated directly from the seed/override values, so the
                caller gets a previewable "initial guess" (used by the GUI to
                show a live preview before optimising). The returned dict has the
                same shape either way; ``errors`` are zero for a guess.
            snap_to_samples (bool): When True (default) an explicit
                ``emission_start`` / ``peak_time`` is quantised to the nearest
                sample time, so the reported timing lands exactly on a data
                point. When False the supplied timestamps are used as continuous
                times (an onset or source-off may fall *between* two samples), so
                the emission duration ``tp`` and the drawn markers vary smoothly
                rather than jumping from sample to sample. Only affects the two
                timing overrides; detection (when they are ``None``) is unchanged.
            series (pandas.Series | None): Optional precomputed, time-indexed
                concentration series to fit instead of the named ``metric`` —
                used to fit an arbitrary series such as a single size bin (for
                per-bin decay fitting) with the same window and overrides. When
                given, ``metric`` is used only for labelling.
            unit (str | None): Unit string for ``series``; when ``series`` is
                given but ``unit`` is ``None`` the named metric's unit is used.

        Returns:
            DecayResult: A typed record (also a read-only mapping, so
            ``result["r_squared"]`` still works) with fields including ``model``,
            ``unit``, ``metric``, ``r_squared`` (whole window), ``decay_r_squared``
            (post-peak fit), ``n_points``, ``params``, ``errors``, ``background``
            (the fitted asymptote), ``start_concentration`` / ``initial_excess``
            (the possibly-elevated level the window opened at),
            ``peak_concentration``, ``peak_excess``, ``emission_rate`` /
            ``emission_rate_unit``, ``decay_rate`` / ``decay_rate_per_hour``
            (first-order-equivalent loss rate, for round-tripping into
            ``decay_rate``), ``loss_rate_per_hour`` / ``half_life_hours``
            (first-order term), ``zeroth_order_rate``, ``second_order_rate``
            (second/combined), ``half_life_hours`` (time for the peak excess to
            halve, for any model), ``wall_loss_rate_per_hour`` (with
            ``air_exchange_rate``), ``source_strength`` / ``total_emitted`` (with
            ``volume``), the timing (``emission_start_s``, ``emission_duration_s``,
            ``peak_time_s``, ``peak_time``) and, for redrawing, ``window_start``
            and ``model_popt`` (see :func:`decay_curve`). Optional model-/volume-
            specific fields are ``None`` (and omitted from the mapping view) when
            not applicable.

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

        # A caller may supply its own precomputed series (e.g. a single size
        # bin, for per-bin decay fitting) to bypass the metric lookup; otherwise
        # resolve the named metric as usual.
        if series is not None:
            if unit is None:
                _s, unit = self._get_metric_series(metric)
        else:
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

        # Continuous (non-snapped) override times: when the caller supplies an
        # explicit onset/source-off and asks not to snap, carry the exact seconds
        # so the emission duration and markers can fall between samples.
        t0_cont = peak_cont = None
        if not snap_to_samples:
            lo, hi = float(t[0]), float(t[-1])
            if emission_start is not None:
                t0_cont = min(max(self._to_seconds(emission_start, times), lo), hi)
            if peak_time is not None:
                peak_cont = min(max(self._to_seconds(peak_time, times), lo), hi)

        wanted = list(_MODELS) if key == "auto" else [_MODEL_ALIASES[key]]
        fits = {}
        for name in wanted:
            outcome = self._fit_two_stage(
                name,
                t,
                values,
                peak_idx,
                t0_idx,
                background=background,
                start_concentration=start_concentration,
                peak_concentration=peak_concentration,
                decay_rate=decay_rate,
                optimize=optimize,
                t0_cont=t0_cont,
                peak_cont=peak_cont,
            )
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
    def _start_level(y: np.ndarray, t0_idx: int, peak_idx: int) -> float:
        """Estimate the pre-emission *start* level from the baseline.

        Uses the median of the samples before the source turns on. When too few
        such samples exist (the window starts on the rise), falls back to a low
        percentile of the whole pre-peak segment.

        This is the level the series sits at when the source is switched on. It
        is **not** the background: an earlier event may still be decaying,
        leaving the start above it, or a later activity may raise the baseline,
        leaving the start below it. The asymptote the decay actually relaxes to
        is fitted separately -- see :meth:`_fit_two_stage`.
        """
        pre = y[: t0_idx + 1]
        if pre.size >= 3:
            level = float(np.median(pre))
        else:
            level = float(np.percentile(y[: peak_idx + 1], 10))
        return max(level, 0.0)

    @staticmethod
    def _seed_within(value: float, lo: float, hi: float) -> float:
        """Clip a seed into ``[lo, hi]``, nudged off the bounds when possible."""
        if not np.isfinite(value):
            value = 0.5 * (lo + hi)
        span = hi - lo
        if span <= 0:
            return lo
        return float(np.clip(value, lo + 1e-6 * span, hi - 1e-6 * span))

    def _fit_two_stage(
        self,
        name,
        t,
        y,
        peak_idx,
        t0_idx,
        *,
        background=None,
        peak_concentration=None,
        start_concentration=None,
        decay_rate=None,
        optimize=True,
        t0_cont=None,
        peak_cont=None,
    ):
        """Fit one model in two stages; return an outcome dict or None.

        The peak carries **two** distinct levels, fitted separately:

        * ``P0`` -- the asymptote the decay relaxes to, i.e. the *true*
          background. Fitted in stage 1 against the post-peak samples.
        * ``P_start`` -- the level the series sat at when the source came on.
          Independent of ``P0``: above it when an earlier event was still
          decaying, below it when a later activity raised the baseline so the
          concentration never returns to its pre-emission level. Carried as the
          signed initial excess ``xi = P_start - P0`` and fitted in stage 2.

        Treating the start as the background (the previous behaviour) forced the
        decay to flatten out at whatever the window happened to open at, biasing
        the loss rate and the peak excess whenever the two differ.

        ``background`` / ``peak_concentration`` / ``start_concentration`` /
        ``decay_rate`` are **initial guesses**, not constraints: they seed the
        optimiser, which is then free to move them. Pass ``optimize=False`` to
        stop at the seed (the GUI's live preview of a manual guess).
        ``t0_cont`` / ``peak_cont`` are continuous (non-snapped) onset/source-off
        seconds: when given they set the emission timing exactly (so ``tp`` and
        the markers can sit between samples) while the post-peak sample
        selection still uses the nearest-sample ``peak_idx``.
        """
        info = _MODELS[name]
        n_loss = info["n_loss"]
        # The decay samples (post-peak) are still chosen by sample index, but the
        # peak *time* used for the arithmetic follows the continuous override when
        # one is supplied, so the fitted curve and markers are not quantised.
        t_peak = float(t[peak_idx]) if peak_cont is None else float(peak_cont)
        td = t[peak_idx:] - t_peak
        yd = y[peak_idx:]
        if td.size < 4:
            return None

        # -- seeds ---------------------------------------------------------
        if start_concentration is not None:
            start_seed = max(float(start_concentration), 0.0)
        else:
            start_seed = self._start_level(y, t0_idx, peak_idx)
        if peak_concentration is not None:
            peak_seed = float(peak_concentration)
        else:
            # A local 3-point median, so one noisy sample cannot set the peak.
            peak_seed = float(np.median(y[max(0, peak_idx - 1) : peak_idx + 2]))

        # A decay approaches its asymptote from above, so a *measured* background
        # can never sit above the lowest post-peak sample -- a tight, physical
        # bound that keeps the extra free parameter well conditioned.
        #
        # An explicit background from the caller overrides it. They may know the
        # level from a separate background measurement, and noise can push a
        # single sample under the true asymptote, so silently clipping their
        # value to the window's minimum would discard information rather than add
        # any. The bound is widened to admit it instead.
        end_hi = max(float(np.nanmin(yd)), 0.0)
        if background is not None:
            end_seed = max(float(background), 0.0)
        else:
            # Seeded from the decay tail alone. Taking the start level as an
            # upper bound here (the old ``min(start_seed, ...)``) quietly
            # reimposed "the background is no higher than where the window
            # opened", so lowering the start dragged the background down with
            # it instead of producing a negative initial excess.
            end_seed = self._seed_within(0.95 * end_hi, 0.0, end_hi)
        end_bound = max(end_hi, end_seed)

        xmax_seed = max(peak_seed - end_seed, 1e-9)
        # Cap a free peak against the *observed* maximum, not against the seed,
        # so a badly dragged guess cannot unbound the fit.
        xmax_hi = _PEAK_HEADROOM * max(float(np.nanmax(y)) - end_seed, xmax_seed, 1e-9)

        if decay_rate is not None:
            rate = max(float(decay_rate), 1e-12)
        else:
            rate = 1.0 / 1800.0
            good = (yd - end_seed) > 0
            if good.sum() >= 3:
                try:
                    slope = np.polyfit(td[good], np.log(yd[good] - end_seed), 1)[0]
                    if slope < 0:
                        rate = min(max(-slope, 1e-6), 1.0)
                except (ValueError, np.linalg.LinAlgError):
                    pass

        loss0 = self._decay_loss_seed(name, rate, xmax_seed)
        kernel = _DECAY_EXCESS[name]
        t0_val = float(t[t0_idx]) if t0_cont is None else float(min(t0_cont, t_peak))
        tp = max(t_peak - t0_val, float(np.median(np.diff(t)) or 1.0))
        t_rise, y_rise = t[: peak_idx + 1], y[: peak_idx + 1]

        def attempt(variant: str, optimise: bool):
            """Fit this model at one rung of the :data:`_LEVEL_PENALTY` ladder.

            ``"anchored"`` keeps the historical behaviour -- the level the window
            opens at is the background and the peak sits where the data put it,
            so only the loss constant(s) are fitted. ``"split"`` frees the
            background (bounded above by the lowest post-peak sample, which a
            decay approaching from above can never go under) and, in stage 2, the
            initial excess. ``"free_peak"`` additionally fits the peak excess.

            Each rung is genuinely useful and each is degenerate somewhere, so
            the caller runs all three and lets the penalised whole-window misfit
            decide -- the information stage 1 does not have on its own.
            """
            anchored = variant == "anchored"
            free_peak = variant == "free_peak"

            # -- stage 1: post-peak decay -> loss kinetics, background, peak --
            if anchored:
                # Background pinned: to the caller's value when one was given
                # (so a typed guess still means something on this rung), else to
                # the level the window opened at -- the classic assumption.
                p_fixed = (
                    end_seed if background is not None else min(start_seed, end_hi)
                )

                def decay_model(td_, *free):
                    return p_fixed + kernel(
                        td_, *free[:n_loss], max(peak_seed - p_fixed, 1e-9)
                    )

                p0 = list(loss0)
                lo = [1e-30] * n_loss
                hi = [np.inf] * n_loss
            elif free_peak:

                def decay_model(td_, *free):
                    return free[n_loss + 1] + kernel(td_, *free[:n_loss], free[n_loss])

                p0 = [*loss0, xmax_seed, end_seed]
                lo = [1e-30] * n_loss + [0.0, 0.0]
                hi = [np.inf] * n_loss + [xmax_hi, max(end_bound, 1e-12)]
            else:  # "split"

                def decay_model(td_, *free):
                    p_end = free[n_loss]
                    return p_end + kernel(
                        td_, *free[:n_loss], max(peak_seed - p_end, 1e-9)
                    )

                p0 = [*loss0, end_seed]
                lo = [1e-30] * n_loss + [0.0]
                hi = [np.inf] * n_loss + [max(end_bound, 1e-12)]

            if optimise:
                # curve_fit needs a starting point strictly inside its bounds.
                # Manual mode skips this, so a value the caller typed or dragged
                # round-trips exactly instead of coming back nudged.
                p0 = [
                    self._seed_within(v, a, b) if np.isfinite(b) else max(v, a)
                    for v, a, b in zip(p0, lo, hi)
                ]
                try:
                    popt_d, pcov_d = curve_fit(
                        decay_model, td, yd, p0=p0, bounds=(lo, hi), maxfev=20000
                    )
                except (RuntimeError, ValueError):
                    return None
                with np.errstate(invalid="ignore"):
                    perr_d = np.nan_to_num(np.sqrt(np.diag(pcov_d)), nan=0.0)
            else:
                # No optimisation: the seeds *are* the answer. This drives the
                # GUI's live preview of a guess before the user optimises.
                popt_d = np.asarray(p0, dtype=float)
                perr_d = np.zeros_like(popt_d)

            loss = list(popt_d[:n_loss])
            loss_err = list(perr_d[:n_loss])
            if anchored:
                P0 = p_fixed
                xmax = max(peak_seed - P0, 1e-9)
                p0_err = xmax_err = 0.0
            elif free_peak:
                xmax = max(float(popt_d[n_loss]), 1e-9)
                P0 = float(popt_d[n_loss + 1])
                xmax_err, p0_err = float(perr_d[n_loss]), float(perr_d[n_loss + 1])
            else:  # "split"
                P0 = float(popt_d[n_loss])
                xmax = max(peak_seed - P0, 1e-9)
                p0_err = float(perr_d[n_loss])
                xmax_err = p0_err  # the peak tracks the background one-for-one

            decay_r2 = _r_squared(yd, P0 + kernel(td, *loss, xmax))
            if not np.isfinite(decay_r2):
                return None

            # -- stage 2: the rise -> the excess already there, and from it E --
            # The two levels are independent: the window may open *above* the
            # background (an earlier event still decaying, xi > 0) or *below* it
            # (a later activity raises the baseline, so the concentration never
            # returns to where it started, xi < 0). Only the concentration
            # itself has a floor, which puts the start at P0 + xi >= 0.
            #
            # Both ends of the range also admit whatever the caller asked for,
            # so a level typed or dragged past them is not snapped back from
            # under the cursor.
            xi_seed = 0.0 if anchored else start_seed - P0
            xi_lo = min(-P0, xi_seed)
            xi_hi = max(xmax, xi_seed, 1e-12)
            xi, xi_err = xi_seed, 0.0

            def rise_model(t_, xi_):
                # E follows from the anchored peak, so the rise has exactly one
                # free parameter: the excess in the air when the source came on.
                e = _invert_emission(name, loss, xmax, tp, xi_)
                return info["func"](t_, *loss, P0, e, t0_val, tp, xi_)

            if not anchored and optimise and t_rise.size >= 3 and xmax > 1e-9:
                try:
                    popt_r, pcov_r = curve_fit(
                        rise_model,
                        t_rise,
                        y_rise,
                        p0=[self._seed_within(xi_seed, xi_lo, xi_hi)],
                        bounds=([xi_lo], [xi_hi]),
                        maxfev=5000,
                    )
                    xi = float(popt_r[0])
                    with np.errstate(invalid="ignore"):
                        xi_err = float(
                            np.nan_to_num(np.sqrt(np.diag(pcov_r))[0], nan=0.0)
                        )
                except (RuntimeError, ValueError):
                    xi = xi_seed

            E = _invert_emission(name, loss, xmax, tp, xi)
            popt = [*loss, P0, E, t0_val, tp, xi]
            r2 = _r_squared(y, info["func"](t, *popt))

            return {
                "variant": variant,
                "popt": popt,
                "loss": loss,
                "loss_err": loss_err,
                "xmax": xmax,
                "xmax_err": xmax_err,
                "P0": P0,
                "P0_err": p0_err,
                "xi": xi,
                "xi_err": xi_err,
                "P_start": P0 + xi,
                "E": E,
                "t0": t0_val,
                "tp": tp,
                "r2": r2,
                "decay_r2": decay_r2,
            }

        if not optimize:
            # A pure preview of the guess: use the rung that honours every
            # supplied level as given.
            return attempt("split", optimise=False)

        candidates = [
            c
            for c in (attempt(v, optimise=True) for v in _LEVEL_PENALTY)
            if c is not None and np.isfinite(c["r2"])
        ]
        if not candidates:
            return None

        # Drop candidates whose background collapsed onto the bottom of its
        # range -- an unidentified asymptote, not a measured one. Keep them only
        # if nothing else fitted at all.
        floor = _COLLAPSED_BACKGROUND_FRACTION * end_hi
        credible = [c for c in candidates if not (end_hi > 0 and c["P0"] <= floor)]
        if credible:
            candidates = credible

        return min(
            candidates,
            key=lambda c: (1.0 - c["r2"]) * _LEVEL_PENALTY[c["variant"]],
        )

    @staticmethod
    def _effective_rate(name, loss, xmax) -> float:
        """First-order-equivalent loss rate (1/s) — inverse of the loss seed.

        Collapses each model's loss constant(s) back to the single ``rate``
        knob :meth:`_decay_loss_seed` expands, so a fit round-trips through the
        ``decay_rate`` argument (the GUI's decay-rate control) regardless of
        model.
        """
        xmax = max(float(xmax), 1e-30)
        if name == "zeroth_order":
            return float(loss[0]) / xmax  # a = rate · Xmax
        if name == "second_order":
            return float(loss[0]) * xmax  # C = rate / Xmax
        return float(loss[0])  # first_order: k; combined: K

    @staticmethod
    def _half_life_hours(name, loss, xmax) -> float:
        """Hours for the peak excess to fall to half, solved per model.

        Every loss model has a well-defined time to halve; only the
        first-order one has it independent of concentration. Reporting it for
        all four keeps the field usable whichever model is selected, instead of
        being ``None`` the moment the fit picks a non-exponential decay.

        * zeroth order  ``X = Xmax - a t``          -> ``t = Xmax / (2a)``
        * first order   ``X = Xmax e^{-kt}``        -> ``t = ln2 / k``
        * second order  ``X = Xmax/(1 + C Xmax t)`` -> ``t = 1 / (C Xmax)``
        * combined      solves ``K X + C X**2`` for the same half-point:
          ``t = ln((2K + C Xmax) / (K + C Xmax)) / K``
        """
        xmax = max(float(xmax), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            if name == "zeroth_order":
                a = float(loss[0])
                seconds = xmax / (2.0 * a) if a > 0 else float("nan")
            elif name == "first_order":
                k = float(loss[0])
                seconds = np.log(2.0) / k if k > 0 else float("nan")
            elif name == "second_order":
                c_x = float(loss[0]) * xmax
                seconds = 1.0 / c_x if c_x > 0 else float("nan")
            else:  # combined
                K, C = float(loss[0]), float(loss[1])
                if K > 0:
                    seconds = np.log((2.0 * K + C * xmax) / (K + C * xmax)) / K
                elif C * xmax > 0:
                    seconds = 1.0 / (C * xmax)
                else:
                    seconds = float("nan")
        return float(seconds) / 3600.0 if np.isfinite(seconds) else float("nan")

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

    def _decay_result(
        self, model, fit, unit, metric, t_start, n, volume, ach
    ) -> "DecayResult":
        """Assemble the typed :class:`DecayResult` from a two-stage fit outcome."""
        info = _MODELS[model]
        names = info["params"]
        popt = fit["popt"]
        # Every parameter but the initial excess is a magnitude; ``xi`` is a
        # signed difference between two independent levels, so it keeps its sign.
        params = {
            nm: (float(v) if nm == "xi" else float(abs(v)))
            for nm, v in zip(names, popt)
        }
        errors = {nm: 0.0 for nm in names}
        for nm, v in zip(names[: info["n_loss"]], fit["loss_err"]):
            errors[nm] = float(v)

        errors["P0"] = float(fit.get("P0_err", 0.0))
        errors["xi"] = float(fit.get("xi_err", 0.0))
        # Not a model parameter, but the peak excess is now fitted too, so its
        # uncertainty is worth carrying alongside them.
        errors["xmax"] = float(fit.get("xmax_err", 0.0))

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
            "start_concentration": fit["P_start"],
            "initial_excess": fit["xi"],
            "peak_concentration": peak_conc,
            "peak_excess": fit["xmax"],
            "emission_rate": E,
            "emission_rate_unit": f"{unit}/s",
            "decay_rate": self._effective_rate(model, fit["loss"], fit["xmax"]),
            "decay_rate_per_hour": self._effective_rate(model, fit["loss"], fit["xmax"])
            * 3600.0,
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
        # The explicit first-order term, present only for the models that have
        # one. The half-life is reported for all four (see _half_life_hours).
        linear_rate = params.get("k", params.get("K"))
        if linear_rate is not None:
            result["loss_rate_per_hour"] = linear_rate * 3600.0
        result["half_life_hours"] = self._half_life_hours(
            model, fit["loss"], fit["xmax"]
        )
        if ach is not None:
            # Split the loss into ventilation and everything else. For the
            # models without a linear term this uses the first-order-equivalent
            # rate, so it is an approximate decomposition -- but a usable one,
            # and better than reporting nothing.
            result["wall_loss_rate_per_hour"] = result["decay_rate_per_hour"] - ach
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

        return DecayResult.from_dict(result)
