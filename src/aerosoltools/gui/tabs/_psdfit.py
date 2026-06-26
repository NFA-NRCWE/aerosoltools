"""Lognormal PSD-fit logic for the PSD tab — pure, Qt-free numerics.

This module owns *the numbers* behind the PSD tab's lognormal fitting, so the
Qt widget in :mod:`psd` is left owning only the view and interaction. It holds:

* the mode parameterisation and the ``peak`` <-> ``factor`` reparameterisation
  (:func:`peak_to_factor` / :func:`factor_to_peak`),
* lognormal evaluation (:func:`lognormal_modes`),
* mode validation / cleaning (:func:`clean_modes`, :func:`valid_modes`,
  :func:`modes_as_triples`),
* the local fit window (:func:`local_mask`),
* the optimisation call wrapper (:func:`run_fit`), and
* the in-window R²/degeneracy scoring (:func:`fit_quality`).

A "mode" is a plain ``dict`` with keys ``mu`` (peak diameter, nm), ``sigma``
(geometric SD), ``peak`` (dx/dlogDp peak height) and ``bound`` (hold μ during
the fit); optimised modes additionally carry ``*_err`` uncertainties.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# A lognormal mode is parameterised by its peak diameter ``mu`` (nm), its
# geometric standard deviation ``sigma`` (dimensionless), and the *peak height*
# of the dx/dlogDp curve. :meth:`Aerosol2D.fit_psd` instead works with a
# ``factor`` scaling parameter; the two are related by the value of the
# lognormal at its peak, ``peak = factor / (sqrt(2π) · log10(sigma))``. Storing
# the peak height (rather than ``factor``) lets the user read and click a value
# that matches the curve, and lets a single click seed a mode that visually
# peaks where the user clicked.
_SQRT_2PI = np.sqrt(2.0 * np.pi)
#: Default geometric standard deviation for a freshly added/placed mode.
DEFAULT_SIGMA = 1.8
#: Clamp range for σ (geometric SD) while shaping a mode by scroll/typing.
SIGMA_MIN, SIGMA_MAX = 1.05, 5.0
#: How far optimisation may move a mode's peak diameter from its placed value
#: (μ ∈ [μ₀/f, μ₀·f]). Keeps "Fit" a local refinement of the user's manual
#: modes so a redundant mode cannot flee far outside the measured range.
FIT_MU_FACTOR = 3.0
#: When "Local" fitting is on, only size bins within this many geometric SDs of
#: a mode are fit/scored — so modes follow the peaks instead of one broad
#: lognormal trying to span the whole spectrum (which never physically exists).
FIT_LOCAL_SIGMAS = 3.0


def peak_to_factor(peak: float, sigma: float) -> float:
    """Convert a desired dx/dlogDp peak height to ``fit_psd``'s ``factor``."""
    return float(peak) * _SQRT_2PI * np.log10(sigma)


def factor_to_peak(factor: float, sigma: float) -> float:
    """Convert ``fit_psd``'s ``factor`` back to a dx/dlogDp peak height."""
    return float(factor) / (_SQRT_2PI * np.log10(sigma))


def lognormal_modes(dp_nm, modes):
    """Evaluate a sum of lognormal modes (dx/dlogDp) at diameters ``dp_nm``.

    Mirrors the lognormal used inside :meth:`Aerosol2D.fit_psd` so the overlay
    matches the fit exactly.

    Args:
        dp_nm: Diameters in nm to evaluate at.
        modes: Iterable of ``(mu, sigma, factor)`` triples.

    Returns:
        ``(total, per_mode)`` where ``total`` is the summed curve and
        ``per_mode`` is the list of each mode's individual curve.
    """
    x = np.log10(np.asarray(dp_nm, dtype=float))
    total = np.zeros_like(x)
    per_mode = []
    for mu, sigma, factor in modes:
        s = np.log10(sigma)
        comp = (
            factor
            * (1.0 / (_SQRT_2PI * s))
            * np.exp(-((x - np.log10(mu)) ** 2) / (2.0 * s**2))
        )
        per_mode.append(comp)
        total = total + comp
    return total, per_mode


def clean_modes(modes) -> List[dict]:
    """Coerce stored/working modes to plain ``{mu, sigma, peak, bound}`` dicts."""
    return [
        {
            "mu": float(m["mu"]),
            "sigma": float(m["sigma"]),
            "peak": float(m["peak"]),
            "bound": bool(m.get("bound")),
        }
        for m in (modes or [])
    ]


def valid_modes(modes) -> List[Tuple[int, float, float, float, bool]]:
    """Return modes with sane values as ``(index, mu, sigma, peak, bound)``."""
    out = []
    for i, m in enumerate(modes):
        try:
            mu, sigma, peak = float(m["mu"]), float(m["sigma"]), float(m["peak"])
        except (TypeError, ValueError):
            continue
        if mu > 0 and sigma > 1.0 and peak > 0:
            out.append((i, mu, sigma, peak, bool(m.get("bound"))))
    return out


def modes_as_triples(modes) -> List[Tuple[float, float, float]]:
    """Valid modes as ``(mu, sigma, factor)`` for plotting/evaluation."""
    return [
        (mu, sigma, peak_to_factor(peak, sigma))
        for _, mu, sigma, peak, _ in valid_modes(modes)
    ]


def local_mask(x, modes, *, enabled: bool = True, local_sigmas: float = FIT_LOCAL_SIGMAS):
    """Boolean mask of bins within the local fit window of any mode.

    Mirrors ``fit_psd``'s ``local_sigmas`` windowing so a reported R² is scored
    on the same bins the fit used. All-True when ``enabled`` is False (or no
    mode covers any bin).
    """
    x = np.asarray(x, dtype=float)
    if not enabled:
        return np.ones(x.shape, dtype=bool)
    logx = np.log10(x)
    keep = np.zeros(x.shape, dtype=bool)
    for _, mu, sigma, _peak, _bound in valid_modes(modes):
        keep |= np.abs(logx - np.log10(mu)) <= local_sigmas * np.log10(sigma)
    return keep if keep.any() else np.ones(x.shape, dtype=bool)


def run_fit(
    obj,
    period: str,
    modes,
    *,
    log_scaling: bool,
    tolerance: float,
    local: bool,
) -> Tuple[List[dict], Optional[str]]:
    """Optimise ``modes`` against ``obj``'s PSD for ``period`` via ``fit_psd``.

    Args:
        obj: The :class:`Aerosol2D` whose PSD is fit.
        period: Activity name selecting the samples to fit.
        modes: Working modes to seed the optimisation.
        log_scaling: Fit against ``log10(dx/dlogDp)`` when True.
        tolerance: Percent window a bound μ may move within.
        local: Restrict each mode to its local window when True.

    Returns:
        ``(new_modes, None)`` on success, or ``([], message)`` when there is
        nothing valid to fit or the optimisation raises.
    """
    valid = valid_modes(modes)
    if not valid:
        return [], "Add at least one mode first (click the plot or 'Add')."

    mu = [v[1] for v in valid]
    sigma = [v[2] for v in valid]
    factor = [peak_to_factor(v[3], v[2]) for v in valid]
    binding: List[bool] = []
    for v in valid:
        binding += [v[4], False, False]  # bind μ only

    try:
        modes_out, err = obj.fit_psd(
            period=period,
            mu=mu,
            sigma=sigma,
            factor=factor,
            log_scaling=log_scaling,
            binding=binding,
            tolerance=tolerance,
            mu_factor=FIT_MU_FACTOR,
            weighting="uniform",  # fit the curve as shown (by-eye fitting)
            local_sigmas=FIT_LOCAL_SIGMAS if local else None,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a message
        return [], f"Fit failed: {exc}"

    new: List[dict] = []
    for i in range(len(modes_out["mu"])):
        ss = float(modes_out["sigma"][i])
        ff = float(modes_out["factor"][i])
        new.append(
            {
                "mu": float(modes_out["mu"][i]),
                "sigma": ss,
                "peak": factor_to_peak(ff, ss),
                "bound": valid[i][4] if i < len(valid) else False,
                "mu_err": float(err["mu"][i]),
                "sigma_err": float(err["sigma"][i]),
                "factor_err": float(err["factor"][i]),
            }
        )
    return new, None


def fit_quality(x, y, modes, *, local: bool):
    """In-window, linear-space R² of the total fit plus degenerate-mode flags.

    Scored in the plot's own (linear) units, and only over the bins the fit used
    (the modes' local windows), so the number matches what the eye sees on the
    curve.

    Returns:
        ``(r2, scope, flagged)`` where ``scope`` is ``"in-window"`` or
        ``"full range"`` and ``flagged`` is a list of 1-based mode numbers that
        look degenerate, or ``None`` when there are too few usable points.
    """
    triples = modes_as_triples(modes)
    if not triples:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    mask &= local_mask(x, modes, enabled=local)
    if mask.sum() < 3:
        return None

    yhat, _ = lognormal_modes(x[mask], triples)
    yv = y[mask]
    ss_res = np.nansum((yv - yhat) ** 2)
    ss_tot = np.nansum((yv - np.nanmean(yv)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Flag degenerate modes so a poor fit is obvious: a σ pinned near the
    # solver's ceiling or a peak that landed outside the measured size range
    # usually means the data needs another mode (or a bound μ).
    lo, hi = float(np.min(x[mask])), float(np.max(x[mask]))
    flagged = [
        str(i + 1)
        for i, m in enumerate(modes)
        if m["sigma"] >= 4.5 or m["mu"] < 0.9 * lo or m["mu"] > 1.1 * hi
    ]
    scope = "in-window" if local else "full range"
    return r2, scope, flagged
