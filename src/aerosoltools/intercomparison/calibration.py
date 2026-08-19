"""Calibrate one dataset (the *target*) so it agrees with a *reference*.

This is the calibration half of the intercomparison flow
``align → assess → fit → apply``. Two identical instruments run side by side
rarely read exactly the same; picking one as the reference and fitting a gain
(optionally an offset) that maps the other onto it is a standard field
correction.

The scientific work is split cleanly:

* **fitting** (cross-dataset) lives here — :func:`fit_calibration` validates the
  pair, time-aligns them via the shared :mod:`._alignment` machinery, fits the
  relationship and returns a :class:`CalibrationModel`;
* **application** (single-dataset) lives on the data object as
  ``data.apply_calibration(model)`` — this module's :func:`apply_calibration`
  is a thin pass-through so the two never diverge.

:func:`calibrate_against_reference` is the one obvious high-level entry point
(re-exported as ``aerosoltools.calibrate_against_reference``); it runs the whole
workflow and returns ``(calibrated, model)``.

Two bases are supported:

* ``"total"`` — one gain (and optional offset) from the total-concentration
  correlation. For a size-resolved target it corrects only the total series and
  leaves the individual bins untouched, so a total-only calibration never
  distorts the distribution.
* ``"per_bin"`` — a separate gain (optional offset) per size bin, applied
  bin-by-bin. Only bins whose fit clears ``minimum_r_squared`` are corrected;
  weaker bins are left unchanged. Requires both datasets to be size-resolved
  with compatible bins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ._alignment import _align_series, _resolve_unit

# Valid calibration bases.
BASIS_TOTAL = "total"
BASIS_PER_BIN = "per_bin"
_VALID_BASES = (BASIS_TOTAL, BASIS_PER_BIN)

#: Default coefficient-of-determination gate for per-bin calibration: a bin's
#: fit must clear this to be applied, otherwise that bin is left unchanged.
DEFAULT_MIN_R2 = 0.5


class CalibrationError(ValueError):
    """Raised when two datasets cannot be calibrated against each other.

    Carries an actionable, domain-specific message (incompatible instrument
    class, bases, units, size bins, or insufficient overlapping data).
    """


# -- the fitted model ---------------------------------------------------------
@dataclass
class CalibrationModel:
    """A fitted calibration transformation plus its diagnostics.

    A model maps a *target* dataset onto a *reference* by a per-quantity linear
    relation ``y = gain·x (+ offset)``. For ``basis="total"`` the lists hold a
    single element (the total-concentration fit); for ``basis="per_bin"`` they
    hold one element per size bin, in bin order.

    Attributes:
        basis: ``"total"`` or ``"per_bin"``.
        gains: Fitted gain(s) ``m``.
        offsets: Fitted offset(s) ``b`` (zero unless ``include_offset``).
        r_squared: Coefficient(s) of determination of the fit(s).
        n_obs: Number of aligned observations used per fit.
        applied: Per-entry flag — whether that gain/offset is actually applied.
            Always ``[True]`` for a total fit; for per-bin it is ``True`` only
            where the fit cleared ``minimum_r_squared``.
        include_offset: Whether an intercept was fitted.
        minimum_r_squared: The per-bin acceptance gate used (``basis="per_bin"``).
        clip_negative: Clip calibrated concentrations at zero on application.
        target_instrument / reference_instrument: Instrument names, for record.
        target_name / reference_name: Human labels (e.g. GUI dataset labels).
        target_unit / reference_unit: Measurement units, for record.
        bin_mids: Size-bin midpoints (nm) the per-bin fit was computed on.
        align: Time-alignment settings used (match/tolerance/rebin/activity).
    """

    basis: str
    gains: list[float]
    offsets: list[float]
    r_squared: list[float]
    n_obs: list[int]
    applied: list[bool]
    include_offset: bool = False
    minimum_r_squared: float = DEFAULT_MIN_R2
    clip_negative: bool = True
    target_instrument: str = ""
    reference_instrument: str = ""
    target_name: str = ""
    reference_name: str = ""
    target_unit: str = ""
    reference_unit: str = ""
    bin_mids: list[float] = field(default_factory=list)
    align: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.basis not in _VALID_BASES:
            raise CalibrationError(
                f"Unknown calibration basis {self.basis!r}; "
                f"expected one of {_VALID_BASES}."
            )

    @property
    def n_applied(self) -> int:
        """Number of entries (bins, or the single total) actually corrected."""
        return int(sum(1 for a in self.applied if a))

    def to_dict(self) -> dict:
        """Return a JSON-safe dict (round-trips through :meth:`from_dict`)."""
        return {
            "basis": self.basis,
            "gains": [float(m) for m in self.gains],
            "offsets": [float(b) for b in self.offsets],
            "r_squared": [float(r) for r in self.r_squared],
            "n_obs": [int(n) for n in self.n_obs],
            "applied": [bool(a) for a in self.applied],
            "include_offset": bool(self.include_offset),
            "minimum_r_squared": float(self.minimum_r_squared),
            "clip_negative": bool(self.clip_negative),
            "target_instrument": self.target_instrument,
            "reference_instrument": self.reference_instrument,
            "target_name": self.target_name,
            "reference_name": self.reference_name,
            "target_unit": self.target_unit,
            "reference_unit": self.reference_unit,
            "bin_mids": [float(d) for d in self.bin_mids],
            "align": dict(self.align),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationModel":
        """Reconstruct a model from :meth:`to_dict` output."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# -- numerics -----------------------------------------------------------------
def _fit_gain(x, y, include_offset: bool) -> Tuple[float, float, float, int]:
    """Fit ``y ≈ m·x (+ b)`` by least squares; return ``(m, b, r2, n)``.

    Without an offset the gain is the best-fit slope through the origin
    (``Σxy / Σx²``) — the natural correction for two instruments that should
    read the same. ``n`` is the number of aligned points used.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    if n < 2:
        return 1.0, 0.0, float("nan"), n
    if include_offset:
        m, b = np.polyfit(x, y, 1)
    else:
        denom = float(np.dot(x, x))
        m = float(np.dot(x, y) / denom) if denom > 0 else 1.0
        b = 0.0
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(m), float(b), float(r2), n


def _is_size_resolved(obj) -> bool:
    """True when ``obj`` exposes size bins (an Aerosol2D-like object)."""
    return bool(getattr(obj, "_sizebin_headers", None))


def _bin_headers(obj) -> list:
    """Size-bin column names of a size-resolved object, in bin order."""
    return list(obj._sizebin_headers)


def _total_column(obj) -> str:
    """Name of the total/primary series calibrated by a ``"total"`` basis."""
    if "Total_conc" in obj.data.columns:
        return "Total_conc"
    primary = getattr(getattr(obj, "_primary", None), "name", None)
    return primary or obj.data.columns[0]


def _align_kwargs(match, tolerance, rebin_freq, rebin_method, activity) -> dict:
    """Bundle the time-alignment settings shared by every fit here."""
    return {
        "match": match,
        "tolerance": tolerance,
        "rebin_freq": rebin_freq,
        "rebin_method": rebin_method,
        "activity": activity,
    }


def _fit_total_calibration(target, reference, include_offset, align):
    """Fit a single total-concentration gain mapping ``target`` onto ``ref``."""
    col_t = _total_column(target)
    col_r = _total_column(reference)
    try:
        x, y = _align_series(target, reference, (col_t, col_r), None, None, **align)
    except (ValueError, KeyError) as exc:
        raise CalibrationError(
            "Could not align the two datasets for a total-concentration "
            f"calibration: {exc}"
        ) from exc
    m, b, r2, n = _fit_gain(x, y, include_offset)
    return [m], [b], [r2], [n], [True]


def _fit_per_bin_calibration(target, reference, include_offset, minimum_r2, align):
    """Fit one gain per size bin mapping ``target`` onto ``ref``.

    Bins whose fit fails (no overlap, too few points, non-positive gain) or does
    not clear ``minimum_r2`` are left unchanged (gain 1, offset 0, ``applied``
    ``False``) so a weak per-bin fit never distorts the data.
    """
    th, rh = _bin_headers(target), _bin_headers(reference)
    gains, offsets, r2s, ns, applied = [], [], [], [], []
    for tcol, rcol in zip(th, rh):
        try:
            x, y = _align_series(target, reference, (tcol, rcol), None, None, **align)
            m, b, r2, n = _fit_gain(x, y, include_offset)
            ok = bool(np.isfinite(m) and m > 0 and n >= 2)
        except Exception:
            m, b, r2, n, ok = 1.0, 0.0, float("nan"), 0, False
        keep = bool(ok and np.isfinite(r2) and r2 > minimum_r2)
        gains.append(m if keep else 1.0)
        offsets.append(b if keep else 0.0)
        r2s.append(r2)
        ns.append(n)
        applied.append(keep)
    return gains, offsets, r2s, ns, applied


# -- validation ---------------------------------------------------------------
def _validate_pair(target, reference, basis: str) -> None:
    """Raise :class:`CalibrationError` if the pair cannot be calibrated."""
    for name, obj in (("target", target), ("reference", reference)):
        if not hasattr(obj, "data"):
            raise CalibrationError(
                f"The {name} does not look like an aerosol dataset "
                "(no 'data' attribute)."
            )
    if basis not in _VALID_BASES:
        raise CalibrationError(
            f"Unknown calibration basis {basis!r}; expected one of {_VALID_BASES}."
        )

    # Units: when both are known they must match — calibrating instruments that
    # report different quantities/units is almost always a mistake.
    ut = _resolve_unit(target, _total_column(target))
    ur = _resolve_unit(reference, _total_column(reference))
    if ut and ur and ut != ur:
        raise CalibrationError(
            f"Incompatible units: target is in '{ut}' but reference is in "
            f"'{ur}'. Calibration expects the two datasets to measure the same "
            "quantity in the same unit."
        )

    if basis == BASIS_PER_BIN:
        if not (_is_size_resolved(target) and _is_size_resolved(reference)):
            raise CalibrationError(
                "Per-bin calibration needs both datasets to be size-resolved "
                "(2D). Use basis='total' for scalar instruments."
            )
        nt, nr = len(_bin_headers(target)), len(_bin_headers(reference))
        if nt != nr:
            raise CalibrationError(
                f"Per-bin calibration needs matching size bins, but the target "
                f"has {nt} bins and the reference has {nr}."
            )
        et = np.asarray(target._meta.get("bin_edges", []), dtype=float)
        er = np.asarray(reference._meta.get("bin_edges", []), dtype=float)
        if et.size and er.size:
            if et.shape != er.shape or not np.allclose(et, er, rtol=0.05):
                raise CalibrationError(
                    "Per-bin calibration needs compatible size-bin edges; the "
                    "two datasets' bin definitions differ. Align their bins "
                    "first (or calibrate on basis='total')."
                )


# -- public API ---------------------------------------------------------------
def fit_calibration(
    target,
    reference,
    *,
    basis: str = BASIS_TOTAL,
    include_offset: bool = False,
    minimum_r_squared: float = DEFAULT_MIN_R2,
    match: str = "exact",
    tolerance="30s",
    rebin_freq: Optional[str] = None,
    rebin_method="mean",
    activity: Optional[str] = None,
) -> CalibrationModel:
    """Fit a :class:`CalibrationModel` mapping ``target`` onto ``reference``.

    Validates the pair, time-aligns the two datasets with the shared alignment
    machinery, fits the relationship on the chosen ``basis``, and returns the
    model **without** modifying either dataset. Use :func:`apply_calibration`
    (or ``target.apply_calibration(model)``) to apply it, or the one-shot
    :func:`calibrate_against_reference` to do both.

    Args:
        target: Dataset to be corrected.
        reference: Dataset kept unchanged, whose readings ``target`` is mapped to.
        basis: ``"total"`` (default) or ``"per_bin"``.
        include_offset: Fit an intercept ``b`` as well as a gain ``m``. Off by
            default — identical instruments should agree through the origin.
        minimum_r_squared: Per-bin acceptance gate (``basis="per_bin"`` only).
        match: Time-alignment strategy (``"exact"``/``"nearest"``/``"rebin"``).
        tolerance: Max timestamp separation for ``match="nearest"``.
        rebin_freq: Target cadence for ``match="rebin"``.
        rebin_method: Aggregation for ``match="rebin"``.
        activity: Restrict the fit to one activity's marked periods.

    Returns:
        CalibrationModel: The fitted model with per-entry diagnostics.

    Raises:
        CalibrationError: If the datasets are incompatible (class, units, bins)
            or have too little overlapping data to fit.
    """
    _validate_pair(target, reference, basis)
    align = _align_kwargs(match, tolerance, rebin_freq, rebin_method, activity)

    if basis == BASIS_PER_BIN:
        gains, offsets, r2s, ns, applied = _fit_per_bin_calibration(
            target, reference, include_offset, minimum_r_squared, align
        )
        bin_mids = [float(d) for d in getattr(target, "bin_mids", [])]
    else:
        gains, offsets, r2s, ns, applied = _fit_total_calibration(
            target, reference, include_offset, align
        )
        bin_mids = []

    return CalibrationModel(
        basis=basis,
        gains=gains,
        offsets=offsets,
        r_squared=r2s,
        n_obs=ns,
        applied=applied,
        include_offset=include_offset,
        minimum_r_squared=minimum_r_squared,
        clip_negative=True,
        target_instrument=str(getattr(target, "instrument", "")),
        reference_instrument=str(getattr(reference, "instrument", "")),
        target_unit=_resolve_unit(target, _total_column(target)),
        reference_unit=_resolve_unit(reference, _total_column(reference)),
        bin_mids=bin_mids,
        align=align,
    )


def apply_calibration(data, model: CalibrationModel, *, inplace: bool = False):
    """Apply a fitted :class:`CalibrationModel` to a single dataset.

    Thin pass-through to ``data.apply_calibration(model)`` so the intercomparison
    workflow and direct scripting share one application implementation.

    Args:
        data: The dataset to correct (typically the calibration *target*).
        model: A model from :func:`fit_calibration`.
        inplace: Modify ``data`` (``True``) or a deep copy (``False``, default).

    Returns:
        The calibrated dataset (``data`` when ``inplace``).
    """
    return data.apply_calibration(model, inplace=inplace)


def calibrate_against_reference(
    target,
    reference,
    *,
    basis: str = BASIS_TOTAL,
    include_offset: bool = False,
    minimum_r_squared: float = DEFAULT_MIN_R2,
    inplace: bool = False,
    match: str = "exact",
    tolerance="30s",
    rebin_freq: Optional[str] = None,
    rebin_method="mean",
    activity: Optional[str] = None,
):
    """Fit and apply a calibration of ``target`` against ``reference`` in one call.

    The obvious high-level entry point (re-exported as
    ``aerosoltools.calibrate_against_reference``). Equivalent to::

        model = fit_calibration(target, reference, ...)
        calibrated = apply_calibration(target, model, inplace=inplace)
        return calibrated, model

    Args:
        target: Dataset to be corrected.
        reference: Dataset kept unchanged.
        basis: ``"total"`` (default) or ``"per_bin"``.
        include_offset: Fit an intercept as well as a gain.
        minimum_r_squared: Per-bin acceptance gate (``basis="per_bin"``).
        inplace: Correct ``target`` in place (``True``) or return a new,
            calibrated copy and leave ``target`` untouched (``False``, default).
        match, tolerance, rebin_freq, rebin_method, activity: Time-alignment
            settings (see :func:`fit_calibration`).

    Returns:
        tuple[Aerosol1D, CalibrationModel]: The calibrated dataset and the model
            used. The dataset keeps ``target``'s own class.

    Raises:
        CalibrationError: If the datasets are incompatible or lack overlap.
    """
    model = fit_calibration(
        target,
        reference,
        basis=basis,
        include_offset=include_offset,
        minimum_r_squared=minimum_r_squared,
        match=match,
        tolerance=tolerance,
        rebin_freq=rebin_freq,
        rebin_method=rebin_method,
        activity=activity,
    )
    calibrated = apply_calibration(target, model, inplace=inplace)
    return calibrated, model
