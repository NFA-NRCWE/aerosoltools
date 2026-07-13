"""Per-instrument calibration derived from a two-dataset correlation.

Calibrates **one** dataset (the *target*) so it agrees with another (the
*reference*) — e.g. two identical OPS run side-by-side before an experiment,
marked as a task period: pick one as the reference and calibrate the other to
match it. The correction is applied to that **single dataset only** (never the
whole project) via the core ``calibrate`` methods.

Two bases are offered, mirroring what the core ``calibrate`` supports:

* **Total concentration** — one gain (and an optional offset) fitted from the
  total-concentration correlation. For a size-resolved (2D) target it corrects
  only the total-concentration series and leaves the individual size bins
  untouched, so a total-only calibration never distorts the distribution.
* **Per size bin** — a separate gain (optionally offset) is fitted for each size
  bin and applied bin-by-bin, so a size-dependent disagreement between two
  otherwise-identical instruments can be corrected. Only bins whose fit clears
  an R² threshold are corrected; weaker bins are left unchanged. Available only
  when both datasets are size-resolved with the same number of bins.

The correction is stored on the dataset as a reversible spec applied to a
snapshot of the uncalibrated object, so it can be toggled on/off or reset from
the Metadata pane and no calibrated value is ever negative.

Time-alignment between the two instruments reuses the Correlation tab's settings
(exact / nearest / rebin, tolerance, and the optional activity restriction) via
the library's own :func:`~aerosoltools.intercomparison._align_series`, so the fit is
computed on exactly the points the correlation plot shows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..intercomparison import _align_series
from . import helpers
from .qt import QtCore, QtWidgets

_BASIS_TOTAL = "Total concentration"
_BASIS_BINS = "Per size bin"

# A per-bin fit is only trustworthy above this coefficient of determination;
# bins whose fit is weaker are left uncorrected (item 4).
_R2_MIN = 0.5


# -- numerics -----------------------------------------------------------------
def _fit_gain(x, y, include_offset: bool):
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


def _bin_headers(obj) -> list:
    """Size-bin column names of a 2D object, in bin order."""
    return list(obj.size_data.columns)


def bins_compatible(a, b) -> bool:
    """True when both objects are size-resolved with the same number of bins."""
    if not (helpers.is_2d(a) and helpers.is_2d(b)):
        return False
    return len(_bin_headers(a)) == len(_bin_headers(b))


def compute_total(target, ref, include_offset: bool, align_kwargs: dict):
    """Fit a single total-concentration gain mapping ``target`` onto ``ref``."""
    x, y = _align_series(target, ref, "Total_conc", None, None, **align_kwargs)
    return _fit_gain(x, y, include_offset)


def compute_per_bin(target, ref, include_offset: bool, align_kwargs: dict):
    """Fit one gain per size bin mapping ``target`` onto ``ref``.

    Returns ``(m_list, b_list, r2_list, n_failed)``. Bins whose fit fails (no
    overlap, too few points, non-positive gain) are left unchanged (m=1, b=0)
    and counted in ``n_failed`` so the caller can warn.
    """
    th, rh = _bin_headers(target), _bin_headers(ref)
    if len(th) != len(rh):
        raise ValueError("The two datasets have different numbers of size bins.")
    ms, bs, r2s, failed = [], [], [], 0
    for tcol, rcol in zip(th, rh):
        try:
            x, y = _align_series(target, ref, (tcol, rcol), None, None, **align_kwargs)
            m, b, r2, n = _fit_gain(x, y, include_offset)
            if not np.isfinite(m) or m <= 0 or n < 2:
                raise ValueError("unusable bin fit")
        except Exception:
            m, b, r2, failed = 1.0, 0.0, float("nan"), failed + 1
        ms.append(m)
        bs.append(b)
        r2s.append(r2)
    return ms, bs, r2s, failed


# -- reversible, spec-based application ---------------------------------------
# Calibration is stored on the dataset as a *spec* and (re)applied to a snapshot
# of the uncalibrated object, so it can be toggled on/off or reset from any pane
# without re-fitting. The spec is a plain dict (JSON-safe) so it also persists.


def _clip_nonnegative(obj, recompute_total: bool) -> None:
    """Clip calibrated data at zero so no negative concentrations remain (item 5).

    For a per-bin calibration the total is re-summed from the clipped bins; for a
    total-only calibration the (independent) total series is clipped directly and
    the bins are left as they are.
    """
    if helpers.is_2d(obj):
        cols = obj._sizebin_headers
        obj._data[cols] = obj._data[cols].clip(lower=0.0)
        if recompute_total:
            obj._data["Total_conc"] = obj._data[cols].sum(axis=1)
        elif "Total_conc" in obj._data:
            obj._data["Total_conc"] = obj._data["Total_conc"].clip(lower=0.0)
    elif "Total_conc" in obj._data:
        obj._data["Total_conc"] = obj._data["Total_conc"].clip(lower=0.0)


def _apply_spec_to_obj(obj, spec: dict) -> None:
    """Apply a calibration ``spec`` to ``obj`` in place (with zero-clipping)."""
    include_offset = bool(spec.get("include_offset"))
    if spec["basis"] == _BASIS_BINS:
        # Only bins whose fit cleared the R² gate are corrected; the rest keep a
        # unit gain / zero offset so a weak per-bin fit never distorts the data.
        applied = spec["applied"]
        ms = [float(m) if ok else 1.0 for m, ok in zip(spec["ms"], applied)]
        variables = {"m": ms}
        if include_offset:
            variables["b"] = [
                float(b) if ok else 0.0 for b, ok in zip(spec["bs"], applied)
            ]
        obj.calibrate(parameter="bins", Variables=variables)
        _clip_nonnegative(obj, recompute_total=True)
    else:
        # Total-concentration calibration must NOT touch the individual bins
        # (item 4): calibrate only the total series for 2D data.
        m = float(spec["m"])
        b = float(spec["b"]) if include_offset else 0.0
        if helpers.is_2d(obj):
            obj.calibrate(parameter="Total_conc", Variables={"m": m, "b": b})
        else:
            obj.calibrate(m=m, b=b)
        _clip_nonnegative(obj, recompute_total=False)


def _rebuild(ds) -> None:
    """Set ``ds.obj`` to the (calibration-enabled) object built from the baseline."""
    if ds._cal_baseline is None:
        return
    obj = ds._cal_baseline.copy_self()
    if ds.calibration_enabled and ds.calibration is not None:
        _apply_spec_to_obj(obj, ds.calibration)
    ds.obj = obj


def apply_calibration(ds, spec: dict) -> None:
    """Apply a calibration spec to a dataset, snapshotting the baseline once.

    The first calibration captures the current (uncalibrated) object as the
    baseline; further calibrations replace the spec but keep that baseline, so
    the correction never compounds and can always be reset to the original.
    """
    if ds.calibration is None or ds._cal_baseline is None:
        ds._cal_baseline = ds.obj.copy_self()
    ds.calibration = spec
    ds.calibration_enabled = True
    _rebuild(ds)


def set_calibration_enabled(ds, on: bool) -> None:
    """Toggle a dataset's calibration on/off (rebuilding ``ds.obj``)."""
    if ds.calibration is None:
        return
    ds.calibration_enabled = bool(on)
    _rebuild(ds)


def reset_calibration(ds) -> None:
    """Drop the calibration entirely and restore the uncalibrated object."""
    if ds._cal_baseline is not None:
        ds.obj = ds._cal_baseline.copy_self()
    ds.calibration = None
    ds.calibration_enabled = False
    ds._cal_baseline = None


# -- human-readable summaries (Metadata pane, item 6) -------------------------
def _fmt_fn(m: float, b: float, include_offset: bool) -> str:
    """Render a linear calibration as ``y = m·x (+ b)`` with 2-decimal params."""
    text = f"y = {float(m):.2f}·x"
    if include_offset and b:
        text += f" {'+' if b >= 0 else '-'} {abs(float(b)):.2f}"
    return text


def calibration_summary(ds) -> Optional[str]:
    """One-line description of a dataset's total-concentration calibration, or None.

    Returned for a total-concentration calibration (shown beneath the metadata);
    per-bin calibrations are described row-by-row via :func:`bin_calibration_texts`.
    """
    spec = ds.calibration
    if not spec:
        return None
    state = "on" if ds.calibration_enabled else "off"
    if spec["basis"] == _BASIS_TOTAL:
        fn = _fmt_fn(spec["m"], spec.get("b", 0.0), spec.get("include_offset"))
        r2 = spec.get("r2")
        r2txt = f", R² = {r2:.2f}" if r2 is not None and np.isfinite(r2) else ""
        return f"Total concentration: {fn}{r2txt}  [{state}]"
    n_ok = sum(1 for a in spec.get("applied", []) if a)
    n_tot = len(spec.get("applied", []))
    return (
        f"Per size bin: {n_ok}/{n_tot} bins corrected (R² > {_R2_MIN})  [{state}] — "
        "see the per-bin table"
    )


def bin_calibration_texts(ds) -> Optional[list]:
    """Per-bin calibration-function strings (or None when not a per-bin calib.).

    Each entry is the bin's ``y = m·x (+ b)`` when its fit cleared the R² gate,
    or a short ``—`` note when the bin was left uncorrected.
    """
    spec = ds.calibration
    if not spec or spec["basis"] != _BASIS_BINS:
        return None
    include_offset = bool(spec.get("include_offset"))
    out = []
    for m, b, ok in zip(spec["ms"], spec["bs"], spec["applied"]):
        out.append(_fmt_fn(m, b, include_offset) if ok else "— (R² too low)")
    return out


# -- dialog -------------------------------------------------------------------
class CalibrationDialog(QtWidgets.QDialog):
    """Choose a reference, preview the fitted factors, and apply the calibration.

    The non-reference dataset is the *target*: it is modified in place (the core
    ``calibrate`` marks it ``calibrated`` in its metadata). Only that one dataset
    changes — calibration is per instrument, not project-wide.
    """

    def __init__(self, tab, xds, yds, align_kwargs: dict):
        """Build the dialog for the Correlation tab's current X/Y pair.

        Args:
            tab: The owning :class:`CorrelationTab` (used to refresh after apply).
            xds, yds: The two datasets being compared.
            align_kwargs: Time-alignment settings (match/tolerance/activity/…)
                shared with the correlation plot.
        """
        super().__init__(tab)
        self.tab = tab
        self.xds = xds
        self.yds = yds
        self.align_kwargs = align_kwargs
        self._bins_ok = bins_compatible(xds.obj, yds.obj)

        self.setWindowTitle("Calibrate instrument")
        self.setMinimumWidth(440)
        outer = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Calibrate one instrument so it matches the other. The reference is "
            "left unchanged; the other dataset is corrected in place."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # Reference selector.
        ref_box = QtWidgets.QGroupBox("Reference (kept unchanged)")
        ref_col = QtWidgets.QVBoxLayout(ref_box)
        self.ref_x = QtWidgets.QRadioButton(f"{xds.label}")
        self.ref_y = QtWidgets.QRadioButton(f"{yds.label}")
        self.ref_x.setChecked(True)
        self.ref_x.toggled.connect(self._on_change)
        ref_col.addWidget(self.ref_x)
        ref_col.addWidget(self.ref_y)
        outer.addWidget(ref_box)

        # Basis + offset.
        form = QtWidgets.QFormLayout()
        self.basis = QtWidgets.QComboBox()
        self.basis.addItems([_BASIS_TOTAL, _BASIS_BINS])
        if not self._bins_ok:
            # Disable the per-bin option when the datasets aren't both
            # size-resolved with matching bins.
            self.basis.model().item(1).setEnabled(False)
            self.basis.setItemData(
                1,
                "Both datasets must be size-resolved with the same bins.",
                QtCore.Qt.ToolTipRole,
            )
        self.basis.currentIndexChanged.connect(self._on_change)
        form.addRow("Basis:", self.basis)

        self.offset = QtWidgets.QCheckBox("Include offset (b), i.e. y = m·x + b")
        self.offset.setToolTip(
            "Fit an intercept as well as a gain. Off by default: identical "
            "instruments should agree through the origin."
        )
        self.offset.toggled.connect(lambda *_: self._clear_preview())
        form.addRow("", self.offset)
        outer.addLayout(form)

        self.preview = QtWidgets.QLabel("Click Preview to fit the calibration factors.")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("padding:4px;")
        outer.addWidget(self.preview)

        # Buttons.
        btns = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.preview_btn.clicked.connect(self._preview)
        self.apply_btn = QtWidgets.QPushButton("Apply calibration")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(self.preview_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        btns.addWidget(self.apply_btn)
        outer.addLayout(btns)

        self._result = None  # cached (basis, factors) from the last preview
        self._on_change()

    # -- helpers -----------------------------------------------------------
    def _target_ref(self):
        """Return ``(target_ds, ref_ds)`` from the reference selection."""
        if self.ref_x.isChecked():
            return self.yds, self.xds
        return self.xds, self.yds

    def _basis(self) -> str:
        return self.basis.currentText()

    def _on_change(self, *_a) -> None:
        """React to a reference/basis change: fix the offset rule, clear preview."""
        target, _ref = self._target_ref()
        self.offset.setEnabled(True)
        if self._basis() == _BASIS_BINS and helpers.is_2d(target.obj):
            # A per-bin offset is generally ill-posed, but still allowed.
            self.offset.setToolTip(
                "Fit an intercept as well as a gain, per size bin (off by "
                "default: identical instruments should agree through the origin)."
            )
        elif self._basis() == _BASIS_TOTAL and helpers.is_2d(target.obj):
            # Total-conc offset can't be distributed across bins, so it is
            # applied to the total series while the bins take the gain only.
            self.offset.setToolTip(
                "Fit an intercept as well as a gain. For a size-resolved "
                "instrument the offset is applied to the total-concentration "
                "series; the size bins are scaled by the gain only (so the "
                "total may differ slightly from the sum of the bins)."
            )
        else:
            self.offset.setToolTip(
                "Fit an intercept as well as a gain. Off by default: identical "
                "instruments should agree through the origin."
            )
        self._clear_preview()

    def _clear_preview(self) -> None:
        self._result = None
        self.preview.setText("Click Preview to fit the calibration factors.")

    # -- preview / apply ---------------------------------------------------
    def _preview(self):
        """Fit the factors for the current settings and describe them; cache them."""
        target, ref = self._target_ref()
        include_offset = self.offset.isChecked() and self.offset.isEnabled()
        # Fit against the uncalibrated baseline when the target already carries a
        # calibration, so a re-fit measures the original data (never compounds).
        tgt_obj = (
            target._cal_baseline if target._cal_baseline is not None else target.obj
        )
        try:
            if self._basis() == _BASIS_BINS:
                ms, bs, r2s, failed = compute_per_bin(
                    tgt_obj, ref.obj, include_offset, self.align_kwargs
                )
                self._result = {
                    "basis": _BASIS_BINS,
                    "ms": ms,
                    "bs": bs,
                    "r2s": r2s,
                    "include_offset": include_offset,
                }
                arr = np.asarray(ms, dtype=float)
                r2arr = np.asarray([r for r in r2s if np.isfinite(r)], dtype=float)
                n_ok = sum(1 for r in r2s if np.isfinite(r) and r > _R2_MIN)
                note = (
                    f"Per-bin gain across {len(ms)} bins: "
                    f"{arr.min():.3g}–{arr.max():.3g} (median {np.median(arr):.3g})."
                )
                if r2arr.size:
                    note += f"  Median R² = {np.median(r2arr):.3f}."
                note += (
                    f"  {n_ok}/{len(ms)} bins clear R² > {_R2_MIN} and will be "
                    "corrected; the rest are left unchanged."
                )
                if failed:
                    note += f"  ({failed} bin(s) had too little overlap.)"
            else:
                m, b, r2, n = compute_total(
                    tgt_obj, ref.obj, include_offset, self.align_kwargs
                )
                self._result = {
                    "basis": _BASIS_TOTAL,
                    "m": m,
                    "b": b,
                    "r2": r2,
                    "include_offset": include_offset,
                }
                eqn = f"y = {m:.4g}·x" + (f" + {b:.4g}" if include_offset else "")
                note = f"Total-concentration fit ({n} points): {eqn},  R² = {r2:.3f}."
        except Exception as exc:
            self._result = None
            self.preview.setText(f"Could not fit: {exc}")
            return
        self.preview.setText(
            f"Calibrating <b>{target.label}</b> to match <b>{ref.label}</b>.<br>{note}"
        )

    def _apply(self):
        """Confirm, apply the cached (or freshly fitted) factors, refresh the GUI."""
        if self._result is None:
            self._preview()
            if self._result is None:
                return
        target, ref = self._target_ref()
        if target.calibration is not None:
            ans = QtWidgets.QMessageBox.question(
                self,
                "Already calibrated",
                f"'{target.label}' already has a calibration. Applying a new one "
                "replaces it (it is fitted from the uncalibrated baseline, so it "
                "does not compound). Continue?",
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
        ans = QtWidgets.QMessageBox.question(
            self,
            "Apply calibration",
            f"Calibrate '{target.label}' to match '{ref.label}'?\n"
            "The correction can be toggled off or reset from the Metadata pane.",
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        try:
            spec = dict(self._result)
            spec["target"] = target.label
            spec["ref"] = ref.label
            if spec["basis"] == _BASIS_BINS:
                # Record which bins clear the R² gate, so both the applied
                # correction and the per-bin display agree on what was used.
                spec["applied"] = [
                    bool(np.isfinite(r) and r > _R2_MIN) for r in spec["r2s"]
                ]
            apply_calibration(target, spec)
            # The object was rebuilt, so re-project the shared tasks onto it.
            self.tab.main.project._apply_activities(target)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration failed", str(exc))
            return
        # Refresh everything so the calibrated data shows everywhere, then
        # recompute the correlation so the user sees the improved agreement.
        self.tab.main.refresh_all()
        self.tab.main._refresh_sidebar()
        if getattr(self.tab, "_has_drawn", False):
            self.tab._draw()
        QtWidgets.QMessageBox.information(
            self,
            "Calibrated",
            f"'{target.label}' calibrated to match '{ref.label}'.",
        )
        self.accept()
