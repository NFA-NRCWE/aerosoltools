"""GUI glue for calibrating one dataset (the *target*) against another.

Calibrates **one** dataset so it agrees with a *reference* — e.g. two identical
OPS run side-by-side before an experiment: pick one as the reference and
calibrate the other to match it. Only that single dataset changes.

All of the science — validating the pair, time-aligning them, fitting the
gain(s)/offset(s), and applying them — lives in the public
:mod:`aerosoltools.intercomparison.calibration` module and on the data object as
``apply_calibration``. A scripting user reproduces exactly what this dialog does
with::

    calibrated, model = at.calibrate_against_reference(target, reference)

This module adds only GUI conveniences on top of that API:

* a reference / basis / offset dialog with a fit **preview**;
* reversible per-dataset state — apply, toggle on/off, reset — built on a
  snapshot of the uncalibrated object, so a correction can be undone without
  re-fitting and survives project save/load (the model is stored as a JSON-safe
  :meth:`CalibrationModel.to_dict` dict).

Time-alignment reuses the Correlation tab's settings so the fit is computed on
exactly the points the correlation plot shows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...intercomparison.calibration import (
    BASIS_PER_BIN,
    BASIS_TOTAL,
    DEFAULT_MIN_R2,
    CalibrationModel,
    fit_calibration,
)
from ...intercomparison.calibration import apply_calibration as _apply_model
from ..qt import QtCore, QtWidgets
from . import helpers

# Display labels for the two bases (the API uses the machine strings).
_BASIS_LABELS = {BASIS_TOTAL: "Total concentration", BASIS_PER_BIN: "Per size bin"}
_R2_MIN = DEFAULT_MIN_R2


def bins_compatible(a, b) -> bool:
    """True when both objects are size-resolved with the same number of bins."""
    if not (helpers.is_2d(a) and helpers.is_2d(b)):
        return False
    return len(a._sizebin_headers) == len(b._sizebin_headers)


# -- reversible, per-dataset state --------------------------------------------
# The calibration is stored on the dataset wrapper as a JSON-safe model dict and
# (re)applied to a snapshot of the uncalibrated object, so it can be toggled
# on/off or reset from any pane without re-fitting, and persists with the project.


def _model_of(ds) -> Optional[CalibrationModel]:
    """Reconstruct the dataset's :class:`CalibrationModel`, or ``None``."""
    if not ds.calibration:
        return None
    return CalibrationModel.from_dict(ds.calibration)


def _rebuild(ds) -> None:
    """Set ``ds.obj`` to the (calibration-enabled) object built from the baseline."""
    if ds._cal_baseline is None:
        return
    obj = ds._cal_baseline.copy_self()
    model = _model_of(ds)
    if ds.calibration_enabled and model is not None:
        obj = _apply_model(obj, model, inplace=True)
    ds.obj = obj
    ds.touch()  # obj swapped → invalidate the derived-copy cache


def apply_to_dataset(ds, model: CalibrationModel) -> None:
    """Apply a fitted model to a dataset, snapshotting the baseline once.

    The first calibration captures the current (uncalibrated) object as the
    baseline; further calibrations replace the model but keep that baseline, so
    the correction never compounds and can always be reset to the original.
    """
    if ds.calibration is None or ds._cal_baseline is None:
        ds._cal_baseline = ds.obj.copy_self()
    ds.calibration = model.to_dict()
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
    ds.touch()  # obj restored → invalidate the derived-copy cache


# -- human-readable summaries (Metadata pane) ---------------------------------
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
    model = _model_of(ds)
    if model is None:
        return None
    state = "on" if ds.calibration_enabled else "off"
    if model.basis == BASIS_TOTAL:
        fn = _fmt_fn(model.gains[0], model.offsets[0], model.include_offset)
        r2 = model.r_squared[0] if model.r_squared else None
        r2txt = f", R² = {r2:.2f}" if r2 is not None and np.isfinite(r2) else ""
        return f"Total concentration: {fn}{r2txt}  [{state}]"
    n_ok = model.n_applied
    n_tot = len(model.applied)
    return (
        f"Per size bin: {n_ok}/{n_tot} bins corrected "
        f"(R² > {model.minimum_r_squared:g})  [{state}] — see the per-bin table"
    )


def bin_calibration_texts(ds) -> Optional[list]:
    """Per-bin calibration-function strings (or None when not a per-bin calib.).

    Each entry is the bin's ``y = m·x (+ b)`` when its fit cleared the R² gate,
    or a short ``—`` note when the bin was left uncorrected.
    """
    model = _model_of(ds)
    if model is None or model.basis != BASIS_PER_BIN:
        return None
    out = []
    for m, b, ok in zip(model.gains, model.offsets, model.applied):
        out.append(_fmt_fn(m, b, model.include_offset) if ok else "— (R² too low)")
    return out


# -- dialog -------------------------------------------------------------------
class CalibrationDialog(QtWidgets.QDialog):
    """Choose a reference, preview the fitted factors, and apply the calibration.

    The non-reference dataset is the *target*: only it changes. The dialog is a
    thin front end over :func:`aerosoltools.intercomparison.fit_calibration` and
    ``apply_calibration`` — it fits nothing itself.
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
        self.basis.addItem(_BASIS_LABELS[BASIS_TOTAL], BASIS_TOTAL)
        self.basis.addItem(_BASIS_LABELS[BASIS_PER_BIN], BASIS_PER_BIN)
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

        self._model: Optional[CalibrationModel] = None  # cached last-preview model
        self._on_change()

    # -- helpers -----------------------------------------------------------
    def _target_ref(self):
        """Return ``(target_ds, ref_ds)`` from the reference selection."""
        if self.ref_x.isChecked():
            return self.yds, self.xds
        return self.xds, self.yds

    def _basis(self) -> str:
        """The machine basis string (``"total"`` / ``"per_bin"``) selected."""
        return self.basis.currentData()

    def _on_change(self, *_a) -> None:
        """React to a reference/basis change: fix the offset rule, clear preview."""
        target, _ref = self._target_ref()
        self.offset.setEnabled(True)
        if self._basis() == BASIS_PER_BIN and helpers.is_2d(target.obj):
            # A per-bin offset is generally ill-posed, but still allowed.
            self.offset.setToolTip(
                "Fit an intercept as well as a gain, per size bin (off by "
                "default: identical instruments should agree through the origin)."
            )
        elif self._basis() == BASIS_TOTAL and helpers.is_2d(target.obj):
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
        self._model = None
        self.preview.setText("Click Preview to fit the calibration factors.")

    def _fit(self) -> CalibrationModel:
        """Fit a model for the current settings against the uncalibrated baseline."""
        target, ref = self._target_ref()
        include_offset = self.offset.isChecked() and self.offset.isEnabled()
        # Fit against the uncalibrated baseline when the target already carries a
        # calibration, so a re-fit measures the original data (never compounds).
        tgt_obj = (
            target._cal_baseline if target._cal_baseline is not None else target.obj
        )
        return fit_calibration(
            tgt_obj,
            ref.obj,
            basis=self._basis(),
            include_offset=include_offset,
            minimum_r_squared=_R2_MIN,
            **self.align_kwargs,
        )

    # -- preview / apply ---------------------------------------------------
    def _preview(self):
        """Fit the factors for the current settings and describe them; cache them."""
        target, ref = self._target_ref()
        try:
            model = self._fit()
        except Exception as exc:
            self._model = None
            self.preview.setText(f"Could not fit: {exc}")
            return
        self._model = model

        if model.basis == BASIS_PER_BIN:
            gains = np.asarray(model.gains, dtype=float)
            r2arr = np.asarray(
                [r for r in model.r_squared if np.isfinite(r)], dtype=float
            )
            n_ok = model.n_applied
            n = len(model.gains)
            note = (
                f"Per-bin gain across {n} bins: "
                f"{gains.min():.3g}–{gains.max():.3g} (median {np.median(gains):.3g})."
            )
            if r2arr.size:
                note += f"  Median R² = {np.median(r2arr):.3f}."
            note += (
                f"  {n_ok}/{n} bins clear R² > {model.minimum_r_squared:g} and will "
                "be corrected; the rest are left unchanged."
            )
            failed = sum(1 for nobs in model.n_obs if nobs < 2)
            if failed:
                note += f"  ({failed} bin(s) had too little overlap.)"
        else:
            m, b = model.gains[0], model.offsets[0]
            r2, n = model.r_squared[0], model.n_obs[0]
            eqn = f"y = {m:.4g}·x" + (f" + {b:.4g}" if model.include_offset else "")
            note = f"Total-concentration fit ({n} points): {eqn},  R² = {r2:.3f}."

        self.preview.setText(
            f"Calibrating <b>{target.label}</b> to match <b>{ref.label}</b>.<br>{note}"
        )

    def _apply(self):
        """Confirm, apply the cached (or freshly fitted) model, refresh the GUI."""
        if self._model is None:
            self._preview()
            if self._model is None:
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
            model = self._model
            model.target_name = target.label
            model.reference_name = ref.label
            apply_to_dataset(target, model)
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
