"""Size-selective particulate-matter (Pₓ) fractions and metric series."""

from math import erf
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover - typing_extensions fallback
    from typing_extensions import override  # noqa: F401

if TYPE_CHECKING:  # only for type hints; avoids a runtime circular import
    from ..aerosol2d import Aerosol2D


class FractionMixin:
    """Compute size-selective Pₓ fractions and named metric series."""

    def _px_fraction_series(
        self,
        dtype: str = "dM",
        upper: float = 4.2,
        lower: float = 0.0,
        work: "Aerosol2D | None" = None,
    ) -> pd.Series:
        """Compute a size-selective Pₓ time series for a given band (private core).

        This is the internal numeric routine used by :meth:`PM_calc` and
        :meth:`summarize` to compute EN 481 / ISO 7708–style size-selective
        metrics (PM/PN/PS/PV). It integrates the requested distribution in
        the interval ``[lower, upper]`` (µm) using the standard lognormal
        penetration curve (GSD = 1.5, 50% cut at ``upper``).

        The result is a 1D :class:`pandas.Series` indexed by :attr:`time`.
        No data are stored in :attr:`extra_data`; callers decide whether to
        persist the series.

        Args:
            dtype (str, optional): Target base distribution kind, one of
                ``{"dN", "dS", "dV", "dM"}``. The helper ensures the working
                data are unnormalized (not ``*/dlogDp``) and converted to this
                dtype before integration. Defaults to ``"dM"``.
            upper (float, optional): Upper cut-off diameter in micrometers
                (µm) corresponding to the nominal 50% penetration point.
                Defaults to ``4.2``.
            lower (float, optional): Lower cut-off diameter in micrometers
                (µm). If ``0.0`` (default), the returned series is cumulative
                from 0 → ``upper``. If in ``(0, upper)``, the returned series
                represents the band-limited contribution from ``lower`` → ``upper``.
            work (Aerosol2D | None, optional): Optional pre-prepared working
                object that is **already** unnormalized (no ``"/dlogDp"`` in
                :attr:`dtype`) and in the desired ``dtype``. This is used by
                performance-critical code (e.g. :meth:`summarize`) to avoid
                repeated conversions. If ``None`` (default), a suitable working
                copy is created internally as needed.

        Returns:
            pandas.Series: Time series of the requested Pₓ metric, indexed by
            :attr:`time`. Empty or invalid time steps (where
            :attr:`total_concentration` is NaN) are returned as NaN.

        Raises:
            ValueError: If ``lower`` is negative, or if ``lower`` is not
                strictly smaller than ``upper``.
            ValueError: If ``dtype`` is not one of the supported base kinds.
            ValueError: If a non-``None`` ``work`` object is passed that is
                still in ``*/dlogDp`` form or not in the requested ``dtype``.

        Notes:
            * This helper never writes to :attr:`extra_data`; it is purely
              functional. :meth:`PM_calc` is the public API that stores Pₓ
              series in :attr:`extra_data` using this core routine.
            * The integration is always performed on the **unnormalized**
              distribution (i.e. ``dN``, ``dM`` etc., not ``*/dlogDp``).
        """
        if lower < 0:
            raise ValueError("Lower limit must be non-negative.")
        if lower >= upper:
            raise ValueError(
                f"Lower limit {lower!r} must be smaller than upper limit {upper!r}."
            )

        # Normalise dtype to base form (strip any spurious '/dlogDp')
        base_dtype = dtype.replace("/dlogDp", "")
        if base_dtype not in {"dN", "dS", "dV", "dM"}:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Expected one of 'dN', 'dS', 'dV', 'dM'."
            )

        # Decide on a working object: either caller-provided or an internal copy
        if work is not None:
            # For safety, ensure the work object is already in a suitable state
            if "/dlogDp" in str(work.dtype):
                raise ValueError(
                    "Internal error: 'work' passed to _px_series must be unnormalized "
                    "(dtype must not contain '/dlogDp')."
                )
            if str(work.dtype) != base_dtype:
                raise ValueError(
                    "Internal error: 'work' passed to _px_series must already be in "
                    f"the requested dtype {base_dtype!r}, got {work.dtype!r}."
                )
            dist = work.size_data.to_numpy(dtype=float)
        else:
            # Fast path: extract and convert the underlying array without
            # deep-copying the full Aerosol2D object.
            base_arr, _, _ = self._as_base_array()
            current_base = str(self.dtype).replace("/dlogDp", "")
            if current_base != base_dtype:
                dist = self._convert_array(
                    base_arr, self.bin_mids, current_base, base_dtype, self.density
                )
            else:
                dist = base_arr

        # Bin midpoints in nm and EN 481/ISO 7708 fractions
        bin_mids = np.asarray(self.bin_mids, dtype=float)
        pm_nm = upper * 1000.0
        Y = np.log(bin_mids / pm_nm) / (np.sqrt(2.0) * np.log(1.5))
        Frac_upper = 0.5 * (1.0 + np.vectorize(erf)(-Y))
        upper_vals = np.nansum(dist * Frac_upper[None, :], axis=1)

        if lower > 0:
            low_nm = lower * 1000.0
            Yl = np.log(bin_mids / low_nm) / (np.sqrt(2.0) * np.log(1.5))
            Frac_lower = 0.5 * (1.0 + np.vectorize(erf)(-Yl))
            lower_vals = np.nansum(dist * Frac_lower[None, :], axis=1)
            vals = upper_vals - lower_vals
        else:
            vals = upper_vals

        # Replace pure zeros with NaN for "empty" steps
        vals = np.where(vals == 0.0, np.nan, vals)
        # Validity mask from the original object (keeps alignment with self.time)
        series = self._ensure_data_robustness(vals)
        return series

    @staticmethod
    def _parse_px_metric_scalar(name_upper: str) -> tuple[str, float, float] | None:
        """Parse cumulative or band-limited Pₓ metrics (for example ``"PM4.2"``).

        This helper interprets metric names for cumulative or band-limited
        Pₓ metrics of the form

        - ``"PMx"``, ``"PNx"``, ``"PSx"``, ``"PVx"``  (cumulative 0 → x),
        - ``"PMa-b"``, ``"PNa-b"``, etc. (band-limited a → b),

        where ``x``, ``a`` and ``b`` are numeric diameters in µm. It
        returns the distribution-kind character and the lower and upper
        cut diameters in micrometres.

        Args:
            name_upper (str): Metric name in uppercase form. Non-Pₓ
                metrics such as ``"PNC"`` and ``"MASS"`` are safely
                ignored and return ``None``.

        Returns:
            tuple[str, float, float] | None: A tuple ``(dchar, lower, upper)``
            where

                - ``dchar`` is one of ``"M"``, ``"N"``, ``"S"``, ``"V"``,

                - ``upper`` is the upper cut diameter in µm,

                - ``lower`` is the lower cut diameter in µm (0.0 for
                  cumulative metrics),

            or ``None`` if the metric name is not a supported Pₓ form.

        Raises:
            ValueError: If the numeric part of the metric cannot be parsed
                as floats (malformed string), or if the resulting band
                does not satisfy ``0 <= lower < upper``.

        Notes:
            - A metric like ``"PM4.2"`` is treated as cumulative from
              0 → 4.2 µm (``lower = 0.0``).
            - Band metrics like ``"PM1-4.2"`` are returned with
              ``upper = 4.2`'', ``lower = 1.0`` and are required to satisfy
              ``0 <= lower < upper``.
        """
        if name_upper in {"PNC", "MASS", "MODE", "MEDIAN", "GMD"}:
            return None
        if not name_upper.startswith(("PM", "PN", "PS", "PV")):
            return None

        suffix = name_upper[2:]
        if "-" in suffix:
            low_str, high_str = suffix.split("-", 1)
            try:
                lower = float(low_str)
                upper = float(high_str)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse band limits from metric '{name_upper}'."
                ) from exc
        else:
            lower = 0.0
            try:
                upper = float(suffix)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse cutoff from metric '{name_upper}'."
                ) from exc

        if lower < 0 or lower >= upper:
            raise ValueError(
                f"Invalid band specification in metric '{name_upper}': "
                f"require 0 <= lower < upper."
            )

        return (
            {"PM": "M", "PN": "N", "PS": "S", "PV": "V"}[name_upper[:2]],
            upper,
            lower,
        )

    def PM_calc(self, dtype: str = "dM", PM: float = 4.2, lower_lim: float = 0):
        """Compute a size-selective Pₓ time series and store it in extra_data.

        Args:
            dtype (str): Base distribution type to integrate, one of
                "dN", "dS", "dV", "dM". Defaults to "dM" (mass-based).
            PM (float): Upper cut-off diameter in micrometres (µm) for the
                size-selective fraction (nominal 50% penetration point).
            lower_lim (float): Optional lower cut-off diameter in µm.
                If 0 (default), the result is cumulative from 0 → PM.
                If in (0, PM), the result represents the band-limited
                contribution from lower_lim → PM.

        Returns:
            Aerosol2D: self, with a new column added to extra_data named
                "P{X}{PM}" for cumulative metrics (for example "PM2.5",
                "PN10") or "P{X}{lower_lim}-{PM}" for band-limited ones
                (for example "PM1-5").

        Raises:
            ValueError: If lower_lim is greater than or equal to PM, or if
                lower_lim is negative. Check the order and magnitude of
                the cut diameters.
            ValueError: If dtype is not one of "dN", "dS", "dV", "dM".

        Notes:
            Detailed description:
                The method converts the internal distribution to the
                requested base kind if needed, ensures it is not in
                dx/dlogDp form, and then integrates it with an
                EN 481 / ISO 7708–style size-selective penetration curve
                between lower_lim and PM. The resulting time series is
                stored in extra_data with a canonical name that encodes
                both the distribution type and cut diameters.

            Theory:
                Pₓ metrics generalize well-known PM₁₀, PM₂.₅, etc., and
                can be defined for number (PN), surface (PS), volume (PV)
                and mass (PM). The underlying helper uses a standard
                lognormal penetration curve (GSD 1.5, 50% cut at PM) to
                approximate workplace sampling conventions (for example
                EN 481 / ISO 7708 respirable/inhalable fractions).

        Examples:
            Add PM₂.₅ and PN₁₀ series for later plotting and summary:

            .. code-block:: python

                elpi.PM_calc(dtype="dM", PM=2.5)   # PM2.5
                elpi.PM_calc(dtype="dN", PM=10.0)  # PN10
                elpi.extra_data[["PM2.5", "PN10"]].head()
        """

        if lower_lim >= PM:
            raise ValueError("lower_lim is larger than or equal to PM.")

        base_dtype = dtype.replace("/dlogDp", "")
        if base_dtype not in {"dN", "dS", "dV", "dM"}:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Expected one of 'dN', 'dS', 'dV', 'dM'."
            )

        dchar = base_dtype[-1].upper()  # 'N'/'S'/'V'/'M'

        # Canonical output label
        if lower_lim == 0:
            out_label = f"P{dchar}{PM:g}"
        else:
            out_label = f"P{dchar}{lower_lim:g}-{PM:g}"

        # Compute the series using the core helper
        series = self._px_fraction_series(dtype=base_dtype, upper=PM, lower=lower_lim)

        # Ensure extra_data is aligned to self.time and assign the new series
        if self._extra_data.empty:
            self._extra_data = pd.DataFrame(index=self.time)
        elif not self._extra_data.index.equals(self.time):
            self._extra_data = self._extra_data.reindex(self.time)

        self._extra_data[out_label] = series

        return self

    @override
    def _get_metric_series(self, metric_name: str) -> tuple[pd.Series, str]:
        """Return a time series and unit for a PNC/MASS/Pₓ metric.

        This helper provides a unified way of obtaining a 1D metric time
        series derived from the underlying 2D PSD together with its
        physical unit. It supports bulk metrics (PNC, MASS) and
        cumulative or band-limited Pₓ metrics (for example ``"PM4.2"``,
        ``"PM1-4.2"``, ``"PN10"``), using the same EN 481 / ISO 7708
        penetration curves as :meth:`PM_calc`.

        The function reuses existing Pₓ series in :attr:`extra_data` if
        present; otherwise it computes them on a working copy.

        Args:
            metric_name (str): Name of the requested metric, case-insensitive.
                Supported values include:

                    - ``"PNC"``: total number concentration.
                    - ``"MASS"``: total mass concentration.
                    - ``"PMx"``, ``"PNx"``, ``"PSx"``, ``"PVx"``:
                      cumulative Pₓ up to diameter x (µm).
                    - ``"PMa-b"``, ``"PNa-b"``, etc.: band-limited Pₓ
                      between diameters a and b (µm).

        Returns:
            tuple[pandas.Series, str]: A tuple ``(series, unit)`` where

                - ``series`` is a 1D time series indexed by :attr:`time`
                  containing the requested metric for each time step,
                - ``unit`` is the corresponding unit string, one of
                  ``"cm⁻³"``, ``"µg/m³"``, ``"nm²/cm³"``, ``"nm³/cm³"``.

        Raises:
            ValueError: If ``metric_name`` cannot be parsed as a supported
                metric string.
            ValueError: If a Pₓ metric is requested and, despite internal
                computation, the corresponding column is not found in
                :attr:`extra_data`.

        Notes:
            - For PNC, the helper converts to a number-based distribution
              and sums across all size bins.
            - For MASS, it converts to a mass-based distribution and sums
              across all size bins.
            - For Pₓ metrics, it uses :meth:`PM_calc` on a working copy,
              which in turn uses :meth:`_px_fraction_series` as its numeric
              core. This preserves the canonical naming and storage of Pₓ
              series in :attr:`extra_data`.
        """
        mu = metric_name.upper()

        # Ensure extra_data is aligned to the main time index before using it
        if self._extra_data.empty:
            aligned_extra = pd.DataFrame(index=self.time)
        elif not self._extra_data.index.equals(self.time):
            aligned_extra = self._extra_data.reindex(self.time)
        else:
            aligned_extra = self._extra_data

        # --- Bulk metrics: PNC and MASS, with optional reuse -------------------
        if mu == "PNC":
            if "PNC" in aligned_extra.columns:
                series = aligned_extra["PNC"].astype(float)
                self._extra_data = aligned_extra
                return series, "cm⁻³"

            # Fast path: convert array without deep-copying the object
            base_arr, _, _ = self._as_base_array()
            current_base = str(self.dtype).replace("/dlogDp", "")
            if current_base != "dN":
                number_arr = self._convert_array(
                    base_arr, self.bin_mids, current_base, "dN", self.density
                )
            else:
                number_arr = base_arr
            series = self._ensure_data_robustness(np.nansum(number_arr, axis=1))

            aligned_extra["PNC"] = series
            self._extra_data = aligned_extra
            return series, "cm⁻³"

        if mu == "MASS":
            if "MASS" in aligned_extra.columns:
                series = aligned_extra["MASS"].astype(float)
                self._extra_data = aligned_extra
                return series, "µg/m³"

            # Fast path: convert array without deep-copying the object
            base_arr, _, _ = self._as_base_array()
            current_base = str(self.dtype).replace("/dlogDp", "")
            if current_base != "dM":
                mass_arr = self._convert_array(
                    base_arr, self.bin_mids, current_base, "dM", self.density
                )
            else:
                mass_arr = base_arr
            series = self._ensure_data_robustness(np.nansum(mass_arr, axis=1))

            aligned_extra["MASS"] = series
            self._extra_data = aligned_extra
            return series, "µg/m³"

        # --- Pₓ metrics: PM, PN, PS, PV (reusing if already present) ----------
        parsed = self._parse_px_metric_scalar(mu)
        if parsed is None:
            raise ValueError(f"Unsupported metric string '{metric_name}'.")

        # dchar, lower_cut, upper_cut = parsed

        dchar, upper_cut, lower_cut = parsed
        dtype_map = {"M": "dM", "N": "dN", "S": "dS", "V": "dV"}
        unit_map = {
            "M": "µg/m³",
            "N": "cm⁻³",
            "S": "nm²/cm³",
            "V": "nm³/cm³",
        }

        # Canonical label used by PM_calc / _px_fraction_series
        if lower_cut <= 0:
            label = f"P{dchar}{upper_cut:g}"
        else:
            label = f"P{dchar}{lower_cut:g}-{upper_cut:g}"

        # Reuse existing series if already present
        if label in aligned_extra.columns:
            series = aligned_extra[label].astype(float)
            self._extra_data = aligned_extra
            return series, unit_map[dchar]

        # Otherwise compute on a working copy
        work = self.copy_self()
        if "/dlogDp" in str(work.dtype):
            work.unnormalize_logdp(inplace=True)
        work.dtype_converter(dtype=dtype_map[dchar], inplace=True)

        if lower_cut <= 0:
            work.PM_calc(dtype=dtype_map[dchar], PM=upper_cut)
        else:
            work.PM_calc(dtype=dtype_map[dchar], PM=upper_cut, lower_lim=lower_cut)

        # Align and store the result back on self for future reuse
        series = work.extra_data[label].astype(float)
        aligned_extra[label] = series
        self._extra_data = aligned_extra

        return series, unit_map[dchar]

    # snake_case alias for PEP 8 consistency
    pm_calc = PM_calc
