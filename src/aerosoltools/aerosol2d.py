from math import erf
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, cast

try:
    from typing import override  # Python 3.12+
except ImportError:  # Python ≤ 3.11
    from typing_extensions import override

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tabulate import tabulate
from scipy.optimize import curve_fit, least_squares

from .aerosol1d import Aerosol1D

params = {
    "legend.fontsize": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": (19, 10),
}
plt.rcParams.update(params)


class Aerosol2D(Aerosol1D):
    """
    A class for managing time-resolved, size-distributed aerosol data.

    This class extends `Aerosol1D` to handle datasets that contain particle
    size distributions (e.g., number, mass, or surface area concentration
    across particle size bins). It supports transformation between physical
    representations (dN, dS, dV, dW), visualization, activity segmentation,
    and summary statistics including PM values and particle size metrics.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame containing the data to load. The first column should
        contain time stamps or be the DataFrame index. The second column should
        be the total concentration. All remaining columns must represent
        concentration values in size bins with bin midpoints as column headers.

    Notes
    -----
    All data handling is done with `pandas`. Input DataFrames are expected to
    have particle size bin midpoints as column headers, and the class assumes
    these are numeric and represent diameters in nanometers.
    """

    def __init__(self, dataframe):
        super().__init__(dataframe)

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Particle size bin edges in nanometers.

        Returns:
            numpy.ndarray: One-dimensional array of bin edge diameters in
            nanometers (dtype ``float64``). Length is ``n + 1`` when there are
            ``n`` size bins. A copy is returned so callers cannot mutate the
            internal metadata.
        """
        # ensure an array of floats; .copy() so callers can't mutate your internal state
        return np.asarray(self._meta["bin_edges"], dtype=np.float64).copy()

    @property
    def bin_mids(self) -> NDArray[np.float64]:
        """Particle size bin midpoints in nanometers.

        Returns:
            numpy.ndarray: One-dimensional array of bin midpoint diameters in
            nanometers (dtype ``float64``). Length is ``n`` for ``n`` size
            bins. A copy is returned so callers cannot mutate the internal
            metadata.
        """
        return np.asarray(self._meta["bin_mids"], dtype=np.float64).copy()

    @property
    def density(self) -> float:
        """Assumed particle density in g/cm³.

        Returns:
            float: Particle density used for conversions between number, volume,
            surface area, and mass distributions. Falls back to the value
            stored in the metadata (typically set at load time or via
            :meth:`set_density`). Defaults to 1.0 g/cm³ if not explicitly set in
            metadata.
        """
        return float(self._meta.get("density", 1.0))

    @property
    def metadata(self) -> dict:
        """Metadata associated with the size-resolved dataset.

        Returns:
            dict: Dictionary of metadata extracted or defined for this object,
            including bin edges/mids, units, data type (dN/dS/dV/dM),
            instrument information, density, and any additional fields stored
            in ``self._meta``.
        """
        return self._meta

    @property
    def size_data(self) -> pd.DataFrame:
        """Size-bin concentration data.

        Returns:
            pandas.DataFrame: Subset of :attr:`data` containing only the
            columns that represent size-resolved concentration values, ordered
            according to :attr:`bin_mids` (via :attr:`_sizebin_headers`). Each
            column corresponds to a size bin, and each row to a time stamp.
        """
        return self.data.loc[:, self._sizebin_headers]

    @property
    def _sizebin_headers(self) -> list[str]:
        """Column labels for size-bin concentration data.

        Returns:
            list[str]: Column names used in :attr:`data` for the size-bin
            distribution, derived from :attr:`bin_mids` (converted to strings).
            These headers define which columns are treated as the size
            distribution in methods such as :meth:`convert_to_number_concentration`,
            :meth:`convert_to_mass_concentration`, and plotting utilities.
        """
        return [str(x) for x in self.bin_mids]

    ###########################################################################
    """###################### Private Helper Functions #####################"""
    ###########################################################################

    def _as_base_array(self) -> tuple[NDArray[np.float64], list[str], bool]:
        """Extract the size-bin data in base (non-/dlogDp) form.

        This helper centralizes the logic needed to work in *physical* (base)
        units, independent of whether the data are stored as ``*/dlogDp`` or
        not. It performs the following steps:

        * Selects the size-bin columns from :attr:`data` using
          :attr:`_sizebin_headers`.
        * Converts them to a dense ``float64`` numpy array of shape
          ``(n_times, n_bins)``.
        * If the current :attr:`dtype` string contains ``"/dlogDp"``, the array
          is multiplied by Δlog₁₀(Dp) (via :meth:`_dlogdp`) to undo the
          normalization and return the underlying base distribution.

        Returns:
            tuple: ``(base_array, headers, was_norm)`` where

                * ``base_array`` (:class:`numpy.ndarray`): Size-bin data in
                  base units (e.g. dN, dS, dV, dM), shape
                  ``(n_times, n_bins)``.
                * ``headers`` (list[str]): Column labels for the size-bin
                  distribution in :attr:`data`.
                * ``was_norm`` (bool): ``True`` if the original data were in
                  ``*/dlogDp`` form, i.e. the :attr:`dtype` string contained
                  ``"/dlogDp"``; ``False`` otherwise.

        Raises:
            ValueError: If the length of the Δlog₁₀(Dp) vector does not match
                the number of size bins (columns).
        """
        headers = self._sizebin_headers
        arr = self._data[headers].to_numpy(dtype=float, copy=True)
        was_norm = "/dlogDp" in str(self.dtype)

        if was_norm:
            dlogdp = self._dlogdp()
            if dlogdp.shape[0] != arr.shape[1]:
                raise ValueError(
                    "Mismatch between Δlog₁₀(Dp) array and number of size bins."
                )
            arr = arr * dlogdp[None, :]

        return arr.astype(np.float64, copy=False), headers, was_norm


    def _ensure_data_robustness(self, vals) -> pd.Series:
        """Validity mask from the original object (keeps alignment with self.time)

        This returns a cleaned series, so that no new data is generated,
        where before the total_conc was NaN.
        Args:
            vals (np.array):
                array of data structured as a column of data from either data
                extra data.
        Returns:
            pd.Series: Time series of the requested Pₓ metric, indexed by
            :attr:`time`. Empty or invalid time steps (where
            :attr:`total_concentration` is NaN) are returned as NaN.
        """

        valid_mask = self.total_concentration.notna()
        series = pd.Series(vals, index=self.time)

        return series.where(valid_mask, np.nan)

    ###########################################################################

    @staticmethod
    def _convert_array(
        arr: NDArray[np.float64],
        bin_mids: NDArray[np.float64],
        from_dtype: str,
        to_dtype: str,
        density: float = 1.0,
    ) -> NDArray[np.float64]:
        """Convert a (n_times × n_bins) size-distribution array between dtype bases.

        Performs the same physical transformation as the ``_convert_to_*``
        instance methods but operates entirely on raw NumPy arrays, avoiding a
        full :class:`Aerosol2D` deep-copy.

        Args:
            arr: 2-D float64 array of shape ``(n_times, n_bins)``.
            bin_mids: 1-D array of bin midpoint diameters in nm.
            from_dtype: Source base dtype — one of ``"dN"``, ``"dS"``,
                ``"dV"``, ``"dM"``. A ``"/dlogDp"`` suffix is stripped.
            to_dtype: Target base dtype (same choices).
            density: Particle density in g/cm³ (used when mass is involved).

        Returns:
            numpy.ndarray: Converted array with the same shape as ``arr``.
        """
        from_base = from_dtype.replace("/dlogDp", "")
        to_base = to_dtype.replace("/dlogDp", "")
        if from_base == to_base:
            return arr

        bin_radii = bin_mids / 2.0  # nm
        volume_per_particle = (4.0 / 3.0) * np.pi * bin_radii**3  # nm³
        surface_area_per_particle = 4.0 * np.pi * bin_radii**2  # nm²

        # Step 1 — normalise to dN
        if from_base == "dN":
            number_arr = arr
        elif from_base == "dM":
            number_arr = (arr / density * 1e9) / volume_per_particle[None, :]
        elif from_base == "dV":
            number_arr = arr / volume_per_particle[None, :]
        elif from_base == "dS":
            number_arr = arr / surface_area_per_particle[None, :]
        else:
            raise ValueError(f"Unknown from_dtype {from_base!r}.")

        # Step 2 — convert from dN to target
        if to_base == "dN":
            return number_arr
        elif to_base == "dM":
            return number_arr * volume_per_particle[None, :] * density * 1e-9
        elif to_base == "dV":
            return number_arr * volume_per_particle[None, :]
        elif to_base == "dS":
            return number_arr * surface_area_per_particle[None, :]
        else:
            raise ValueError(f"Unknown to_dtype {to_base!r}.")

    ###########################################################################

    def _dlogdp(self) -> NDArray[np.float64]:
        """Compute bin widths in log₁₀(Dp) space.

        This helper derives the logarithmic bin widths Δlog₁₀(Dp) from
        :attr:`bin_edges`. It is the fundamental quantity used when converting
        between base distributions (e.g. dN, dM) and their log-diameter–
        normalized counterparts (e.g. dN/dlogDp).

        Returns:
            numpy.ndarray: One-dimensional float64 array of Δlog₁₀(Dp) values
            with length ``n`` for ``n`` size bins.

        Raises:
            ValueError: If :attr:`bin_edges` does not contain at least two
                elements or is not one-dimensional.
        """
        be = np.asarray(self.bin_edges, dtype=float)
        if be.ndim != 1 or be.size < 2:
            raise ValueError(
                "bin_edges must be a one-dimensional array of length >= 2 to "
                "compute Δlog₁₀(Dp)."
            )
        return np.diff(np.log10(be)).astype(np.float64)

    ###########################################################################

    def _emit(
        self,
        base_arr: NDArray[np.float64],
        headers: list[str],
        was_norm: bool,
        unit: str,
        dtype_base: str,
        inplace: bool = True,
    ) -> "Aerosol2D":
        """Write converted data back to :attr:`data`, with optional /dlogDp.

        This helper takes a size distribution expressed in **base** form
        (e.g. dN, dS, dV, dM; *not* divided by Δlog₁₀(Dp)) and writes it back to
        the internal :attr:`data` frame. If the original data were in
        ``*/dlogDp`` form (``was_norm=True``), the method re-applies the
        normalization before storing the values, so that the representation
        (normalized vs. non-normalized) is preserved across conversions.

        Regardless of normalization, the ``Total_conc`` column is recomputed
        as the sum over the supplied base distribution, ensuring that the total
        reflects the **true physical** distribution rather than the normalized
        one.

        Args:
            base_arr: 2D float64 array of shape ``(n_times, n_bins)``
                containing the converted size distribution in base units,
                e.g. dN, dS, dV, or dM.
            headers: Column labels corresponding to the size-bin data in
                :attr:`data`.
            was_norm: Whether the input object was originally stored as
                ``*/dlogDp`` (``True``) or not (``False``). If ``True``,
                ``base_arr`` will be divided by Δlog₁₀(Dp) prior to storage
                and the resulting :attr:`dtype` will have ``"/dlogDp"`` appended.
            unit: Unit string to write to ``metadata["unit"]``.
            dtype_base: Base dtype string (e.g. ``"dN"``, ``"dS"``, ``"dV"``,
                ``"dM"``) to write to ``metadata["dtype"]``. The suffix
                ``"/dlogDp"`` is added automatically when ``was_norm`` is
                ``True``.
            inplace: If ``True`` (default), modify this instance in-place and
                return it. If ``False``, operate on a deep copy and return the
                new object, leaving the original unchanged.

        Returns:
            Aerosol2D: The updated object (``self`` or a new copy) with

            * size-bin data in :attr:`data` updated (in base or ``*/dlogDp``
              form, consistent with the original),
            * :attr:`metadata["unit"]` and :attr:`metadata["dtype"]` updated,
            * :attr:`_TOTAL_COL` (``"Total_conc"``) recomputed from the base
              distribution.

        Raises:
            ValueError: If ``base_arr`` is not 2D or its second dimension does
                not match the number of provided ``headers``. Also raised if
                Δlog₁₀(Dp) does not match the number of bins when
                re-normalizing.
        """
        if base_arr.ndim != 2:
            raise ValueError("base_arr must be a 2D array (n_times × n_bins).")
        if base_arr.shape[1] != len(headers):
            raise ValueError(
                "Number of columns in base_arr does not match number of headers."
            )

        out = self if inplace else self.copy_self()

        if was_norm:
            dlogdp = self._dlogdp()
            if dlogdp.shape[0] != base_arr.shape[1]:
                raise ValueError(
                    "Mismatch between Δlog₁₀(Dp) array and number of size bins."
                )
            arr_out = base_arr / dlogdp[None, :]
            out._meta["dtype"] = f"{dtype_base}/dlogDp"
        else:
            arr_out = base_arr
            out._meta["dtype"] = dtype_base

        out._meta["unit"] = unit
        out._data[headers] = pd.DataFrame(
            arr_out, index=out._data.index, columns=headers
        )

        # Total_conc is always computed from the base (non-/dlogDp) distribution
        new_total = np.nansum(base_arr, axis=1)

        series = self._ensure_data_robustness(new_total)

        out._data["Total_conc"] = series

        return out

    ###########################################################################

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

    # ------------------------------------------------------------------
    # Shared helpers for dtype_converter
    # ------------------------------------------------------------------

    def _convert_to_mass_concentration(self, inplace: bool = True):
        """Convert size distribution to mass concentration (dM).

        Converts the current size-resolved distribution to a **mass-based**
        distribution (``dM``) using the particle density and assuming spherical
        particles. The resulting size-bin data are stored in the main data
        frame and the metadata are updated:

        * ``dtype`` → ``"dM"`` (or ``"dM/dlogDp"`` if the input was stored as
          ``*/dlogDp``),
        * ``unit`` → ``"µg/m³"``.

        Internally, the conversion is always performed on an *unnormalized*
        base distribution (e.g. dN, dS, dV), even if the data are stored as
        ``*/dlogDp``. If the original representation was normalized,
        Δlog₁₀(Dp) is removed before conversion and re-applied when writing the
        converted data back. The ``Total_conc`` column is always computed from
        the underlying base mass distribution.

        Depending on the current data type (:attr:`dtype`), the following
        transformations are applied per size bin (using bin radius in nm):

        * ``"dN"`` → number → volume → mass
        * ``"dS"`` → surface area → number → volume → mass
        * ``"dV"`` → volume → mass

        If the data are already mass-based (``"dM"`` in :attr:`dtype`), the
        method returns either ``self`` or a deep copy without modification.

        Args:
            inplace: If ``True`` (default), modify the current instance and
                return it. If ``False``, perform the conversion on a deep copy
                and return the new instance.

        Returns:
            Aerosol2D: The object containing a mass-based size distribution and
            updated total mass concentration. This is ``self`` when
            ``inplace=True``, otherwise a new instance.

        Raises:
            ValueError: If :attr:`dtype` does not contain one of ``"dN"``,
                ``"dS"``, or ``"dV"`` and does not already indicate mass.

        Notes:
            * The conversion assumes spherical particles and uses
              :attr:`density` (in g/cm³).
            * The conversion internally works in nm and nm³; the factor
              ``1e-9`` is used to obtain units consistent with µg/m³, matching
              the rest of this module.
        """
        current_dtype = str(self.dtype)
        if "dM" in current_dtype:
            return self if inplace else self.copy_self()

        base_arr, headers, was_norm = self._as_base_array()
        bin_radii = self.bin_mids / 2.0  # nm

        if "dS" in current_dtype:
            # Surface area -> Number -> Volume -> Mass
            surface_area_per_particle = 4.0 * np.pi * bin_radii**2  # nm²
            number_distribution = base_arr / surface_area_per_particle[None, :]

            volume_per_particle = (4.0 / 3.0) * np.pi * bin_radii**3  # nm³
            volume_distribution = number_distribution * volume_per_particle[None, :]

            mass_distribution = volume_distribution * self.density * 1e-9

        elif "dV" in current_dtype:
            # Volume -> Mass (direct)
            mass_distribution = base_arr * self.density * 1e-9

        elif "dN" in current_dtype:
            # Number -> Volume -> Mass
            volume_per_particle = (4.0 / 3.0) * np.pi * bin_radii**3  # nm³
            volume_distribution = base_arr * volume_per_particle[None, :]

            mass_distribution = volume_distribution * self.density * 1e-9

        else:
            raise ValueError("Unknown data type for conversion to mass.")

        return self._emit(
            base_arr=mass_distribution,
            headers=headers,
            was_norm=was_norm,
            unit="µg/m³",
            dtype_base="dM",
            inplace=inplace,
        )

    ###########################################################################

    def _convert_to_number_concentration(self, inplace: bool = True):
        """Convert size distribution to number concentration (dN).

        Converts the current size-resolved distribution to a **number-based**
        distribution (``dN``). The resulting size-bin data are stored in the
        main data frame and the metadata are updated:

        * ``dtype`` → ``"dN"`` (or ``"dN/dlogDp"`` if the input was stored as
          ``*/dlogDp``),
        * ``unit`` → ``"cm⁻³"``.

        Internally, the conversion is always performed on an unnormalized base
        distribution (not divided by Δlog₁₀(Dp)), and any prior
        log-diameter normalization is re-applied on output if needed. The
        ``Total_conc`` column is computed from the base number distribution.

        Depending on the current data type (:attr:`dtype`), the following
        transformations are applied per size bin (using bin radius in nm):

        * ``"dV"`` → volume → number
        * ``"dM"`` → mass → volume → number
        * ``"dS"`` → surface area → number

        If the data are already number-based (``"dN"`` in :attr:`dtype`), the
        method returns either ``self`` or a deep copy without modification.

        Args:
            inplace: If ``True`` (default), modify the current instance and
                return it. If ``False``, perform the conversion on a deep copy
                and return the new instance.

        Returns:
            Aerosol2D: The object containing a number-based size distribution
            and updated total number concentration. This is ``self`` when
            ``inplace=True``, otherwise a new instance.

        Raises:
            ValueError: If :attr:`dtype` does not contain one of ``"dV"``,
                ``"dM"``, or ``"dS"`` and does not already indicate number
                concentration.

        Notes:
            * Conversions that involve volume or mass assume spherical
              particles and use :attr:`density` for mass–volume relationships.
        """
        current_dtype = str(self.dtype)
        if "dN" in current_dtype:
            return self if inplace else self.copy_self()

        base_arr, headers, was_norm = self._as_base_array()
        bin_radii = self.bin_mids / 2.0  # nm

        volume_per_particle = (4.0 / 3.0) * np.pi * bin_radii**3  # nm³
        surface_area_per_particle = 4.0 * np.pi * bin_radii**2  # nm²

        if "dV" in current_dtype:
            # Volume -> Number
            number_distribution = base_arr / volume_per_particle[None, :]

        elif "dM" in current_dtype:
            # Mass -> Volume -> Number
            volume_distribution = base_arr / self.density * 1e9  # nm³/cm³
            number_distribution = volume_distribution / volume_per_particle[None, :]

        elif "dS" in current_dtype:
            # Surface Area -> Number
            number_distribution = base_arr / surface_area_per_particle[None, :]

        else:
            raise ValueError("Unknown data type for conversion to number.")

        return self._emit(
            base_arr=number_distribution,
            headers=headers,
            was_norm=was_norm,
            unit="cm⁻³",
            dtype_base="dN",
            inplace=inplace,
        )

    ###########################################################################

    def _convert_to_surface_concentration(self, inplace: bool = True):
        """Convert size distribution to surface area concentration (dS).

        Converts the current size-resolved distribution to a **surface-area–
        based** distribution (``dS``) expressed as nm²/cm³. The resulting
        size-bin data are stored in the main data frame and the metadata are
        updated:

        - ``dtype`` → ``"dS"`` (or ``"dS/dlogDp"`` if the input was stored as
          ``*/dlogDp``),
        - ``unit`` → ``"nm²/cm³"``.

        The conversion is performed on an unnormalized base distribution and
        then, if the original data were in ``*/dlogDp`` form, the result is
        re-normalized before being written back. ``Total_conc`` is always
        computed from the base surface-area distribution.

        Depending on the current data type (:attr:`dtype`), the following
        transformations are applied per size bin (using bin radius in nm):

        - ``"dV"`` → volume → number → surface area
        - ``"dM"`` → mass → volume → number → surface area
        - ``"dN"`` → number → surface area

        If the data are already surface-area–based (``"dS"`` in
        :attr:`dtype`), the method returns either ``self`` or a deep copy
        without modification.

        Args:
            inplace: If ``True`` (default), modify the current instance and
                return it. If ``False``, perform the conversion on a deep copy
                and return the new instance.

        Returns:
            Aerosol2D: The object containing a surface-area–based size
            distribution and updated total surface area concentration. This is
            ``self`` when ``inplace=True``, otherwise a new instance.

        Raises:
            ValueError: If :attr:`dtype` does not contain one of ``"dV"``,
                ``"dM"``, or ``"dN"`` and does not already indicate surface
                area.

        Notes:
            * Conversions that involve volume or mass assume spherical
              particles and use :attr:`density` where mass is involved.
        """
        current_dtype = str(self.dtype)
        if "dS" in current_dtype:
            return self if inplace else self.copy_self()

        base_arr, headers, was_norm = self._as_base_array()
        bin_radii = self.bin_mids / 2.0  # nm

        surface_area_per_particle = 4.0 * np.pi * bin_radii**2  # nm²
        volume_per_particle = (4.0 / 3.0) * np.pi * bin_radii**3  # nm³

        if "dV" in current_dtype:
            # Volume -> Number -> Surface Area
            number_distribution = base_arr / volume_per_particle[None, :]
            surface_area_distribution = (
                number_distribution * surface_area_per_particle[None, :]
            )

        elif "dM" in current_dtype:
            # Mass -> Volume -> Number -> Surface Area
            volume_distribution = base_arr / self.density * 1e9  # nm³/cm³
            number_distribution = volume_distribution / volume_per_particle[None, :]
            surface_area_distribution = (
                number_distribution * surface_area_per_particle[None, :]
            )

        elif "dN" in current_dtype:
            # Number -> Surface Area
            surface_area_distribution = base_arr * surface_area_per_particle[None, :]

        else:
            raise ValueError("Unknown data type for conversion to surface area.")

        return self._emit(
            base_arr=surface_area_distribution,
            headers=headers,
            was_norm=was_norm,
            unit="nm²/cm³",
            dtype_base="dS",
            inplace=inplace,
        )

    ###########################################################################

    def _convert_to_volume_concentration(self, inplace: bool = True):
        """Convert size distribution to volume concentration (dV).

        Converts the current size-resolved distribution to a **volume-based**
        distribution (``dV``) expressed as nm³/cm³. The resulting size-bin data
        are stored in the main data frame and the metadata are updated:

        - ``dtype`` → ``"dV"`` (or ``"dV/dlogDp"`` if the input was stored as
          ``*/dlogDp``),
        - ``unit`` → ``"nm³/cm³"``.

        The conversion operates on an unnormalized base distribution and then,
        if the original data were in ``*/dlogDp`` form, re-applies the
        normalization before writing back. ``Total_conc`` is computed from the
        base volume distribution.

        Depending on the current data type (:attr:`dtype`), the following
        transformations are applied per size bin (using bin radius in nm):

        - ``"dS"`` → surface area → number → volume
        - ``"dM"`` → mass → volume
        - ``"dN"`` → number → volume

        If the data are already volume-based (``"dV"`` in :attr:`dtype`), the
        method returns either ``self`` or a deep copy without modification.

        Args:
            inplace: If ``True`` (default), modify the current instance and
                return it. If ``False``, perform the conversion on a deep copy
                and return the new instance.

        Returns:
            Aerosol2D: The object containing a volume-based size distribution
            and updated total volume concentration. This is ``self`` when
            ``inplace=True``, otherwise a new instance.

        Raises:
            ValueError: If :attr:`dtype` does not contain one of ``"dS"``,
                ``"dM"``, or ``"dN"`` and does not already indicate volume.

        Notes:
            Conversions that involve number, surface, or mass assume
            spherical particles and use :attr:`density` where mass is
            involved.
        """
        current_dtype = str(self.dtype)
        if "dV" in current_dtype:
            return self if inplace else self.copy_self()

        base_arr, headers, was_norm = self._as_base_array()
        bin_radii = self.bin_mids / 2.0  # nm

        volume_per_particle = (4.0 / 3.0) * np.pi * bin_radii**3  # nm³
        surface_area_per_particle = 4.0 * np.pi * bin_radii**2  # nm²

        if "dS" in current_dtype:
            # Surface Area -> Number -> Volume
            number_distribution = base_arr / surface_area_per_particle[None, :]
            volume_distribution = number_distribution * volume_per_particle[None, :]

        elif "dM" in current_dtype:
            # Mass -> Volume
            volume_distribution = base_arr / self.density * 1e9  # nm³/cm³

        elif "dN" in current_dtype:
            # Number -> Volume
            volume_distribution = base_arr * volume_per_particle[None, :]

        else:
            raise ValueError("Unknown data type for conversion to volume.")

        return self._emit(
            base_arr=volume_distribution,
            headers=headers,
            was_norm=was_norm,
            unit="nm³/cm³",
            dtype_base="dV",
            inplace=inplace,
        )

    # ------------------------------------------------------------------
    # Shared helpers for summarize_activities / summarize_exposure
    # ------------------------------------------------------------------

    ###########################################################################

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

    ###########################################################################

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

    ###########################################################################
    """############################# Functions #############################"""

    ###########################################################################
    @override
    def calibrate(self, parameter: str | int = 'bins',
                  fit_function = None,
                  Variables: dict = {'m': 1},
                  inplace: bool = True):
        #m: Union[int, float, list] = 1, b: Union[int, float, list] = 0, inplace: bool = True):
        """
        Apply a correction to the total conc and mark the data as calibrated
        by a linear function. The calibration value is applied to the size data.

        Args:
            parameter (int | str, optional):
                Index or column name of the signal to plot. If ``int``, it is
                interpreted as a positional index into :attr:`data.columns`. If
                ``str``, it is treated as a column label. Defaults to ``0``.
                If 'bin' is chosen, the calibration will go through each size bin,
                and apply the calibration function.
            fit_function (function):
                A defined function to apply to the calibration.
                If none is chosen, an assumed linear calibration is used using y = m*x +b
            Variables (dict):
                The calibration value to be multiplied to the data for correction.
                If m is provided as a list, it should be of equal length to the number
                of bins. The total concentration is then recalculated as the sum.
            inplace (bool): If ``True`` (default), modify the current instance and
                return it. If ``False``, perform the conversion on a deep copy
                and return the new instance.

        Returns:
            out (Aerosold2D):
                If inplace ``True`` the calibration function applies the calibration
                to the acted upon dataset, if inplace ``False`` a copy of the calibrated
                dataset is returned. 
                In addition to the data with the applied calibration, a 
        None

        """
        if fit_function is None:
            def fit_function(x,m,b=0):
                #Calculates a first order equation.
                return m*x + b
        
        values = list(Variables.values())
        all_lists = all(isinstance(v, list) for v in values)
        all_scalars = all(not isinstance(v, list) for v in values)
        
        if not (all_lists or all_scalars):
            raise ValueError("Variables must contain either all lists or all scalars")
        
        out = self if inplace else self.copy_self()

        # Resolve which column to use based on the requested parameter.
        if isinstance(parameter, int):
            if parameter >= len(self.data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter = self.data.columns[parameter]
        elif isinstance(parameter, str):
            if parameter != 'bins' and parameter not in out._data and parameter not in out._extra_data:
                raise LookupError(f"Chosen parameter '{parameter}' is invalid")
        else:
            raise LookupError("Chosen parameter is invalid")

        # Apply the correction to the chosen parameter
        if parameter != 'bins' and all_lists:
            raise ValueError("List-valued Variables are only supported when parameter='bins'")
        
        elif parameter == 'bins':

            if all_lists:
                lengths = [len(v) for v in values]
                if len(set(lengths)) != 1:
                    raise ValueError("All parameter lists must have same length")
        
                if lengths[0] != len(out._sizebin_headers):
                    raise ValueError("Parameter length must match number of bins")
        
                for i, params in enumerate(zip(*values)):
                    kwargs_i = dict(zip(Variables.keys(), params))
                    header = out._sizebin_headers[i]
                    out._data[header] = self._ensure_data_robustness(
                        fit_function(out.data[header], **kwargs_i)
                    )
            else:
                for header in out._sizebin_headers:
                    out._data[header] = self._ensure_data_robustness(
                        fit_function(out.data[header], **Variables)
                    )
        
            out_sum = out.data[out._sizebin_headers].sum(axis=1)
            out._data['Total_conc'] = self._ensure_data_robustness(out_sum)
        else: 
            if parameter in out._data:
                out._data[parameter]=self._ensure_data_robustness(fit_function(out.data[parameter],**Variables))
            elif parameter in out._extra_data:
                out._extra_data[parameter]=self._ensure_data_robustness(fit_function(out._extra_data[parameter],**Variables))
            else:
                raise KeyError(f"Parameter '{parameter}' not found in data or extra_data")
        
        if 'calibrated' not in out._meta:
            out._meta['calibrated'] = {}

        out._meta['calibrated'][parameter]=Variables.copy()

        return out
    
    ###########################################################################

    def dtype_converter(self, dtype: str = "dN", inplace: bool = True):
        """Description:
            Convert the size distribution to a chosen base data type.

        Args:
            dtype (str): Target data type string, one of "dN", "dS",
                "dV", or "dM" (case-sensitive). "dN" is number-based,
                "dS" surface-area–based, "dV" volume-based, and "dM"
                mass-based.
            inplace (bool): If True, convert this object in place and
                return it. If False, perform the conversion on a deep
                copy and return the new instance.

        Returns:
            Aerosol2D: Object whose size-bin data, total_conc, dtype and
                unit fields have been converted to the requested type
                (self when inplace=True, otherwise a new instance).

        Raises:
            ValueError: If dtype is not one of "dN", "dS", "dV", "dM".
                Check the spelling and letter case of the requested type.

        Notes:
            Detailed description:
                The method converts the current size-resolved distribution
                between number, surface, volume and mass representations
                using the stored particle density and the bin midpoints.
                Any existing normalization by dlogDp is preserved: data
                stored as dx/dlogDp remain in dx/dlogDp form after
                conversion, and Total_conc is recomputed from the
                underlying base distribution.

            Theory:
                Conversions assume spherical particles and use the usual
                geometric relationships between radius, surface area,
                volume and mass (with the density given in g/cm³).
                Number-based distributions can be transformed into volume
                or mass by multiplying with per-particle volume and
                density; surface-area distributions scale similarly via
                4πr².

        Examples:
            Convert a number distribution to mass concentration for
            comparison with gravimetric limits:

            .. code-block:: python

                elpi.dtype_converter("dM")
                elpi.plot_psd()
        """

        if dtype == "dN":
            return self._convert_to_number_concentration(inplace)
        elif dtype == "dS":
            return self._convert_to_surface_concentration(inplace)
        elif dtype == "dV":
            return self._convert_to_volume_concentration(inplace)
        elif dtype == "dM":
            return self._convert_to_mass_concentration(inplace)
        else:
            raise ValueError(
                f"Unknown target dtype {dtype!r}. Expected one of 'dN', 'dS', 'dV', 'dM'."
            )

    ###########################################################################

    def correct_diffusion_losses(
        self,
        D_tube: float,
        L: float,
        Q: float,
        T: float = 293,
        P: float = 101300,
        inplace: bool = True,
    ):
        """Description:
            Correct size distributions for diffusion losses in sampling tubes.

        Args:
            D_tube (float): Inner diameter of the sampling tube in metres.
            L (float): Length of the sampling tube in metres.
            Q (float): Volumetric flow through the tube in L/min.
            T (float): Gas temperature in Kelvin. Defaults to 293 K.
            P (float): Gas pressure in Pascal. Defaults to 101300 Pa.
            inplace (bool): If True, apply the correction to this object
                and return it. If False, perform the correction on a deep
                copy and return the new instance.

        Returns:
            Aerosol2D: Object with diffusion-loss–corrected size-bin data
                and updated Total_conc (self when inplace=True, otherwise
                a new instance).

        Raises:
            None: The method does not explicitly raise custom exceptions,
                but non-physical values (for example Q or D_tube close to
                zero) can lead to infinities or NaNs in the correction
                factors. Always use positive, realistic geometry and flow
                parameters.

        Notes:
            Detailed description:
                For each size bin, a transmission efficiency between the
                tube inlet and outlet is computed based on geometry,
                volumetric flow and particle diffusivity. The recorded
                distribution is divided by this efficiency to estimate the
                upstream concentration, and total_conc is recomputed from
                the corrected bins. The size-dependent efficiency curve
                and a flag indicating that diffusion correction has been
                applied are stored in metadata.

            Theory:
                The correction builds on classical mass-transfer
                correlations in straight circular tubes. Particle
                diffusivity is estimated via the Stokes–Einstein relation
                with a Cunningham slip correction, Reynolds and Schmidt
                numbers describe the flow, and a Sherwood number
                correlation is used to obtain the mass transfer
                coefficient. The residence parameter and Sherwood number
                define the deposition loss and thus the transmission
                efficiency per size.

        Examples:
            Correct ELPI or SMPS data for diffusion losses in a long
            sampling line:

            .. code-block:: python

                elpi.correct_diffusion_losses(
                    D_tube=0.004,  # 4 mm ID
                    L=2.0,        # 2 m tube
                    Q=10.0,       # 10 L/min
                )
                elpi.plot_psd()
        """

        # --- Geometry and flow definition ------------------------------------
        k = 1.380649e-23  # Boltzmann constant (J/K)
        Dp = np.array(self.bin_mids) * 1e-9  # particle diameters (m)
        Q_m3s = Q / (1000 * 60)  # volumetric flow (m³/s) from L/min
        A = 0.25 * np.pi * D_tube**2  # tube cross-sectional area (m²)
        V = Q_m3s / A  # average flow velocity (m/s)

        # --- Gas properties and mean free path (T, P dependent) --------------
        mfp_std = 66.5e-9  # reference mean free path (m)
        mfp = (
            mfp_std * (101e3 / P) * (T / 293.15) * ((1 + 110 / 293.15) / (1 + 110 / T))
        )

        eta_std = 1.708e-5  # reference dynamic viscosity (Pa·s)
        eta = (
            eta_std * (T / 273.15) ** 1.5 * (393.396 / (T + 120.246))
        )  # dynamic viscosity at (T, P)
        rho = 1.293 * (273.15 / T) * (P / 101300)  # gas density (kg/m³)

        # --- Dimensionless groups: slip, Re, diffusivity, Sc, xi -------------
        Kn = 2 * mfp / Dp  # Knudsen number
        Cc = 1 + Kn * (1.142 + 0.558 * np.exp(-0.999 / Kn))  # slip correction

        Re = rho * V * D_tube / eta  # Reynolds number

        Dc = (
            k * T * Cc / (3 * np.pi * eta * Dp)
        )  # particle diffusion coefficient (m²/s)
        Sc = eta / (rho * Dc)  # Schmidt number
        xi = np.pi * Dc * L / Q_m3s  # dimensionless residence parameter

        # --- Sherwood number (mass transfer) and diffusion efficiency --------
        if Re < 2000:
            # Laminar tube flow correlation
            Sh = 3.66 + 0.2672 / (xi + 0.10079 * xi ** (1 / 3))
        else:
            # Turbulent correlation: Re and Sc dependent
            Sh = 0.0118 * Re ** (7 / 8) * Sc ** (1 / 3)

        eff = np.exp(-Sh * xi)  # size-dependent transmission efficiency

        # --- Apply correction to size bins and update total concentration ----
        corrected = self.copy_self() if not inplace else self
        size_cols = corrected._sizebin_headers
        corrected._data[size_cols] = corrected._data[size_cols].div(eff, axis=1)
        corrected._data["Total_conc"] = corrected._data[size_cols].sum(axis=1)

        # --- Store efficiency and flag in metadata ---------------------------
        corrected._meta["diffusion_efficiency"] = eff.tolist()
        corrected._meta["diffusion_loss_corrected"] = True

        return corrected
    
    ###########################################################################

    def fit_psd(self, period = 'All data',
                      mu=[150],
                      sigma=[2],
                      factor=[1000],
                      log_scaling=True,
                      binding=None,
                      tolerance=10.0,
                      mu_factor=None,
                      weighting="variance"):
        """
        A function to fit one or multiple peaks following lognormal distribution,
        with the option for tethering values to set values. 
        
        An example would be an OPS dataset, with a pronounced shoulder from a mode 
        below its diameter range. A guess for mu1 could then be 100nm,
        which can be bound by providing the "binding" list of [True]
        
        Parameters
        ----------
        mu : list of floats, optinonal
            If specified, acts as the initial guess of the particle modes, meaning
            the size where the particle size distribution peaks.
            The default is 150, but more modes can be added to the list.
        sigma : list of floats, optional
            Initial guess for the geometric standard deviation factor. A good guess
            is the size at peak height divided by the size at 2/3 peak height in
            the decending direction. E.g. the PSD peaks at 200 nm and is at 2/3 
            height at 140 nm, so the sigma_guess parameter should be 200/140 = 1.4.
            The default is 2, but more modes can be added to the list. 
        factor : list of floats, optional
            Initial guess for the parameter used to scale the lognormal distribution.
            Getting a good estimate can be difficult, but a guess in the same order
            of magnitude as the peak height, is a good start. 
            The default is 1000, but more modes can be added to the list.
        log_scaling: boolean, optional
            Value to designate whether the fit should be done against log10 data, 
            or the regular values. Using true values run the risk of larger modes
            dominating the fit, potentially lossing structure for low populated 
            modes. Default is True.
        sort: str, optional
            Value to designate whether the reported modes should be structured from
            smallest to largest diameter or from most to least populated mode,
            with the designations "Diameter" or "Number" respectively. Default is 
            "Diameter" 
        binding: bool list, optional
            A boolean list for each parameter whether they must be bound or not.
            If True the tolerance limit is put  on the bound value(s).
            The list has the following association:
            (mu1, sigma1, factor1, mu2, sigma2, factor2....factorN)
            The list only needs to be filled up, to the last True value.
            Default is 0 with no bound values.        
        tolerance: float, optional
            Percentage value around which the bound values can be fitted
        mu_factor: float, optional
            If given (and > 1), constrain each fitted peak diameter to the
            window ``[mu0 / mu_factor, mu0 * mu_factor]`` around its initial
            guess ``mu0``. This refines the supplied modes locally instead of
            allowing a redundant mode to drift far outside the measured size
            range. Default is None (the original wide diameter bounds).
        weighting: str, optional
            Residual weighting used by the optimiser. ``"variance"`` (default)
            weights each bin by the inverse of its temporal standard error;
            ``"uniform"`` weights every bin equally, i.e. fits the mean curve as
            plotted, so a fit started from a good visual guess cannot end up
            worse than that guess. Use ``"uniform"`` for by-eye fitting.

        Returns
        -------
        popt : list
            A list containing the sorted fitted parameters (mu1, sigma1, factor1, mu2...)
            either sorted by mode diameter or from most to least populated mode
        perr : list
            Error estimates for the fitted parameters in the same order as the fit.
        """
        
        
        # Added functions for the fitting
        """
        The mathmatical expression of a lognormal distribution. The function can be
        used to genereate a theoretical lognormal distribution and is also used by
        the Fit_lognormal function. It assumes a minimum of 2 peaks, but can fit
        additional peaks if given prompt.

        Parameters
        ----------
        bin_mid : numpy.array
            An array of size bin midpoints used as the x values of the lognormal fit.
        *params: list
            params is a list contaning the triplet of information making out a:
            peak center: mu, peak spread: sigma, and population: factor
            The list should be structured:
                parameters=[mu1,sigma1,factor1,mu2,sigma2,factor2...]

        Returns
        -------
        lognormal_function : numpy.array
            Returns an array of the same size as bin_mid populated by the sum of
            the desired peaks at diameter size in bin_mid.

        """
        def Lognormal(bin_mid, *params):

            bin_mid = np.log10(bin_mid)
            mu      = np.log10(np.array(params[0::3]))
            sigma   = np.log10(np.array(params[1::3]))
            factor  = np.array(params[2::3])
            
            population = 0
            for i in range(0,len(mu)):
                    
                population+=((1/(np.sqrt(2*np.pi) * sigma[i])) *
                        np.exp(-((bin_mid - mu[i])**2) /
                        (2*sigma[i]**2)))*factor[i] 
           
            return np.log10(population)
        
        def Normal(bin_mid, *params):

            bin_mid = np.log10(bin_mid)
            mu      = np.log10(np.array(params[0::3]))
            sigma   = np.log10(np.array(params[1::3]))
            factor  = np.array(params[2::3])
            
            population = 0
            for i in range(0,len(mu)):

                population+=((1/(np.sqrt(2*np.pi) * sigma[i])) * 
                             np.exp(-((bin_mid - mu[i])**2) /
                             (2*sigma[i]**2)))*factor[i] 
           
            return population
        ###
        data=self.copy_self()
        data.normalize_logdp()
        # Specify x and y data to fit
        xdata = np.array(data.bin_mids,dtype='float64')
        if period in data.activities: 
            
            ydata = np.array(pd.DataFrame(data.get_activity_data(period),
                    columns=data.bin_mids.astype(str)),dtype='float64')
        elif type(period) == tuple:
            ydata = np.array(pd.DataFrame(data.timecrop(start = period[0], end = period[1],
                    inplace = False ).data,columns=data.bin_mids.astype(str)),dtype='float64')
        else:
            raise ValueError("Period chosen is neither an activity or a range of data")
            

        ymean = np.nanmean(ydata, axis=0)
        
        if len(xdata)!=len(ymean):
            raise ValueError("Discrepency between number of bins and data")
        
        # Removes values of 0 or below from fitting
        mask=ymean>0
        
        xdata=xdata[mask]
        ymean=ymean[mask]
        ydata_masked = ydata[:, mask]
        
        if len(ymean) == 0:
            raise ValueError("Empty dataset")
        
        n = ydata_masked.shape[0]
        
        mu=mu.copy()
        sigma=sigma.copy()
        factor=factor.copy()
        
        peak_number=len(mu)
        
        if peak_number*3>=len(ymean):
            raise ValueError("Peak number will lead to overfitting")
        
        if peak_number!=len(sigma):
            raise ValueError("Missing input for initial guess")
        
        # Gather all the initial guesses for parameters to fit in a list
        init_guess = []
        for i in range(peak_number):
            init_guess.extend([mu[i], sigma[i], factor[i]])
            
        # Generate bounds to reduce the risk of producing impossible or irrelevant modes
        low_bounds=[0.1*min(xdata),1.15,0]*peak_number
        up_bounds=[10*max(xdata),5,max(max(ymean),max(factor))*2.5]*peak_number

        # Optionally keep each mode's peak diameter within a multiplicative
        # window of its initial guess (mu in [mu0/mu_factor, mu0*mu_factor]).
        # This refines the supplied modes locally instead of letting a redundant
        # mode flee far outside the measured range, where it stops contributing
        # to the fit. Useful when the modes were placed deliberately (e.g. by
        # eye in the GUI). Default (None) keeps the original wide bounds.
        if mu_factor is not None and mu_factor > 1:
            for k in range(peak_number):
                low_bounds[3 * k] = init_guess[3 * k] / mu_factor
                up_bounds[3 * k] = init_guess[3 * k] * mu_factor

        if tolerance>0:
          tolerance=tolerance/100
            
        # binding=number_to_bool_list(binding,peak_number*3)
        if binding is None:
            binding = []
            
        for i in range(0,len(binding)):
            if binding[i]==True:
                low_bounds[i]=init_guess[i]*(1-tolerance)
                up_bounds[i]=init_guess[i]*(1+tolerance)
                
        # Build the model and the (optional) per-bin weights. log_scaling fits
        # log10(dx/dlogDp) so modes of very different population get comparable
        # leverage; otherwise the raw values are fit.
        if log_scaling == True:
            model, yfit = Lognormal, np.log10(ymean)
            sigma_y = np.nanstd(ydata_masked, axis=0, ddof=1) / np.sqrt(n)
            sigma_fit = sigma_y / (ymean * np.log(10))
        elif log_scaling == False:
            model, yfit = Normal, ymean
            sigma_fit = np.nanstd(ydata_masked, axis=0, ddof=1) / np.sqrt(n)
        else:
            raise ValueError("log_scaling not set")

        # weighting selects the residual that is minimised:
        #   "variance" — weight each bin by 1/(temporal standard error). This is
        #     statistically principled but lets temporally steady bins dominate,
        #     so the fit can drift away from the visible mean curve (and flatten
        #     a mode whose region happens to be noisy).
        #   "uniform" — weight every bin equally, i.e. fit the mean curve as
        #     plotted. An optimisation started from a good by-eye guess then
        #     cannot land on a visibly worse result.
        if weighting == "uniform":
            valid = np.isfinite(yfit)
            popt, pcov = curve_fit(
                model, xdata[valid], yfit[valid],
                p0=init_guess, bounds=(low_bounds, up_bounds),
            )
        elif weighting == "variance":
            sigma_fit = np.where(np.isfinite(sigma_fit) & (sigma_fit > 0), sigma_fit, np.nan)
            valid = np.isfinite(sigma_fit)
            popt, pcov = curve_fit(
                model, xdata[valid], yfit[valid],
                p0=init_guess, bounds=(low_bounds, up_bounds), sigma=sigma_fit[valid],
            )
        else:
            raise ValueError("weighting must be 'uniform' or 'variance'")
        
        # Get error estimates of the fits
        perr = np.sqrt(np.diag(pcov))
        
        #The next line of code sorts the data according to the desired focus; mode or population
        
        Modes= {'mu':popt[0::3],
                'sigma':popt[1::3],
                'factor':popt[2::3]}
        Error={'mu':perr[0::3],
                'sigma':perr[1::3],
                'factor':perr[2::3]}

        # Returns the sorted fitted modes and their uncertainty.
        return Modes, Error

    ###########################################################################

    def set_density(self, density: Union[float, int] = 1.0):
        """Description:
            Set or update the assumed particle density (g/cm³).

        Args:
            density (float | int): New particle density in g/cm³.

        Returns:
            Aerosol2D: The updated object with metadata["density"]
                set to the new value. If the current dtype is mass-based
                ("dM" in dtype), the mass distribution and Total_conc are
                rescaled immediately.

        Raises:
            ValueError: If the existing stored density is non-positive
                while the data are mass-based, so rescaling is undefined.
                In that case, manually fix metadata["density"] or reload
                the data before changing density.

        Notes:
            Detailed description:
                For non-mass-based data (dN, dS, dV), the method simply
                updates the stored density used in later conversions. For
                mass-based data (dM), it rescales all size-bin values and
                the Total_conc column so that the mass distribution is
                consistent with the new density.

            Theory:
                Mass concentration scales linearly with particle density
                for a fixed volume distribution (M = ρ · V). Updating the
                density therefore requires rescaling existing mass values
                to preserve the implied volume distribution.

        Examples:
            Update density when reinterpreting a measurement for a
            specific material:

            .. code-block:: python

                elpi.dtype_converter("dM")
                elpi.set_density(1.6)  # g/cm³
        """

        density = float(density)
        old = float(self.density)

        if "dM" in str(self.dtype):
            if old <= 0:
                raise ValueError(
                    f"Existing density must be positive for rescaling, got {old!r}."
                )

            factor = density / old
            # Rescale mass bins in-place
            self._data[self._sizebin_headers] *= factor

            # Rescale Total_conc mass concentration
            if "Total_conc" in self._data.columns:
                self._data["Total_conc"] *= factor

        self._meta["density"] = density
        return self

    ###########################################################################

    def normalize_logdp(self, inplace: bool = True):
        """Description:
            Normalize the size distribution by Δlog₁₀(Dp) (dx/dlogDp).

        Args:
            inplace (bool): If True, normalize this object in place and
                return it. If False, perform the normalization on a deep
                copy and return the new instance.

        Returns:
            Aerosol2D: The normalized object (self when ``inplace=True``,
                otherwise a new copy). If the dtype already contains
                ``"/dlogDp"`` the object is already normalized and is
                returned unchanged (no modification is made).

        Raises:
            ValueError: If the number of size-bin columns does not match
                the number of Δlog₁₀(Dp) widths derived from bin_edges.
                Check that bin_edges and the PSD columns are consistent.

        Notes:
            Detailed description:
                The method computes Δlog₁₀(Dp) from bin_edges and divides
                each size-bin column by its corresponding width. The dtype
                string is updated to append "/dlogDp" (for example
                "dN" → "dN/dlogDp"). Only size-bin columns are modified;
                other columns in data (including Total_conc and activity
                masks) are left unchanged.

            Theory:
                Plotting or comparing size distributions on a logarithmic
                diameter axis is often done using dN/dlogDp, dM/dlogDp,
                etc., so that equal logarithmic bin widths represent equal
                contributions when integrating over logDp. This method
                implements that per-bin normalization.

        Examples:
            Prepare a PSD for log-diameter plotting:

            .. code-block:: python

                elpi.normalize_logdp()
                elpi.plot_psd()
        """

        # Only normalize if not already in */dlogDp form
        if "/dlogDp" in str(self.dtype):
            return self if inplace else self.copy_self()

        dlog_dp = self._dlogdp()
        bin_columns = self._sizebin_headers

        # Sanity check: ensure one width per size bin
        if dlog_dp.shape[0] != len(bin_columns):
            raise ValueError("Mismatch between number of bins and dlogDp array.")

        # Apply to self or to a copy
        target = self if inplace else self.copy_self()
        target._data[bin_columns] = target._data[bin_columns].div(dlog_dp, axis=1)
        target._meta["dtype"] = f"{self.dtype}/dlogDp"

        return target

    ###########################################################################

    def unnormalize_logdp(self, inplace: bool = True):
        """Description:
            Undo Δlog₁₀(Dp) normalization (dx/dlogDp → base form).

        Args:
            inplace (bool): If True, unnormalize this object in place and
                return it. If False, perform the operation on a deep copy
                and return the new instance.

        Returns:
            Aerosol2D: The unnormalized object (self when ``inplace=True``,
                otherwise a new copy). If the dtype does not contain
                ``"/dlogDp"`` the data are already in base form and the
                object is returned unchanged (no modification is made).

        Raises:
            ValueError: If the number of PSD columns does not match the
                Δlog₁₀(Dp) array derived from bin_edges.

        Notes:
            Detailed description:
                The method multiplies each size-bin column by the
                corresponding Δlog₁₀(Dp), recovering the original base
                distribution (for example dN, dM). The "/dlogDp" suffix is
                removed from the dtype string. This is typically used
                before performing physical integrations or conversions
                that expect base distributions.

            Theory:
                This is the exact inverse of normalize_logdp: the integral
                over logDp of dX/dlogDp is equal to the integral over Dp
                of dX when the same Δlog₁₀(Dp) widths are used. Removing
                the normalization restores the original dX.

        Examples:
            Convert a normalized PSD back to base units for further
            processing (safe to call even if already in base form):

            .. code-block:: python

                elpi.unnormalize_logdp()
                elpi.dtype_converter("dM")
        """

        # Only unnormalize if current dtype indicates */dlogDp form
        if "/dlogDp" not in str(self.dtype):
            return self if inplace else self.copy_self()

        dlog_dp = self._dlogdp()
        bin_columns = self._sizebin_headers

        # Sanity check: ensure one width per size bin
        if dlog_dp.shape[0] != len(bin_columns):
            raise ValueError("Mismatch between number of bins and dlogDp array.")

        # Apply to self or to a copy
        target = self if inplace else self.copy_self()
        target._data[bin_columns] = target._data[bin_columns].mul(dlog_dp, axis=1)
        target._meta["dtype"] = str(self.dtype).replace("/dlogDp", "")

        return target

    ###########################################################################

    def PM_calc(self, dtype: str = "dM", PM: float = 4.2, lower_lim: float = 0):
        """Description:
            Compute a size-selective Pₓ time series and store it in extra_data.

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

    ###########################################################################

    def plot_psd(
        self,
        activities: Optional[list[str]] = None,
        normalize: bool = True,
        ax=None,
        dtype: str | None = None,
    ):
        """Description:
            Plot mean particle size distributions for one or more activities.

        Args:
            activities (list[str] | None): Names of activities to include.
                If None, all activities in self.activities are considered.
                Activities that do not exist are skipped.
            normalize (bool): If True, plot PSDs in log-diameter–
                normalized form (dx/dlogDp). If the underlying data are not
                normalized, a temporary division by Δlog₁₀(Dp) is applied.
                If False, PSDs are shown in base units, undoing any stored
                normalisation if needed.
            ax (matplotlib.axes.Axes | None): Axis to plot into. If None,
                a new figure and axes are created.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
                figure and axes with the PSD plot.

        Raises:
            None: Aside from Matplotlib or data consistency errors (for
                example invalid bin_edges or empty activities).

        Notes:
            Detailed description:
                For each selected activity, the method filters rows where
                the activity mask is True, optionally converts the data to
                normalized or base form for plotting, and computes mean
                and standard deviation across time in each size bin. It
                then plots the mean PSD as a line on a logarithmic
                diameter axis with a shaded ±1σ envelope. Colors are
                assigned per activity and a legend is added.

            Theory:
                Log-diameter–normalized PSDs (for example dN/dlogDp) are
                often preferred for visual comparison because equal
                horizontal distances represent equal decades in size. The
                method supports both normalized and base distributions so
                that you can inspect either representation.

        Examples:
            Compare PSDs during two tasks:

            .. code-block:: python

                elpi.mark_activities({
                    "Task A": [("2025-01-24 09:00", "2025-01-24 10:00")],
                    "Task B": [("2025-01-24 10:00", "2025-01-24 11:00")],
                })
                elpi.plot_psd(activities=["Task A", "Task B"], normalize=True)
        """

        clas = self if dtype is None else self.dtype_converter(dtype, False)

        new_fig_created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            new_fig_created = True
        else:
            fig = ax.figure

        # Set up axes: log diameter on x, grid on both scales
        ax.set_xscale("log")
        ax.set_xlabel("Particle diameter (nm)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Determine current normalization state and Δlog₁₀(Dp)
        is_already_normalized = "/dlogDp" in clas.dtype
        bin_columns = clas._sizebin_headers
        bin_mids = clas.bin_mids
        log_bin_edges = np.log10(clas.bin_edges)
        dlog_dp = np.diff(log_bin_edges)
        factor_series = pd.Series(dlog_dp, index=bin_columns)

        # Choose y-label based on requested vs. current normalization
        if normalize and not is_already_normalized:
            y_label_dtype = f"{clas.dtype}/dlogDp"
        elif not normalize and is_already_normalized:
            y_label_dtype = clas.dtype.replace("/dlogDp", "")
        else:
            y_label_dtype = clas.dtype
        ax.set_ylabel(f"{y_label_dtype}, {clas.unit}")

        # Assign colors per activity
        all_activities = sorted(clas._activity_periods.keys())
        color_map = plt.colormaps.get_cmap("gist_ncar")
        activity_colors = {
            activity: color_map(i / max(1, len(all_activities)))
            for i, activity in enumerate(all_activities)
        }

        # Determine which activities to plot
        selected_activities = activities if activities is not None else clas.activities

        for activity in selected_activities:
            if activity not in clas.activities:
                print(f"Activity '{activity}' not found. Skipping.")
                continue

            subset = clas.data[clas.data[activity]]
            if subset.empty:
                continue

            # Adjust normalization form if requested
            if normalize:
                if not is_already_normalized:
                    act_data = subset[bin_columns].copy().div(factor_series, axis=1)
                else:
                    act_data = subset[bin_columns].copy()
            else:
                if is_already_normalized:
                    act_data = subset[bin_columns].copy().mul(factor_series, axis=1)
                else:
                    act_data = subset[bin_columns].copy()

            # Average and spread across time for the current activity
            avg_act = act_data.mean()
            std_act = act_data.std()
            color = activity_colors.get(activity, None)

            ax.plot(bin_mids, avg_act, label=activity, color=color or "black")
            ax.fill_between(
                bin_mids,
                avg_act - std_act,
                avg_act + std_act,
                color=color or "black",
                alpha=0.3,
            )

        ax.legend()
        if new_fig_created:
            fig.tight_layout()

        return fig, ax

    ###########################################################################

    def plot_PM_timeseries(
        self,
        PM_values: list[float] | None = None,
        dtype: str = "dM",
        activity: str = "All data",
        fraction: bool = False,
        cummulative: bool = False,
        mark_activities: bool | Sequence[str] = False,
    ):
        """Description:
            Plot time series of one or more size-selective Pₓ metrics.

        Args:
            PM_values (list[float]): Cut diameters in µm defining the
                Pₓ series to compute (for example [0.5, 2.5, 10]).
            dtype (str): Base distribution type for Pₓ evaluation, one of
                "dN", "dS", "dV", "dM". Defaults to "dM" (mass-based).
            activity (str): Name of the activity mask selecting which time
                steps to plot. Must be a boolean column in data.
            fraction (bool): If False (default), plot Pₓ in absolute
                units and stack bands between successive PM_values. If
                True, plot the largest Pₓ on the primary axis and the
                fractional contributions of each Pₓ on a secondary axis.
            cummulative (bool): Controls how bands/legend values are
                interpreted:

                    * False: legend reports band-wise contributions between
                      successive PM_values (for example PM10 − PM2.5).
                    * True: legend reports cumulative Pₓ at each cut
                      (for example PM2.5, PM10).

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
                figure and primary axes. When fraction=True, a secondary
                y-axis for fractions is also created.

        Raises:
            Exception: If the number of PM_values exceeds the internal
                color palette. Reduce PM_values or extend the color list.
            KeyError: If activity is not a defined mask column in data.
                Check data.columns and mark_activities/Peak_finder calls.
            ValueError: If dtype is not one of "dN", "dS", "dV", "dM".

        Notes:
            Detailed description:
                The method works on a converted copy of the data (using
                dtype_converter) to compute Pₓ series for each requested
                cut diameter via PM_calc. It then restricts to the chosen
                activity and draws either stacked absolute bands or
                fractional contributions relative to the largest Pₓ. Mean
                ± standard deviation for each series (or band) are shown
                in the legend for quick comparison.

            Theory:
                Pₓ metrics reflect the contribution of different size
                ranges to overall exposure, following EN 481 / ISO 7708
                penetration curves for the chosen base distribution.
                Visualising absolute vs fractional Pₓ helps understand
                whether coarse or fine particles dominate during a task.

        Examples:
            Examine how PM0.5, PM2.5 and PM10 evolve during a shift:

            .. code-block:: python

                fig, ax = elpi.plot_PM_timeseries(
                    PM_values=[0.5, 2.5, 10],
                    dtype="dM",
                    activity="All data",
                    fraction=False,
                )
        """

        if PM_values is None:
            PM_values = [0.5, 2.5, 10]

        # Color palette for stacking PM bands
        colors = [
            "brown",
            "chocolate",
            "darkorange",
            "gold",
            "olive",
            "darkgreen",
            "teal",
            "deepskyblue",
            "darkblue",
        ]
        if len(PM_values) > len(colors):
            raise Exception(
                "Number of PM values are above the limit. Reduce the number of PM-values"
            )

        # Work on a copy to avoid modifying base distributions
        data_copy = self.copy_self()
        data_copy.dtype_converter(dtype=dtype)

        # Compute P-series for each requested PM limit
        for i in PM_values:
            data_copy.PM_calc(dtype=dtype, PM=i)

        # Restrict to selected activity
        mask = self.data[activity]
        PM_data = data_copy.extra_data.loc[mask]

        # Create figure/axes and set font sizes
        figure, ax = plt.subplots()
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        # Highlight activities
        if mark_activities and hasattr(self, "_activity_periods"):
            # Exclude "All data" unless explicitly requested
            all_activities = sorted(self._activity_periods.keys())
            color_map = plt.colormaps.get_cmap("gist_ncar")
            activity_colors = {
                activity: color_map(i / max(1, len(all_activities)))
                for i, activity in enumerate(all_activities)
            }

            if mark_activities is True:
                selected_activities = [a for a in all_activities if a != "All data"]
            elif isinstance(mark_activities, list):
                selected_activities = [
                    a for a in mark_activities if a in self._activity_periods
                ]
            else:
                selected_activities = []

            for activity in selected_activities:
                color = activity_colors[activity]
                first = True
                for start, end in self._activity_periods[activity]:
                    ax.axvspan(
                        cast(float, mdates.date2num(pd.Timestamp(start))),
                        cast(float, mdates.date2num(pd.Timestamp(end))),
                        color=color,
                        alpha=0.3,
                        label=activity if first else None,
                        zorder=3,
                    )
                    first = False
            # Clip x-axis to actual data range
            left = float(mdates.date2num(self.time.min()))
            right = float(mdates.date2num(self.time.max()))
            ax.set_xlim(left, right)
            ax.legend()

        # Fractional mode: total on primary axis, fractions on secondary axis
        if fraction:
            total_name = f"P{dtype[-1]}{PM_values[-1]}"
            total = PM_data[total_name]
            ax.plot(total, color="k", label="Total", lw=3)

            ax2 = ax.twinx()
            for i, pmv in enumerate(PM_values):
                pm = f"P{dtype[-1]}{pmv}"
                # Safe ratio
                ratio = PM_data[pm].where(total != 0, np.nan) / total.where(
                    total != 0, np.nan
                )

                if i == 0:
                    avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                    ax2.fill_between(
                        self.time[mask],
                        ratio,
                        alpha=0.75,
                        color=colors[i],
                        label=f"{pm}: {avg:.2f}±{sd:.2f}",
                    )
                else:
                    pm_1 = f"P{dtype[-1]}{PM_values[i-1]}"
                    if cummulative:
                        avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                        ax2.fill_between(
                            self.time[mask],
                            PM_data[pm_1].where(total != 0, np.nan)
                            / total.where(total != 0, np.nan),
                            ratio,
                            alpha=0.75,
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )
                    else:
                        band = PM_data[pm] - PM_data[pm_1]
                        avg, sd = float(band.mean()), float(band.std())
                        ax2.fill_between(
                            self.time[mask],
                            PM_data[pm_1].where(total != 0, np.nan)
                            / total.where(total != 0, np.nan),
                            ratio,
                            alpha=0.75,
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )

            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.set_ylim(0, 1)
            ax2.set_ylabel(f"P{dtype[-1]} fraction")
            ax2.legend(
                loc="best",
                title=f"Average values ({data_copy.unit})",
                fontsize=25,
                title_fontsize=25,
            )
        else:
            for i, pmv in enumerate(PM_values):
                pm = f"P{dtype[-1]}{pmv}"
                if i == 0:
                    avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                    ax.fill_between(
                        self.time[mask],
                        PM_data[pm],
                        alpha=1,
                        color=colors[i],
                        label=f"{pm}: {avg:.2f}±{sd:.2f}",
                    )
                else:
                    pm_1 = f"P{dtype[-1]}{PM_values[i-1]}"
                    if cummulative:
                        avg, sd = float(PM_data[pm].mean()), float(PM_data[pm].std())
                        ax.fill_between(
                            self.time[mask],
                            PM_data[pm_1],
                            PM_data[pm],
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )
                    else:
                        band = PM_data[pm] - PM_data[pm_1]
                        avg, sd = float(band.mean()), float(band.std())
                        ax.fill_between(
                            self.time[mask],
                            PM_data[pm_1],
                            PM_data[pm],
                            color=colors[i],
                            label=f"{pm}: {avg:.2f}±{sd:.2f}",
                        )
            ax.legend(
                loc="best",
                title=f"Average values ({data_copy.unit})",
                fontsize=25,
                title_fontsize=25,
            )

        ax.set_ylim(0)
        ax.set_ylabel(f"P{dtype[-1]}, {data_copy.unit}")
        loc = mdates.AutoDateLocator()
        fmt = mdates.ConciseDateFormatter(loc)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)

        return ax.figure, ax

    ###########################################################################

    def plot_timeseries(
        self,
        y_tot: tuple[float, float] = (0, 0),
        y_3d: tuple[float, float] = (0, 0),
        log: bool = True,
        ax1: Axes | None = None,
        ax2: Axes | None = None,
        mark_activities: bool | Sequence[str] = False,
        dtype: str | None = None,
    ) -> tuple[Figure, NDArray[Any]]:
        """Description:
            Plot total concentration and a time–size heatmap in one figure.

        Args:
            y_tot (tuple[float, float]): Y-limits for the total
                concentration panel (ymin, ymax). Use (0, 0) for automatic
                limits; if a non-zero entry is given with zero partner,
                the non-zero value is used directly, while the zero is
                replaced by the max/min in the data.
            y_3d (tuple[float, float]): Color-scale limits for the 2D PSD
                panel (zmin, zmax). Use (0, 0) for automatic limits. To
                enforce a strictly positive lower limit for log scaling,
                set zmin > 0 and zmax = 0 for automatic upper limit, e.g.
                y_3d = (1, 0).
            log (bool): If True, the function attempts to use a logarithmic
                color scale for the 2D panel. If the PSD values used for
                the mesh (after any clipping from y_3d) are not strictly
                positive or the lower limit is ≤ 0, the function
                automatically falls back to a linear color scale and prints
                a warning to the terminal. If False, a linear color scale
                is used directly.
            ax1 (matplotlib.axes.Axes | None): Axis for the top (total
                concentration) plot. If provided, ax2 must also be
                provided.
            ax2 (matplotlib.axes.Axes | None): Axis for the bottom
                (time–size) plot. If provided, ax1 must also be provided.
            mark_activities (bool | Sequence[str]): Passed to
                plot_total_conc to control activity highlighting. True
                shades all activities except "All data"; a sequence
                restricts shading to specific activities.
            dtype (str | None): Designator for the desired datatype to be
                plotted, independent from current datatype.
                Chose between; 'dN', 'dS', 'dV', 'dM'
        Returns:
            tuple[matplotlib.figure.Figure, numpy.ndarray]: A tuple with
                the figure and a NumPy array [ax1, ax2, colorbar].

        Raises:
            ValueError: If only one of ax1 or ax2 is supplied. Provide
                both or neither.

        Notes:
            Detailed description:
                The method first draws the total concentration time series
                (via plot_total_conc) in the top panel, optionally with
                activity shading and custom y-limits. The bottom panel
                shows a pcolormesh of particle size distribution (PSD)
                with values as a function of time and particle diameter.
                A shared colorbar is added and labeled with the
                current dtype and unit, and the y-axis for the PSD is
                log-scaled in diameter.

                By default, the color scale for the PSD uses a logarithmic
                normalization (log=True). If the PSD values (after any
                clipping via y_3d) include zeros or negatives, or if the
                color-scale lower limit is ≤ 0, the method automatically
                falls back to a linear color scale and prints a clearly
                visible warning to the terminal. To avoid this fallback and
                enforce log scaling, you can specify a strictly positive
                lower limit via y_3d, for example y_3d = (1, 0), which
                clips all values below 1 and lets the method safely use a
                log color scale.

            Theory:
                The heatmap represents the evolution of the size
                distribution over time. Combining this with total
                concentration in one figure makes it easier to relate bulk
                peaks to specific size modes or shifts in the
                distribution.

        Examples:
            Create an overview plot of an ELPI or SMPS data set:

            .. code-block:: python

                fig, (ax1, ax2, cbar) = elpi.plot_timeseries()
                fig.savefig("elpi_timeseries.png", dpi=150)
        """

        clas = self if dtype is None else self.dtype_converter(dtype, False)

        # Require both axes or neither when passing external axes
        if (ax1 is None) != (ax2 is None):
            raise ValueError("You must provide both ax1 and ax2, or neither.")

        # Create figure/axes if not provided
        if ax1 is None and ax2 is None:
            newplot = True
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
        else:
            assert ax1 is not None and ax2 is not None
            fig: Figure = ax1.get_figure()  # type: ignore
            newplot = False
        ax1 = cast(Axes, ax1)
        ax2 = cast(Axes, ax2)

        time = clas.time
        total = clas.total_concentration
        data = clas.size_data
        bin_edges = clas.bin_edges

        # --- Top panel: total concentration (with optional activity shading) ---
        _, ax_new = clas.plot_total_conc(ax=ax1, mark_activities=mark_activities)
        ax1 = ax_new
        ax1 = cast(Axes, ax1)

        # Optionally fix y-limits of the total concentration plot
        if y_tot != (0, 0):
            ymin = y_tot[0] if y_tot[0] != 0 else total.min() * 0.98
            ymax = y_tot[1] if y_tot[1] != 0 else total.max() * 1.02
            ax1.set_ylim(ymin, ymax)

        # --- Construct time edges for pcolormesh (center ± ½Δt) ---------------
        dt = (time[1] - time[0]) / 2
        time_edges = pd.DatetimeIndex(np.append(time - dt, [time[-1] + dt]))  # type: ignore

        x_grid, y_grid = np.meshgrid(time_edges, bin_edges, indexing="ij")

        # --- Handle color scale limits ----------------------------------------
        z_data = data.copy()
        if y_3d != (0, 0):
            zmin, zmax = y_3d
            if zmin != 0:
                z_data = z_data.clip(lower=zmin)
            if zmax == 0:
                zmax = z_data.max().max()
        else:
            zmin = z_data.min().min()
            zmax = z_data.max().max()

        # --- Choose color normalization (log or linear, with fallback) --------
        use_log = log
        if log:
            # Check for non-positive values or non-positive lower limit
            has_non_positive = (z_data <= 0).any().any()
            if has_non_positive or zmin <= 0:
                # Fallback to linear with a big, clear warning
                print(
                    "\n" + "=" * 72 + "\nWARNING (plot_timeseries):"
                    "\n  PSD data contain zero or negative values, or the lower"
                    "\n  color-scale limit is ≤ 0. Falling back to *linear*"
                    "\n  color scale for the concentration heatmap."
                    "\n"
                    "\n  To enforce log-scaled concentration, set y_3d to use a"
                    "\n  strictly positive lower limit, for example:"
                    "\n      y_3d = (1, 0)"
                    "\n  which clips values below 1 and allows log scaling."
                    "\n" + "=" * 72 + "\n"
                )
                use_log = False

        if use_log:
            # Safety: ensure vmin > 0 for LogNorm
            vmin = max(zmin, np.nextafter(0, 1))
            norm = LogNorm(vmin=vmin, vmax=zmax)
        else:
            norm = Normalize(vmin=zmin, vmax=zmax)

        # --- Bottom panel: size–time pcolormesh -------------------------------
        mesh = ax2.pcolormesh(
            x_grid, y_grid, z_data, cmap="jet", norm=norm, shading="flat"
        )

        # Axis labels and scales
        ax2.set_yscale("log")
        ax2.set_ylabel("Dp, nm")
        ax2.set_xlabel("Time")
        if newplot:
            ax1.set_xlabel("")
        ax2.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )

        # Shared colorbar for both panels
        col = fig.colorbar(mesh, ax=[ax1, ax2])
        col.set_label(f"{clas.dtype}, {clas.unit}")

        # Basic styling
        ax1.tick_params(axis="y", which="both", direction="out", length=6, width=2)
        ax2.tick_params(axis="y", which="both", direction="out", length=6, width=2)

        return fig, np.array([ax1, ax2, col], dtype=object)

    ###########################################################################

    @override
    def summarize_activities(
        self,
        filename: Optional[str] = None,
        sheet_name: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize size-resolved aerosol metrics per activity.

        Args:
            filename (str | None): Optional Excel file path. If provided,
                the summary table is written to this file (one sheet,
                activities as rows). If None, no file is written.
            sheet_name (str | None): Optional sheet name. If provided
                the data is written in a sheet named as sheet_name.
                If one already exsits it overwrites the exsisitng sheet.
            metrics (list[str] | None): List of metric names to compute.
                If None, a default set is used: ["PNC", "PM1", "PM2.5",
                "PM4", "PM10", "MASS", "MODE", "MEDIAN", "GMD"].

        Returns:
            pandas.DataFrame: Summary table with:
                * "Segment"
                * "Duration (HH:MM)"
                * For each metric M: "M mean [unit]" and "M std [unit]".

        Raises:
            ValueError: If a metric name cannot be interpreted (for
                example malformed PMx string) or is unsupported.
            ValueError: If internal preparation for a Pₓ metric fails
                (for example missing PSD columns or inconsistent bin
                metadata).

        Notes:
            Detailed description:
                For each activity, the method computes the total duration
                and the requested set of metrics. PNC and MASS are total
                number and mass concentrations. MODE, MEDIAN and GMD are
                size metrics derived from the number distribution. Metrics
                of the form PMx, PNx, PSx, PVx are Pₓ values at cut
                diameter x (in µm) for mass, number, surface area and
                volume, respectively. Each metric is reported as mean and
                standard deviation over the activity. A transposed version
                of the table is printed to the terminal for quick inspection.

            Theory:
                The method combines time-weighted activity durations with
                standard PSD-derived metrics: bulk concentrations, Pₓ, and
                central size metrics (mode, median, geometric mean).
                Pₓ metrics are determined using penetration curves given in
                EN 481 / ISO 7708, mimicing cyclone cut-offs rather than
                sharp size cut-offs for PMₓ calculations.

        Examples:
            Generate a task-level summary of exposure metrics:

            .. code-block:: python

                elpi.summarize_activities(
                    filename="activity_summary_elpi.xlsx",
                    metrics=["PNC", "PM2.5", "PM10", "MODE", "GMD"],
                )
        """

        # --- defaults --------------------------------------------------------
        if metrics is None:
            metrics = [
                "PNC",
                "PM1",
                "PM2.5",
                "PM4",
                "PM10",
                "MASS",
                "MODE",
                "MEDIAN",
                "GMD",
            ]

        # --- helper: duration in minutes per time step (shared helper) -------
        dt_mins = self._dt_minutes()

        def _px_label(dchar: str, cutoff: float, lower_lim: float = 0) -> str:
            if lower_lim == 0:
                return f"P{dchar}{cutoff:g}"
            else:
                return f"P{dchar}{lower_lim:g}-{cutoff:g}"

        metrics_upper = [m.upper() for m in metrics]

        want_pnc = "PNC" in metrics_upper
        want_mass = "MASS" in metrics_upper
        want_mode = "MODE" in metrics_upper
        want_median = "MEDIAN" in metrics_upper
        want_gmd = "GMD" in metrics_upper
        want_any_size_metric = want_mode or want_median or want_gmd

        # Gather requested Pₓ cutoffs by dtype char (M/N/S/V) using shared parser
        pm_requests: dict[str, set[float]] = {
            "M": set(),
            "N": set(),
            "S": set(),
            "V": set(),
        }
        for name_upper in metrics_upper:
            parsed = self._parse_px_metric_scalar(name_upper)
            if parsed:
                dchar, cutoff, lower_lim = parsed
                pm_requests[dchar].add((cutoff, lower_lim))

        # --- prepare prerequisite data only if needed ------------------------
        # Extract the base array once and convert to target dtypes via numpy,
        # avoiding expensive full-object deep copies.
        dtype_map = {"M": "dM", "N": "dN", "S": "dS", "V": "dV"}
        current_base = str(self.dtype).replace("/dlogDp", "")

        number_df = None
        mass_df = None
        if want_pnc or want_any_size_metric or want_mass:
            base_arr, _, _ = self._as_base_array()

        if want_pnc or want_any_size_metric:
            if current_base != "dN":
                number_arr = self._convert_array(
                    base_arr, self.bin_mids, current_base, "dN", self.density
                )
            else:
                number_arr = base_arr
            number_df = pd.DataFrame(
                number_arr, index=self.time, columns=self._sizebin_headers
            )

        if want_mass:
            if current_base != "dM":
                mass_arr = self._convert_array(
                    base_arr, self.bin_mids, current_base, "dM", self.density
                )
            else:
                mass_arr = base_arr
            mass_df = pd.DataFrame(
                mass_arr, index=self.time, columns=self._sizebin_headers
            )

        # Precompute Pₓ series directly via _px_fraction_series (no copy_self needed)
        px_extra: dict[str, dict[str, pd.Series]] = {}
        for dchar, cutoffs in pm_requests.items():
            if not cutoffs:
                continue
            px_extra[dchar] = {}
            for pm, lower_lim in sorted(cutoffs):
                label = _px_label(dchar, pm, lower_lim)
                px_extra[dchar][label] = self._px_fraction_series(
                    dtype=dtype_map[dchar], upper=pm, lower=lower_lim
                )

        # --- compute per-activity --------------------------------------------
        rows: list[list[float | str]] = []
        bin_mids = np.asarray(self.bin_mids, dtype=float)

        for activity in self.activities:
            mask = self.data[activity]
            if mask.sum() == 0:
                continue

            # Duration of this activity (min and HH:MM)
            duration_minutes = float(dt_mins.loc[mask].sum())
            duration_hhmm = self._format_hhmm(duration_minutes)

            # Initialize with NaNs
            pnc = pnc_std = float("nan")
            total_mass = total_mass_std = float("nan")
            mode_d = mode_d_std = float("nan")
            median_d = median_d_std = float("nan")
            gmd = gmd_std = float("nan")

            # PNC (from number_df)
            if want_pnc and number_df is not None:
                num_act = number_df.loc[mask]
                s = self._ensure_data_robustness(num_act.sum(axis=1))
                pnc, pnc_std = float(s.mean()), float(s.std())

            # Total mass (from mass_df)
            if want_mass and mass_df is not None:
                mass_act = mass_df.loc[mask]
                s = self._ensure_data_robustness(mass_act.sum(axis=1))
                total_mass, total_mass_std = float(s.mean()), float(s.std())

            # Size metrics (mode/median/GMD) on number distribution
            if want_any_size_metric and number_df is not None:
                num_act = number_df.loc[mask]

                mode_list: list[float] = []
                med_list: list[float] = []
                gmd_list: list[float] = []

                for _, row in num_act.iterrows():  # type: ignore
                    dist = row.to_numpy(dtype=float)  # type: ignore
                    tot = float(dist.sum())
                    if tot <= 0:
                        continue

                    if want_mode:
                        mode_idx = int(np.argmax(dist))
                        mode_list.append(float(bin_mids[mode_idx]))

                    if want_median:
                        cum = np.cumsum(dist)
                        cum /= cum[-1]
                        med_idx = int(np.searchsorted(cum, 0.5))
                        med_list.append(float(bin_mids[med_idx]))

                    if want_gmd:
                        positive = dist > 0
                        if not np.any(positive):
                            continue
                        dpos = dist[positive]
                        mpos = bin_mids[positive]
                        gval = float(np.exp(np.sum(np.log(mpos) * dpos) / dpos.sum()))
                        gmd_list.append(gval)

                if want_mode and mode_list:
                    mode_d, mode_d_std = float(np.mean(mode_list)), float(
                        np.std(mode_list)
                    )
                if want_median and med_list:
                    median_d, median_d_std = float(np.mean(med_list)), float(
                        np.std(med_list)
                    )
                if want_gmd and gmd_list:
                    gmd, gmd_std = float(np.mean(gmd_list)), float(np.std(gmd_list))

            # Pₓ metrics collected in requested order
            px_values: dict[str, tuple[float, float]] = {}
            for name, name_upper in zip(metrics, metrics_upper):
                parsed = self._parse_px_metric_scalar(name_upper)
                if not parsed:
                    continue
                dchar, cutoff, lower_lim = parsed
                dchar_dict = px_extra.get(dchar)
                if dchar_dict is None:
                    raise ValueError(
                        f"Internal error: dtype '{dchar}' was not prepared."
                    )
                label = _px_label(dchar, cutoff, lower_lim)
                if label not in dchar_dict:
                    raise ValueError(f"Internal error: PX column '{label}' not found.")
                ser = dchar_dict[label].loc[mask]
                px_values[label] = (float(ser.mean()), float(ser.std()))

            # Assemble row
            row: list[float | str] = [activity, duration_hhmm]
            for name, name_upper in zip(metrics, metrics_upper):
                if name_upper == "PNC":
                    row += [round(pnc, 2), round(pnc_std, 2)]
                elif name_upper == "MASS":
                    row += [round(total_mass, 2), round(total_mass_std, 2)]
                elif name_upper == "MODE":
                    row += [round(mode_d, 1), round(mode_d_std, 1)]
                elif name_upper == "MEDIAN":
                    row += [round(median_d, 1), round(median_d_std, 1)]
                elif name_upper == "GMD":
                    row += [round(gmd, 1), round(gmd_std, 1)]
                else:
                    parsed = self._parse_px_metric_scalar(name_upper)
                    if not parsed:
                        raise ValueError(f"Unsupported metric '{name}'.")
                    dchar, cutoff, lower_lim = parsed
                    label = _px_label(dchar, cutoff, lower_lim)
                    mean_val, std_val = px_values[label]
                    row += [round(mean_val, 2), round(std_val, 2)]
            rows.append(row)

        # --- column headers with explicit units ---------------------------------
        columns: list[str] = ["Segment", "Duration (HH:MM)"]

        unit_map_px = {
            "M": "µg/m³",
            "N": "cm⁻³",
            "S": "nm²/cm³",
            "V": "nm³/cm³",
        }

        for name, name_upper in zip(metrics, metrics_upper):
            parsed = self._parse_px_metric_scalar(name_upper)
            if parsed:
                dchar, _, _ = parsed
                unit = unit_map_px[dchar]
                label = f"{name} [{unit}]"
            elif name_upper == "PNC":
                label = "PNC [cm⁻³]"
            elif name_upper == "MASS":
                label = "Total Mass [µg/m³]"
            elif name_upper == "MODE":
                label = "Mode Dp [nm]"
            elif name_upper == "MEDIAN":
                label = "Median Dp [nm]"
            elif name_upper == "GMD":
                label = "GMD [nm]"
            else:
                label = name
            columns += [label, f"{label} std"]

        summary = pd.DataFrame(rows, columns=columns)
        # --- append to file if requested ---------------------------------------
        if filename and not summary.empty:
            fname = str(filename)
            lower = fname.lower()
            if sheet_name:
                shname = str(sheet_name)
            else:
                shname = f"{self.instrument} summary"

            if lower.endswith(".csv"):
                if os.path.exists(fname):
                    summary.to_csv(fname, mode="a", header=False, index=False)
                else:
                    summary.to_csv(fname, mode="w", header=True, index=False)
            elif lower.endswith((".xlsx", ".xls")):
                if os.path.exists(fname):
                    with pd.ExcelWriter(
                        filename,
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace",  # requires pandas ≥ 1.4
                    ) as writer:
                        summary.to_excel(writer, sheet_name=shname, index=False)
                else:
                    summary.to_excel(fname, sheet_name=shname, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension for '{filename}'. Use .csv or .xlsx."
                )
            print(f"\nActivity summary saved to: {filename}")

        summary_t = summary.set_index("Segment").T
        print("\nSummary of aerosol properties (transposed):\n")
        print(tabulate(summary_t, headers="keys", tablefmt="pretty", floatfmt=".3f"))  # type: ignore

        return summary

    ###########################################################################

    @override
    def summarize_exposure(
        self,
        metric: str = "PM4.2",
        background: Union[float, str, None] = None,
        exposure_hours: Optional[float] = None,
        short_limit: float = 1.0,
        long_limit: float = 1.0,
        short_window: str = "15min",
        twa_window: str = "8h",
        peak_ratio: float = 2.5,
        filename: Optional[str] = None,
        sheet_name: Optional[str] = None,
        activities: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Description:
            Summarize exposure metrics for one PSD-derived metric across activities.

        Args:
            metric (str): Exposure metric name derived from the underlying
                particle size distribution (PSD). Default is "PM4.2",
                corresponding to respirable dust.Supported forms:

                    - "PNC": total number concentration.
                    - "MASS": total mass concentration.
                    - "PM<x>", "PN<x>", "PS<x>", "PV<x>": cumulative Pₓ at cut
                      diameter <x> in µm for mass, number, surface, volume
                      (for example "PM4.2").
                    - "PM<a>-<b>", "PN<a>-<b>", "PS<a>-<b>", "PV<a>-<b>":
                      band-limited Pₓ between diameters <a> and <b> in µm
                      (for example "PM1-4", "PM1-4.2"). These are computed
                      using the EN 481 / ISO 7708 penetration curves.

            background (float | str | None): Background level used when
                computing the time-weighted average over ``twa_window``.
                The same background level is used for all activities in
                the output. Default is None. Possible entries are:

                    * None: assume zero background.
                    * float: constant background level in metric units.
                    * str: name of an activity; the TWA of ``metric`` over
                      that activity is used as background.

            exposure_hours (float | None): Assumed duration of exposure for
                each activity, in hours, when embedding it into the TWA
                window. If None, the measured activity duration is used for
                that activity. If a positive value is given, the same
                exposure duration is applied for all activities.
                Default is None.

            short_limit (float): Short-term concentration limit in metric
                units (for example a 15-min STEL). This value is reported in
                the output as ``"STEL [unit]"``. Default is 1.0 (in metric
                units).

            long_limit (float): Long-term concentration limit in metric
                units (for example an 8-h OEL). This value is reported in the
                output as ``"Exposure limit [unit]"``. Default is 1.0 (in
                metric units).

            short_window (str): Rolling window used for short-term (STEL)
                evaluation, given as a pandas offset string (for example
                "15min"). This is reported as ``"STEL window"``.
                Default is "15min" (15 minutes).

            twa_window (str): Total duration of the TWA window as a pandas
                offset string (for example "8h"). This is reported as
                ``"TWA window"``. Default is "8h" (8 hours).

            peak_ratio (float): Factor used in peak detection; peaks are
                flagged when the metric exceeds::

                    baseline + peak_ratio * rolling_std,

                where ``baseline`` is a rolling median and ``rolling_std`` is
                a rolling standard deviation over a short window.
                Default is 2.5.

            filename (str | None): Optional path to a CSV/Excel file to which
                the non-transposed result rows are appended. If the file
                exists, rows are appended; otherwise the file is created with
                a header. Supported extensions are ".csv", ".xls", ".xlsx".

            activities (Sequence[str] | None): Activities to summarize.
                If None (default), all defined activities in
                :attr:`activities` are summarized (for example "All data",
                "Background", "Emission", "Decay"). Activities with no
                marked time steps are skipped.

        Returns:
            pandas.DataFrame: One row per activity segment with summary
            statistics for the chosen metric. Column names embed their units
            in square brackets. See Notes below (Detailed description) for a
            complete list of columns.

        Raises:
            ValueError: If ``metric`` cannot be parsed (unsupported string).
            ValueError: If a background activity name is given but does not
                exist or has no samples.
            ValueError: If ``short_window`` or ``twa_window`` cannot be
                parsed as pandas-style durations.
            ValueError: If ``exposure_hours`` is negative.
            TypeError: If ``background`` is not None, float or str.

        Notes:
            Detailed description:
                The method first derives the requested metric time series
                from the underlying 2D PSD (for example PNC from a number
                distribution, PM4.2 from a mass-based Pₓ, or a band metric
                such as PM1–4). Where possible, existing Pₓ series already
                stored in :attr:`extra_data` are reused; otherwise they are
                computed on a working copy and then used to build the metric
                time series.

                For each selected activity, the method:

                    1. Extracts the metric time series within the activity
                       using the activity mask.
                    2. Computes per-step durations from the actual sampling
                       times.
                    3. Forms a time-weighted segment mean, used internally
                       when embedding the activity into the TWA window.
                    4. Computes instantaneous percentiles, peaks, STEL-style
                       exceedances and long-term time-above-limit measures.
                    5. Combines the segment mean with the specified background
                       level to obtain an overall TWA for the chosen window.

                The returned DataFrame contains the following columns
                (per activity):

                    - "Segment"
                        Name of the activity segment.

                    - "Metric"
                        Name of the metric that was summarized, e.g.
                        "PM4.2", "PM1-4", "PNC", "MASS".

                    - "Duration [HH:MM]"
                        Duration of the activity derived from the actual
                        sampling intervals, formatted as "HH:MM" (hours and
                        minutes).

                    - "Max [unit]"
                        Maximum of the instantaneous metric values during the
                        activity. The unit is the metric unit (e.g. "µg/m³",
                        "cm⁻³", "nm²/cm³", "nm³/cm³").

                    - "95th percentile [unit]"
                    - "75th percentile [unit]"
                    - "50th percentile [unit]"
                    - "25th percentile [unit]"
                    - "5th percentile [unit]"
                        Percentiles of the instantaneous metric values during
                        the activity, all in the same unit as the metric.

                    - "Peaks [count]"
                        Number of distinct peak events detected in the
                        activity, where a peak is a contiguous period in which
                        the metric exceeds::

                            baseline + peak_ratio * rolling_std,

                        with ``baseline`` the rolling median and
                        ``rolling_std`` the rolling standard deviation over a
                        short sliding window, covering at least 3 datapoints
                        and maximum 15 datapoints, depending on the number
                        of available datapoints in the given segment.

                    - "STEL [unit]"
                        Short-term exposure limit used in the STEL
                        exceedance calculations (echoing ``short_limit``),
                        with the same concentration unit as the metric.

                    - "STEL window"
                        Rolling window length used for STEL evaluation,
                        echoing ``short_window`` (for example "15min").

                    - "STEL exceedance [min]"
                        Total time in minutes during the activity where the
                        **full-window** rolling mean is greater than or equal
                        to the STEL. Only complete windows are counted in
                        this measure.

                    - "STEL episodes [count]"
                        Number of distinct STEL exceedance episodes based on
                        the full-window rolling mean. Each contiguous block of
                        time where the full-window mean is at or above the
                        STEL is counted as one episode.

                    - "Exposure limit [unit]"
                        Long-term exposure limit value used in the long-term
                        evaluation (echoing ``long_limit``), with the same
                        unit as the metric.

                    - "Exposure limit exceedance [min]"
                        Total time in minutes during the activity where the
                        instantaneous metric values are greater than or equal
                        to the exposure limit.

                    - "TWA concentration [unit]"
                        Time-weighted average concentration of the metric over
                        the full TWA window (``"TWA window"``), combining the
                        internal segment mean (optionally stretched to
                        ``exposure_hours``) with the specified background
                        level for the remainder of the window::

                            TWA = (segment_mean * exposure_minutes
                                   + background * (twa_minutes - exposure_minutes))
                                  / twa_minutes

                        where ``twa_minutes`` is the duration of
                        ``twa_window`` in minutes and ``exposure_minutes`` is
                        either the measured activity duration or
                        ``exposure_hours * 60``, whichever is used.

                    - "TWA window"
                        Duration of the TWA window over which
                        "TWA concentration [unit]" is evaluated, echoing
                        ``twa_window`` (for example "8h").

            Theory:
                Conceptually, the method mirrors typical occupational hygiene
                practice:

                    * Exposure within an activity is represented by
                      time-weighted averages and distributional
                      descriptors (max and percentiles).
                    * Short-term limits (STEL) are evaluated using rolling
                      means over a specified window, counting both total time
                      above the limit and distinct exceedance episodes.
                    * Long-term limits are represented by cumulative
                      time-above-limit measures.
                    * An overall TWA for an 8-h (or user-specified) reference
                      period is constructed by mixing the task exposure with a
                      background level for the rest of the window.

                This ensures a transparent link between raw time series
                behaviour and regulatory style metrics such as STEL and
                8-hour OELs.

        Examples:
            Summarize respirable PM4.2 exposure for all activities in an
            ELPI dataset, using the "Background" activity as background
            level and a 15-min STEL with an 8-h TWA window:

            .. code-block:: python

                elpi.summarize_exposure(
                    metric="PM4.2",
                    background="Background",
                    short_limit=5.0,
                    long_limit=10.0,
                    short_window="15min",
                    twa_window="8h",
                )

            Band-limited metrics are handled in the same way, e.g. for a
            PM1–4.2 band::

            .. code-block:: python

                elpi.summarize_exposure(metric="PM1-4.2")
        """

        # --- obtain the time series for the requested metric -------------------
        series_source, unit = self._get_metric_series(metric)

        # --- time deltas in minutes for integration (irregular sampling safe) ---
        dt_mins = self._dt_minutes()

        # --- parse windows ------------------------------------------------------
        try:
            short_minutes = pd.to_timedelta(short_window).total_seconds() / 60.0
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse short_window {short_window!r}.") from exc

        try:
            twa_minutes = pd.to_timedelta(twa_window).total_seconds() / 60.0
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse twa_window {twa_window!r}.") from exc

        # --- background level (one value, reused for all activities) -----------
        if background is None:
            bg = 0.0
        elif isinstance(background, (int, float)):
            bg = float(background)
        elif isinstance(background, str):
            if background not in self.activities:
                raise ValueError(
                    f"Background activity '{background}' not found in activities."
                )
            mask_bg = self.data[background].astype(bool)
            if mask_bg.sum() == 0:
                raise ValueError(
                    f"Background activity '{background}' has no marked time steps."
                )
            s_bg = series_source.loc[mask_bg]
            dt_bg = dt_mins.loc[mask_bg]
            dur_bg = float(dt_bg.sum())
            bg = float((s_bg * dt_bg).sum() / max(dur_bg, 1e-9))
        else:
            raise TypeError(
                "background must be None, a float, or an activity name (str)."
            )

        # --- assumed exposure duration within the TWA window --------------------
        if exposure_hours is not None and exposure_hours < 0:
            raise ValueError("exposure_hours must be non-negative.")

        # --- which activities to summarize -------------------------------------
        if activities is None:
            activity_list = list(self.activities)
        else:
            activity_list = list(activities)

        # Column labels with units embedded (same for all rows)
        duration_label = "Duration [HH:MM]"
        max_label = f"Max [{unit}]"
        p95_label = f"95th percentile [{unit}]"
        p75_label = f"75th percentile [{unit}]"
        p50_label = f"50th percentile [{unit}]"
        p25_label = f"25th percentile [{unit}]"
        p5_label = f"5th percentile [{unit}]"
        peaks_label = "Peaks [count]"
        stel_label = f"STEL [{unit}]"
        stel_window_label = "STEL window [offset]"
        stel_exceed_label = "STEL exceedance [min]"
        stel_episodes_label = "STEL episodes [count]"
        exp_limit_label = f"Exposure limit [{unit}]"
        exp_exceed_label = "Exposure limit exceedance [min]"
        twa_conc_label = f"TWA concentration [{unit}]"
        twa_window_label = "TWA window [offset]"

        rows: list[dict[str, Any]] = []

        for activity in activity_list:
            if activity not in self.activities:
                raise ValueError(f"Activity '{activity}' not found in activities.")

            mask = self.data[activity].astype(bool)
            if mask.sum() == 0:
                # Skip empty activities
                continue

            s = series_source.loc[mask]
            dtm = dt_mins.loc[mask]

            # --- basic duration & segment mean ----------------------------------
            duration_min = float(dtm.sum())
            duration_hhmm = self._format_hhmm(duration_min)
            if duration_min <= 0:
                # Skip pathological segments rather than crashing everything
                continue

            seg_mean = float((s * dtm).sum() / duration_min)
            seg_max = float(s.max())

            # percentiles in metric units
            p95, p75, p50, p25, p5 = map(
                float, np.nanpercentile(s, [95, 75, 50, 25, 5])
            )

            # --- short-term limit: full-window rolling mean (episodes) ----------
            if len(dtm) > 0 and np.isfinite(dtm.median()) and dtm.median() > 0:
                dt_med = float(dtm.median())
            else:
                dt_med = short_minutes  # fall back to something non-zero

            if dt_med <= 0:
                dt_med = short_minutes

            n_win = max(1, int(round(short_minutes / dt_med)))
            if n_win > s.size:
                n_win = s.size

            if s.size == 1:
                s_roll_full = s.copy()
            else:
                s_roll_full = s.rolling(window=n_win, min_periods=n_win).mean()

            full_exceed_mask = s_roll_full >= float(short_limit)
            short_full_exceed_min = float(dtm.where(full_exceed_mask, 0.0).sum())
            short_full_episodes = int(
                ((full_exceed_mask) & (~full_exceed_mask.shift(fill_value=False))).sum()
            )

            # --- peaks using rolling baseline -----------------------------------
            window_size = max(3, min(15, s.size))
            baseline = s.rolling(
                window=window_size, center=True, min_periods=1
            ).median()
            spread = s.rolling(window=window_size, center=True, min_periods=1).std()
            peak_mask = (s - baseline) > (spread * float(peak_ratio))
            peak_count = int(((peak_mask) & (~peak_mask.shift(fill_value=False))).sum())

            # --- window TWA with background -------------------------------------
            if exposure_hours is None:
                exposure_minutes = duration_min
            else:
                exposure_minutes = float(exposure_hours) * 60.0

            if exposure_minutes >= twa_minutes:
                window_twa = seg_mean
            else:
                window_twa = float(
                    (
                        seg_mean * exposure_minutes
                        + bg * (twa_minutes - exposure_minutes)
                    )
                    / twa_minutes
                )

            # instantaneous time above long-term limit within the activity
            long_exceed_min = float(dtm.where(s >= float(long_limit), 0.0).sum())

            rows.append(
                {
                    "Segment": activity,
                    "Metric": metric,
                    duration_label: duration_hhmm,
                    max_label: round(seg_max, 3),
                    p95_label: round(p95, 3),
                    p75_label: round(p75, 3),
                    p50_label: round(p50, 3),
                    p25_label: round(p25, 3),
                    p5_label: round(p5, 3),
                    peaks_label: peak_count,
                    stel_label: float(short_limit),
                    stel_window_label: short_window,
                    stel_exceed_label: round(short_full_exceed_min, 2),
                    stel_episodes_label: short_full_episodes,
                    exp_limit_label: float(long_limit),
                    exp_exceed_label: round(long_exceed_min, 2),
                    twa_conc_label: round(window_twa, 3),
                    twa_window_label: twa_window,
                }
            )

        result = pd.DataFrame(rows)

        # --- append to file if requested ---------------------------------------
        if filename and not result.empty:
            fname = str(filename)
            lower = fname.lower()
            if sheet_name:
                shname = str(sheet_name)
            else:
                shname = f"{metric} summary"

            if lower.endswith(".csv"):
                if os.path.exists(fname):
                    result.to_csv(fname, mode="a", header=False, index=False)
                else:
                    result.to_csv(fname, mode="w", header=True, index=False)
            elif lower.endswith((".xlsx", ".xls")):
                if os.path.exists(fname):
                    with pd.ExcelWriter(
                        filename,
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace",  # requires pandas ≥ 1.4
                    ) as writer:
                        result.to_excel(writer, sheet_name=shname, index=False)
                else:
                    result.to_excel(fname, sheet_name=shname, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension for '{filename}'. Use .csv or .xlsx."
                )
            print(f"\nExposure summary saved to: {filename}")

        # --- pretty terminal print (transposed) --------------------------------
        if not result.empty:
            result_t = result.set_index("Segment").T
            display_df = result_t.reset_index().rename(columns={"index": "Metric"})
            table_data = cast(
                Mapping[str, Iterable[Any]],
                display_df.to_dict(orient="list"),
            )

            print(f"\nExposure summary by segment for metric '{metric}' ({unit}):\n")
            print(
                tabulate(
                    table_data,
                    headers="keys",
                    tablefmt="pretty",
                    floatfmt=".3f",
                )
            )

        return result

    # snake_case aliases for PEP 8 consistency
    pm_calc = PM_calc
    
    ###########################################################################
    
    def rebin_bin_edges(self, new_bin_edges, inplace: bool = True):
        """Description:
            Normalize the size distribution by Δlog₁₀(Dp) (dx/dlogDp).
    
        Args:
            new_bin_edges (np.array): A list of 
            inplace (bool): If True, normalize this object in place and
                return it. If False, perform the normalization on a deep
                copy and return the new instance.
    
        Returns:
            Aerosol2D | None: The normalized object (self or a new copy)
                when normalization is applied. If the dtype already
                contains "/dlogDp", no changes are made and None is
                returned.
    
        Raises:
            ValueError: If the number of size-bin columns does not match
                the number of Δlog₁₀(Dp) widths derived from bin_edges.
                Check that bin_edges and the PSD columns are consistent.
    
        Notes:
            Detailed description:
                The method computes Δlog₁₀(Dp) from bin_edges and divides
                each size-bin column by its corresponding width. The dtype
                string is updated to append "/dlogDp" (for example
                "dN" → "dN/dlogDp"). Only size-bin columns are modified;
                other columns in data (including Total_conc and activity
                masks) are left unchanged.
    
            Theory:
                Plotting or comparing size distributions on a logarithmic
                diameter axis is often done using dN/dlogDp, dM/dlogDp,
                etc., so that equal logarithmic bin widths represent equal
                contributions when integrating over logDp. This method
                implements that per-bin normalization.
    
        Examples:
            Prepare a PSD for log-diameter plotting:
    
            .. code-block:: python
    
                elpi.normalize_logdp()
                elpi.plot_psd()
        """
        
        out = self if inplace else self.copy_self()
        
        out.unnormalize_logdp()

        # Convert inputs
        bin_edges = np.asarray(out.bin_edges, dtype=float)
        new_bin_edges = np.asarray(new_bin_edges, dtype=float)
        
        #Ensure compatibility of new_bind_edges
        if new_bin_edges.ndim != 1:
            raise ValueError("new_bin_edges must be 1D sequences.")
    
        if np.any(new_bin_edges <= 0):
            raise ValueError("All bin edges must be > 0 for log-space rebinning.")
    
        if np.any(np.diff(new_bin_edges) <= 0):
            raise ValueError("new_bin_edges must be strictly increasing.")
        
        if new_bin_edges[0]<bin_edges[0]:
            raise ValueError("smallest new bin must be equal to or larger than the smallest old bin.")
            
        if new_bin_edges[-1]>bin_edges[-1]:
            raise ValueError("largest new bin must be equal to or smaller than the largest old bin.")
        
        # Log10 edges and widths
        log_edges = np.log10(bin_edges)
        log_new_edges = np.log10(new_bin_edges)
    
        old_dlog = np.diff(log_edges)
        new_dlog = np.diff(log_new_edges)
    
        n_old = len(old_dlog)
        n_new = len(new_dlog)
    
        # Build transfer matrix T such that:
        # new_totals = old_totals @ T
        T = np.zeros((n_old, n_new), dtype=float)
    
        for i in range(n_old):
            old_lo = log_edges[i]
            old_hi = log_edges[i + 1]
            old_width = old_hi - old_lo
    
            for j in range(n_new):
                new_lo = log_new_edges[j]
                new_hi = log_new_edges[j + 1]
    
                overlap = min(old_hi, new_hi) - max(old_lo, new_lo)
                if overlap > 0:
                    T[i, j] = overlap / old_width
    
        # Convert dataframe to numeric array
        values = out.data[out.bin_mids.astype(str)].to_numpy(dtype=float)
    
        # Conservative rebinning
        new_totals = values @ T

        # Use geometric midpoints for new column labels
        new_bin_mids = np.round(np.sqrt(new_bin_edges[:-1] * new_bin_edges[1:]),2)
    
        # --- assemble object -----------------------------------------------------
        
        
        # Distribution block (positions may vary in different exports; adjust if needed)
        total_conc = pd.DataFrame(out._ensure_data_robustness(np.nansum(new_totals, axis=1)), columns=["Total_conc"])
        dist_df = pd.DataFrame(new_totals, columns=new_bin_mids.astype(str)).set_index([out.time])
        final_df = pd.concat([ total_conc, dist_df], axis=1)
        
        out._data = final_df
        out._meta["bin_edges"] = new_bin_edges
        out._meta["bin_mids"] = new_bin_mids
        out.mark_activities(out.activity_periods)

        return out
