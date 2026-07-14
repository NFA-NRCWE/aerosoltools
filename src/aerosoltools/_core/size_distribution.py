"""Size-distribution maths for 2D data: basis conversions (dN/dS/dV/dM),
log-normalisation, density rescaling and bin-edge rebinning."""

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:  # only for type hints; avoids a runtime circular import
    from ..aerosol2d import Aerosol2D


class SizeConversionMixin:
    """Convert between distribution bases and rescale/rebin the size axis."""

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

    def dtype_converter(self, dtype: str = "dN", inplace: bool = True):
        """Convert the size distribution to a chosen base data type.

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

    def normalize_logdp(self, inplace: bool = True):
        """Normalize the size distribution by Δlog₁₀(Dp) (dx/dlogDp).

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

    def unnormalize_logdp(self, inplace: bool = True):
        """Undo Δlog₁₀(Dp) normalization (dx/dlogDp → base form).

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

    def _recompute_diameters_for_density(self, density: float, old: float) -> bool:
        """Hook: recompute the size axis for a new density (default: no-op).

        Instruments whose reported diameter is density-dependent (e.g. the ELPI)
        override this to rebuild the diameters — and, where needed, the per-bin
        number — for the new density. Return ``True`` if the recompute was
        handled here (in which case :meth:`set_density` only applies the residual
        mass rescale); return ``False`` (the default) to fall through to the
        standard mass-only rescale.

        Args:
            density (float): The new particle density (g/cm³).
            old (float): The previous density (g/cm³).

        Returns:
            bool: Whether this hook handled the diameter recompute.
        """
        return False

    def set_density(self, density: Union[float, int] = 1.0):
        """Set or update the assumed particle density (g/cm³).

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

        # Some instruments report a density-dependent particle size, so a density
        # change must recompute the diameters (and possibly the number) rather
        # than only rescaling mass. Those classes override the hook below (e.g.
        # ELPI); the default is a no-op, so the standard mass rescale runs.
        if self._recompute_diameters_for_density(density, old):
            # Mass still also scales with density (M ∝ ρ·V) when mass-based.
            if "dM" in str(self.dtype) and old > 0:
                factor = density / old
                self._data[self._sizebin_headers] *= factor
                if "Total_conc" in self._data.columns:
                    self._data["Total_conc"] *= factor
            return self

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

    def rebin_bin_edges(self, new_bin_edges, inplace: bool = True):
        """Normalize the size distribution by Δlog₁₀(Dp) (dx/dlogDp).

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

        # Ensure compatibility of new_bind_edges
        if new_bin_edges.ndim != 1:
            raise ValueError("new_bin_edges must be 1D sequences.")

        if np.any(new_bin_edges <= 0):
            raise ValueError("All bin edges must be > 0 for log-space rebinning.")

        if np.any(np.diff(new_bin_edges) <= 0):
            raise ValueError("new_bin_edges must be strictly increasing.")

        if new_bin_edges[0] < bin_edges[0]:
            raise ValueError(
                "smallest new bin must be equal to or larger than the smallest old bin."
            )

        if new_bin_edges[-1] > bin_edges[-1]:
            raise ValueError(
                "largest new bin must be equal to or smaller than the largest old bin."
            )

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
        new_bin_mids = np.round(np.sqrt(new_bin_edges[:-1] * new_bin_edges[1:]), 2)

        # --- assemble object -----------------------------------------------------

        # Distribution block (positions may vary in different exports; adjust if needed)
        total_conc = pd.DataFrame(
            out._ensure_data_robustness(np.nansum(new_totals, axis=1)),
            columns=["Total_conc"],
        )
        dist_df = pd.DataFrame(new_totals, columns=new_bin_mids.astype(str)).set_index(
            [out.time]
        )
        final_df = pd.concat([total_conc, dist_df], axis=1)

        out._data = final_df
        out._meta["bin_edges"] = new_bin_edges
        out._meta["bin_mids"] = new_bin_mids
        out.mark_activities(out.activity_periods)

        return out
