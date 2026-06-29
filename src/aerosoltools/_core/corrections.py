"""Instrument corrections for 2D data: calibration and diffusion losses."""

import numpy as np

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover - typing_extensions fallback
    from typing_extensions import override  # noqa: F401


class CorrectionMixin:
    """Apply calibration factors and diffusion-loss corrections."""

    @override
    def calibrate(
        self,
        parameter: str | int = "bins",
        fit_function=None,
        Variables: dict = {"m": 1},
        inplace: bool = True,
    ):
        # m: Union[int, float, list] = 1, b: Union[int, float, list] = 0, inplace: bool = True):
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

            def fit_function(x, m, b=0):
                # Calculates a first order equation.
                return m * x + b

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
            if (
                parameter != "bins"
                and parameter not in out._data
                and parameter not in out._extra_data
            ):
                raise LookupError(f"Chosen parameter '{parameter}' is invalid")
        else:
            raise LookupError("Chosen parameter is invalid")

        # Apply the correction to the chosen parameter
        if parameter != "bins" and all_lists:
            raise ValueError(
                "List-valued Variables are only supported when parameter='bins'"
            )

        elif parameter == "bins":

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
            out._data["Total_conc"] = self._ensure_data_robustness(out_sum)
        else:
            if parameter in out._data:
                out._data[parameter] = self._ensure_data_robustness(
                    fit_function(out.data[parameter], **Variables)
                )
            elif parameter in out._extra_data:
                out._extra_data[parameter] = self._ensure_data_robustness(
                    fit_function(out._extra_data[parameter], **Variables)
                )
            else:
                raise KeyError(
                    f"Parameter '{parameter}' not found in data or extra_data"
                )

        if "calibrated" not in out._meta:
            out._meta["calibrated"] = {}

        out._meta["calibrated"][parameter] = Variables.copy()

        return out

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
