"""Instrument corrections for 2D data: diffusion-loss transmission.

Calibration application now lives on the data object itself as
``apply_calibration`` (see :meth:`aerosoltools.aerosol1d.Aerosol1D.apply_calibration`)
and the cross-dataset fitting in
:mod:`aerosoltools.intercomparison.calibration`.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Typing-only host contract (the size-resolved facade this mixin runs on);
    # at runtime the base is ``object`` so composition/MRO is unchanged.
    from ._protocols import SizeResolvedData as _Host
else:
    _Host = object


class CorrectionMixin(_Host):
    """Apply diffusion-loss transmission corrections to size-resolved data."""

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
