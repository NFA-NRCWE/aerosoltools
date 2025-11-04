from math import erf
from typing import Any, Optional, Sequence, Union, cast

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
        """
        1D array of bin edges in nanometers.

        Returns
        -------
        numpy.ndarray
            Shape (n,), dtype float64.
        """
        # ensure an array of floats; .copy() so callers can't mutate your internal state
        return np.asarray(self._meta["bin_edges"], dtype=np.float64).copy()

    @property
    def bin_mids(self) -> NDArray[np.float64]:
        """
        1D array of bin mids in nanometers.

        Returns
        -------
        numpy.ndarray
            Shape (n,), dtype float64.
        """
        return np.asarray(self._meta["bin_mids"], dtype=np.float64).copy()

    @property
    def density(self) -> float:
        """
        Unit of the measurements.

        Returns
        -------
        float
            Particle density in "g/cm³".
        """
        return self._meta.get("density")  # type: ignore

    @property
    def metadata(self):
        """
        Meta data as extracted upon loading the data

        Returns
        -------
        dict
            Meta data related to the loaded data.

        """
        return self._meta

    @property
    def size_data(self) -> pd.DataFrame:
        """
        Size-bin concentration data.

        Returns
        -------
        pandas.DataFrame
            Columns for all size bins in the dataset, in the order of `_sizebin_headers`.
        """
        return self.data.loc[:, self._sizebin_headers]

    @property
    def _sizebin_headers(self) -> list:
        """
        Headers of the sizebin concentration columns within main DataFrame

        Returns
        -------
        list
            Headers to access sizebin columns.
        """
        return [str(x) for x in self.bin_mids]

    ###########################################################################
    """############################# Functions #############################"""
    ###########################################################################

    def convert_to_mass_concentration(self, inplace: bool = True):
        """
        Convert particle size distribution data to mass concentration (ug/m³) based on current data type.

        Parameters
        ----------
        inplace : bool, optional
            If True (default), modifies the current instance in-place.
            If False, returns a new instance with converted mass concentration data.

        Returns
        -------
        self or aerosolxd
            Updated instance with mass concentration data, either in-place or as a copy.
        """
        if "dM" in self.dtype:
            print("Data is already in mass concentration (ug/m³).")
            return self if inplace else self.copy_self()

        bin_radii = self.bin_mids / 2.0  # convert diameter to radius in nm

        if "dS" in self.dtype:
            # Convert from surface area to number, then to volume, then to mass
            surface_area_per_particle = 4 * np.pi * bin_radii**2
            number_distribution = self.size_data.copy() / surface_area_per_particle
            volume_per_particle = (4 / 3) * np.pi * bin_radii**3
            volume_distribution = number_distribution * volume_per_particle
            mass_distribution = (
                volume_distribution * self.density * 1e-9
            )  # convert nm³ to ug

        elif "dV" in self.dtype:
            # Convert from volume to mass directly
            mass_distribution = self.size_data.copy() * self.density * 1e-9

        elif "dN" in self.dtype:
            # Convert from number to volume, then to mass
            volume_per_particle = (4 / 3) * np.pi * bin_radii**3
            volume_distribution = self.size_data.copy() * volume_per_particle
            mass_distribution = volume_distribution * self.density * 1e-9
        else:
            raise ValueError("Unknown data type for conversion to mass.")

        # Apply the results
        target_instance = self if inplace else self.copy_self()
        target_instance._data[self._sizebin_headers] = mass_distribution
        target_instance._meta["unit"] = "ug/m³"
        target_instance._meta["dtype"] = "dM"

        # Update total concentration
        # Ensure unnormalized data before summing
        if "/dlogDp" in target_instance.dtype:
            unnormalized = target_instance.unnormalize_logdp(inplace=False)
            sum_data = unnormalized.size_data.sum(axis=1)  # type: ignore
        else:
            sum_data = mass_distribution.sum(axis=1)

        target_instance._data["Total_conc"] = sum_data

        return target_instance

    ###########################################################################

    def convert_to_number_concentration(self, inplace: bool = True):
        """
        Convert particle size distribution data to number concentration (cm⁻³)
        from the current data type.

        Parameters
        ----------
        inplace : bool, optional
            If True (default), modifies the current instance in-place.
            If False, returns a new instance with number concentration data.

        Returns
        -------
        self or aerosolxd
            Updated instance with number concentration data, either in-place or as a copy.
        """
        if "dN" in self.dtype:
            print("Data is already in number concentration (cm⁻³).")
            return self if inplace else self.copy_self()

        bin_radii = self.bin_mids / 2.0  # nm

        if "dV" in self.dtype:
            # Convert from volume to number
            volume_per_particle = (4 / 3) * np.pi * bin_radii**3  # nm³
            number_distribution = self.size_data.copy() / volume_per_particle

        elif "dM" in self.dtype:
            # Convert from mass to volume, then to number
            volume_distribution = self.size_data.copy() / self.density * 1e9  # nm³
            volume_per_particle = (4 / 3) * np.pi * bin_radii**3
            number_distribution = volume_distribution / volume_per_particle

        elif "dS" in self.dtype:
            # Convert from surface area to number
            surface_area_per_particle = 4 * np.pi * bin_radii**2
            number_distribution = self.size_data.copy() / surface_area_per_particle

        else:
            raise ValueError("Unknown data type for conversion to number.")

        # Apply the results
        target_instance = self if inplace else self.copy_self()
        target_instance._data[self._sizebin_headers] = number_distribution
        target_instance._meta["unit"] = "cm⁻³"
        target_instance._meta["dtype"] = "dN"

        # Update total concentration
        # Ensure unnormalized data before summing
        if "/dlogDp" in target_instance.dtype:
            unnormalized = target_instance.unnormalize_logdp(inplace=False)
            sum_data = unnormalized.size_data.sum(axis=1)  # type: ignore
        else:
            sum_data = number_distribution.sum(axis=1)

        target_instance._data["Total_conc"] = sum_data

        return target_instance

    ###########################################################################

    def convert_to_surface_concentration(self, inplace: bool = True):
        """
        Convert particle size distribution data to surface area concentration (nm²/cm³)
        based on the current data type.

        Parameters
        ----------
        inplace : bool, optional
            If True (default), modifies the current instance in-place.
            If False, returns a new instance with surface area concentration data.

        Returns
        -------
        self or aerosolxd
            Updated instance with surface area concentration data, either in-place or as a copy.
        """
        if "dS" in self.dtype:
            print("Data is already in surface area concentration (nm²/cm³).")
            return self if inplace else self.copy_self()

        bin_radii = self.bin_mids / 2.0  # in nm
        surface_area_per_particle = 4 * np.pi * bin_radii**2
        volume_per_particle = (4 / 3) * np.pi * bin_radii**3

        if "dV" in self.dtype:
            # Volume -> Number -> Surface Area
            number_distribution = self.size_data.copy() / volume_per_particle
            surface_area_distribution = number_distribution * surface_area_per_particle

        elif "dM" in self.dtype:
            # Mass -> Volume -> Number -> Surface Area
            volume_distribution = self.size_data.copy() / self.density * 1e9  # nm³/cm³
            number_distribution = volume_distribution / volume_per_particle
            surface_area_distribution = number_distribution * surface_area_per_particle

        elif "dN" in self.dtype:
            # Number -> Surface Area
            surface_area_distribution = (
                self.size_data.copy() * surface_area_per_particle
            )

        else:
            raise ValueError("Unknown data type for conversion to surface area.")

        # Apply the results
        target_instance = self if inplace else self.copy_self()
        target_instance._data[self._sizebin_headers] = surface_area_distribution
        target_instance._meta["unit"] = "nm²/cm³"
        target_instance._meta["dtype"] = "dS"

        # Update total concentration
        # Ensure unnormalized data before summing
        if "/dlogDp" in target_instance.dtype:
            unnormalized = target_instance.unnormalize_logdp(inplace=False)
            sum_data = unnormalized.size_data.sum(axis=1)  # type: ignore
        else:
            sum_data = surface_area_distribution.sum(axis=1)
        target_instance._data["Total_conc"] = sum_data

        return target_instance

    ###########################################################################

    def convert_to_volume_concentration(self, inplace: bool = True):
        """
        Convert particle size distribution data to volume concentration (nm³/cm³)
        based on the current data type.

        Parameters
        ----------
        inplace : bool, optional
            If True (default), modifies the current instance in-place.
            If False, returns a new instance with volume concentration data.

        Returns
        -------
        self or aerosolxd
            Updated instance with volume concentration data, either in-place or as a copy.
        """
        if "dV" in self.dtype:
            print("Data is already in volume concentration (nm³/cm³).")
            return self if inplace else self.copy_self()

        bin_radii = self.bin_mids / 2.0  # in nm
        volume_per_particle = (4 / 3) * np.pi * bin_radii**3
        surface_area_per_particle = 4 * np.pi * bin_radii**2

        if "dS" in self.dtype:
            # Surface Area -> Number -> Volume
            number_distribution = self.size_data.copy() / surface_area_per_particle
            volume_distribution = number_distribution * volume_per_particle

        elif "dM" in self.dtype:
            # Mass -> Volume
            volume_distribution = self.size_data.copy() / self.density * 1e9  # nm³/cm³

        elif "dN" in self.dtype:
            # Number -> Volume
            volume_distribution = self.size_data.copy() * volume_per_particle

        else:
            raise ValueError("Unknown data type for conversion to volume.")

        # Apply the results
        target_instance = self if inplace else self.copy_self()
        target_instance._data[self._sizebin_headers] = volume_distribution
        target_instance._meta["unit"] = "nm³/cm³"
        target_instance._meta["dtype"] = "dV"

        # Update total concentration
        # Ensure unnormalized data before summing
        if "/dlogDp" in target_instance.dtype:
            unnormalized = target_instance.unnormalize_logdp(inplace=False)
            sum_data = unnormalized.size_data.sum(axis=1)  # type: ignore
        else:
            sum_data = volume_distribution.sum(axis=1)
        target_instance._data["Total_conc"] = sum_data

        return target_instance

    ###########################################################################
    def dtype_converter(self, dtype: str = "dN", inplace: bool = True):
        """
        Convert particle size distribution data the chosen datatype based on current data type.

        Parameters
        ----------

        dtype : str, optional
            Designate the desired datatype for the data to be converted into.
            Defaults to 'dN'

        inplace : bool, optional
            If True (default), modifies the current instance in-place.
            If False, returns a new instance with converted mass concentration data.

        Returns
        -------
        self or aerosolxd
            Updated instance with mass concentration data, either in-place or as a copy.
        """
        if dtype == "dN":
            return self.convert_to_number_concentration(inplace)
        elif dtype == "dS":
            return self.convert_to_surface_concentration(inplace)
        elif dtype == "dV":
            return self.convert_to_volume_concentration(inplace)
        elif dtype == "dM":
            return self.convert_to_mass_concentration(inplace)

    ###########################################################################
    def set_density(self, density: Union[float, int] = 1.0):
        """
        Set density of the aerosol particles in g/cm3

        Parameters
        ----------
        density : float
            Density of the aerosol particles.

        Returns
        -------
        class: Aerosol2D
            The updated density data. If the data was already mass-based then the
            updated density is applied immidiatly.
        """
        if "dM" in self.dtype:
            unit_density_data = self.size_data.copy() / self.density
            new_density_data = unit_density_data * density
            self._data[self._sizebin_headers] = new_density_data

        self._meta["density"] = density
        return self

    ###########################################################################
    def PM_calc(self, dtype: str = "dM", PM: float = 4.2, Lower_lim: float = 0):
        """
        Function to calculate Px values from size bins from an array.

        Parameters
        ----------
        dtype : str, optional
            A key to designate what datatype the PM_calc should return.
            It adds the selected datetype to the dtype_converter function.
            Options are: 'dN', 'dS', 'dV', 'dM'.Defaults to 'dM'.
            This change will not affect the original data.

        PM : integer or float
            PM indicates the D50% for calculating the PM value according to ISO 7708:1995(E)
            Typical values would be; 1, 2.5, 4.2 or 10. (all in µm)
            Default is 4.2 µm.

        Lower_lim: integer or float
            Lower_lim calculates D50% according to ISO 7708:1995(E), for a diameter value
            smaller than the chosen PM. This value is then sustracted from the value caluclated from the PM.
            Example:  PM_calc('dN',PM=10,Lower_lim=4.2) would return the number of particles between 4.2 and 10 µm.
            Default is 0, which returns the unaltered PM.

        Returns
        -------
        Data_return : numpy.array
            An array of mass concentration data. The first column is datetime.
            The other column or columns are PM values corresponding to each time for
            the specified PM limit.
        """
        # Prepare the data for
        data_copy = self.copy_self()
        # Maintaines total concentration mask
        mask = data_copy.data["Total_conc"].notna()
        # Unnormalize and ensure correct dtype
        data_copy.unnormalize_logdp()
        data_copy.dtype_converter(dtype)

        # Convert to
        # Remove the datetime and total conc columns as these are not needed for PM
        # calculations
        data_copy = data_copy.size_data

        # Run through the or all of the PM limits specified

        # convert the current PM limit frim um to nm
        PM_lim = PM * 1000

        # Sizebins
        bin_mids = np.array(self._sizebin_headers).astype(float)
        # EN 481 calculation
        Y = np.log(bin_mids / PM_lim) / (2**0.5 * np.log(1.5))

        Frac = 0.5 * (1 + np.vectorize(erf)(-Y))
        # Apply the fraction to the mass concentration of the bin.
        mass_frac = np.nansum(data_copy * Frac, axis=1)

        if Lower_lim > 0:
            if Lower_lim < PM:

                # convert the current PM limit from um to nm
                lower_lim = Lower_lim * 1000
                # EN 481 calculation
                Y = np.log(bin_mids / lower_lim) / (2**0.5 * np.log(1.5))

                Frac = 0.5 * (1 + np.vectorize(erf)(-Y))
                # Apply the fraction to the mass concentration of the bin.
                lower_mass_frac = np.nansum(data_copy * Frac, axis=1)
                mass_frac = mass_frac - lower_mass_frac
            else:
                raise ValueError("Lower_lim is larger than the chosen PM")

        mass_frac[mass_frac == 0] = np.nan  # where(mass_frac==0,mass_frac,np.nan)

        # Add the two values for a final mass concentration
        self._extra_data[f"P{dtype[-1]}{PM}"] = mass_frac  # .where(mask,np.nan)

        self._extra_data.set_index(self.time, inplace=True)
        self._extra_data[f"P{dtype[-1]}{PM}"] = self._extra_data[
            f"P{dtype[-1]}{PM}"
        ].where(mask, np.nan)

        return self

    ###########################################################################

    def normalize_logdp(self, inplace: bool = True):
        """
        Normalize the size distribution data by dlogDp to obtain, e.g., dN/dlogDp.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies current instance in-place.
            If False, returns a new instance with normalized data.

        Returns
        -------
        self or aerosolxd
            Instance with normalized size distribution data.
        """
        if "/dlogDp" not in self.dtype:

            log_bin_edges = np.log10(self.bin_edges)
            dlog_dp = np.diff(log_bin_edges)

            bin_columns = self._sizebin_headers

            if len(dlog_dp) != len(bin_columns):
                raise ValueError("Mismatch between number of bins and dlogDp array.")

            normalized_data = self._data[bin_columns].copy().div(dlog_dp, axis=1)

            target = self if inplace else self.copy_self()
            target._data[bin_columns] = normalized_data

            target._meta["dtype"] = f"{self.dtype}/dlogDp"

            return target
        elif "/dlogDp" in self.dtype:
            return print("Warning: data already normalized; nothing was changed.")

    ###########################################################################

    def unnormalize_logdp(self, inplace: bool = True):
        """
        Reverse dlogDp normalization (e.g., convert dN/dlogDp to dN).

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies current instance in-place.
            If False, returns a new instance with unnormalized data.

        Returns
        -------
        self or aerosolxd
            Instance with unnormalized size distribution data.
        """

        if "/dlogDp" in self.dtype:

            log_bin_edges = np.log10(self.bin_edges)
            dlog_dp = np.diff(log_bin_edges)

            bin_columns = self._sizebin_headers

            if len(dlog_dp) != len(bin_columns):
                raise ValueError("Mismatch between number of bins and dlogDp array.")

            unnormalized_data = self._data[bin_columns].copy().mul(dlog_dp, axis=1)

            target = self if inplace else self.copy_self()
            target._data[bin_columns] = unnormalized_data
            target._meta["dtype"] = self.dtype.replace("/dlogDp", "")

            return target
        else:
            return print(
                "Warning: dtype does not contain '/dlogDp'; nothing was changed."
            )

    ###########################################################################

    def plot_psd(
        self, activities: Optional[list[str]] = None, normalize: bool = True, ax=None
    ):
        """
        Plot the average particle size distribution (PSD) for the entire dataset and optionally selected activities.

        Parameters
        ----------
        activities : list of str, optional
            List of activity names to include. If None, all defined activities are plotted.
        normalize : bool, optional
            Whether to normalize PSD to dlogDp before plotting. If data is already normalized, will respect that.
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object.
        ax : matplotlib.axes.Axes
            The matplotlib Axes object.
        """
        new_fig_created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            new_fig_created = True
        else:
            fig = ax.figure

        ax.set_xscale("log")
        ax.set_xlabel("Particle diameter (nm)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Determine normalization state
        is_already_normalized = "/dlogDp" in self.dtype
        bin_columns = self._sizebin_headers
        bin_mids = self.bin_mids
        log_bin_edges = np.log10(self.bin_edges)
        dlog_dp = np.diff(log_bin_edges)
        factor_series = pd.Series(dlog_dp, index=bin_columns)

        # Determine label based on normalization intent
        if normalize and not is_already_normalized:
            y_label_dtype = f"{self.dtype}/dlogDp"
        elif not normalize and is_already_normalized:
            y_label_dtype = self.dtype.replace("/dlogDp", "")
        else:
            y_label_dtype = self.dtype
        ax.set_ylabel(f"{y_label_dtype}, {self.unit}")

        # Colormap for activities
        all_activities = sorted(self._activity_periods.keys())
        color_map = plt.colormaps.get_cmap("gist_ncar")
        activity_colors = {
            activity: color_map(i / max(1, len(all_activities)))
            for i, activity in enumerate(all_activities)
        }

        # Plot selected or all activities
        selected_activities = activities if activities is not None else self.activities

        for activity in selected_activities:
            if activity not in self.activities:
                print(f"Activity '{activity}' not found. Skipping.")
                continue

            subset = self.data[self.data[activity]]
            if subset.empty:
                continue

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

    def correct_diffusion_losses(
        self,
        D_tube: float,
        L: float,
        Q: float,
        T: float = 293,
        P: float = 101300,
        inplace: bool = True,
    ):
        """
        Correct for diffusion losses in a sampling tube based on tubing geometry,
        flow conditions, and particle sizes.

        Parameters
        ----------
        D_tube : float
            Diameter of the tubing (in meters).
        L : float
            Length of the tubing (in meters).
        Q : float
            Volumetric flow through the tubing (in L/min).
        T : float, optional
            Temperature in Kelvin. Default is 293 K.
        P : float, optional
            Pressure in Pascals. Default is 101300 Pa.
        inplace : bool, optional
            Whether to modify the current instance or return a new one. Default is True.

        Returns
        -------
        Aerosol2D
            Instance with diffusion-corrected sizebin data.
        """
        # Constants
        k = 1.380649e-23  # Boltzmann constant
        Dp = np.array(self.bin_mids) * 1e-9  # Convert nm to meters
        Q_m3s = Q / (1000 * 60)  # Convert L/min to m³/s
        A = 0.25 * np.pi * D_tube**2
        V = Q_m3s / A  # Flow velocity (m/s)

        # Mean free path (adjusted for P, T)
        mfp_std = 66.5e-9  # m
        mfp = (
            mfp_std * (101e3 / P) * (T / 293.15) * ((1 + 110 / 293.15) / (1 + 110 / T))
        )

        # Gas properties
        eta_std = 1.708e-5
        eta = (
            eta_std * (T / 273.15) ** 1.5 * (393.396 / (T + 120.246))
        )  # dynamic viscosity
        rho = 1.293 * (273.15 / T) * (P / 101300)  # gas density

        # Knudsen number and slip correction
        Kn = 2 * mfp / Dp
        Cc = 1 + Kn * (1.142 + 0.558 * np.exp(-0.999 / Kn))

        # Reynolds number
        Re = rho * V * D_tube / eta

        # Diffusion coefficient
        Dc = k * T * Cc / (3 * np.pi * eta * Dp)
        Sc = eta / (rho * Dc)
        xi = np.pi * Dc * L / Q_m3s

        # Sherwood number
        if Re < 2000:
            Sh = 3.66 + 0.2672 / (xi + 0.10079 * xi ** (1 / 3))
        else:
            Sh = 0.0118 * Re ** (7 / 8) * Sc ** (1 / 3)

        # Diffusion efficiency
        eff = np.exp(-Sh * xi)

        # Apply correction
        corrected = self.copy_self() if not inplace else self
        size_cols = corrected._sizebin_headers
        corrected._data[size_cols] = corrected._data[size_cols].div(eff, axis=1)
        corrected._data["Total Concentration"] = corrected._data[size_cols].sum(axis=1)

        # Store efficiency in metadata for reference
        corrected._meta["diffusion_efficiency"] = eff.tolist()
        corrected._meta["diffusion_loss_corrected"] = True

        return corrected

    ###########################################################################
    def plot_PM_timeseries(
        self,
        PM_values=[0.5, 2.5, 10],
        dtype="dM",
        activity="All data",
        fraction=False,
        cummulative=False,
    ):
        """
        Plot the total concentration

        Parameters
        ----------

        PM_values : list, optional
            A list of Dp values for which PM should be plotted.
            The default is [0.5,2.5,10].

        dtype : string, optional
            Keyword to specify the datatype. The available options are "number",
            "surface" and "mass". Default is "number"
        activity : str
            Designates which activity should be plotted. Default is "All data"
        fraction : boolean
            Decides whether the PM should be plotted as values, or as fraction of the biggest PM value.
            By default the fraction is off. If True, the total value will be plotted as a line.
        cummulative : boolean
            Determines whether the PM value reported in the legend should be PM values or the difference
            between two PM values. Default is False, which PM_values[0.5,2.5,10] returns the values:
            P0.5= PM0.5,
            P2.5=PM2.5-PM0.5
            P10=PM10-PM2.5
            Chose True if regular PM values should be used.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle for the returned figure for saving.
        ax : matplotlib.axes._subplots.AxesSubplot
            Handles for the axis of the plot.

        """
        # Set up standardized values for later use

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
        # y_label={"Number"   :   "N$_{Total}$, cm$^{-3}$",
        #          "Surface"  :   "S$_{Total}$, nm$^{2}$ cm$^{-3}$",
        #          "Mass"     :   "m$_{Total}$, $\mu$g/m$^{3}$"}
        # Legend_label={'Number': 'PN',
        #               'Surface':'PS',
        #               'Mass':   'PM'}

        # Corrects for the first letter not being capital in the datatype

        data_copy = self.copy_self()
        data_copy.dtype_converter(dtype=dtype)

        # Generate a dictionary of the lists of PM values
        # PM={}
        for i in PM_values:
            data_copy.PM_calc(dtype=dtype, PM=i)

        mask = self.data[activity]

        PM_data = data_copy.extra_data.loc[mask]
        # If no ax is provided, figure and ax is generated here

        figure, ax = plt.subplots()
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        """
        Determines the type of plot. The default is a plot with total concentration
        and the total concentration each pn/pm values reases.
        If the Fraction has been turned on, the fractional values will be calculated instead
        """
        if fraction:
            ax.plot(
                PM_data[f"P{dtype[-1]}{PM_values[-1]}"], color="k", label="Total", lw=3
            )
            ax2 = ax.twinx()
            PM_fr = PM_data.copy()
            for i in range(0, len(PM_values)):
                pm = f"P{dtype[-1]}{PM_values[i]}"

                PM_fr[pm] = PM_data[pm] / PM_data[f"P{dtype[-1]}{PM_values[-1]}"]
                if i == 0:
                    avg_PM = round(PM_data[pm].mean(), 2)
                    sem_PM = round(PM_data[pm].std(), 2)
                    Label = f"{pm}: {avg_PM}+/-{sem_PM}"
                    ax2.fill_between(
                        self.time[mask],
                        PM_fr[pm],
                        alpha=0.75,
                        color=colors[i],
                        label=Label,
                    )

                else:
                    pm_1 = f"P{dtype[-1]}{PM_values[i-1]}"
                    if cummulative:
                        avg_PM = round(PM_data[pm].mean(), 2)
                        sem_PM = round(PM_data[pm].std(), 2)
                        Label = f"{pm}: {avg_PM}+/-{sem_PM}"
                    else:
                        avg_PM = round(
                            np.nanmean(PM_data[pm] - PM_data[pm_1], axis=0), 2
                        )
                        sem_PM = round(
                            np.nanstd(PM_data[pm] - PM_data[pm_1], axis=0), 2
                        )
                        Label = f"{pm}: {avg_PM}+/-{sem_PM}"
                    ax2.fill_between(
                        self.time[mask],
                        PM_fr[pm_1],
                        PM_fr[pm],
                        alpha=0.75,
                        color=colors[i],
                        label=Label,
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
            for i in range(0, len(PM_values)):
                pm = f"P{dtype[-1]}{PM_values[i]}"

                # Label=f"{pm} $\mu$m: {rounder(np.nanmean(PM[pm]),sem_PM)}"
                if i == 0:
                    avg_PM = round(PM_data[pm].mean(), 2)
                    sem_PM = round(PM_data[pm].std(), 2)
                    Label = f"{pm}: {avg_PM}+/-{sem_PM}"
                    ax.fill_between(
                        self.time[mask],
                        PM_data[pm],
                        alpha=1,
                        color=colors[i],
                        label=Label,
                    )
                else:
                    pm_1 = f"P{dtype[-1]}{PM_values[i-1]}"
                    if cummulative:
                        avg_PM = round(PM_data[pm].mean(), 2)
                        sem_PM = round(PM_data[pm].std(), 2)
                        Label = f"{pm}: {avg_PM}+/-{sem_PM}"
                    else:
                        avg_PM = round(
                            np.nanmean(PM_data[pm] - PM_data[pm_1], axis=0), 2
                        )
                        sem_PM = round(
                            np.nanstd(PM_data[pm] - PM_data[pm_1], axis=0), 2
                        )
                        Label = f"{pm}: {avg_PM}+/-{sem_PM}"
                    ax.fill_between(
                        self.time[mask],
                        PM_data[pm_1],
                        PM_data[pm],
                        color=colors[i],
                        label=Label,
                    )
            ax.legend(
                loc="best",
                title=f"Average values ({data_copy.unit})",
                fontsize=25,
                title_fontsize=25,
            )

        # Set the axis settings

        ax.set_ylim(0)
        ax.set_ylabel(f"P{dtype[-1]}, {data_copy.unit}")

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

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
    ) -> tuple[Figure, NDArray[Any]]:
        """
        Plot total concentration (top) and a size-resolved time series (bottom).

        Parameters
        ----------
        y_tot : tuple, optional
            Y-axis limits for total concentration (min, max). Default auto.
        y_3d : tuple, optional
            Colorbar scale limits for 2D mesh (min, max). Default auto.
        log : bool, optional
            Whether to apply logarithmic color scaling. Default True.
        ax1 : matplotlib.axes.Axes, optional
            Axis for the top plot. If provided, ax2 must also be provided.
        ax2 : matplotlib.axes.Axes, optional
            Axis for the mesh plot. If provided, ax1 must also be provided.
        mark_activities : bool or list of str, optional
            Passed to `plot_total_conc()` to highlight activity periods.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axs : np.ndarray
            Array of axes and colorbar handle: [ax1, ax2, colorbar].
        """
        if (ax1 is None) != (ax2 is None):
            raise ValueError("You must provide both ax1 and ax2, or neither.")

        if ax1 is None and ax2 is None:
            newplot = True
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
        else:
            assert ax1 is not None and ax2 is not None
            fig: Figure = ax1.get_figure()  # type: ignore
            newplot = False
        ax1 = cast(Axes, ax1)
        ax2 = cast(Axes, ax2)

        time = self.time
        total = self.total_concentration
        data = self.size_data
        bin_edges = self.bin_edges

        # Top panel: total concentration
        _, ax_new = self.plot_total_conc(ax=ax1, mark_activities=mark_activities)

        ax1 = ax_new

        # Set y-limits for the total_conc plot
        if y_tot != (0, 0):
            ymin = y_tot[0] if y_tot[0] != 0 else total.min() * 0.98
            ymax = y_tot[1] if y_tot[1] != 0 else total.max() * 1.02
            ax1.set_ylim(ymin, ymax)

        dt = (time[1] - time[0]) / 2

        # Generate edges: center ± half step
        dt = (time[1] - time[0]) / 2
        time_edges = pd.DatetimeIndex(np.append(time - dt, [time[-1] + dt]))  # type: ignore

        x_grid, y_grid = np.meshgrid(time_edges, bin_edges, indexing="ij")

        # Handle color scale limits
        z_data = data
        if y_3d != (0, 0):
            zmin, zmax = y_3d
            if zmin != 0:
                z_data = z_data.clip(lower=zmin)
            if zmax == 0:
                zmax = z_data.max().max()
        else:
            zmin = z_data.min().min()
            zmax = z_data.max().max()

        # Define color scale
        if log:
            if (z_data <= 0).any().any():
                raise ValueError(
                    "Data contains zeros or negatives; cannot use log color scale."
                )
            norm = LogNorm(vmin=zmin, vmax=zmax)
        else:
            norm = Normalize(vmin=zmin, vmax=zmax)

        # Mesh plot
        mesh = ax2.pcolormesh(
            x_grid, y_grid, z_data, cmap="jet", norm=norm, shading="flat"
        )

        # Set axis labels and scale
        ax2.set_yscale("log")
        ax2.set_ylabel("Dp, nm")
        ax2.set_xlabel("Time")
        if newplot:
            ax1.set_xlabel("")
        # Use matplotlib's default date handling
        ax2.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )

        # # Add colorbar
        col = fig.colorbar(mesh, ax=[ax1, ax2])
        col.set_label(f"{self.dtype}, {self.unit}")

        # Styling
        ax1.tick_params(axis="y", which="both", direction="out", length=6, width=2)
        ax2.tick_params(axis="y", which="both", direction="out", length=6, width=2)

        return fig, np.array([ax1, ax2, col], dtype=object)

    ###########################################################################

    def summarize(self, filename=None):
        """
        Summarize aerosol characteristics for each activity period.

        Metrics included for each segment:
        - PNC (cm⁻³): Particle number concentration (sum across number bins)
        - PM1, PM2.5, PM4 (respirable), PM10 (inhalable) in µg/m³, calculated using partial bin inclusion
        - Total mass concentration (µg/m³)
        - Mode diameter (nm): Bin midpoint with max number conc per timestep
        - Median diameter (nm): 50% of cumulative number distribution
        - GMD (nm): Geometric mean diameter (log-space, number-weighted)

        All metrics include standard deviation across the segment.

        Parameters
        ----------
        filename : str, optional
            If provided, saves the summary to Excel.

        Returns
        -------
        pd.DataFrame
            Summary statistics for all defined activities.
        """

        number_data = self.convert_to_number_concentration(inplace=False)
        mass_data = self.convert_to_mass_concentration(inplace=False)

        # Make a copy and populate it with PM values
        data_copy = self.copy_self()
        for PM in [1, 2.5, 4, 10]:
            data_copy.PM_calc(PM=PM)

        rows = []

        for activity in self.activities:
            mask = self.data[activity]
            if mask.sum() == 0:
                continue

            num_df = number_data.size_data.loc[mask]
            mass_df = mass_data.size_data.loc[mask]

            # PNC
            pnc_series = num_df.sum(axis=1)
            pnc = pnc_series.mean()
            pnc_std = pnc_series.std()

            # PM fractions (with partial bins)

            PM_copy = data_copy.extra_data.loc[mask]

            pm1, pm1_std = PM_copy["PM1"].mean(), PM_copy["PM1"].std()
            pm2_5, pm2_5_std = PM_copy["PM2.5"].mean(), PM_copy["PM2.5"].std()
            pm4, pm4_std = PM_copy["PM4"].mean(), PM_copy["PM4"].std()
            pm10, pm10_std = PM_copy["PM10"].mean(), PM_copy["PM10"].std()

            # Total mass
            total_mass_series = mass_df.sum(axis=1)
            total_mass = total_mass_series.mean()
            total_mass_std = total_mass_series.std()

            # Per-time-step size metrics
            mode_d_list = []
            median_d_list = []
            gmd_list = []

            for _, row in num_df.iterrows():
                dist = row.values
                total = dist.sum()
                if total == 0:
                    continue

                # Mode diameter
                mode_d = np.array(self.bin_mids)[np.argmax(dist)]
                mode_d_list.append(mode_d)

                # Median diameter
                cum = np.cumsum(dist)
                cum /= cum[-1]
                median_idx = np.searchsorted(cum, 0.5)
                median_d = np.array(self.bin_mids)[median_idx]
                median_d_list.append(median_d)

                # GMD
                gmd = np.exp(np.sum(np.log(np.array(self.bin_mids)) * dist) / total)
                gmd_list.append(gmd)

            # Final size stats
            mode_d = np.mean(mode_d_list)
            mode_d_std = np.std(mode_d_list)
            median_d = np.mean(median_d_list)
            median_d_std = np.std(median_d_list)
            gmd = np.mean(gmd_list)
            gmd_std = np.std(gmd_list)

            # Store row
            rows.append(
                [
                    activity,
                    round(pnc, 2),
                    round(pnc_std, 2),
                    round(pm1, 2),
                    round(pm1_std, 2),
                    round(pm2_5, 2),
                    round(pm2_5_std, 2),
                    round(pm4, 2),
                    round(pm4_std, 2),
                    round(pm10, 2),
                    round(pm10_std, 2),
                    round(total_mass, 2),
                    round(total_mass_std, 2),
                    round(mode_d, 1),
                    round(mode_d_std, 1),
                    round(median_d, 1),
                    round(median_d_std, 1),
                    round(gmd, 1),
                    round(gmd_std, 1),
                ]
            )

        summary = pd.DataFrame(
            rows,
            columns=[
                "Segment",
                "PNC (cm⁻³)",
                "PNC std (cm⁻³)",
                "PM1 (µg/m³)",
                "PM1 std (µg/m³)",
                "PM2.5 (µg/m³)",
                "PM2.5 std (µg/m³)",
                "PM4 (µg/m³)",
                "PM4 std (µg/m³)",
                "PM10 (µg/m³)",
                "PM10 std (µg/m³)",
                "Total Mass (µg/m³)",
                "Total Mass std (µg/m³)",
                "Mode Dp (nm)",
                "Mode Dp std (nm)",
                "Median Dp (nm)",
                "Median Dp std (nm)",
                "GMD (nm)",
                "GMD std (nm)",
            ],
        )

        if filename:
            summary.to_excel(filename, index=False)
            print(f"Summary saved to: {filename}")

        summary_t = summary.set_index("Segment").T
        print("\nSummary of aerosol properties (transposed):\n")
        print(tabulate(summary_t, headers="keys", tablefmt="pretty", floatfmt=".3f"))  # type: ignore
        return summary
