"""Lognormal multi-mode fitting of particle size distributions."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class FitMixin:
    """Fit one or more lognormal modes to a mean PSD."""

    def fit_psd(
        self,
        period="All data",
        mu=[150],
        sigma=[2],
        factor=[1000],
        log_scaling=True,
        binding=None,
        tolerance=10.0,
        mu_factor=None,
        weighting="variance",
        local_sigmas=None,
    ):
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
        local_sigmas: float, optional
            If given (and > 0), restrict the fit to size bins lying within
            ``local_sigmas`` geometric standard deviations of any mode's initial
            guess (in log10-diameter space). This fits each mode locally to the
            region where it responds, rather than forcing the lognormals to also
            describe non-modal background far from any peak. Default is None
            (use the full measured range).

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
            mu = np.log10(np.array(params[0::3]))
            sigma = np.log10(np.array(params[1::3]))
            factor = np.array(params[2::3])

            population = 0
            for i in range(0, len(mu)):

                population += (
                    (1 / (np.sqrt(2 * np.pi) * sigma[i]))
                    * np.exp(-((bin_mid - mu[i]) ** 2) / (2 * sigma[i] ** 2))
                ) * factor[i]

            return np.log10(population)

        def Normal(bin_mid, *params):

            bin_mid = np.log10(bin_mid)
            mu = np.log10(np.array(params[0::3]))
            sigma = np.log10(np.array(params[1::3]))
            factor = np.array(params[2::3])

            population = 0
            for i in range(0, len(mu)):

                population += (
                    (1 / (np.sqrt(2 * np.pi) * sigma[i]))
                    * np.exp(-((bin_mid - mu[i]) ** 2) / (2 * sigma[i] ** 2))
                ) * factor[i]

            return population

        ###
        data = self.copy_self()
        data.normalize_logdp()
        # Specify x and y data to fit
        xdata = np.array(data.bin_mids, dtype="float64")
        if period in data.activities:

            ydata = np.array(
                pd.DataFrame(
                    data.get_activity_data(period), columns=data.bin_mids.astype(str)
                ),
                dtype="float64",
            )
        elif type(period) == tuple:
            ydata = np.array(
                pd.DataFrame(
                    data.timecrop(start=period[0], end=period[1], inplace=False).data,
                    columns=data.bin_mids.astype(str),
                ),
                dtype="float64",
            )
        else:
            raise ValueError("Period chosen is neither an activity or a range of data")

        ymean = np.nanmean(ydata, axis=0)

        if len(xdata) != len(ymean):
            raise ValueError("Discrepency between number of bins and data")

        # Removes values of 0 or below from fitting
        mask = ymean > 0

        xdata = xdata[mask]
        ymean = ymean[mask]
        ydata_masked = ydata[:, mask]

        if len(ymean) == 0:
            raise ValueError("Empty dataset")

        n = ydata_masked.shape[0]

        mu = mu.copy()
        sigma = sigma.copy()
        factor = factor.copy()

        peak_number = len(mu)

        if peak_number * 3 >= len(ymean):
            raise ValueError("Peak number will lead to overfitting")

        if peak_number != len(sigma):
            raise ValueError("Missing input for initial guess")

        # Optional local fit: keep only the size bins within a window of each
        # mode (|log10(Dp) - log10(mu0)| <= local_sigmas * log10(sigma0)). This
        # makes the fit follow the placed modes instead of trying to also span
        # the non-modal background with one broad, unphysical lognormal — the
        # window scales with each mode's (geometric) width, so a narrower mode
        # is fit more locally.
        if local_sigmas is not None and local_sigmas > 0:
            logx = np.log10(xdata)
            keep = np.zeros(len(xdata), dtype=bool)
            for i in range(peak_number):
                keep |= np.abs(logx - np.log10(mu[i])) <= local_sigmas * np.log10(
                    sigma[i]
                )
            if keep.sum() < peak_number * 3:
                raise ValueError(
                    "Local fit window covers too few size bins; widen the "
                    "mode(s) or reduce the number of modes."
                )
            xdata = xdata[keep]
            ymean = ymean[keep]
            ydata_masked = ydata_masked[:, keep]

        # Gather all the initial guesses for parameters to fit in a list
        init_guess = []
        for i in range(peak_number):
            init_guess.extend([mu[i], sigma[i], factor[i]])

        # Generate bounds to reduce the risk of producing impossible or irrelevant modes
        low_bounds = [0.1 * min(xdata), 1.15, 0] * peak_number
        up_bounds = [
            10 * max(xdata),
            5,
            max(max(ymean), max(factor)) * 2.5,
        ] * peak_number

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

        if tolerance > 0:
            tolerance = tolerance / 100

        # binding=number_to_bool_list(binding,peak_number*3)
        if binding is None:
            binding = []

        for i in range(0, len(binding)):
            if binding[i] == True:
                low_bounds[i] = init_guess[i] * (1 - tolerance)
                up_bounds[i] = init_guess[i] * (1 + tolerance)

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
                model,
                xdata[valid],
                yfit[valid],
                p0=init_guess,
                bounds=(low_bounds, up_bounds),
            )
        elif weighting == "variance":
            sigma_fit = np.where(
                np.isfinite(sigma_fit) & (sigma_fit > 0), sigma_fit, np.nan
            )
            valid = np.isfinite(sigma_fit)
            popt, pcov = curve_fit(
                model,
                xdata[valid],
                yfit[valid],
                p0=init_guess,
                bounds=(low_bounds, up_bounds),
                sigma=sigma_fit[valid],
            )
        else:
            raise ValueError("weighting must be 'uniform' or 'variance'")

        # Get error estimates of the fits
        perr = np.sqrt(np.diag(pcov))

        # The next line of code sorts the data according to the desired focus; mode or population

        Modes = {"mu": popt[0::3], "sigma": popt[1::3], "factor": popt[2::3]}
        Error = {"mu": perr[0::3], "sigma": perr[1::3], "factor": perr[2::3]}

        # Returns the sorted fitted modes and their uncertainty.
        return Modes, Error
