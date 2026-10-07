"""Lognormal multi-mode fitting of particle size distributions."""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def lognormal_modes(dp_nm, modes):
    """Evaluate a sum of lognormal modes (dx/dlogDp) at diameters ``dp_nm``.

    This is the exact model :meth:`Aerosol2D.fit_psd` fits, exposed as a public
    helper so a script (or the GUI) can reconstruct the fitted curve from the
    modes ``fit_psd`` returns and plot it against the measured PSD::

        modes, err = data.fit_psd("Task 1", mu=[80, 200])
        triples = list(zip(modes["mu"], modes["sigma"], modes["factor"]))
        total, per_mode = at.lognormal_modes(data.bin_mids, triples)

    Args:
        dp_nm: Diameters (nm) to evaluate at.
        modes: Iterable of ``(mu, sigma, factor)`` triples — the peak diameter
            (nm), geometric standard deviation, and scaling factor of each mode,
            as returned by :meth:`fit_psd`.

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray]]: ``(total, per_mode)`` where
        ``total`` is the summed dx/dlogDp curve and ``per_mode`` holds each
        mode's individual curve, all in the same order as ``modes``.
    """
    x = np.log10(np.asarray(dp_nm, dtype=float))
    total = np.zeros_like(x)
    per_mode = []
    for mu, sigma, factor in modes:
        s = np.log10(sigma)
        comp = (
            factor
            * (1.0 / (_SQRT_2PI * s))
            * np.exp(-((x - np.log10(mu)) ** 2) / (2.0 * s**2))
        )
        per_mode.append(comp)
        total = total + comp
    return total, per_mode


class PSDFitResult(NamedTuple):
    """Typed result of :meth:`Aerosol2D.fit_psd`.

    A ``NamedTuple``, so it stays 100 % backward compatible with the historical
    ``modes, errors = data.fit_psd(...)`` two-tuple unpacking and ``result[0]`` /
    ``result[1]`` indexing, while adding self-documenting names and the helpers a
    script needs to reconstruct the fitted curve.

    Attributes:
        modes: Fitted parameters ``{"mu": array, "sigma": array, "factor":
            array}`` (peak diameters nm, geometric SDs, scaling factors), one
            entry per mode in the same order.
        errors: 1σ uncertainties in the same ``{"mu"/"sigma"/"factor": array}``
            shape.
    """

    modes: dict
    errors: dict

    @property
    def n_modes(self) -> int:
        """Number of fitted modes."""
        return len(self.modes.get("mu", []))

    def triples(self) -> list:
        """Modes as ``(mu, sigma, factor)`` triples for evaluation/plotting."""
        return list(zip(self.modes["mu"], self.modes["sigma"], self.modes["factor"]))

    def evaluate(self, dp_nm):
        """The fitted dx/dlogDp curve at ``dp_nm``: ``(total, per_mode)``.

        Uses the same :func:`lognormal_modes` model the fit is built on, so a
        script reproduces exactly the curve the GUI overlays::

            res = data.fit_psd("Task 1", mu=[80, 200])
            total, per_mode = res.evaluate(data.bin_mids)
        """
        return lognormal_modes(dp_nm, self.triples())


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
        PSDFitResult
            A ``NamedTuple`` ``(modes, errors)`` — so it still unpacks as
            ``modes, errors = data.fit_psd(...)`` — where ``modes`` and
            ``errors`` are ``{"mu": array, "sigma": array, "factor": array}``
            dicts (fitted parameters and their 1σ uncertainties). Use
            ``result.evaluate(dp)`` to reconstruct the fitted dx/dlogDp curve.
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

        def Normal(bin_mid, *params):
            # The measured-space model: the shared lognormal evaluator so the
            # fit, the public helper, and the GUI overlay are one implementation.
            triples = list(zip(params[0::3], params[1::3], params[2::3]))
            total, _ = lognormal_modes(bin_mid, triples)
            return total

        def Lognormal(bin_mid, *params):
            # log10 of the same model, so modes of very different population get
            # comparable leverage in the least-squares fit.
            return np.log10(Normal(bin_mid, *params))

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
        elif isinstance(period, tuple):
            ydata = np.array(
                pd.DataFrame(
                    data.timecrop(start=period[0], end=period[1], inplace=False).data,
                    columns=data.bin_mids.astype(str),
                ),
                dtype="float64",
            )
        else:
            raise ValueError("Period chosen is neither an activity or a range of data")

        if ydata.shape[0]==1:
            ydata = np.vstack((ydata[0]/1.01,ydata[0]*1.01))

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
            if binding[i]:
                low_bounds[i] = max(low_bounds[i],init_guess[i] * (1 - tolerance))
                up_bounds[i] = min(up_bounds[i],init_guess[i] * (1 + tolerance))

        # Build the model and the (optional) per-bin weights. log_scaling fits
        # log10(dx/dlogDp) so modes of very different population get comparable
        # leverage; otherwise the raw values are fit.
        if log_scaling:
            model, yfit = Lognormal, np.log10(ymean)
            sigma_y = np.nanstd(ydata_masked, axis=0, ddof=1) / np.sqrt(n)
            sigma_fit = sigma_y / (ymean * np.log(10))
        elif not log_scaling:
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

        # Typed result (a NamedTuple, so `modes, errors = fit_psd(...)` and
        # `result[0]`/`result[1]` keep working) carrying the fitted modes, their
        # uncertainty, and an evaluate() helper for the fitted curve.
        return PSDFitResult(Modes, Error)


    def dataset_psd_fitting(self,
                            def_fit: dict = 
                            {'mu'   : [50,150,1500],
                            'sigma' : [2,2,2],
                            'factor': [5E6,2E7,1E7]},
                            error_lim: float|int = 5,
                            nan_lim: int = 9,
                            inplace: bool = True,):
    
        """
        A function to fit one or multiple peaks following lognormal distribution,
        throughout a complete dataset. The function is made with NS+OPS in mind, as
        the NS data can suffer from empty bins based on the type of particles
        and abundance. For that reason the dtype is also converted to dS,
        as that shifts the relevence of modes towards medium sized particles
        (100-1000 nm).The function first changes all values of 0 to np.nan. followed
        by running through each column and change a bin to np.nan if the following
        bin is np.nan. The reason for this is that when NS has empty bins,
        the preceding bins population also is reduced from what is expected
        based on the observable mode.
        With these bin data set tp np.nan, the script will go through each row,
        and if the number of empty bins is not equal or above the nan_lim value,
        it will try to fit a PSD based on the default fit. The quality of the fit will
        be compared to the error_lim which discards insuficient fits. If the quality
        is not good enough, another attempt of fitting will be done with one peak less.
        If fit passes the quality control, it will be used to generate bin values
        for the empty bins of that row. This fit will then be used as the first guess
        for the next row, as the following row is expected to be similiar to the previous.
        An additional column has also been generated to the extra_data called 'PSD_fit',
        which is equal to the binary number of the bin with a fitted value.
        eg. if bin 1,4,5 and 11 is fitted, "PSD_fit"= 2^1 + 2^4 + 2^5 + 2^11 = 2097
        Currently this value is not usable, but is intended for potentially marking
        fitted | true values.
        Finally the total value will be recalculatde based on returning to the
        dtype to dN, and the old has been saved in extra_data as "Old_total".

        Parameters
        ----------
        def_fit : dict, optional
            Default guess for particle distribution. Buildt up from mu, sigma and factor.
            The base number of modes will be determined by this, by the number of values
            within the three variables. Default is three modes.
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
            As the dtype is changed to dS, the factors are high compared to the
            usual levels for dN. The value is in nm2/cm3.
        error_lim: float, optional
            Determines the maximum value of error allowed for the fit parameters.
            error_lim < error_par[i]/value_par[i] * weight_par.
            With weight_par: 'mu'=2, 'sigma'=5, 'factor'=1.
        nan_lime: int, optional
            Percentage value around which the bound values can be fitted
        inplace (bool): If True, modify the current object and return
            it. If False, return a cropped deep copy.

        Returns
        -------
        data
            Aerodsol2D, with an updated dataset where values of zero has been
            replaced by fitted values if sufficient fit has been found. A marker
            for the fitted values has been saved in extra_data as "PSD_fit" together
            with the old total concentration saved in extra_data as "Old_total"
        """
        data = self if inplace else self.copy_self()
        bins=data._sizebin_headers
        #Turn all 0 values to nan.
        data._data[data.data[bins]==0] = np.nan
        #Save old total data, to allow for comparison
        data._extra_data['Old_total']=data._data['Total_conc']
        # Due to the relevant bins being in the region of 150-300 nm, the data is
        # converted to dS, where the middle sized modes become the most dominant.
        data.dtype_converter('dS')
    
        #Make a fit based off of the def_fit to one 
        try:
            fit,error=data.fit_psd(
                period="All data",
                mu      = def_fit['mu'],
                sigma   = def_fit['sigma'],
                factor  = def_fit['factor'],
                log_scaling=True,
                weighting = "uniform"
            )
        except Exception as e:
            raise ValueError("Chosen fitting parameters not suitbale. Try new ones.")
        
        def_fit=fit
        #Make a column called 'PSD fit' that will be used to designate fitted bins
        data._extra_data['PSD fit']=[0]*len(data.data['All data'])
    
        fit_test=0
        for t in range(0,len(data.time)):
            Time=data.time[t]
            for b in range(0,12):
                #Removes all the low bins seen right before empty bins for ns
                bin_more=bins[b:b+2]
                bin_m=bin_more[0]
                bin_more=bin_more[1]
                if data.data.loc[Time,bin_more] > 0:
                    pass
        
                else:
                    data._data.loc[Time,bin_m] = np.nan
                    # data._data[bin_m][time] = np.nan
                    
            if data.data.loc[Time].isnull().sum()>=nan_lim:
                if np.mod(t,200)==0:
                    print('testing testing')
                pass #If number of nan along the row is too high no fit is attempted.
            elif data.data.loc[Time][bins[0:12]].isnull().sum()<2:
                if np.mod(t,200)==0:
                    print('testing testing')
                pass #If number of nan along the row is too high no fit is attempted.
            else:
                if fit_test==0:
                    #If there is no succesfull fit to the previous rows initial guess goes to default.
                    try:
                        if np.mod(t,200)==0:
                            print('we are trying')
                        fit,error=data.fit_psd(
                            period=(Time,Time),#"All data",
                            mu      = def_fit['mu'],
                            sigma   = def_fit['sigma'],
                            factor  = def_fit['factor'],
                            log_scaling=True,
                        )
                        fit_test=1
                    except Exception as e:
                        pass
                else: #Use previous points with binding
                    try:
                        if len(fit['mu'])==len(def_fit['mu']):
                            fit,error=data.fit_psd(
                                    period=(Time,Time),#"All data",
                                    mu      = fit['mu'],
                                    sigma   = fit['sigma'],
                                    factor  = fit['factor'],
                                    log_scaling=True,
                                    binding=[True,True,True]*len(fit['mu']),
                                    tolerance=40,
                                )
                            fit_test=1
                        else: #if fit uses fewer points than def, re-add the last mode from default.
                            fit,error=data.fit_psd(
                                    period=(Time,Time),
                                    mu      = def_fit['mu'],
                                    sigma   = def_fit['sigma'],
                                    factor  = def_fit['factor'],
                                    log_scaling=True,
                                )
                            fit_test=1 
                    except Exception as e:
                        fit_test=0
                #Test the quality of the fit 
                if fit_test==1:
                    for i in range(0,len(fit['mu'])):
                        if abs(error['mu'][i] /fit['mu'][i]*2) > error_lim:
                            fit_test=2
                            break
                        elif abs(error['sigma'][i]/fit['sigma'][i]*5)>error_lim:
                            fit_test=2
                            break 
                        elif abs(error['factor'][i]/fit['factor'][i])>error_lim:
                            fit_test=2
                            break 
                     
                if fit_test==2:
                    try:
                        fit,error=data.fit_psd(
                                period=(Time,Time),
                                mu      = fit['mu'][:-1],
                                sigma   = fit['sigma'][:-1],
                                factor  = fit['factor'][:-1],
                                log_scaling=True,
                                binding=[True,True,True]*(len(fit['mu'])-1),
                                tolerance=40,
                            )
                        fit_test=1
                        for i in range(0,len(fit['mu'])):
                            if abs(error['mu'][i] /fit['mu'][i]*4) > error_lim:
                                fit_test=0
                                break
                            elif abs(error['sigma'][i]/fit['sigma'][i]*10)>error_lim:
                                fit_test=0
                                break 
                            elif abs(error['factor'][i]/fit['factor'][i]*2)>error_lim:
                                fit_test=0
                                break 
                    except Exception as e:
                        fit_test=0
                                                       
                if fit_test==1:
                    modes = list(zip(fit["mu"], fit["sigma"], fit["factor"]))
                    bin_fit, additional_modes=lognormal_modes(bins, modes)
                    bin_fit = bin_fit*np.diff(np.log10(data.bin_edges)).astype(np.float64)
                    
                    if min(bin_fit)<0:
                        raise ValueError('Fit has resulted in negative values at time {Time}')
                    for i in range(0,len(bins)):
                        b=bins[i]
                        if data._data.loc[Time,b]>0:
                            pass
                        else:
                            data._data.loc[Time,b]=bin_fit[i]
                            #Mark the fitted row with a number equal to the bins changed. 
                            data._extra_data.loc[Time,'PSD fit']=data._extra_data.loc[Time,'PSD fit']+2**i             
                            
        data.dtype_converter('dN')
        
        return data
