aerosoltools.utility
====================

.. py:module:: aerosoltools.utility




Module Contents
---------------

.. py:function:: Combine_NS_OPS(NS_data, OPS_data, start = None, end = None, *, match = 'rebin', tolerance = '30s', rebin_freq = None, rebin_method = 'mean')

   Description:
       Combine NanoScan (NS) and OPS number size distributions into one
       time-aligned Aerosol2D spectrum.

   :param NS_data: NanoScan measurement as an :class:`~aerosoltools.aerosol2d.Aerosol2D`
                   instance, containing a time-resolved size distribution.
   :type NS_data: Aerosol2D
   :param OPS_data: OPS measurement as an :class:`~aerosoltools.aerosol2d.Aerosol2D`
                    instance, containing a time-resolved size distribution.
   :type OPS_data: Aerosol2D
   :param start: Start time of the period used for combining the two instruments.
                 If ``None``, the later of the two available start times
                 (NS vs OPS) is used. Strings are parsed with
                 :func:`pandas.to_datetime`. Default is None.
   :type start: pandas.Timestamp | str | None, optional
   :param end: End time of the period used for combining the two instruments.
               If ``None``, the earlier of the two available end times
               (NS vs OPS) is used. Default is None.
   :type end: pandas.Timestamp | str | None, optional
   :param match: Strategy for aligning the two time series in time. Default is
                 ``"rebin"``. Options are:

                 * ``"rebin"`` (default): Rebin both instruments to a common
                   time step using :meth:`Aerosol2D.timerebin`, then intersect
                   timestamps.
                 * ``"exact"``: Use only timestamps that are present in both
                   datasets without resampling.
                 * ``"nearest"``: Match OPS values to NS timestamps using the
                   nearest available OPS point within ``tolerance``.
   :type match: str, optional
   :param tolerance: Maximum allowed separation between NS and OPS timestamps when
                     ``match="nearest"`` is used. Can be a pandas offset string
                     (e.g. ``"30s"``) or a :class:`pandas.Timedelta`. Ignored
                     for other ``match`` modes. Default is 30s.
   :type tolerance: str | pandas.Timedelta, optional
   :param rebin_freq: Target resampling rule for ``match="rebin"`` (e.g. ``"1min"``).
                      If ``None``, the coarser of the inferred NS and OPS cadences is
                      chosen automatically. Default is None.
   :type rebin_freq: str | None, optional
   :param rebin_method: Aggregation method passed to :meth:`Aerosol2D.timerebin` when
                        ``match="rebin"`` is used (e.g. ``"mean"``, ``"median"``,
                        or a custom function). Default is ``"mean"``.
   :type rebin_method: str | Callable, optional

   :returns:     A new :class:`~aerosoltools.aerosol2d.Aerosol2D` object containing
                 the merged NS+OPS number size distribution.
   :rtype: Aerosol2D

   :raises ValueError: If the requested time interval has no overlap between NS and OPS,
       or if the chosen ``match`` strategy produces no common timestamps.
       Also raised if the lowest OPS bin edge falls outside the NS bin
       range so that no consistent splice point can be defined.
   :raises TypeError: If ``NS_data`` or ``OPS_data`` does not behave like an
       :class:`Aerosol2D` instance (e.g. missing required attributes such
       as ``time``, ``bin_edges``, or ``timerebin``).

   .. rubric:: Notes

   Detailed description:
       This function is intended for combining data from a NanoScan SMPS
       and an OPS into a single, continuous number size distribution in
       diameter space. Internally, the following steps are carried out:

       * Both inputs are copied to avoid modifying the originals.
       * Each dataset is converted to number concentration (dN) in
         ``cm⁻³`` and de-normalized from ``/dlogDp`` if needed.
       * The time series from NS and OPS are aligned using the specified
         ``match`` mode and the chosen time window (``start``/``end``).
       * The overlapping size range is determined from the NS and OPS
         bin edges. The OPS lower bin edge defines the splice point.
       * The NanoScan bin that overlaps the OPS lower edge is truncated,
         and its concentration is scaled by the fraction of bin width
         that remains below the splice point.
       * All NS bins below the splice point and all OPS bins are
         concatenated to form a new set of bin edges and midpoints.
       * The size-resolved concentrations are merged, and the total
         number concentration is recomputed from the combined size bins
         for each aligned timestamp.
       * NanoScan activities and metadata are propagated to the result,
         and the instrument metadata is set to ``"NS_OPS"``.

       The resulting class object includes:

       * Combined size-bin edges and midpoints covering the NS+OPS range.
       * Recomputed total number concentration in ``cm⁻³`` for each
         timestamp.
       * Propagated NanoScan activities and metadata, with the instrument
         set to ``"NS_OPS"``.

   Theory:
       The function assumes both instruments measure size-resolved
       particle number concentration and uses a simple geometric
       bin-splicing approach:

       * Number concentration per bin is first expressed as dN (cm⁻³)
         without ``/dlogDp`` normalization.
       * The shared diameter range is determined from the reported bin
         edges; a single splice bin in the NanoScan is truncated so that
         the OPS lower edge becomes the exact boundary between NS and OPS.
       * Conservation of particle number within the truncated bin is
         approximated by scaling with the retained width fraction.

       This produces a piecewise distribution that is directly usable for
       further number-based metrics (e.g. total PNC, modal fits, etc.).

   .. rubric:: Examples

   A typical use case is combining co-located NanoScan and OPS
   measurements into a single spectrum before analysis or reporting:

   .. code-block:: python

       import aerosoltools as at

       NS = at.Load_NS_file("nanoscan.csv")
       OPS = at.Load_OPS_file("ops.csv")

       # Combine on a 1-minute grid using time rebinning
       combined = at.Combine_NS_OPS(
           NS, OPS,
           start="2023-10-01 08:00",
           end="2023-10-01 16:00",
       )

       # Plot the resulting size distribution
       fig, ax = combined.plot_timeseries()


.. py:function:: Plot_correlation(X, Y, ax_in = None, *, start_time = None, end_time = None, column = 'Total_conc', match = 'exact', tolerance = '30s', rebin_freq = None, rebin_method = 'mean', intercept = True, uniform_scaling = True, outlier_influence = True)

   Description:
       Create a correlation plot between the same variable from two aerosol
       datasets, including regression line, 1:1 line, and R².

   :param X: First aerosol dataset. Typically an :class:`Aerosol1D` or
             :class:`Aerosol2D` instance exposing ``data`` (and optionally
             ``extra_data`` and ``timerebin``).
   :param Y: Second aerosol dataset with the same interface requirements as
             ``X``.
   :param ax_in: Existing Matplotlib axes to draw on. If ``None``, a new figure
                 and axes are created. Default is None.
   :type ax_in: matplotlib.axes.Axes | None, optional
   :param start_time: Inclusive start of the analysis window. If provided together with
                      ``end_time`` and the objects implement ``timecrop``, the data are
                      cropped to this period before correlation is computed. Strings are
                      parsed with :func:`pandas.to_datetime`. Default is None, meaning
                      start from first common timestamp.
   :type start_time: pandas.Timestamp | str | None, optional
   :param end_time: Inclusive end of the analysis window. Same parsing rules as
                    ``start_time``.
   :type end_time: pandas.Timestamp | str | None, optional
   :param column: Name of the variable to correlate. The function first looks for
                  this column in ``obj.data`` and then in ``obj.extra_data``.
                  The default is ``"Total_conc"``
   :type column: str, optional
   :param match: Strategy for aligning the two time series in time. One of:

                 - ``"exact"`` (default): Keep only timestamps that are present
                   in both series.
                 - ``"nearest"``: Match values from ``Y`` to the timeline of
                   ``X`` using nearest timestamps within ``tolerance``.
                 - ``"rebin"``: Rebin both datasets to a common time step using
                   ``timerebin`` and then join on timestamps.
   :type match: str, optional
   :param tolerance: Maximum allowed separation between timestamps when
                     ``match="nearest"`` is used. Can be a pandas offset string
                     (e.g. ``"30s"``) or a :class:`pandas.Timedelta`. Ignored for
                     other ``match`` modes.
   :type tolerance: str | pandas.Timedelta, optional
   :param rebin_freq: Target resampling rule for ``match="rebin"`` (e.g. ``"1min"``).
                      If ``None``, the coarser cadence inferred from the two series is
                      chosen automatically. Default is None.
   :type rebin_freq: str | None, optional
   :param rebin_method: Aggregation method passed to ``timerebin`` when ``match="rebin"``
                        is used (e.g. ``"mean"``, ``"median"``, or a custom function).
                        Default is ``"mean"``.
   :type rebin_method: str | Callable, optional
   :param intercept: If ``True`` (default), fit a full linear model
                     ``y = A·x + B``. If ``False``, constrain the fit to pass through
                     the origin (``y = A·x``).
   :type intercept: bool, optional
   :param uniform_scaling: If ``True`` (default), both axes are scaled by a common factor so
                           that the same numerical range is shown on x and y. If ``False``,
                           each axis is scaled independently.
   :type uniform_scaling: bool, optional
   :param outlier_influence: If ``True`` (default), use standard least-squares regression
                             (:func:`scipy.optimize.curve_fit`) and draw a 1σ confidence band
                             around the fitted line. If ``False``, use the robust
                             Theil–Sen estimator (:func:`scipy.stats.theilslopes`) without a
                             confidence band.
   :type outlier_influence: bool, optional

   :returns:     The figure and axes containing the correlation scatter plot, the
                 1:1 line, and the regression line with its equation and R² in
                 the legend.
   :rtype: tuple[Figure, Axes]

   :raises ValueError: If one or both objects contain no data for the requested
       ``column`` and time window, if the chosen alignment strategy
       yields no matching timestamps, or if all overlapping points are
       non-finite (NaN/inf) after cleaning.
   :raises KeyError: If ``column`` is not found in either ``data`` or ``extra_data`` of
       one or both objects.
   :raises RuntimeError: If the regression fit fails to converge (e.g. due to degenerate
       or extremely ill-conditioned data) and :func:`curve_fit` or the
       Theil–Sen estimator raises a fitting-related error.

   .. rubric:: Notes

   Detailed description:
       ``Plot_correlation`` is a convenience function for quickly
       comparing two aerosol datasets measuring the same physical
       quantity, such as total particle number concentration from two
       instruments. The function:

       * Extracts the requested ``column`` from each object.
       * Aligns the series in time using the selected ``match`` mode
         (exact timestamps, nearest neighbors, or common rebinned
         cadence).
       * Removes rows where either series is NaN or infinite.
       * Fits a linear model relating ``Y`` to ``X``, optionally
         including an intercept and using either standard or robust
         regression.
       * Computes and reports the coefficient of determination (R²).
       * Plots the scatter of aligned data points, the 1:1 line, the
         fitted regression line, and (optionally) a confidence band
         around the fit.

       Axis labels are automatically derived from ``X.instrument`` and
       ``Y.instrument`` (if available), giving a quick visual summary of
       how well two instruments agree.

   Theory:
       The regression models used are simple linear relationships:

       * With intercept: ``y = A·x + B``
       * Without intercept: ``y = A·x``

       When ``outlier_influence=True``, the parameters ``A`` and ``B``
       are obtained by minimizing the least-squares error using
       :func:`scipy.optimize.curve_fit`. Standard errors of the fit
       parameters are derived from the covariance matrix and propagated
       to form an approximate 1σ confidence band.

       When ``outlier_influence=False``, the Theil–Sen estimator is used
       (:func:`scipy.stats.theilslopes`). This approach is more robust to
       outliers, but no confidence band is drawn.

   .. rubric:: Examples

   A typical use case is to compare the agreement between two
   instruments over the same time period:

   .. code-block:: python

       import aerosoltools as at

       # Load two datasets measuring total number concentration
       smps = at.Load_SMPS_file("smps_data.txt")
       ops = at.Load_OPS_file("ops_data.txt")

       # Plot correlation of total concentration over a work shift
       fig, ax = at.Plot_correlation(
           smps,
           ops,
           start_time="2023-10-01 08:00",
           end_time="2023-10-01 16:00",
           column="Total_conc",
           match="nearest",
           tolerance="60s",
           intercept=True,
           uniform_scaling=True,
           outlier_influence=False,
       )


