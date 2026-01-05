aerosoltools.aerosol2d
======================

.. py:module:: aerosoltools.aerosol2d






Module Contents
---------------

.. py:data:: params

.. py:class:: Aerosol2D(dataframe)

   Bases: :py:obj:`aerosoltools.aerosol1d.Aerosol1D`


   A class for managing time-resolved, size-distributed aerosol data.

   This class extends `Aerosol1D` to handle datasets that contain particle
   size distributions (e.g., number, mass, or surface area concentration
   across particle size bins). It supports transformation between physical
   representations (dN, dS, dV, dW), visualization, activity segmentation,
   and summary statistics including PM values and particle size metrics.

   :param dataframe: A DataFrame containing the data to load. The first column should
                     contain time stamps or be the DataFrame index. The second column should
                     be the total concentration. All remaining columns must represent
                     concentration values in size bins with bin midpoints as column headers.
   :type dataframe: pandas.DataFrame

   .. rubric:: Notes

   All data handling is done with `pandas`. Input DataFrames are expected to
   have particle size bin midpoints as column headers, and the class assumes
   these are numeric and represent diameters in nanometers.


   .. py:property:: bin_edges
      :type: numpy.typing.NDArray[numpy.float64]


      Particle size bin edges in nanometers.

      :returns: One-dimensional array of bin edge diameters in
                nanometers (dtype ``float64``). Length is ``n + 1`` when there are
                ``n`` size bins. A copy is returned so callers cannot mutate the
                internal metadata.
      :rtype: numpy.ndarray


   .. py:property:: bin_mids
      :type: numpy.typing.NDArray[numpy.float64]


      Particle size bin midpoints in nanometers.

      :returns: One-dimensional array of bin midpoint diameters in
                nanometers (dtype ``float64``). Length is ``n`` for ``n`` size
                bins. A copy is returned so callers cannot mutate the internal
                metadata.
      :rtype: numpy.ndarray


   .. py:property:: density
      :type: float


      Assumed particle density in g/cm³.

      :returns: Particle density used for conversions between number, volume,
                surface area, and mass distributions. Falls back to the value
                stored in the metadata (typically set at load time or via
                :meth:`set_density`). Defaults to 1.0 g/cm³ if not explicitly set in
                metadata.
      :rtype: float


   .. py:property:: metadata
      :type: dict


      Metadata associated with the size-resolved dataset.

      :returns: Dictionary of metadata extracted or defined for this object,
                including bin edges/mids, units, data type (dN/dS/dV/dM),
                instrument information, density, and any additional fields stored
                in ``self._meta``.
      :rtype: dict


   .. py:property:: size_data
      :type: pandas.DataFrame


      Size-bin concentration data.

      :returns: Subset of :attr:`data` containing only the
                columns that represent size-resolved concentration values, ordered
                according to :attr:`bin_mids` (via :attr:`_sizebin_headers`). Each
                column corresponds to a size bin, and each row to a time stamp.
      :rtype: pandas.DataFrame


   .. py:method:: dtype_converter(dtype = 'dN', inplace = True)

      Description:
          Convert the size distribution to a chosen base data type.

      :param dtype: Target data type string, one of "dN", "dS",
                    "dV", or "dM" (case-sensitive). "dN" is number-based,
                    "dS" surface-area–based, "dV" volume-based, and "dM"
                    mass-based.
      :type dtype: str
      :param inplace: If True, convert this object in place and
                      return it. If False, perform the conversion on a deep
                      copy and return the new instance.
      :type inplace: bool

      :returns:

                Object whose size-bin data, total_conc, dtype and
                    unit fields have been converted to the requested type
                    (self when inplace=True, otherwise a new instance).
      :rtype: Aerosol2D

      :raises ValueError: If dtype is not one of "dN", "dS", "dV", "dM".
          Check the spelling and letter case of the requested type.

      .. rubric:: Notes

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

      .. rubric:: Examples

      Convert a number distribution to mass concentration for
      comparison with gravimetric limits:

      .. code-block:: python

          elpi.dtype_converter("dM")
          elpi.plot_psd()



   .. py:method:: correct_diffusion_losses(D_tube, L, Q, T = 293, P = 101300, inplace = True)

      Description:
          Correct size distributions for diffusion losses in sampling tubes.

      :param D_tube: Inner diameter of the sampling tube in metres.
      :type D_tube: float
      :param L: Length of the sampling tube in metres.
      :type L: float
      :param Q: Volumetric flow through the tube in L/min.
      :type Q: float
      :param T: Gas temperature in Kelvin. Defaults to 293 K.
      :type T: float
      :param P: Gas pressure in Pascal. Defaults to 101300 Pa.
      :type P: float
      :param inplace: If True, apply the correction to this object
                      and return it. If False, perform the correction on a deep
                      copy and return the new instance.
      :type inplace: bool

      :returns:

                Object with diffusion-loss–corrected size-bin data
                    and updated Total_conc (self when inplace=True, otherwise
                    a new instance).
      :rtype: Aerosol2D

      :raises None: The method does not explicitly raise custom exceptions,
          but non-physical values (for example Q or D_tube close to
          zero) can lead to infinities or NaNs in the correction
          factors. Always use positive, realistic geometry and flow
          parameters.

      .. rubric:: Notes

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

      .. rubric:: Examples

      Correct ELPI or SMPS data for diffusion losses in a long
      sampling line:

      .. code-block:: python

          elpi.correct_diffusion_losses(
              D_tube=0.004,  # 4 mm ID
              L=2.0,        # 2 m tube
              Q=10.0,       # 10 L/min
          )
          elpi.plot_psd()



   .. py:method:: set_density(density = 1.0)

      Description:
          Set or update the assumed particle density (g/cm³).

      :param density: New particle density in g/cm³.
      :type density: float | int

      :returns:

                The updated object with metadata["density"]
                    set to the new value. If the current dtype is mass-based
                    ("dM" in dtype), the mass distribution and Total_conc are
                    rescaled immediately.
      :rtype: Aerosol2D

      :raises ValueError: If the existing stored density is non-positive
          while the data are mass-based, so rescaling is undefined.
          In that case, manually fix metadata["density"] or reload
          the data before changing density.

      .. rubric:: Notes

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

      .. rubric:: Examples

      Update density when reinterpreting a measurement for a
      specific material:

      .. code-block:: python

          elpi.dtype_converter("dM")
          elpi.set_density(1.6)  # g/cm³



   .. py:method:: normalize_logdp(inplace = True)

      Description:
          Normalize the size distribution by Δlog₁₀(Dp) (dx/dlogDp).

      :param inplace: If True, normalize this object in place and
                      return it. If False, perform the normalization on a deep
                      copy and return the new instance.
      :type inplace: bool

      :returns:

                The normalized object (self or a new copy)
                    when normalization is applied. If the dtype already
                    contains "/dlogDp", no changes are made and None is
                    returned.
      :rtype: Aerosol2D | None

      :raises ValueError: If the number of size-bin columns does not match
          the number of Δlog₁₀(Dp) widths derived from bin_edges.
          Check that bin_edges and the PSD columns are consistent.

      .. rubric:: Notes

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

      .. rubric:: Examples

      Prepare a PSD for log-diameter plotting:

      .. code-block:: python

          elpi.normalize_logdp()
          elpi.plot_psd()



   .. py:method:: unnormalize_logdp(inplace = True)

      Description:
          Undo Δlog₁₀(Dp) normalization (dx/dlogDp → base form).

      :param inplace: If True, unnormalize this object in place and
                      return it. If False, perform the operation on a deep copy
                      and return the new instance.
      :type inplace: bool

      :returns:

                The unnormalized object (self or a new copy)
                    when dtype contains "/dlogDp". If the data are already in
                    base form (no "/dlogDp" in dtype), the method returns None
                    and does not modify the object.
      :rtype: Aerosol2D | None

      :raises ValueError: If the number of PSD columns does not match the
          Δlog₁₀(Dp) array derived from bin_edges.

      .. rubric:: Notes

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

      .. rubric:: Examples

      Convert a normalized PSD back to base units for further
      processing:

      .. code-block:: python

          if "/dlogDp" in elpi.dtype:
              elpi.unnormalize_logdp()
          elpi.dtype_converter("dM")



   .. py:method:: PM_calc(dtype = 'dM', PM = 4.2, Lower_lim = 0)

      Description:
          Compute a size-selective Pₓ time series and store it in extra_data.

      :param dtype: Base distribution type to integrate, one of
                    "dN", "dS", "dV", "dM". Defaults to "dM" (mass-based).
      :type dtype: str
      :param PM: Upper cut-off diameter in micrometres (µm) for the
                 size-selective fraction (nominal 50% penetration point).
      :type PM: float
      :param Lower_lim: Optional lower cut-off diameter in µm.
                        If 0 (default), the result is cumulative from 0 → PM.
                        If in (0, PM), the result represents the band-limited
                        contribution from Lower_lim → PM.
      :type Lower_lim: float

      :returns:

                self, with a new column added to extra_data named
                    "P{X}{PM}" for cumulative metrics (for example "PM2.5",
                    "PN10") or "P{X}{Lower_lim}-{PM}" for band-limited ones
                    (for example "PM1-5").
      :rtype: Aerosol2D

      :raises ValueError: If Lower_lim is greater than or equal to PM, or if
          Lower_lim is negative. Check the order and magnitude of
          the cut diameters.
      :raises ValueError: If dtype is not one of "dN", "dS", "dV", "dM".

      .. rubric:: Notes

      Detailed description:
          The method converts the internal distribution to the
          requested base kind if needed, ensures it is not in
          dx/dlogDp form, and then integrates it with an
          EN 481 / ISO 7708–style size-selective penetration curve
          between Lower_lim and PM. The resulting time series is
          stored in extra_data with a canonical name that encodes
          both the distribution type and cut diameters.

      Theory:
          Pₓ metrics generalize well-known PM₁₀, PM₂.₅, etc., and
          can be defined for number (PN), surface (PS), volume (PV)
          and mass (PM). The underlying helper uses a standard
          lognormal penetration curve (GSD 1.5, 50% cut at PM) to
          approximate workplace sampling conventions (for example
          EN 481 / ISO 7708 respirable/inhalable fractions).

      .. rubric:: Examples

      Add PM₂.₅ and PN₁₀ series for later plotting and summary:

      .. code-block:: python

          elpi.PM_calc(dtype="dM", PM=2.5)   # PM2.5
          elpi.PM_calc(dtype="dN", PM=10.0)  # PN10
          elpi.extra_data[["PM2.5", "PN10"]].head()



   .. py:method:: plot_psd(activities = None, normalize = True, ax=None)

      Description:
          Plot mean particle size distributions for one or more activities.

      :param activities: Names of activities to include.
                         If None, all activities in self.activities are considered.
                         Activities that do not exist are skipped.
      :type activities: list[str] | None
      :param normalize: If True, plot PSDs in log-diameter–
                        normalized form (dx/dlogDp). If the underlying data are not
                        normalized, a temporary division by Δlog₁₀(Dp) is applied.
                        If False, PSDs are shown in base units, undoing any stored
                        normalisation if needed.
      :type normalize: bool
      :param ax: Axis to plot into. If None,
                 a new figure and axes are created.
      :type ax: matplotlib.axes.Axes | None

      :returns:

                The
                    figure and axes with the PSD plot.
      :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

      :raises None: Aside from Matplotlib or data consistency errors (for
          example invalid bin_edges or empty activities).

      .. rubric:: Notes

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

      .. rubric:: Examples

      Compare PSDs during two tasks:

      .. code-block:: python

          elpi.mark_activities({
              "Task A": [("2025-01-24 09:00", "2025-01-24 10:00")],
              "Task B": [("2025-01-24 10:00", "2025-01-24 11:00")],
          })
          elpi.plot_psd(activities=["Task A", "Task B"], normalize=True)



   .. py:method:: plot_PM_timeseries(PM_values=[0.5, 2.5, 10], dtype = 'dM', activity = 'All data', fraction = False, cummulative = False)

      Description:
          Plot time series of one or more size-selective Pₓ metrics.

      :param PM_values: Cut diameters in µm defining the
                        Pₓ series to compute (for example [0.5, 2.5, 10]).
      :type PM_values: list[float]
      :param dtype: Base distribution type for Pₓ evaluation, one of
                    "dN", "dS", "dV", "dM". Defaults to "dM" (mass-based).
      :type dtype: str
      :param activity: Name of the activity mask selecting which time
                       steps to plot. Must be a boolean column in data.
      :type activity: str
      :param fraction: If False (default), plot Pₓ in absolute
                       units and stack bands between successive PM_values. If
                       True, plot the largest Pₓ on the primary axis and the
                       fractional contributions of each Pₓ on a secondary axis.
      :type fraction: bool
      :param cummulative: Controls how bands/legend values are
                          interpreted:

                              * False: legend reports band-wise contributions between
                                successive PM_values (for example PM10 − PM2.5).
                              * True: legend reports cumulative Pₓ at each cut
                                (for example PM2.5, PM10).
      :type cummulative: bool

      :returns:

                The
                    figure and primary axes. When fraction=True, a secondary
                    y-axis for fractions is also created.
      :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

      :raises Exception: If the number of PM_values exceeds the internal
          color palette. Reduce PM_values or extend the color list.
      :raises KeyError: If activity is not a defined mask column in data.
          Check data.columns and mark_activities/Peak_finder calls.
      :raises ValueError: If dtype is not one of "dN", "dS", "dV", "dM".

      .. rubric:: Notes

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

      .. rubric:: Examples

      Examine how PM0.5, PM2.5 and PM10 evolve during a shift:

      .. code-block:: python

          fig, ax = elpi.plot_PM_timeseries(
              PM_values=[0.5, 2.5, 10],
              dtype="dM",
              activity="All data",
              fraction=False,
          )



   .. py:method:: plot_timeseries(y_tot = (0, 0), y_3d = (0, 0), log = True, ax1 = None, ax2 = None, mark_activities = False)

      Description:
          Plot total concentration and a time–size heatmap in one figure.

      :param y_tot: Y-limits for the total
                    concentration panel (ymin, ymax). Use (0, 0) for automatic
                    limits; if a non-zero entry is given with zero partner,
                    the non-zero value is used directly, while the zero is
                    replaced by the max/min in the data.
      :type y_tot: tuple[float, float]
      :param y_3d: Color-scale limits for the 2D PSD
                   panel (zmin, zmax). Use (0, 0) for automatic limits. To
                   enforce a strictly positive lower limit for log scaling,
                   set zmin > 0 and zmax = 0 for automatic upper limit, e.g.
                   y_3d = (1, 0).
      :type y_3d: tuple[float, float]
      :param log: If True, the function attempts to use a logarithmic
                  color scale for the 2D panel. If the PSD values used for
                  the mesh (after any clipping from y_3d) are not strictly
                  positive or the lower limit is ≤ 0, the function
                  automatically falls back to a linear color scale and prints
                  a warning to the terminal. If False, a linear color scale
                  is used directly.
      :type log: bool
      :param ax1: Axis for the top (total
                  concentration) plot. If provided, ax2 must also be
                  provided.
      :type ax1: matplotlib.axes.Axes | None
      :param ax2: Axis for the bottom
                  (time–size) plot. If provided, ax1 must also be provided.
      :type ax2: matplotlib.axes.Axes | None
      :param mark_activities: Passed to
                              plot_total_conc to control activity highlighting. True
                              shades all activities except "All data"; a sequence
                              restricts shading to specific activities.
      :type mark_activities: bool | Sequence[str]

      :returns:

                A tuple with
                    the figure and a NumPy array [ax1, ax2, colorbar].
      :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray]

      :raises ValueError: If only one of ax1 or ax2 is supplied. Provide
          both or neither.

      .. rubric:: Notes

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

      .. rubric:: Examples

      Create an overview plot of an ELPI or SMPS data set:

      .. code-block:: python

          fig, (ax1, ax2, cbar) = elpi.plot_timeseries()
          fig.savefig("elpi_timeseries.png", dpi=150)



   .. py:method:: summarize_activities(filename = None, metrics = None)

      Description:
          Summarize size-resolved aerosol metrics per activity.

      :param filename: Optional Excel file path. If provided,
                       the summary table is written to this file (one sheet,
                       activities as rows). If None, no file is written.
      :type filename: str | None
      :param metrics: List of metric names to compute.
                      If None, a default set is used: ["PNC", "PM1", "PM2.5",
                      "PM4", "PM10", "MASS", "MODE", "MEDIAN", "GMD"].
      :type metrics: list[str] | None

      :returns:

                Summary table with:
                    * "Segment"
                    * "Duration (HH:MM)"
                    * For each metric M: "M mean [unit]" and "M std [unit]".
      :rtype: pandas.DataFrame

      :raises ValueError: If a metric name cannot be interpreted (for
          example malformed PMx string) or is unsupported.
      :raises ValueError: If internal preparation for a Pₓ metric fails
          (for example missing PSD columns or inconsistent bin
          metadata).

      .. rubric:: Notes

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

      .. rubric:: Examples

      Generate a task-level summary of exposure metrics:

      .. code-block:: python

          elpi.summarize_activities(
              filename="activity_summary_elpi.xlsx",
              metrics=["PNC", "PM2.5", "PM10", "MODE", "GMD"],
          )



   .. py:method:: summarize_exposure(metric = 'PM4.2', background = None, exposure_hours = None, short_limit = 1.0, long_limit = 1.0, short_window = '15min', twa_window = '8h', peak_ratio = 2.5, filename = None, activities = None)

      Description:
          Summarize exposure metrics for one PSD-derived metric across activities.

      :param metric: Exposure metric name derived from the underlying
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
      :type metric: str
      :param background: Background level used when
                         computing the time-weighted average over ``twa_window``.
                         The same background level is used for all activities in
                         the output. Default is None. Possible entries are:

                             * None: assume zero background.
                             * float: constant background level in metric units.
                             * str: name of an activity; the TWA of ``metric`` over
                               that activity is used as background.
      :type background: float | str | None
      :param exposure_hours: Assumed duration of exposure for
                             each activity, in hours, when embedding it into the TWA
                             window. If None, the measured activity duration is used for
                             that activity. If a positive value is given, the same
                             exposure duration is applied for all activities.
                             Default is None.
      :type exposure_hours: float | None
      :param short_limit: Short-term concentration limit in metric
                          units (for example a 15-min STEL). This value is reported in
                          the output as ``"STEL [unit]"``. Default is 1.0 (in metric
                          units).
      :type short_limit: float
      :param long_limit: Long-term concentration limit in metric
                         units (for example an 8-h OEL). This value is reported in the
                         output as ``"Exposure limit [unit]"``. Default is 1.0 (in
                         metric units).
      :type long_limit: float
      :param short_window: Rolling window used for short-term (STEL)
                           evaluation, given as a pandas offset string (for example
                           "15min"). This is reported as ``"STEL window"``.
                           Default is "15min" (15 minutes).
      :type short_window: str
      :param twa_window: Total duration of the TWA window as a pandas
                         offset string (for example "8h"). This is reported as
                         ``"TWA window"``. Default is "8h" (8 hours).
      :type twa_window: str
      :param peak_ratio: Factor used in peak detection; peaks are
                         flagged when the metric exceeds::

                             baseline + peak_ratio * rolling_std,

                         where ``baseline`` is a rolling median and ``rolling_std`` is
                         a rolling standard deviation over a short window.
                         Default is 2.5.
      :type peak_ratio: float
      :param filename: Optional path to a CSV/Excel file to which
                       the non-transposed result rows are appended. If the file
                       exists, rows are appended; otherwise the file is created with
                       a header. Supported extensions are ".csv", ".xls", ".xlsx".
      :type filename: str | None
      :param activities: Activities to summarize.
                         If None (default), all defined activities in
                         :attr:`activities` are summarized (for example "All data",
                         "Background", "Emission", "Decay"). Activities with no
                         marked time steps are skipped.
      :type activities: Sequence[str] | None

      :returns: One row per activity segment with summary
                statistics for the chosen metric. Column names embed their units
                in square brackets. See Notes below (Detailed description) for a
                complete list of columns.
      :rtype: pandas.DataFrame

      :raises ValueError: If ``metric`` cannot be parsed (unsupported string).
      :raises ValueError: If a background activity name is given but does not
          exist or has no samples.
      :raises ValueError: If ``short_window`` or ``twa_window`` cannot be
          parsed as pandas-style durations.
      :raises ValueError: If ``exposure_hours`` is negative.
      :raises TypeError: If ``background`` is not None, float or str.

      .. rubric:: Notes

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

      .. rubric:: Examples

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



   .. py:property:: activities
      :type: List[str]


      Names of all defined activity labels.

      :returns: Activity names for which a boolean mask column exists
                in :attr:`data` (for example ``"All data"``, ``"Peak"``, or
                user-defined labels created via :meth:`mark_activities`).
      :rtype: list[str]


   .. py:property:: activity_periods
      :type: Dict


      Time periods associated with each activity label.

      :returns: Mapping from activity name (str) to a list of
                ``(start, end)`` timestamp tuples, where each tuple defines a
                contiguous period during which the activity is active.
      :rtype: dict


   .. py:property:: data
      :type: pandas.DataFrame


      Main data frame with time index and measurement columns.

      :returns: The full data table, including the time index,
                the primary measurement columns (for example total concentration),
                and any boolean activity mask columns.
      :rtype: pandas.DataFrame


   .. py:property:: dtype
      :type: str


      Data type descriptor for the primary measurements.

      :returns: A short description of the data type, for example
                ``"dN"`` (number concentration), ``"dM"`` (mass concentration),
                or ``"dN/dlogDp"`` when normalized. Falls back to
                ``"Unknown dtype"`` if not set in the metadata.
      :rtype: str


   .. py:property:: extra_data
      :type: pandas.DataFrame


      Additional non-core data columns.

      :returns: Data frame containing extra columns that were
                extracted from the raw data file but are not considered part of the
                core aerosol time series (for example environmental or metadata
                channels). The index is aligned with :attr:`data`.
      :rtype: pandas.DataFrame


   .. py:property:: instrument
      :type: str


      Instrument used to acquire the measurements.

      :returns: Instrument name or description (for example model/type).
                Falls back to ``"Unknown instrument"`` if not set in the metadata.
      :rtype: str


   .. py:property:: original_data
      :type: pandas.DataFrame


      Unmodified original main data frame.

      :returns: A copy of the raw data as it was immediately after
                loading and initial normalization of the time index and column
                names, before any further processing steps (such as cropping,
                smoothing, or rebinning).
      :rtype: pandas.DataFrame


   .. py:property:: original_extra_data
      :type: pandas.DataFrame


      Unmodified original extra-data frame.

      :returns: A copy of the raw extra-data table (if any) before
                any processing steps. This will typically be empty if no extra data
                were extracted at load time.
      :rtype: pandas.DataFrame


   .. py:property:: serial_number
      :type: str


      Instrument serial number.

      :returns: Serial number of the instrument, if available in the metadata.
                Falls back to ``"Unknown serial number"`` if not set.
      :rtype: str


   .. py:property:: time
      :type: pandas.DatetimeIndex


      Time index of the measurements.

      :returns: The time stamps corresponding to each row in
                :attr:`data`. This is the index of the main data frame.
      :rtype: pandas.DatetimeIndex


   .. py:property:: total_concentration
      :type: pandas.Series


      Total aerosol concentration time series.

      :returns: The total concentration over time. If a column named
                ``"Total_conc"`` exists in :attr:`data`, that column is
                returned; otherwise the first data column is used as a fallback.
      :rtype: pandas.Series


   .. py:property:: unit
      :type: str


      Measurement unit for the primary data.

      :returns: Unit string, for example ``"#/cm³"`` for number concentration
                or ``"µg/m³"`` for mass concentration. Falls back to
                ``"Unknown unit"`` if not set in the metadata.
      :rtype: str


   .. py:method:: copy_self()

      Description:
          Create and return a deep copy of the aerosol time-series object.

      :returns:

                A new object with independent copies of all data,
                    extra_data, metadata, and activity definitions.
      :rtype: Aerosol1D

      .. rubric:: Notes

      Detailed description:
          The returned object is completely detached from the original:
          changing masks, cropping, rebinning, or smoothing on the copy
          does not affect the source instance, and vice versa. Both
          the original and the copy retain their own raw_data and
          raw_extra_data snapshots.

      .. rubric:: Examples

      Use this when you want to experiment with processing steps
      without changing your original dataset:

      .. code-block:: python

          cpc_raw = cpc_data
          cpc_proc = cpc_data.copy_self()
          cpc_proc.timerebin("1min").timesmooth(window=11)



   .. py:method:: get_activity_data(activity_name)

      Description:
          Return main data restricted to a given activity period.

      :param activity_name: Name of the activity/boolean mask column
                            to use (for example "All data", "Peak", or a user-defined
                            label created via mark_activities).
      :type activity_name: str

      :returns:

                Copy of the main data for time steps where the
                    selected activity is True, with all activity mask columns
                    removed. The index is the filtered DatetimeIndex.
      :rtype: pandas.DataFrame

      :raises ValueError: If activity_name is not present in self.activities.
          Ensure you call mark_activities or Peak_finder first, or
          check available labels via the activities property.

      .. rubric:: Notes

      Detailed description:
          This method filters self.data using the chosen activity mask
          and then drops all boolean activity columns so that the result
          only contains “measurement-like” columns (for example
          Total_conc and any other numeric channels). The returned
          DataFrame is safe to modify without affecting the internal
          storage of the Aerosol1D object.

      Theory:
          The method performs a simple time-based selection of rows
          flagged by an activity mask.

      .. rubric:: Examples

      Extract a clean dataset for a specific task before further
      analysis:

      .. code-block:: python

          task_df = data.get_activity_data("Task")
          task_df["Total_conc"].plot()



   .. py:method:: get_activity_extra_data(activity_name)

      Description:
          Return extra_data restricted to a given activity period.

      :param activity_name: Name of the activity/boolean mask column
                            in self.data to use as a selector.
      :type activity_name: str

      :returns:

                Subset of extra_data aligned to the time
                    periods where the chosen activity is True. If extra_data is
                    empty, an empty DataFrame is returned.
      :rtype: pandas.DataFrame

      :raises ValueError: If activity_name is not present in self.activities.

      .. rubric:: Notes

      Detailed description:
          The method uses the activity mask stored in self.data but
          returns rows from self.extra_data. This is useful to inspect
          auxiliary channels (for example temperature, pressure, or
          instrument status signals) corresponding to specific tasks
          or events, without mixing them with the primary aerosol
          time series.

      .. rubric:: Examples

      Correlate environmental parameters with a task:

      .. code-block:: python

          env_task = data.get_activity_extra_data("Task")
          env_task[["Temperature", "RelativeHumidity"]].plot()



   .. py:method:: mark_activities(activity_periods, mode = 'union')

      Description:
          Define or update boolean activity masks on the time axis.

      :param activity_periods: Mapping from activity name (str) to
                               one of:

                                   - a (start, end) tuple,
                                   - a list of (start, end) tuples, or
                                   - None (no active periods).

                               Each start/end can be a pandas.Timestamp, datetime, or any
                               string understood by pandas.to_datetime.
      :type activity_periods: dict
      :param mode: How to combine new periods with existing masks of
                   the same name. One of:

                       - "union": existing OR new periods (default),
                       - "replace": overwrite any existing mask,
                       - "intersection": keep only overlapping periods.
      :type mode: str

      :returns:

                The object is modified in place by adding/updating
                    boolean columns in self.data and entries in
                    self.activity_periods.
      :rtype: None

      :raises ValueError: If mode is not "union", "replace", or "intersection".
          Check the spelling of mode and choose among the supported
          options.

      .. rubric:: Notes

      Detailed description:
          For each activity key, a boolean Series is built over the
          current time index. All timestamps falling between any of
          the supplied (start, end) pairs (inclusive) are set to True,
          and others to False. Depending on mode, this new mask is
          merged with any existing column of the same name. The
          normalized list of (start, end) periods is stored in
          self.activity_periods[activity].

      Theory:
          The method implements generic time segmentation. Activities
          can represent tasks, locations, process states, or anything
          else defined by the user.

      .. rubric:: Examples

      Mark a measurement into “Task” and “Background” segments:

      .. code-block:: python

          periods = {
              "Task": [("2025-01-24 09:00", "2025-01-24 11:00"),
                          ("2025-01-24 12:30", "2025-01-24 14:00")
                          ],
              "Background": [("2025-01-24 08:30", "2025-01-24 09:00")]
          }
          data.mark_activities(periods, mode="replace")
          data.plot_total_conc(mark_activities=True)



   .. py:method:: Peak_finder(window = 15, ratio = 2.5, method = 'median', specific_data = '')

      Description:
          Detect peaks and mark them as an activity labeled 'Peak'.

      :param window: Rolling window size in number of samples used to
                     compute the local baseline and spread. Defaults to 15.
      :type window: int
      :param ratio: Threshold factor on the rolling standard
                    deviation. A time step is flagged as peak when
                    value - baseline > ratio * rolling_std. Defaults to 2.5.
      :type ratio: float
      :param method: Aggregation used as the rolling baseline, one of
                     "mean", "median", "sum", "min", "max". Defaults to "median".
      :type method: str
      :param specific_data: Optional column name to use for peak
                            detection. If empty, the total_concentration time series is
                            used. Otherwise, the name is looked up first in self.data
                            (numeric columns) and then in extra_data.
      :type specific_data: str

      :returns:

                The method adds/updates a boolean "Peak" column in
                    self.data and updates self.activity_periods["Peak"].
      :rtype: None

      :raises ValueError: If method is not one of the supported names. Choose
          one of "mean", "median", "sum", "min", "max".
      :raises ValueError: If specific_data is not found as a numeric column
          in either self.data or extra_data. Check column names via
          data.columns or extra_data.columns.

      .. rubric:: Notes

      Detailed description:
          The method computes a rolling baseline and rolling standard
          deviation over the chosen data series. Points where the
          deviation from the baseline exceeds ratio times the rolling
          standard deviation are marked as peaks. Contiguous True
          segments in this mask are converted to (start, end) time
          intervals and stored as the "Peak" activity.

      Theory:
          This is a simple statistical peak detector based on local
          outlier detection relative to a moving baseline and spread.
          It highlights periods where the signal is unusually high
          compared to its recent history.

      .. rubric:: Examples

      Automatically tag high-exposure episodes:

      .. code-block:: python

          data.Peak_finder(window=31, ratio=3.0)
          data.plot_total_conc(mark_activities=["Peak"])
          peak_df = data.get_activity_data("Peak")



   .. py:method:: plot_total_conc(ax = None, mark_activities = False)

      Description:
          Plot the time series of total aerosol concentration.

      :param ax: Optional axis to draw the plot on.
                 If None, a new figure and axes are created.
      :type ax: matplotlib.axes.Axes | None
      :param mark_activities: Controls highlighting
                              of activity periods:

                                  - False: no highlighting.
                                  - True: shade all activities except "All data".
                                  - sequence of str: shade only the named activities.
      :type mark_activities: bool | Sequence[str]

      :returns:

                The
                    figure and axes containing the plot.
      :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

      :raises None: Any errors usually stem from invalid Matplotlib axes or
          malformed time indices.

      .. rubric:: Notes

      Detailed description:
          The method draws total_concentration versus time with a
          grid and labels derived from dtype and unit. The x-axis is
          formatted using Matplotlib's concise date formatter. If
          mark_activities is enabled, each selected activity is drawn
          as semi-transparent vertical spans covering its active
          periods, with a legend entry per activity.

      .. rubric:: Examples

      Inspect an instrument time series with highlighted tasks:

      .. code-block:: python

          fig, ax = data.plot_total_conc(mark_activities=True)
          fig.savefig("total_concentration.png", dpi=150)



   .. py:method:: timecrop(start = None, end = None, inplace = True, focus = True, crop_extra = True)

      Description:
          Crop the time axis to include or exclude a time interval.

      :param start: Start of the interval.
                    If None, cropping starts at the first timestamp.
      :type start: str | pandas.Timestamp | None
      :param end: End of the interval.
                  If None, cropping ends at the last timestamp.
      :type end: str | pandas.Timestamp | None
      :param inplace: If True, modify the current object and return
                      it. If False, return a cropped deep copy.
      :type inplace: bool
      :param focus: If True, keep only the interval [start, end].
                    If False, remove that interval and keep data before start
                    and after end.
      :type focus: bool
      :param crop_extra: If True, also crop extra_data to the same
                         time mask (when inplace=True or on the copy).
      :type crop_extra: bool

      :returns:

                The cropped object (self when inplace=True, or a
                    new instance when inplace=False).
      :rtype: Aerosol1D

      :raises ValueError: May be raised by pandas.to_datetime if start or
          end strings cannot be parsed. Check the date format
          (for example "YYYY-MM-DD HH:MM:SS").

      .. rubric:: Notes

      Detailed description:
          A boolean mask is built over the current time index based
          on start and end. Depending on focus, this mask either
          selects or removes the specified interval. All columns in
          self.data are subset accordingly. When crop_extra is True,
          extra_data is cropped using the same mask, keeping time
          alignment between main and extra channels.

      .. rubric:: Examples

      Drop a calibration period at the start of a measurement:

      .. code-block:: python

          data.timecrop(end="2025-01-24 09:00", focus=False)
          data.plot_total_conc()



   .. py:method:: timerebin(freq = 's', start = None, end = None, method = 'mean', inplace = True)

      Description:
          Resample the time series to a new regular frequency.

      :param freq: Target sampling interval as a pandas offset alias
                   (for example "30s", "1min", "5min", "1H"). Defaults to "s".
      :type freq: str
      :param start: Optional start time
                    for the resampling window. If None, the first timestamp is
                    used.
      :type start: datetime | pandas.Timestamp | None
      :param end: Optional end time for
                  the resampling window. If None, the last timestamp is used.
      :type end: datetime | pandas.Timestamp | None
      :param method: Aggregation function for numeric
                     columns. Any aggregation accepted by pandas.DataFrame.resample
                     (for example "mean", "sum", "median") or a custom callable.
      :type method: str | Callable
      :param inplace: If True, update this object and return it. If
                      False, leave this instance unchanged and return a rebinned
                      deep copy.
      :type inplace: bool

      :returns:

                The rebinned object (self when inplace=True, or a
                    new instance otherwise) with a regular DatetimeIndex and
                    resampled main and extra data.
      :rtype: Aerosol1D

      :raises ValueError: May be raised by pandas if freq is not a valid
          offset alias or if start/end cannot be interpreted as
          timestamps.

      .. rubric:: Notes

      Detailed description:
          Numeric columns in self.data are resampled using the chosen
          method, while boolean columns are aggregated with a logical
          maximum so that any True within a bin keeps the bin True.
          The resulting DataFrame is reindexed to a fully regular
          time grid between the floored start and end times. Extra
          data, if present, are resampled in the same way and aligned
          to the new index. Activity masks such as "All data" are
          maintained and missing boolean values are filled with False.

      .. rubric:: Examples

      Convert a high‑frequency logger to 1‑minute averages:

      .. code-block:: python

          data.timerebin(freq="1min", method="mean")
          data.plot_total_conc()



   .. py:method:: timeshift(seconds = 0, minutes = 0, hours = 0, inplace = True, shift_extra = True)

      Description:
          Shift the time index by a constant offset.

      :param seconds: Offset in seconds; positive shifts the series
                      forward in time, negative backward. Defaults to 0.
      :type seconds: float
      :param minutes: Offset in minutes, converted internally to
                      seconds and added to seconds. Defaults to 0.
      :type minutes: float
      :param hours: Offset in hours, converted internally to
                    seconds and added to seconds. Defaults to 0.
      :type hours: float
      :param inplace: If True, shift this object in place and return
                      it. If False, return a shifted deep copy. Default is True.
      :type inplace: bool
      :param shift_extra: If True, also shift the index of
                          extra_data so that main and extra channels remain aligned.
                          Default is True.
      :type shift_extra: bool

      :returns:

                Object with shifted time index (self when
                    inplace=True, otherwise a new instance).
      :rtype: Aerosol1D

      :raises None: Unless invalid types are passed for the offset arguments.

      .. rubric:: Notes

      Detailed description:
          A pandas.Timedelta is constructed from the combined offset:
          total_shift = seconds + 60*minutes + 3600*hours.
          This timedelta is added to the index of self.data, and
          optionally to self.extra_data. Activity masks and other
          columns are left unchanged; only timestamps move.

      .. rubric:: Examples

      Align measurements from two instruments with a known clock
      offset:

      .. code-block:: python

          cpc.timeshift(minutes=+2)
          elpi.timeshift(minutes=-1)
          cpc.plot_total_conc()
          elpi.plot_total_conc()



   .. py:method:: timesmooth(window = 5, method = 'mean', inplace = True)

      Description:
          Apply a centered rolling smoother to numeric columns.

      :param window: Rolling window size in number of samples.
                     Defaults to 5.
      :type window: int
      :param method: Aggregation used for smoothing, one of
                     "mean", "median", "sum", "min", "max". Defaults to "mean".
      :type method: str
      :param inplace: If True, replace the numeric columns of this
                      object with their smoothed versions. If False, return a
                      smoothed deep copy.
      :type inplace: bool

      :returns:

                Object with smoothed numeric columns and unchanged
                    boolean/activity columns (self when inplace=True, otherwise
                    a new instance).
      :rtype: Aerosol1D

      :raises ValueError: If method is not among the supported names.
          Choose one of "mean", "median", "sum", "min", "max".

      .. rubric:: Notes

      Detailed description:
          The method separates numeric and boolean columns in
          self.data. A centered rolling window with at least one
          valid sample (min_periods=1) is applied to numeric columns
          using the requested aggregation, producing a smoothed
          version of each numeric time series. Boolean columns
          untouched.

      Theory:
          The smoothing is a simple moving-window operation. It
          can help reduce noise in total concentration or other
          instrument channels prior to further analysis.

      .. rubric:: Examples

      Smooth a noisy CPC time series while preserving activities:

      .. code-block:: python

          smooth = data.timesmooth(window=11, method="median", inplace=False)
          smooth.plot_total_conc(mark_activities=True)



