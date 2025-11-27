aerosoltools.aerosol1d
======================

.. py:module:: aerosoltools.aerosol1d






Module Contents
---------------

.. py:data:: params

.. py:data:: DateLike

.. py:class:: Aerosol1D(dataframe)

   Handle 1D aerosol time-series measurements.

   This class manages time-indexed aerosol concentration data (for example
   total particle number or mass concentration) and provides utilities for
   resampling, smoothing, marking activity segments, cropping, shifting,
   summarizing, and plotting.

   It is intended for pre- and post-processing of aerosol datasets collected
   with portable or stationary particle counters that report a single
   concentration value per time step.

   :param dataframe: Input data. If the index is not a
                     :class:`pandas.DatetimeIndex`, the first column is interpreted as
                     timestamps and converted using :func:`pandas.to_datetime`, then
                     set as the index. The first data column is assumed to represent
                     total particle concentration.
   :type dataframe: pandas.DataFrame

   .. attribute:: data

      Main data frame containing the time index,
      the total concentration column, and any activity masks (e.g.
      ``"All data"``, ``"Peak"``, user-defined activities).

      :type: pandas.DataFrame

   .. attribute:: extra_data

      Additional columns extracted from the
      raw file that are not considered core aerosol measurements
      (e.g. environmental or meta signals).

      :type: pandas.DataFrame

   .. attribute:: activities

      List of defined activity labels for which
      boolean mask columns exist in :attr:`data`.

      :type: list[str]

   .. attribute:: activity_periods

      Mapping from activity name to a list of (start, end) time intervals
      where that activity is considered active.

      :type: dict[str, list[tuple[pandas.Timestamp, pandas.Timestamp]]]

   .. attribute:: metadata

      Dictionary of metadata such as unit, data type,
      instrument type and serial number. Internally stored in
      ``self._meta``.

      :type: dict

   .. rubric:: Notes

   Users are encouraged to interact with the class via its public
   properties and methods (e.g. :attr:`data`, :meth:`mark_activities`,
   :meth:`timerebin`, :meth:`plot_total_conc`) rather than modifying
   private attributes directly.


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


   .. py:property:: metadata
      :type: Dict


      Metadata associated with the dataset.

      :returns: Dictionary containing metadata such as unit, data type,
                instrument type, and serial number. Internally stored in
                ``self._meta``.
      :rtype: dict


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



   .. py:method:: summarize_activities(filename = None, metrics = None)

      Description:
          Summarize 1D aerosol metrics per activity.

      :param filename: Optional path to a CSV or Excel file.
                       If provided, the summary table is appended to the file
                       (creating it if it does not exist). If None, nothing is
                       written to disk.
      :type filename: str | None
      :param metrics: List of metric names to summarize.
                      If None, a default set is used: ["PNC"].

                      * "PNC" refers to total_concentration.
                      * Any other name is looked up in data or extra_data
                        (numeric columns only).
      :type metrics: list[str] | None

      :returns: Summary table with one row per activity and
                columns:

                    * "Segment"
                    * "Duration (HH:MM)"
                    * For each metric M: "M [unit] mean" and "M [unit] std".
      :rtype: pandas.DataFrame

      :raises ValueError: If a requested metric (other than "PNC") cannot
          be found in data or extra_data.

      .. rubric:: Notes

      The per-activity means and standard deviations are simple
      sample statistics over the selected time steps. Durations are
      based on the actual sampling intervals (via _dt_minutes).

      .. rubric:: Examples

      .. code-block:: python

          data.summarize_activities()



   .. py:method:: summarize_exposure(metric = 'PNC', background = None, exposure_hours = None, short_limit = 1.0, long_limit = 1.0, short_window = '15min', twa_window = '8h', peak_ratio = 2.5, filename = None, activities = None)

      Description:
          Summarize exposure metrics for one 1D metric across activities.

      :param metric: Exposure metric name based on the underlying
                     1D time series. Default is "PNC", corresponding to the
                     total particle number concentration. Supported forms:

                         * "PNC": total number concentration, mapped to
                           :attr:`total_concentration`.
                         * Any other string: interpreted as the name of a
                           numeric column in :attr:`data` or :attr:`extra_data`
                           (case-sensitive). For example, "MASS" or "TEMP"
                           if such columns exist.
      :type metric: str
      :param background: Background level used when
                         computing the time-weighted average over ``twa_window``.
                         The same background level is used for all activities in
                         the output. Possible entries are:

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
      :type exposure_hours: float | None
      :param short_limit: Short-term concentration limit in metric
                          units (for example a 15-min STEL). This value is reported
                          in the output as ``"STEL [unit]"``.
      :type short_limit: float
      :param long_limit: Long-term concentration limit in metric
                         units (for example an 8-h OEL). This value is reported in
                         the output as ``"Exposure limit [unit]"``.
      :type long_limit: float
      :param short_window: Rolling window used for short-term (STEL)
                           evaluation, given as a pandas offset string (for example
                           "15min"). This is reported as ``"STEL window [offset]"`` and
                           used to construct a time-based rolling mean.
      :type short_window: str
      :param twa_window: Total duration of the TWA window as a pandas
                         offset string (for example "8h"). This is reported as
                         ``"TWA window [offset]"`` and used to embed the segment
                         exposure into a reference window (for example an 8-h shift).
      :type twa_window: str
      :param peak_ratio: Factor used in peak detection; peaks are
                         flagged when the metric exceeds::

                             baseline + peak_ratio * rolling_std

                         where ``baseline`` is a rolling median and ``rolling_std`` is
                         a rolling standard deviation over a short window.
      :type peak_ratio: float
      :param filename: Optional path to a CSV/Excel file to
                       which the non-transposed result rows are appended. If the
                       file exists, rows are appended; otherwise the file is
                       created with a header. Supported extensions are ".csv",
                       ".xls", ".xlsx".
      :type filename: str | None
      :param activities: Activities to summarize.
                         If None (default), all defined activities in
                         :attr:`activities` are summarized (for example "All data",
                         "Background", "Task"). Activities with no marked time steps
                         are skipped.
      :type activities: Sequence[str] | None

      :returns: One row per activity segment with summary
                statistics for the chosen metric. Column names embed their units
                in square brackets where applicable (for example "Max [µg/m³]").
                The set of columns matches that of
                :meth:`Aerosol2D.summarize_exposure`, restricted to a single
                1D metric.
      :rtype: pandas.DataFrame

      :raises ValueError: If ``metric`` cannot be found (except "PNC").
      :raises ValueError: If a background activity name is given but does not
          exist or has no samples.
      :raises ValueError: If ``short_window`` or ``twa_window`` cannot be
          parsed as pandas-style durations.
      :raises ValueError: If ``exposure_hours`` is negative.
      :raises TypeError: If ``background`` is not None, float, or str.

      .. rubric:: Notes

      Detailed description:
          The method first obtains the requested metric time series
          from the 1D data:

              - For "PNC", the series is taken from
                :attr:`total_concentration`.
              - For any other metric name, the series is taken from a
                numeric column in :attr:`data` or :attr:`extra_data`
                (if present).

          For each selected activity, the method:

              1. Extracts the metric time series within the activity
                 using the corresponding boolean mask.
              2. Computes per-step durations from the actual sampling
                 times via :meth:`_dt_minutes`.
              3. Forms a time-weighted segment mean (using the per-step
                 durations), which is used internally when embedding
                 the activity into the TWA window.
              4. Computes instantaneous percentiles, peaks, STEL-style
                 exceedances, and long-term time-above-limit measures.
              5. Combines the segment mean with the specified background
                 level to obtain an overall TWA for the chosen window.

          The returned DataFrame contains the following columns
          (per activity):

              - "Segment"
                  Name of the activity segment.

              - "Metric"
                  Name of the metric that was summarized, e.g.
                  "PNC", "MASS", or any other 1D column name.

              - "Duration [HH:MM]"
                  Duration of the activity derived from the actual
                  sampling intervals, formatted as "HH:MM" (hours and
                  minutes).

              - "Max [unit]"
                  Maximum of the instantaneous metric values during the
                  activity. The unit is the metric unit (for example
                  "cm⁻³" or "µg/m³").

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

                      baseline + peak_ratio * rolling_std

                  with ``baseline`` the rolling median and
                  ``rolling_std`` the rolling standard deviation over a
                  short sliding window, covering at least 3 datapoints
                  and at most 15 datapoints, depending on how many
                  samples are available in the segment.

              - "STEL [unit]"
                  Short-term exposure limit used in the STEL
                  exceedance calculations (echoing ``short_limit``),
                  with the same concentration unit as the metric.

              - "STEL window [offset]"
                  Rolling window length used for STEL evaluation,
                  echoing ``short_window`` (for example "15min").

              - "STEL exceedance [min]"
                  Total time in minutes during the activity where the
                  **full-window** rolling mean is greater than or equal
                  to the STEL. Only complete windows are counted in this
                  measure.

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
                  the full TWA window (``"TWA window [offset]"``),
                  combining the internal segment mean (optionally
                  stretched to ``exposure_hours``) with the specified
                  background level for the remainder of the window::

                      TWA = (segment_mean * exposure_minutes
                             + background * (twa_minutes - exposure_minutes))
                            / twa_minutes

                  where ``twa_minutes`` is the duration of
                  ``twa_window`` in minutes and ``exposure_minutes`` is
                  either the measured activity duration or
                  ``exposure_hours * 60``, whichever is used.

              - "TWA window [offset]"
                  Duration of the TWA window over which
                  "TWA concentration [unit]" is evaluated, echoing
                  ``twa_window`` (for example "8h").

      Theory:
          Conceptually, the method mirrors typical occupational hygiene
          practice for a single 1D metric:

              - Exposure within an activity is represented by
                time-weighted averages and distributional descriptors
                (max and percentiles).
              - Short-term limits (STEL) are evaluated using rolling
                means over a specified window, counting both total time
                above the limit and distinct exceedance episodes.
              - Long-term limits are represented by cumulative
                time-above-limit measures.
              - An overall TWA for an 8-h (or user-specified) reference
                period is constructed by mixing the task exposure with a
                background level for the rest of the window.

          This ensures a transparent link between raw 1D time-series
          behaviour and regulatory-style metrics such as STEL and
          8-hour OELs.

      .. rubric:: Examples

      Summarize total particle number concentration (PNC) exposure
      for all activities in a CPC dataset, using the "Background"
      activity as background level and a 15-min STEL with an 8-h
      TWA window:

      .. code-block:: python

          cpc.summarize_exposure(
                  metric="PNC",
                  background="Background",
                  short_limit=5e4,
                  long_limit=2e4,
                  short_window="15min",
                  twa_window="8h",
          )

      Other 1D metrics stored as columns (for example "MASS") can
      be handled in the same way simply by changing the ``metric``
      argument.



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



