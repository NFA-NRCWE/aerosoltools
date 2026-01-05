aerosoltools.loaders.DustTrak
=============================

.. py:module:: aerosoltools.loaders.DustTrak




Module Contents
---------------

.. py:function:: Load_DustTrak_file(file, extra_data = False)

   Description:
       Load a DustTrak DRX export file and return PM mass concentrations and
       total mass as an :class:`AerosolAlt` time series.

   :param file: Path to the DustTrak DRX data file (typically a ``.csv`` export).
   :type file: str
   :param extra_data: If ``True``, non-core columns (e.g. alarms, errors, diagnostics)
                      are stored in ``extra_data``. Defaults to ``False``.
   :type extra_data: bool, optional

   :returns:     DustTrak measurements with a datetime index and PM channels
                 (PM1, PM2.5, PM4, PM10, Total) in µg/m³.
   :rtype: AerosolAlt

   :raises FileNotFoundError: If ``file`` does not exist or cannot be opened.
   :raises UnicodeDecodeError: If the file cannot be decoded with the inferred or fallback
       encoding.
   :raises ValueError: If the start datetime cannot be parsed from the header, or if
       elapsed time cannot be converted to timestamps.
   :raises KeyError: If expected columns (elapsed time or PM channels) are missing or
       renamed unexpectedly.

   .. rubric:: Notes

   Detailed description:
       This loader assumes a standard DustTrak DRX ASCII/CSV export with
       a metadata header followed by a tabular data block. Internally it:

       - Tries to infer file encoding and delimiter using
         :func:`_detect_delimiter`; if this fails, it falls back to
         ``encoding="latin-1"`` and ``delimiter=","``.
       - Reads the data block starting at row 35 and renames key columns:

         - ``"Elapsed Time [s]"`` → ``"Datetime"`` (elapsed seconds),
         - ``"PM1 [mg/m3]"`` → ``"PM1"``,
         - ``"PM2.5 [mg/m3]"`` → ``"PM2.5"``,
         - ``"PM4 [mg/m3]"`` → ``"PM4"``,
         - ``"PM10 [mg/m3]"`` → ``"PM10"``,
         - ``"TOTAL [mg/m3]"`` → ``"Total"``.

       - Reads the first 8 header lines to extract instrument name,
         model number, serial number, and the start date/time. The
         start datetime is parsed using ``"%d/%m/%Y %H:%M:%S"``.
       - Converts the elapsed seconds in ``"Datetime"`` to absolute
         timestamps by adding a time delta to the parsed start datetime.
       - Converts all PM channels from mg/m³ to µg/m³.
       - Creates an :class:`AerosolAlt` object using ``Datetime`` and the
         PM channels as the main data frame, and attaches metadata:

         - ``instrument``, ``model_number``, ``serial_number``,
         - per-channel units (µg/m³),
         - per-channel dtype (``"dM"``).

       - If ``extra_data=True``, any remaining columns are moved to
         ``extra_data`` (and ``_raw_extra_data``) indexed by ``Datetime``,
         so alarms or diagnostic signals are preserved.

   .. rubric:: Examples

   Typical usage is to load one or more DustTrak DRX files for QA/QC
   or exposure assessment:

   .. code-block:: python

       import aerosoltools as at

       # Load core PM fractions
       dust = at.Load_DustTrak_file("data/DustTrak_DRX_2023-10-01.csv")

       # Inspect PM2.5 and PM10 time series
       print(dust.data[["PM2.5", "PM10"]].head())

       # Plot total mass concentration
       fig, ax = dust.plot_timeseries()


