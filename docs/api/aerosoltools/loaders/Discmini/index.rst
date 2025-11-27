aerosoltools.loaders.Discmini
=============================

.. py:module:: aerosoltools.loaders.Discmini




Module Contents
---------------

.. py:function:: Load_DiSCmini_file(file, extra_data = False)

   Description:
       Load a converted DiSCmini export file and return total number
       concentration, mean size, and LDSA as an :class:`AerosolAlt` time
       series.

   :param file: Path to the DiSCmini ``.txt`` file exported and converted by the
                vendor software.
   :type file: str
   :param extra_data: If ``True``, columns that are not part of the core time series
                      (datetime, total concentration, mean size, LDSA) are stored in
                      the returned object's ``.extra_data`` (and ``._raw_extra_data``)
                      DataFrames, indexed by ``Datetime``. Defaults to ``False``.
   :type extra_data: bool, optional

   :returns:     An :class:`~aerosoltools.aerosolalt.AerosolAlt` instance with
                 the loaded data
   :rtype: AerosolAlt

   :raises FileNotFoundError: If ``file`` does not exist or cannot be opened. Check the path
       and file permissions.
   :raises UnicodeDecodeError: If the file cannot be decoded using the encodings tried by
       :func:`_detect_delimiter`. This usually indicates a corrupted
       or non-text file.
   :raises ValueError: If timestamps or header-derived start date/time strings are in
       an unsupported format and cannot be parsed. Check that the file
       was exported using a supported DiSCmini software version and
       that the regional date/time settings are compatible.
   :raises KeyError: If expected columns such as ``"TimeStamp"``/``"Time"``,
       ``"Number"``, or size/LDSA columns are missing or renamed in an
       unexpected way. Verify that the file is an unmodified DiSCmini
       export.
   :raises Exception: If neither direct parsing nor reconstruction of the datetime
       column succeeds, or if a valid ``"Datetime"`` column cannot be
       produced after all attempts. The raised message will typically
       indicate that the file has not been converted correctly or that
       the datetime format is unsupported.

   .. rubric:: Notes

   Detailed description:
       ``Load_DiSCmini_file`` is designed for DiSCmini data that have
       already been converted by the vendor software to a tab-delimited
       text format.

       Internally, the function:

       - Reads a subset of columns (up to 7) as strings to robustly
         handle locale-specific decimal separators and whitespace.
       - Normalizes column names.
       - Attempts to parse the ``"Datetime"`` column using two common
         formats:

         - ``"%d-%b-%Y %H:%M:%S"`` (e.g. ``01-Oct-2023 12:00:00``)
         - ``"%d-%m-%Y %H:%M:%S"`` (e.g. ``01-10-2023 12:00:00``)

       - If both direct parses fail, it falls back to reconstructing
         absolute time from header information:

         - Searches the header for lines containing ``"start date:"``
           and ``"start time:"``.
         - Parses those as a start datetime.
         - Interprets the ``"Datetime"`` column as elapsed seconds
           since that start time, then creates a real timestamp series.

       - Normalizes numeric fields by:

         - Replacing commas with dots as decimal separators.
         - Removing extraneous whitespace.
         - Coercing invalid entries to ``NaN``.

       - Extracts the serial number by scanning the early header lines
         for text containing ``"serial"`` and, if needed, falling back
         to a small :func:`numpy.genfromtxt` read.

       - Builds the core :class:`AerosolAlt` object from the subset of
         columns that includes

         - ``"Datetime"`` — measurement timestamps.
         - ``"Total_conc"`` — particle number concentration (cm⁻³).
         - ``"Size"`` — mean particle diameter (nm).
         - ``"LDSA"`` — lung-deposited surface area (nm²/cm³).

       - Populates the ``.meta`` dictionary with, e.g.:

         - ``"instrument"`` — set to ``"DiSCmini"``.
         - ``"serial_number"`` — serial number parsed from the header.
         - ``"unit"`` — mapping of variable names to units.
         - ``"dtype"`` — mapping of variable names to data types
           (e.g. ``"dN"`` for number concentration, ``"dS"`` for LDSA).

       - Optionally collects all remaining non-core columns into
         ``.extra_data`` (and ``._raw_extra_data``) when
         ``extra_data=True``.

   .. rubric:: Examples

   A typical workflow is to convert a DiSCmini file using the vendor
   software and then load it for analysis or plotting:

   .. code-block:: python

       import aerosoltools as at

       # Load DiSCmini data with core metrics only
       dm = at.Load_DiSCmini_file("data/discmini_converted.txt")

       # Quick look at total concentration and LDSA
       print(dm.data[["Total_conc", "LDSA"]])

       # List available extra channels
       print(dm_full.extra_data.columns)

       # Plot LDSA over time
       fig, ax = dm.plot_timeseries()


