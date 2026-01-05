aerosoltools.loaders.ELPI
=========================

.. py:module:: aerosoltools.loaders.ELPI




Module Contents
---------------

.. py:function:: Load_ELPI_file(file, extra_data = False)

   Description:
       Load an ELPI size-distribution export and return it as an
       :class:`Aerosol2D` number-size distribution with metadata.

   :param file: Path to the ELPI data file (typically a ``.txt`` or ``.dat`` export).
   :type file: str
   :param extra_data: If ``True``, non-distribution columns (operational parameters,
                      status, etc.) are stored in ``extra_data``. Defaults to ``False``.
   :type extra_data: bool, optional

   :returns:     ELPI size distributions with a datetime index, total concentration,
                 size-resolved bins, and associated metadata.
   :rtype: Aerosol2D

   :raises FileNotFoundError: If ``file`` does not exist or cannot be opened.
   :raises UnicodeDecodeError: If the file cannot be decoded using the encodings tried by
       :func:`_detect_delimiter`.
   :raises ValueError: If the ``[Data]`` marker or header line cannot be found, or if
       datetime parsing fails for a substantial fraction of rows.
   :raises KeyError: If required metadata entries (e.g. cutpoints, moments) are missing
       from the header.
   :raises Exception: If the calculated moment/type in the ELPI metadata cannot be mapped
       to a known combination (Nu, Su, Vo, Ma), indicating an unsupported
       or unconverted export.

   .. rubric:: Notes

   Detailed description:
       This loader is tailored to ELPI (Electrical Low Pressure Impactor)
       exports generated via the ELPI Dekati software. The file should
       contain a header region with key–value metadata and a subsequent
       data block marked by ``[Data]``.

       Internally, the function:

       - Uses :func:`_detect_delimiter` to infer file encoding and field
         delimiter.
       - Calls :func:`_load_ELPI_metadata` to parse the header region into
         a metadata dictionary. In particular, it reads:

         - ``"D50values(um)"`` — used to construct ``bin_edges`` (cutpoints),
         - ``"CalculatedDi(um)"`` — used as geometric bin mid-diameters,
         - ``"Density(g/cm^3)"`` — particle density used for cutpoints,
         - ``"CalculatedMoment"`` / ``"CalculatedType"`` — to determine
           which moment the data represent (Nu, Su, Vo, Ma).

       - Converts bin edges and mid-diameters from µm to nm and, if the
         density differs from 1 g/cm³, recalculates internal bin edges
         using geometric means of adjacent midpoints before continuing.
       - Reads the data block after ``[Data]`` and assigns the aligned
         header as column names.
       - Renames the first column to ``"Datetime"`` and parses timestamps
         by trying several explicit formats (with and without fractional
         seconds), then a permissive fallback. If more than ~20% of rows
         fail, a ``ValueError`` is raised.
       - Extracts the size-distribution block (columns 34–47) as the core
         distribution data and removes these columns from a copy used as
         potential extra data.
       - Determines the physical unit and data type from
         ``"CalculatedMoment"`` / ``"CalculatedType"`` using a small
         lookup (Nu → dN, Su → dS, Vo → dV, Ma → dM). If this fails, an
         exception is raised: A likely cause is that the data is given in
         current (fA) and has not yet been converted with the Dekati software.
       - Computes ``"Total_conc"`` as the sum over all size bins, rounds
         bin mid-diameters to one decimal place, and renames the
         distribution columns to the stringified midpoints.
       - Assembles the final data frame as:

         - ``Datetime``,
         - ``Total_conc``,
         - size-bin columns named by bin midpoint (nm).

       - Constructs an :class:`Aerosol2D` object from this table and
         populates metadata:

         - ``bin_edges`` and ``bin_mids`` in nm,
         - ``density`` (g/cm³),
         - ``instrument`` set to ``"ELPI"``,
         - ``serial_number`` extracted from the first header line,
         - ``unit`` (string) and ``dtype`` (string) for the distribution.

       - Calls ``_convert_to_number_concentration()`` followed by
         :meth:`Aerosol2D.unnormalize_logdp` to ensure the distribution
         is expressed as number concentration per bin (dN, cm⁻³) rather
         than a moment normalized by ``dlogDp``.
       - If ``extra_data=True``, all non-distribution columns are stored
         in ``extra_data`` (with ``Datetime`` as index) and preserved in
         ``_raw_extra_data`` for later use.

   Theory:
       ELPI data can represent different geometric moments of the particle
       size distribution depending on the export settings:

       - ``Nu`` — number-based moment (dN),
       - ``Su`` — surface-based moment (dS),
       - ``Vo`` — volume-based moment (dV),
       - ``Ma`` — mass-based moment (dM).

       The metadata fields ``"CalculatedMoment"`` and ``"CalculatedType"``
       encode which of these is present. The loader uses this to assign
       the correct unit and dtype strings and then converts the
       distribution to number concentration (dN, cm⁻³) using internal
       helpers.

       When the density differs from 1 g/cm³, the nominal cutpoints are
       mass-based; to approximate number-based cutpoints, inner bin edges
       are recomputed using geometric means of adjacent mid-diameters.
       This preserves a consistent bin structure when changing from mass
       to number metrics.

   .. rubric:: Examples

   Typical usage is to load an ELPI export and directly access a
   number-size distribution for further analysis:

   .. code-block:: python

       import aerosoltools as at

       # Load ELPI data as a 2D number-size distribution
       elpi = at.Load_ELPI_file("data/ELPI_export.txt", extra_data=True)

       # Inspect the data
       print(elpi.data)

       # Inspect bin_edges and metadata
       print(elpi.bin_edges)
       print(elpi.metadata)

       # Plot the time-integrated size distribution
       fig, ax = elpi.plot_psd()


