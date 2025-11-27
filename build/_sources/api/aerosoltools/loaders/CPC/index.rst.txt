aerosoltools.loaders.CPC
========================

.. py:module:: aerosoltools.loaders.CPC




Module Contents
---------------

.. py:function:: Load_CPC_file(file, extra_data = False)

   Description:
       Load a TSI CPC (Condensation Particle Counter) export file, automatically
       detect its format, and return a time series of total particle number
       concentration as an :class:`Aerosol1D` object.

   :param file: Path to the CPC data file (typically ``.txt`` or ``.csv``) exported
                from the instrument software.
   :type file: str
   :param extra_data: If ``True`` and the file is in *full* format, non-core diagnostic
                      and operational parameters are stored in the returned object's
                      ``.extra_data`` attribute. For *focused* format files, this flag
                      has no effect because only minimal channels are available.
                      Defaults to ``False``.
   :type extra_data: bool, optional

   :returns:     An :class:`~aerosoltools.aerosol1d.Aerosol1D` instance containing:

                 - ``.data`` with a datetime index and a ``"Total_conc"`` column
                   holding particle number concentration in ``cm⁻³``.
                 - ``.meta`` (metadata) populated with, e.g.:

                   - ``"instrument"`` — set to ``"CPC"``.
                   - ``"serial_number"`` — instrument serial number parsed from
                     the header.
                   - ``"unit"`` — set to ``"cm$^{-3}$"``.
                   - ``"dtype"`` — set to ``"dN"`` (number concentration).

                 If ``extra_data=True`` and the file is a *full* CPC export,
                 additional diagnostic channels are provided in ``.extra_data``.
   :rtype: Aerosol1D

   :raises FileNotFoundError: If the specified ``file`` path does not exist or cannot be opened.
       Verify the path and that you have read permissions.
   :raises UnicodeDecodeError: If the file cannot be decoded using the encodings tried by
       :func:`_detect_delimiter`. Check that the file is a valid CPC
       export and not a binary or corrupted file.
   :raises ValueError: If :func:`_detect_delimiter` cannot reliably determine a delimiter
       from the sampled lines. This can happen for very short or malformed
       files.
   :raises Exception: If the detected column structure does not match any supported CPC
       format (currently focused with 4 columns or full with 14 columns).
       In this case, check that the file is an unmodified CPC export and
       that the header layout has not changed.

   .. rubric:: Notes

   Detailed description:
       ``Load_CPC_file`` is the main entry point for reading CPC data.
       It is designed to handle at least two common export layouts:

       - Focused format directly from the instrument (compact output):

         - Contains only a time column and total number concentration.
         - Identified by having 4 columns in the numeric data section.
         - Internally parsed by a specialized focused-format loader.

       - Full format via TSI AIM software (diagnostic-rich output):

         - Contains additional operational and diagnostic parameters
           (e.g. flow, temperature, status flags) in separate columns.
         - Identified by having 14 columns in the numeric data section.
         - Internally parsed by a specialized full-format loader, which
           also exposes optional diagnostics via ``.extra_data`` when
           requested.

   .. rubric:: Examples

   Typical usage is to load a single CPC file (regardless of its
   export format) and immediately access a clean time series:

   .. code-block:: python

       import aerosoltools as at

       # Load a CPC export (focused or full format)
       cpc = at.Load_CPC_file("data/CPC_2023-10-01.txt", extra_data=True)

       # Inspect the main total concentration time series
       print(cpc.data)

       # Plot the time series of total concentration
       fig, ax = cpc.plot_timeseries()


