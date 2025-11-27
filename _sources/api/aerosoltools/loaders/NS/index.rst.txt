aerosoltools.loaders.NS
=======================

.. py:module:: aerosoltools.loaders.NS




Module Contents
---------------

.. py:function:: Load_NS_file(file, extra_data = False)

   Description:
       Load a NanoScan SMPS (NS) CSV export and return it as an
       :class:`Aerosol2D` number-size distribution with metadata.

   :param file: Path to the NanoScan CSV export file.
   :type file: str
   :param extra_data: If ``True``, non-size-bin columns (e.g. status, density) are stored
                      in ``extra_data`` indexed by ``Datetime``. Defaults to ``False``.
   :type extra_data: bool, optional

   :returns:     NanoScan size distributions with a datetime index, total
                 concentration, size-resolved bins, and associated metadata.
   :rtype: Aerosol2D

   :raises FileNotFoundError: If ``file`` does not exist or cannot be opened.
   :raises UnicodeDecodeError: If the CSV file cannot be decoded using the encodings tried by
       :func:`_detect_delimiter`.
   :raises ValueError: If the reported data type/unit string cannot be mapped to a known
       NanoScan format (e.g. dN, dS, dV, dM), or if the datetime column
       cannot be parsed.

   .. rubric:: Notes

   Detailed description:
       This loader is tailored to NanoScan SMPS CSV exports produced by
       the TSI AIM instrument software. The file typically contains several
       header rows, followed by a table with:

       - a ``"Date Time"`` column,
       - size-bin columns labelled by mid diameters (nm),
       - additional metadata columns (e.g. particle density).

       Internally, the function:

       - Uses :func:`_detect_delimiter` to infer file encoding and field
         delimiter.
       - Reads the main data table, starting at the NanoScan data header
         (``header=5``).
       - Drops columns that are not part of the size distribution or time:
         ``"File Index"``, ``"Sample #"``, and ``"Total Conc"``.
       - Extracts bin mid diameters from the first 13 size-bin column
         headers and converts them to floats. From these, it computes bin
         edges in nm using geometric means between adjacent mids, with
         fixed outer edges at 10 nm and 420 nm.
       - Renames ``"Date Time"`` to ``"Datetime"`` and parses timestamps
         using the format ``"%Y/%m/%d %H:%M:%S"``.
       - Selects the size-distribution columns (those matching the
         bin-mid labels) as numeric data.
       - Optionally collects all non-size-bin columns into ``extra_data``
         when ``extra_data=True``, with ``Datetime`` as the index.
       - Reads a header line to extract the instrument serial number.
       - Reads the first data row to infer the data-type string
         (e.g. ``"dN/dlogDp"``) and extracts particle density.
       - Maps the data-type prefix (``"dN"``, ``"dS"``, ``"dV"``, ``"dM"``)
         to a physical unit via a small lookup, raising a ``ValueError``
         if the type is unknown.
       - Computes ``"Total_conc"`` as the sum over all size bins for each
         time step and concatenates ``Datetime``, ``Total_conc`` and the
         size-bin columns into a single DataFrame.
       - Creates an :class:`Aerosol2D` object from this DataFrame and
         populates metadata:

         - ``instrument`` set to ``"NS"``,
         - ``bin_edges`` and ``bin_mids`` (rounded to one decimal place),
         - ``density`` (float),
         - ``serial_number``,
         - ``unit`` (string) and ``dtype`` (raw type string from header).

       - Calls ``_convert_to_number_concentration()`` followed by
         :meth:`Aerosol2D.unnormalize_logdp` so that the final
         distribution is expressed as number concentration per bin
         (dN, cm⁻³) without ``/dlogDp`` normalisation.
       - If ``extra_data=True``, attaches the non-size-bin columns as
         ``extra_data`` indexed by ``Datetime``.

   Theory:
       NanoScan exports can represent different moments of the particle
       size distribution (number, surface, volume, mass) and may be
       normalised by ``/dlogDp``. The raw type string typically encodes
       this, for example:

       - ``dN/dlogDp`` — number concentration per logarithmic diameter
         interval,
       - ``dS/dlogDp`` — surface-based moment,
       - ``dV/dlogDp`` — volume-based moment,
       - ``dM/dlogDp`` — mass-based moment.

       From the two-letter prefix (``dN``, ``dS``, ``dV``, ``dM``), the
       loader assigns the corresponding unit (cm⁻³, nm²/cm³, nm³/cm³,
       µg/m³). The internal helper then converts any supported moment to
       number concentration and removes the ``/dlogDp`` normalisation so
       that the resulting distribution is directly comparable to other
       number-size distributions in aerosoltools.

     Examples:
         Typical usage is to load a NanoScan CSV file and work directly
         with the resulting number-size distribution:

         .. code-block:: python

             import aerosoltools as at

             # Load NanoScan data (keep extra metadata columns)
             ns = at.Load_NS_file("data/NanoScan_export.csv", extra_data=True)

             # Inspect the first few rows
             print(ns.data.head())

             # Check bin edges and metadata
             print(ns.bin_edges)
             print(ns.metadata)

             # Plot a time-integrated or mean size distribution
             fig, ax = ns.plot_psd()


