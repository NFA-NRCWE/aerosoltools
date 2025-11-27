aerosoltools.loaders.OPS
========================

.. py:module:: aerosoltools.loaders.OPS




Module Contents
---------------

.. py:function:: Load_OPS_file(file, extra_data = False)

   Description:
       Load a TSI OPS spectrometer export and return it as an
       :class:`Aerosol2D` number-size distribution with metadata.

   :param file: Path to the OPS data file exported either via the AIM software or
                directly from the OPS instrument.
   :type file: str
   :param extra_data: If ``True``, auxiliary channels (e.g. status, environmental
                      variables, Bin 17 for direct exports) are stored in ``extra_data``
                      when supported by the underlying loader. Defaults to ``False``.
   :type extra_data: bool, optional

   :returns:     OPS size distributions with a datetime index, total concentration,
                 size-resolved bins, and associated metadata.
   :rtype: Aerosol2D

   :raises FileNotFoundError: If ``file`` does not exist or cannot be opened.
   :raises UnicodeDecodeError: If the file cannot be decoded using the encodings tried by
       :func:`_detect_delimiter`.
   :raises Exception: If the first header line cannot be recognised as either an AIM
       export (``"Sample File"``) or a direct OPS export
       (``"Instrument Name"``), and the file therefore cannot be routed
       to a supported loader.

   .. rubric:: Notes

   Detailed description:
       This loader supports two main OPS export flavours:

       - AIM software exports (AIM-generated CSV),
       - direct instrument exports written by the OPS itself.

       Internally, the function:

       - Uses :func:`_detect_delimiter` to infer file encoding and field
         delimiter.
       - Reads the first line of the file using :func:`numpy.genfromtxt`
         and inspects the first token:

         - If the line starts with ``"Sample File"``, the file is treated
           as an AIM export and passed to :func:`_Load_OPS_AIM`, which:

           - reads the main data table starting at the AIM header,
           - reconstructs a single ``Datetime`` column from ``"Date"`` and
             ``"Start Time"``,
           - extracts OPS mid diameters (in µm) from specific columns,
             converts them to nm, and builds bin edges from lower/upper
             cutpoints in the header,
           - sums over the size bins to compute ``"Total_conc"`` for each
             time step,
           - interprets the metadata block to infer the underlying moment
             (Nu, Su, Vo, Ma) and normalisation (e.g. ``/dlogDp``),
           - constructs an :class:`Aerosol2D` with ``Datetime``,
             ``Total_conc`` and one column per size bin (named by bin
             midpoint in nm), and attaches metadata such as bin edges,
             bin mids, density, serial number, unit and dtype,
           - converts the distribution to number concentration and removes
             any ``/dlogDp`` normalisation.

         - If the line starts with ``"Instrument Name"``, the file is
           treated as a direct OPS export and passed to
           :func:`_Load_OPS_Direct`, which:

           - reads a metadata block from the top of the file (including
             test start date/time, sample interval, bin cutpoints, flow
             and density),
           - reconstructs absolute timestamps from the test start time and
             the ``"Elapsed Time [s]"`` column,
           - converts counts in each size bin (Bin 1–16) to number
             concentration in ``cm⁻³`` using the nominal flow rate and
             sample interval adjusted for dead time,
           - computes ``"Total_conc"`` as the sum over bins,
           - defines bin edges from the reported bin cut points and
             derives bin mid diameters in nm,
           - builds an :class:`Aerosol2D` with ``Datetime``, ``Total_conc``
             and size-bin columns, and attaches metadata including bin
             edges/mids, density, serial number, unit and dtype.

       - If ``extra_data=True``:

         - AIM exports preserve non-distribution columns (e.g. flags,
           additional channels) in ``extra_data``.
         - Direct exports store Bin 17 (converted to concentration when
           available) and other non-size-bin columns in ``extra_data``
           indexed by ``Datetime``.

       The returned :class:`Aerosol2D` is therefore a number-size
       distribution (dN, cm⁻³) with OPS-specific binning and metadata,
       ready for further analysis or plotting.

   Theory:
       OPS exports provide binned particle counts and, depending on the
       export type, may encode different moments or normalisations:

       - AIM exports may report number, surface, volume or mass based
         moments (Nu, Su, Vo, Ma) and can be normalised by ``dlogDp`` or
         ``dDp``. The internal AIM loader interprets this from the
         metadata and uses internal helpers to convert the distribution to
         number concentration (dN, cm⁻³) and remove any ``/dlogDp`` or
         ``/dDp`` normalisation.
       - Direct OPS exports report counts per bin over a known sample
         interval and flow. These are converted to number concentrations
         by

         .. math::

             C = \frac{N}{Q \cdot (\Delta t - t_\mathrm{dead})},

         where :math:`N` is the count in the bin, :math:`Q` is the
         volumetric flow (cm³/s), :math:`\Delta t` is the sample
         interval, and :math:`t_\mathrm{dead}` is the recorded dead time.

       Bin edges (in µm) are taken from the OPS metadata and converted to
       nm; mid diameters are defined as geometric means of neighbouring
       edges and used to label the size-bin columns.

   .. rubric:: Examples

   Typical usage is to load OPS data from either AIM or direct
   instrument exports and work directly with the resulting
   number-size distribution:

   .. code-block:: python

       import aerosoltools as at

       # Load OPS data (AIM or direct export)
       ops = at.Load_OPS_file("data/OPS_export.csv", extra_data=True)

       # Inspect the first few rows
       print(ops.data.head())

       # Inspect bin edges and metadata
       print(ops.bin_edges)
       print(ops.metadata)

       # Plot a time-integrated or mean size distribution
       fig, ax = ops.plot_psd()


