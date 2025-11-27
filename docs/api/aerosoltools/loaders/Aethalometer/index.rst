aerosoltools.loaders.Aethalometer
=================================

.. py:module:: aerosoltools.loaders.Aethalometer




Module Contents
---------------

.. py:function:: Load_Aethalometer_file(file, extra_data = False)

   Description:
       Load time-resolved black carbon data from an Aethalometer export file
       into an :class:`aerosoltools.AerosolAlt` object.

   :param file: Path to the Aethalometer export file (typically ``.csv`` or
                ``.txt``) containing time-resolved concentration data.
   :type file: str
   :param extra_data: If ``True``, non-core variables (e.g. diagnostics, sensor status
                      or quality flags) are stored in ``AerosolAlt.extra_data`` with
                      a ``Datetime`` index. If ``False`` (default), only the main
                      measurement channels are kept in ``.data``.
   :type extra_data: bool, optional

   :returns:     An :class:`~aerosoltools.aerosolalt.AerosolAlt` instance with:

                 - ``.data`` containing the primary BC concentration channels
                   (at least IR/UV/blue/green/red) indexed by ``Datetime``.
                 - ``.metadata`` (``.meta``) populated with, e.g.:

                   - ``"serial_number"`` — Aethalometer serial number.
                   - ``"optical_config"`` — optical configuration string.
                   - ``"instrument"`` — set to ``"Aethalometer"``.
                   - ``"unit"`` — set to ``"ng/m³"``.
                   - ``"dtype"`` — set to ``"dm"`` (mass concentration).

                 If ``extra_data=True``, additional columns that are not part of
                 the core BC channels are stored in ``.extra_data``.
   :rtype: AerosolAlt

   :raises FileNotFoundError: If the specified ``file`` path does not exist.
   :raises UnicodeDecodeError: If the file encoding cannot be correctly decoded using the
       detected encoding. Check that the file is not corrupted and
       matches the expected export format.
   :raises KeyError: If expected columns (e.g. ``"Date / time local"``,
       ``"Serial number"``, ``"Optical config"``, or the BC channels)
       are missing or renamed in an unexpected way. Verify that the
       file is an unmodified Aethalometer export.
   :raises ValueError: If the datetime strings in the ``"Date / time local"`` column
       cannot be parsed using the expected format
       ``"%Y-%m-%dT%H:%M:%S"``. Check that the locale and export
       format match the loader assumptions.
   :raises Exception: If the file is read successfully but, after dropping empty rows,
       no data remain (i.e. the dataset is effectively empty). This
       typically indicates an export problem or a file containing only
       headers.

   .. rubric:: Notes

   Detailed description:
       This loader is tailored to Aethalometer (e.g. MicroAeth) ASCII or
       CSV exports produced by the vendor software. Internally, it:

       - Auto-detects file delimiter and encoding.
       - Reads the file and drops the ``"Readable status"`` column
         if present, and removes fully empty rows.
       - Extracts metadata from the first row (serial number, optical
         configuration) and stores it in the internal ``.meta``
         dictionary.
       - Detects the file structure: for wide exports (many columns,
         typically ``>= 70``), it standardizes verbose column names
         such as ``"Biomass BCc  (ng/m^3)"`` and
         ``"Fossil fuel BCc  (ng/m^3)"`` to shorter labels
         (``"Biomass BCc"``, ``"Fossil fuel BCc"``) and includes
         them along with ``"AAE"`` as core channels. For narrower
         exports, only the main spectral BC channels are treated as
         core.
       - Builds a new :class:`AerosolAlt` object from these core
         columns, with ``Datetime`` as the time index.
       - Optionally collects all remaining non-core columns into
         ``.extra_data`` if ``extra_data=True``.

   Theory:
       The loader does not perform any transformations on the physical
       values themselves beyond basic type conversion; it assumes the
       Aethalometer has already converted raw optical attenuation to BC
       mass concentration (BCc) in units of ``ng/m³`` using the
       manufacturer’s algorithms and calibration constants.

       The different color channels (IR, UV, blue, green, red) provide
       wavelength-resolved BC mass concentrations. Optional channels
       such as ``"Biomass BCc"`` and ``"Fossil fuel BCc"`` are derived
       quantities based on multi-wavelength absorption (e.g. via Ångström
       exponent or source apportioning models) as implemented in the
       vendor software.

   .. rubric:: Examples

   Load a single Aethalometer file

   .. code-block:: python

       import aerosoltools as at

       # Load Aethalometer data with core BC channels only
       aeth = at.Load_Aethalometer_file("data/aeth_export.csv")


