aerosoltools.loaders
====================

.. py:module:: aerosoltools.loaders

.. autoapi-nested-parse::

   Loaders for instrument-specific aerosol data files.

   This subpackage provides a collection of functions for reading and parsing
   raw exports from common aerosol instruments. Each loader normalizes the
   instrument-specific file format into one of the core classes in
   :mod:`aerosoltools` (:class:`Aerosol1D`, :class:`Aerosol2D`,
   or :class:`AerosolAlt`).

   Supported instruments include:

   * Aethalometer (Magee Scientific)
   * CPC – Condensation Particle Counter (TSI)
   * DiSCmini – Electrostatic dosimeter (Testo)
   * DustTrak – Optical particle counter (TSI)
   * ELPI – Electric Low Pressure Impactor (Dekati)
   * FMPS – Fast Mobility Particle Sizer (TSI)
   * Fourtec – Bluefish temperature / RH loggers
   * Grimm – Optical particle counters (Grimm Aerosol)
   * NS – NanoScan SMPS (TSI)
   * OPC-N3 – Optical particle counter (Alphasense)
   * OPS – Optical Particle Sizer (TSI)
   * Partector – Partector / PartectorTEM (Naneos)
   * SMPS – Scanning Mobility Particle Sizer (TSI)

   In addition, :func:`Load_data_from_folder` provides a convenience wrapper
   for batch-loading and concatenating multiple compatible files from a
   directory into a single aerosol object.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/aerosoltools/loaders/Aethalometer/index
   /api/aerosoltools/loaders/CPC/index
   /api/aerosoltools/loaders/Common/index
   /api/aerosoltools/loaders/Discmini/index
   /api/aerosoltools/loaders/DustTrak/index
   /api/aerosoltools/loaders/ELPI/index
   /api/aerosoltools/loaders/FMPS/index
   /api/aerosoltools/loaders/Fourtec/index
   /api/aerosoltools/loaders/Grimm/index
   /api/aerosoltools/loaders/NS/index
   /api/aerosoltools/loaders/OPCN3/index
   /api/aerosoltools/loaders/OPS/index
   /api/aerosoltools/loaders/Partector/index
   /api/aerosoltools/loaders/SMPS/index


