Welcome to aerosoltools's documentation!
========================================

**aerosoltools** is a Python package developed by NFA for working with aerosol data from various aerosol instruments.  
It provides a unified interface for:

* Loading and parsing raw measurement files (ELPI, OPS, CPC, DiSCmini, etc.)
* Structuring data into 1D or 2D time-resolved formats
* Performing unit conversions (number, volume, mass, surface) and normalization (dx/dlogDp)
* Applying transformations like cropping, shifting, smoothing, and rebinning with respect to time
* Indexing the datasets into time specific segments, corresponding to tasks or processes
* Quick statistical summary of PNC, PM levels etc. of the entire dataset and the defined time segments 
* Visualizing particle size distributions and time series

The package is designed to support both research analysis and instrument post-processing workflows.

Installation
------------

You can install the package using `pip`:

.. code-block:: bash

   pip install aerosoltools

Quick Start Example
-------------------

Here's how to load ELPI data and plot it:

.. code-block:: python

   import aerosoltools as at

   data = at.Load_ELPI_file("elpi_data.txt")
   data.plot_timeseries()

API Reference
-------------

To dive into the functionalities of aerosoltools, look through the documenation below, or visit
the examples.

.. toctree::
   :maxdepth: 1
   :caption: Aerosol Classes

   api/aerosol1d
   api/aerosol2d
   api/aerosolalt

.. toctree::
   :maxdepth: 1
   :caption: Loaders

   api/loaders

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples