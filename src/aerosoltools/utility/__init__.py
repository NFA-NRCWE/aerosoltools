"""Utility functions for combining and comparing aerosol datasets.

Re-exports the public helpers from the topic submodules so the historical
``aerosoltools.utility`` import path is unchanged:

* :mod:`~aerosoltools.utility.combine` - combine instruments into one dataset.
* :mod:`~aerosoltools.utility.correlation` - correlation / Bland-Altman analysis.
"""

from .combine import Combine_NS_OPS, combine_measurements
from .correlation import (
    Plot_correlation,
    bland_altman_analysis,
    fit_data,
)

# Re-exported for the GUI; the `as` self-alias marks it an intentional re-export.
from .correlation import _align_series as _align_series

__all__ = [
    "Combine_NS_OPS",
    "fit_data",
    "Plot_correlation",
    "bland_altman_analysis",
    "combine_measurements",
]
