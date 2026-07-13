"""Public multi-dataset (intercomparison) workflows.

Everything here operates on **two or more** independent datasets or instruments,
following the conceptual flow ``align → assess agreement → fit → apply``:

* :mod:`~aerosoltools.intercomparison.combination` — stitch/concatenate
  instruments into one dataset (:func:`combine_measurements`,
  :func:`combine_size_ranges`).
* :mod:`~aerosoltools.intercomparison.correlation` — correlation, regression
  and Bland-Altman agreement (:func:`plot_correlation`,
  :func:`bland_altman_analysis`, :func:`fit_data`).
* :mod:`~aerosoltools.intercomparison._alignment` — private temporal-alignment
  machinery shared by the above (and by calibration).

Single-dataset behaviour is **not** here — it lives on the data objects
themselves (``data.timecrop(...)``, ``data.summarize(...)``, …).
"""

from ._alignment import _align_series as _align_series  # re-exported for the GUI
from .combination import combine_measurements, combine_size_ranges
from .correlation import bland_altman_analysis, fit_data, plot_correlation

__all__ = [
    "combine_measurements",
    "combine_size_ranges",
    "plot_correlation",
    "bland_altman_analysis",
    "fit_data",
]
