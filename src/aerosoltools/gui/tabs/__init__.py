"""Data-tab widgets for the aerosoltools GUI.

Each tab is a self-contained widget that reads the current aerosol object (or,
for the comparison tabs, the whole project) from the parent ``MainWindow`` and
renders on :meth:`refresh`. This package re-exports every tab so callers can keep
importing ``from aerosoltools.gui.tabs import SummaryTab`` etc.
"""

from __future__ import annotations

from .aero_optical import AeroOpticalTab
from .correlation import CorrelationTab
from .decay import DecayTab
from .heatmap import HeatmapTab
from .metadata import MetadataTab
from .overlay import OverlayTab
from .pmbands import PMBandsTab
from .psd_compare import ComparisonPSDTab
from .psd_single import SinglePSDTab
from .raw import RawDataTab
from .summary import SummaryTab
from .timeseries import ActivityEditorDialog, TimeSeriesTab

__all__ = [
    "AeroOpticalTab",
    "ComparisonPSDTab",
    "CorrelationTab",
    "DecayTab",
    "HeatmapTab",
    "MetadataTab",
    "OverlayTab",
    "PMBandsTab",
    "RawDataTab",
    "SinglePSDTab",
    "SummaryTab",
    "TimeSeriesTab",
    "ActivityEditorDialog",
]
