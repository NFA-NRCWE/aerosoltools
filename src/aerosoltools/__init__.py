"""
Tools for loading and analyzing aerosol instrument data.

This package provides core data structures and loader functions for
post-processing data from a variety of aerosol instruments. It is
intended for workflows involving time-resolved, size-resolved, and
instrument-specific particle measurements.

Naming convention
-----------------
All public functions are available under both their original
``PascalCase`` names (e.g. ``Load_ELPI_file``) and PEP 8-compliant
``snake_case`` aliases (e.g. ``load_elpi_file``). Both forms are
equivalent; the snake_case names are preferred for new code.

Classes
-------
Aerosol1D
    Time-resolved scalar aerosol data (e.g., total particle concentration).
Aerosol2D
    Time- and size-resolved aerosol data (e.g., particle size distributions).
AerosolAlt
    Time-resolved data for instruments reporting alternative metrics
    (e.g., black carbon mass, optical depth, or custom channels).

Loader functions
----------------
Each loader is available as both ``Load_<Instrument>_file`` and
``load_<instrument>_file``.

load_cpc_file / Load_CPC_file
    Load data from condensation particle counters (CPC, TSI).
load_devlabs_file / Load_Devlabs_file
    Load weather station data from DevLabs instrument.
load_discmini_file / Load_DiSCmini_file
    Load data from DiSCmini personal dosimeters (Testo).
load_elpi_file / Load_ELPI_file
    Load data from electrical low-pressure impactors (ELPI, Dekati).
load_fmps_file / Load_FMPS_file
    Load data from fast mobility particle sizers (FMPS, TSI).
load_fourtec_file / Load_Fourtec_file
    Load environmental logger data (e.g., temperature / relative humidity).
load_grimm_file / Load_Grimm_file
    Load data from Grimm optical particle counters.
load_ns_file / Load_NS_file
    Load data from NanoScan SMPS (NS, TSI).
load_opcn3_file / Load_OPCN3_file
    Load data from Alphasense OPC-N3 optical particle counters.
load_ops_file / Load_OPS_file
    Load data from optical particle sizers (OPS, TSI).
load_partector_file / Load_Partector_file
    Load data from Naneos Partector particle dosimeters.
load_smps_file / Load_SMPS_file
    Load data from scanning mobility particle sizers (SMPS, TSI).
load_aethalometer_file / Load_Aethalometer_file
    Load black carbon mass data from MicroAeth / aethalometers.
load_dusttrak_file / Load_DustTrak_file
    Load PM mass concentration data from DustTrak instruments.
load_data_from_folder / Load_data_from_folder
    Dispatch the appropriate loader over all files in a folder and
    return the combined dataset(s).

Utilities
---------
combine_ns_ops / Combine_NS_OPS
    Combine NanoScan and OPS measurements into a single size-resolved
    dataset with harmonized bin structure.
plot_correlation / Plot_correlation
    Plot and fit correlations between two aerosol time series (e.g.,
    instrument inter-comparisons).

Typical usage example
---------------------
    >>> import aerosoltools as at
    >>> ns = at.load_ns_file("nanoscan_example.txt")
    >>> ops = at.load_ops_file("ops_example.txt")
    >>> combined = at.combine_ns_ops(ns, ops)
"""

from .aerosol1d import Aerosol1D
from .aerosol2d import Aerosol2D
from .aerosol3d import Aerosol3d
from .aerosolalt import AerosolAlt
from .loaders import (
    Load_Aethalometer_file,
    Load_APS_file,
    Load_CPC_file,
    Load_data_from_folder,
    Load_Devlabs_file,
    Load_DiSCmini_file,
    Load_DiSCmini_raw_file,
    Load_DustTrak_file,
    Load_ELPI_file,
    Load_FMPS_file,
    Load_Fourtec_file,
    Load_Grimm_file,
    Load_NS_file,
    Load_OPCN3_file,
    Load_OPS_file,
    Load_Partector_file,
    Load_Ranger_file,
    Load_SMPS_file,
)
from .utility import (
    Combine_NS_OPS,
    Plot_correlation,
    bland_altman_analysis,
    combine_measurements,
    combine_size_ranges,
    fit_data,
)

# snake_case aliases for PEP 8 consistency
combine_ns_ops = Combine_NS_OPS
Combine_size_ranges = combine_size_ranges  # PascalCase alias
plot_correlation = Plot_correlation
Combine_measurements = combine_measurements  # PascalCase alias
load_aps_file = Load_APS_file
load_aethalometer_file = Load_Aethalometer_file
load_cpc_file = Load_CPC_file
load_devlabs_file = Load_Devlabs_file
load_discmini_file = Load_DiSCmini_file
load_discmini_raw_file = Load_DiSCmini_raw_file
load_dusttrak_file = Load_DustTrak_file
load_elpi_file = Load_ELPI_file
load_fmps_file = Load_FMPS_file
load_fourtec_file = Load_Fourtec_file
load_grimm_file = Load_Grimm_file
load_ns_file = Load_NS_file
load_opcn3_file = Load_OPCN3_file
load_ops_file = Load_OPS_file
load_partector_file = Load_Partector_file
load_ranger_file = Load_Ranger_file
load_smps_file = Load_SMPS_file
load_data_from_folder = Load_data_from_folder

__all__ = [
    # Classes
    "Aerosol1D",
    "Aerosol2D",
    "Aerosol3d",
    "AerosolAlt",
    # Utilities
    "bland_altman_analysis",
    "Combine_NS_OPS",
    "combine_size_ranges",
    "Combine_size_ranges",
    "fit_data",
    "Plot_correlation",
    "combine_ns_ops",
    "plot_correlation",
    "combine_measurements",
    "Combine_measurements",
    # Loaders (original names)
    "Load_APS_file",
    "Load_Aethalometer_file",
    "Load_CPC_file",
    "Load_Devlabs_file",
    "Load_DiSCmini_file",
    "Load_DiSCmini_raw_file",
    "Load_DustTrak_file",
    "Load_ELPI_file",
    "Load_FMPS_file",
    "Load_Fourtec_file",
    "Load_Grimm_file",
    "Load_NS_file",
    "Load_OPCN3_file",
    "Load_OPS_file",
    "Load_Partector_file",
    "Load_Ranger_file",
    "Load_SMPS_file",
    "Load_data_from_folder",
    # Loaders (snake_case aliases)
    "load_aps_file",
    "load_aethalometer_file",
    "load_cpc_file",
    "load_discmini_file",
    "load_discmini_raw_file",
    "load_dusttrak_file",
    "load_elpi_file",
    "load_fmps_file",
    "load_fourtec_file",
    "load_grimm_file",
    "load_ns_file",
    "load_opcn3_file",
    "load_ops_file",
    "load_partector_file",
    "load_ranger_file",
    "load_smps_file",
    "load_data_from_folder",
]
