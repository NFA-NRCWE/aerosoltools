
import numpy as np
import pandas as pd

from ..aerosolalt import AerosolAlt
from .support.reading import sniff

###############################################################################

def load_simple_acsm_file(file: str) -> AerosolAlt:
    """Description:
        Load a acsm dM text export and return it as an
        :class:`AerosolAlt` time series with TEM sampling metadata.

    Args:
        file (str):
            Path to the acsm ``.txt`` export file.

    Returns:
        AerosolAlt:
            acsm total VOC time series with a datetime index, TVOC
            and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            :func:`sniff`.

    Notes:
        Detailed description:
            This loader is tailored to acsm text exports that provide
            a simle timeseries of the ascibed categories SO4, Organic, NO3, NH4
            and chorine.
            
            Internally, the function:

            - Attempts to infer encoding and delimiter via
              :func:`sniff`. If delimiter detection fails.
            - Reads the main data block with :func:`numpy.genfromtxt`, starting
              at the acsm data header (``header=14``).
            - Renames core columns:

              - ``"t_base"`` → ``"Datetime"``,
              - ``"SO4_11000"`` → ``"SO4"``.

            - Converts the ``"Datetime"`` columns from str to absolute 
              timestamps by suptracting the difference in year from
              the recorded year.

            - Constructs an :class:`AerosolAlt` object using the core columns:

              - ``Datetime``,
              - ``SO4``,
              - ``Org``,
              - ``NO3``,
              - ``NH4``,
              - ``Chlorine``,

            - Populates metadata:

              - ``instrument`` set to ``"acsm"``,

              - ``unit`` set to ``"µg/m³"``,

              - ``dtype`` set to ``"dM"``,

        Theory:
            The acsm logs dM as acribed to five catagories in µg/m³ 

      Examples:
          Typical usage is to load a acsm file and look at the timeseries and
          sampling information:

          .. code-block:: python

              import aerosoltools as at

              # Load acsm VOC data
              acsm = at.Load_acsm_file("data/acsm_export.txt",
                                            extra_data=True)

              # Inspect the main time series
              print(acsm.data.head())

              # Plot VOC over time
              fig, ax = acsm.plot_total_conc()
    """

    enc, delim = sniff(file)

    # Read main data
    df = pd.read_csv(file, delimiter=delim, header=0)
    df.columns=['Datetime','SO4','Org','NO3','NH4','Chlorine']
    df["Datetime"] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M:%S")

    # Create AerosolAlt object
    acsm = AerosolAlt(df)
    acsm._meta["instrument"] = "ACSM"
    # acsm._meta["serial_number"] = [meta_lines[h] for h in meta_lines if 'IRN' in h]
    acsm._meta["unit"] = "µg/m³"
    acsm._meta["dtype"] = "dM"

    return acsm
