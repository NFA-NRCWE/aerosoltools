
import numpy as np
import pandas as pd

from ..aerosolalt import AerosolAlt
from .support.reading import sniff

###############################################################################

def load_tiger_file(file: str, year=2026, extra_data: bool = False) -> AerosolAlt:
    """Description:
        Load a Tiger LDSA text export and return it as an
        :class:`AerosolAlt` time series with TEM sampling metadata.

    Args:
        file (str):
            Path to the Tiger ``.txt`` export file.
        extra_data (bool, optional):
            If ``True``, additional columns (beyond ``LDSA``, ``TEM`` and
            ``Flow``) are stored in ``extra_data`` indexed by ``Datetime``.
            Defaults to ``False``.

    Returns:
        AerosolAlt:
            Tiger total VOC time series with a datetime index, TVOC
            and associated metadata.

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            :func:`sniff`.

    Notes:
        Detailed description:
            This loader is tailored to Tiger text exports that provide
            Total VOC measurements as a function of time.

            Internally, the function:

            - Attempts to infer encoding and delimiter via
              :func:`sniff`. If delimiter detection fails.
            - Reads the main data block with :func:`numpy.genfromtxt`, starting
              at the Tiger data header (``header=14``).
            - Renames core columns:

              - ``"Date"`` → ``"Datetime"``,
              - ``"TVOC (ppb)"`` → ``"TVOC"``.

            - Reads the header region (first 8 lines) via
              :func:`numpy.genfromtxt`.
            - Converts the ``"Date"`` and ``"Time"`` columns from date and time
              to absolute timestamps by suptracting the difference in year from
              the recorded year.

            - Constructs an :class:`AerosolAlt` object using the core columns:

              - ``Datetime``,
              - ``TVOC``,

            - Populates metadata:

              - ``instrument`` set to ``"Tiger"``,
              - ``serial_number`` from the first header line (last 3
                characters),

              - ``unit`` mapping:

                - ``"VOC"`` → ``"ppb"`` or ``"ppb"``,

            - If ``extra_data=True``, all remaining columns (not ``LDSA``,
              ``TEM``, ``Flow``) are stored in ``extra_data`` with
              ``Datetime`` as index.

        Theory:
            The Tiger logs total VOC in either ppb or ppm 

      Examples:
          Typical usage is to load a Tiger file and inspect VOC
          sampling information:

          .. code-block:: python

              import aerosoltools as at

              # Load Tiger VOC data
              tiger = at.Load_Tiger_file("data/Tiger_export.txt",
                                            extra_data=True)

              # Inspect the main time series
              print(tiger.data.head())

              # Plot VOC over time
              fig, ax = tiger.plot_total_conc()
    """

    enc, delim = sniff(file)

    # Read main data
    df = pd.read_csv(file, delimiter=delim, header=14)
    df.rename(columns={"Date":"Datetime",df.columns[2]:"TVOC"}, inplace=True)

    # # Read header metadata
    meta_lines = dict(np.genfromtxt(file, delimiter=delim, max_rows=8, dtype="str"))
    # Create new columns
    parts = df["Datetime"].str.split("-", expand=True)
    
    df["DayMonth"] = parts[0] + "-" + parts[1]
    #Corrects the year, as it is incorrectly stored in the software.
    df["Year"] = (parts[2].astype(int)-[int(parts[2][0])-year]).astype(str)
    # Convert time column to absolute datetime
    df["Datetime"] = pd.to_datetime(df["DayMonth"] + "-" + df['Year'] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S")

    # Create AerosolAlt object
    Tiger = AerosolAlt(df[["Datetime", "TVOC"]])
    Tiger._meta["instrument"] = "Tiger"
    Tiger._meta["serial_number"] = [meta_lines[h] for h in meta_lines if 'IRN' in h]
    Tiger._meta["unit"] = meta_lines['Units']
    Tiger._meta["dtype"] = "TVOC"

    return Tiger
