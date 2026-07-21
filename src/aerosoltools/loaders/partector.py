import datetime

import numpy as np
import pandas as pd
from matplotlib.dates import date2num

from ..partector import Partector
from .support.construct import build_1d
from .support.exceptions import TimestampError
from .support.reading import sniff
from .support.units import resolve_extra_columns

###############################################################################


def _tem_sampling_periods(tem_flag, datetimes) -> list[tuple]:
    """Return a ``(start, end)`` timestamp pair for each TEM-sampling run.

    The Partector's ``TEM`` channel is 1 while the sampler is actively
    collecting particles onto its TEM grid. This finds every *contiguous* run of
    ``TEM == 1`` (a distinct grid-sampling period), so multiple samples in one
    recording are separated rather than merged into a single span.

    Args:
        tem_flag: The ``TEM`` column (0/1, possibly with NaN).
        datetimes: The matching ``Datetime`` column (same positional order).

    Returns:
        A list of ``(start, end)`` :class:`pandas.Timestamp` pairs, one per run
        (empty when the sampler never ran).
    """
    flag = (tem_flag == 1).to_numpy()
    if not flag.any():
        return []
    # Rising/falling edges of the padded 0/1 signal bound each run (inclusive).
    edges = np.diff(np.concatenate([[0], flag.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    dt = np.asarray(datetimes)
    return [(pd.Timestamp(dt[s]), pd.Timestamp(dt[e])) for s, e in zip(starts, ends)]


def load_partector_file(
    file: str, extra_data: bool = True, tem_activities: bool = True
) -> Partector:
    """Load a Partector LDSA text export and return it as a
    :class:`Partector` time series with TEM sampling metadata.

    Args:
        file (str):
            Path to the Partector ``.txt`` export file.
        extra_data (bool, optional):
            If ``True``, additional columns (beyond ``LDSA``, ``TEM`` and
            ``Flow``) are stored in ``extra_data`` indexed by ``Datetime`` (the
            ambient T/RH channels get canonical units). Defaults to ``True`` —
            everything in the file is kept.
        tem_activities (bool, optional):
            If ``True``, each contiguous ``TEM == 1`` grid-sampling run is
            registered as an activity ("TEM grid sampling 1", 2, …). Defaults to
            ``True``.

    Returns:
        Partector:
            Partector LDSA time series with a datetime index, LDSA, TEM flag,
            flow, and associated metadata (including a TEM sampling summary).

    Raises:
        FileNotFoundError:
            If ``file`` does not exist or cannot be opened.
        UnicodeDecodeError:
            If the file cannot be decoded using the encodings tried by
            ``_detect_delimiter``.
        ValueError:
            If the start datetime cannot be parsed from the header, or if
            there are too few TEM sampling points to estimate a sampling
            volume.

    Notes:
        Detailed description:
            This loader is tailored to Partector text exports that provide
            lung-deposited surface area (LDSA), a TEM sampling flag, and flow
            information as a function of time.

            Internally, the function:

            - Attempts to infer encoding and delimiter via
              ``_detect_delimiter``. If delimiter detection fails, it
              falls back to tab (``"\\t"``).
            - Reads the main data block with :func:`pandas.read_csv`, starting
              at the Partector data header (``header=10``).
            - Renames core columns:

              - ``"time"`` → ``"Datetime"`` (elapsed seconds),
              - ``"flow"`` → ``"Flow"``.

            - Reads the header region (first 10 lines) via
              :func:`numpy.genfromtxt` and extracts the measurement start
              datetime from the line containing ``"Start: "`` using the format
              ``"%d.%m.%Y %H:%M:%S"``.
            - Converts the ``"Datetime"`` column from elapsed seconds to
              absolute timestamps by adding the parsed start time.
            - Uses the ``"TEM"`` column (1 during TEM sampling, 0 otherwise) to
              identify the TEM sampling interval:

              - selects all rows where ``TEM == 1``,
              - takes the first and last such timestamps as sampling start/end,
              - raises a ``ValueError`` if fewer than two TEM points are found.

            - Computes the average flow during the TEM sampling period as
              ``Flow`` (L/min) × 1000 → mL/min.
            - Estimates the sampling duration in minutes from the difference
              between the start and end timestamps (using
              :func:`matplotlib.dates.date2num`), and calculates an approximate
              TEM sampling volume in mL as:

              .. code-block:: python

                  duration_min = (tem_end - tem_start) in minutes
                  sample_volume_ml = avg_flow_ml_min * duration_min

            - Packages this information into a small ``TEM_samples`` table with
              columns ``"Start"``, ``"End"`` and ``"Sample_vol [ml]"`` and
              stores it in ``metadata["TEM_samples"]``.
            - Constructs a :class:`Partector` object using the core columns:

              - ``Datetime``,
              - ``LDSA``,
              - ``TEM``,
              - ``Flow``.

            - Populates metadata:

              - ``instrument`` set to ``"Partector"``,
              - ``serial_number`` from the first header line (last 3
                characters),
              - ``TEM_samples`` as described above.
              - ``unit`` mapping:

                - ``"LDSA"`` → ``"nm$^{2}$/cm$^{3}$"``,
                - ``"TEM"`` → ``"bool"``,
                - ``"Flow"`` → ``"l/min"``,

            - If ``extra_data=True``, all remaining columns (not ``LDSA``,
              ``TEM``, ``Flow``) are stored in ``extra_data`` with
              ``Datetime`` as index.

        Theory:
            The Partector logs LDSA and a binary TEM sampling flag as a
            function of time. To estimate the TEM sampling volume, the loader
            assumes:

            - the TEM filter is sampling whenever ``TEM == 1``,
            - the effective sampling interval is from the first to the last
              timestamp where ``TEM == 1``,
            - the flow is reasonably constant over that interval so that the
              mean flow during ``TEM == 1`` can be used.

            The estimated TEM sampling volume :math:`V_\\mathrm{TEM}` in mL is
            then given by:

            .. math::

                V_\\mathrm{TEM} \\approx \\bar{Q} \\cdot \\Delta t,

            where :math:`\\bar{Q}` is the mean flow (mL/min) during
            :math:`TEM = 1` and :math:`\\Delta t` is the sampling duration in
            minutes between the first and last TEM-flagged timestamps.

      Examples:
          Typical usage is to load a Partector file and inspect LDSA and TEM
          sampling information:

          .. code-block:: python

              import aerosoltools as at

              # Load Partector LDSA and TEM information
              par = at.load_partector_file("data/Partector_export.txt",
                                            extra_data=True)

              # Inspect the main time series
              print(par.data.head())

              # Inspect the estimated TEM sampling volume(s)
              print(par.metadata["TEM_samples"])

              # Plot LDSA over time
              fig, ax = par.plot_total_conc(parameter="LDSA")
    """

    try:
        encoding, delimiter = sniff(file, sample_lines=30)
    except Exception:
        delimiter = "\t"

    # Read main data
    df = pd.read_csv(file, delimiter=delimiter, header=10)
    df.rename(columns={"time": "Datetime", "flow": "Flow"}, inplace=True)

    # Read header metadata
    meta_lines = np.genfromtxt(file, delimiter=delimiter, max_rows=10, dtype="str")
    try:
        start_str = meta_lines[4].split("Start: ")[1].split("\n")[0]
        start_time = datetime.datetime.strptime(start_str, "%d.%m.%Y %H:%M:%S")
    except Exception as e:
        raise TimestampError(f"Unable to parse start datetime from metadata: {e}")

    # Convert time column to absolute datetime
    df["Datetime"] = pd.to_timedelta(df["Datetime"], unit="s") + start_time

    # Sample duration estimate from TEM flag
    is_templog = df["TEM"] == 1
    if is_templog.sum() < 2:
        raise ValueError("Not enough TEM sampling points to estimate volume.")

    if is_templog.sum() > 0 and is_templog.sum() != len(is_templog):
        if (df["TEM"].diff() == 1).sum() > 1:
            print(f"⚠️ Warning: More than one TEM sampling period detected in {file}")

    tem_times = df.loc[is_templog, "Datetime"]
    tem_start, tem_end = tem_times.iloc[0], tem_times.iloc[-1]
    avg_flow_ml_min = df.loc[is_templog, "Flow"].mean() * 1000  # l/min → ml/min

    duration_min = (date2num(tem_end) - date2num(tem_start)) * 24 * 60
    sample_volume_ml = avg_flow_ml_min * duration_min

    # Package sample info
    sample_meta = {
        "Start": [tem_start],
        "End": [tem_end],
        "Sample_vol [ml]": [sample_volume_ml],
    }

    # Create Partector object
    Par = build_1d(
        df[["Datetime", "LDSA", "TEM", "Flow"]],
        cls=Partector,
        instrument="Partector",
        serial_number=meta_lines[0][-3:],
        unit={"LDSA": "nm$^{2}$/cm$^{3}$", "TEM": "bool", "Flow": "l/min"},
        extra_meta={"TEM_samples": pd.DataFrame(sample_meta)},
    )

    # Attach extra data. The Partector writes bare headers with no in-header
    # units; curate the ambient T/RH channels (whose units are fixed by the
    # instrument) so they resolve too. The diagnostic currents/voltages are kept
    # but left unitless rather than guess a unit.
    if extra_data:
        extra_df = df.drop(columns=["LDSA", "TEM", "Flow"]).set_index("Datetime")
        extra_df, column_units = resolve_extra_columns(
            extra_df, curated={"T": "°C", "RH": "%"}
        )
        Par._extra_data = extra_df
        Par._meta["column_units"] = column_units

    # Register each TEM grid-sampling run as its own activity so users get the
    # sampling periods directly, without re-extracting them by hand.
    if tem_activities:
        periods = _tem_sampling_periods(df["TEM"], df["Datetime"])
        if periods:
            Par.mark_activities(
                {f"TEM grid sampling {i + 1}": [p] for i, p in enumerate(periods)},
                mode="replace",
            )

    return Par
