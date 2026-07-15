"""Loader for TSI Aerodynamic Particle Sizer (APS 3321) AIM exports."""

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from ..aerosol3d import Aerosol3d
from .support.construct import build_2d
from .support.exceptions import FileFormatError
from .support.reading import read_table, sniff

###############################################################################

# APS aerosol (sample) flow in cm³/min — used to turn "Raw Counts" exports into
# number concentration (dN/cm³ = counts / (flow · sample_time)). Verified
# against the instrument's own reported Total Conc. on the example files.
_APS_SAMPLE_FLOW_CM3_MIN = 1000.0
_AERO_SLICE = slice(4, 56)  # the 52 aerodynamic size-bin columns

# Light-scattering (side-scatter) particle range in nm. Per the APS 3321 manual,
# the optical signal spans 0.5–20 µm. The optical "channels" are equal bins of
# the raw side-scatter *intensity* (their number is the SCR setting, e.g. 16),
# NOT calibrated diameters — the instrument only calibrates the aerodynamic axis
# (SCA). We therefore assign a nominal log-spaced optical diameter over this
# range so the two axes can be compared; it is not a scatter→size calibration.
_APS_OPTICAL_RANGE_NM = (500.0, 20000.0)


def _aps_header(file, delimiter, encoding) -> dict:
    """Parse the 6-line APS AIM header (sample time, density, channel bounds)."""
    meta = {}
    with open(file, "r", encoding=encoding, errors="replace") as f:
        for _ in range(6):
            key, _sep, value = f.readline().rstrip("\n").partition(delimiter)
            meta[key.strip()] = value.strip()
    return meta


def _aps_bins_nm(headers, lower_um, upper_um):
    """Aerodynamic bin midpoints and edges (nm) from the AIM column headers.

    ``headers`` are the 52 size-column labels: a "<x" catch-all followed by 51
    numeric midpoints in µm. Edges are arithmetic midpoints between bin centres,
    closed with the instrument's lower/upper channel bounds.
    """
    mids_um = np.array(headers[1:], dtype=float)  # 51 numeric midpoints
    lowest = (mids_um[0] - lower_um) / 2.0 + lower_um  # centre of the catch-all
    mids_um = np.append(lowest, mids_um)
    inner = (mids_um[1:] - mids_um[:-1]) / 2.0 + mids_um[:-1]
    edges_um = np.concatenate([[lower_um], inner, [upper_um]])
    return np.round(mids_um * 1000.0, 1), np.round(edges_um * 1000.0, 1)


def load_aps_file(file: str) -> Aerosol2D:
    """Load a TSI APS 3321 AIM text export and return the size distribution.

        The APS sizes each particle both **aerodynamically** (time of flight)
        and, in *correlated* mode, **optically** (side scatter). This loader
        returns an :class:`~aerosoltools.aerosol3d.Aerosol3d` carrying both axes
        when the file is correlated, and a plain
        :class:`~aerosoltools.aerosol2d.Aerosol2D` (aerodynamic) when the APS was
        run in aerodynamic-only mode. The delimiter (comma or tab — an export
        choice) is sniffed automatically.

    Args:
        file (str): Path to the APS ``.txt`` export.

    Returns:
        Aerosol2D | Aerosol3d: The aerodynamic distribution (dN, cm⁻³). A
        correlated file additionally carries the optical distribution
        (``.optical``) and the optical×aerodynamic matrix (``.correlation``),
        for :meth:`~aerosoltools.aerosol3d.Aerosol3d.plot_aero_vs_optical`.

    Raises:
        ValueError: If the file does not look like an APS AIM export.

    Notes:
        Units: exports weighted as "Raw Counts" are converted to number
        concentration with the APS 1 L/min aerosol flow and the header's sample
        time (verified against the instrument's reported Total Conc.);
        concentration-weighted (correlated) exports are used as-is.

        Optical axis: per the APS 3321 manual the optical "channels" are equal
        bins of the raw side-scatter *intensity* (their count set by the SCR
        command — 16 in these files), and the instrument calibrates only the
        aerodynamic axis (SCA), not scatter→size. The optical channels are
        therefore given a nominal log-spaced diameter over the manual's
        0.5–20 µm light-scattering range (``optical_nominal`` flag; the raw
        1–N channel indices are kept in ``optical_channel_index``).

    Examples:
        .. code-block:: python

            import aerosoltools as at

            aps = at.load_aps_file("KOBOT correlated AIM1.txt")
            aps.plot_psd()                 # aerodynamic PSD (inherited 2D API)
            if aps.is_correlated:
                aps.plot_aero_vs_optical()  # optical vs aerodynamic comparison
    """
    encoding, delimiter = sniff(file)
    meta = _aps_header(file, delimiter, encoding)
    try:
        sample_time = float(meta.get("Sample Time", 1.0))
        lower_um = float(meta["Lower Channel Bound"])
        upper_um = float(meta["Upper Channel Bound"])
    except (KeyError, ValueError) as exc:
        raise FileFormatError("File does not look like an APS AIM export.") from exc

    df = read_table(file, delimiter=delimiter, encoding=encoding, header=6)
    if df.shape[1] < 56:
        raise FileFormatError("APS export has fewer size columns than expected.")

    headers = [str(c) for c in df.columns[_AERO_SLICE]]
    bin_mids, bin_edges = _aps_bins_nm(headers, lower_um, upper_um)

    weight = str(df.columns[3]).strip().lower()
    is_correlated = "correlat" in weight
    aero_block = df.iloc[:, _AERO_SLICE].apply(pd.to_numeric, errors="coerce")

    # Counts -> concentration for "Raw Counts" exports; correlated exports are
    # already in concentration.
    col3_vals = df.iloc[:, 3].astype(str).str.lower()
    raw_counts = col3_vals.str.contains("raw").any()
    if raw_counts:
        volume_cm3 = _APS_SAMPLE_FLOW_CM3_MIN * sample_time / 60.0
        aero_block = aero_block / volume_cm3

    dt = pd.to_datetime(
        df["Date"].ffill().astype(str) + " " + df["Start Time"].ffill().astype(str),
        format="%m/%d/%y %H:%M:%S",
        errors="coerce",
    )

    if not is_correlated:
        aero_block.columns = [str(m) for m in bin_mids]
        aero_block.insert(0, "Total_conc", aero_block.sum(axis=1))
        aero_block.insert(0, "Datetime", dt.to_numpy())
        aero_block = aero_block.dropna(subset=["Datetime"])
        return _build_aero_2d(aero_block, bin_mids, bin_edges, meta, Aerosol2D)

    # -- correlated: 16 optical channels per sample -> reshape ---------------
    n_opt = int(pd.to_numeric(df.iloc[:, 3], errors="coerce").max())
    n_rows = (len(df) // n_opt) * n_opt
    aero_arr = aero_block.to_numpy()[:n_rows].reshape(-1, n_opt, aero_block.shape[1])
    times = dt.to_numpy()[:n_rows][::n_opt]

    aero_time_bin = aero_arr.sum(axis=1)  # (n_samples, n_aero): sum over optical
    opt_time_chan = aero_arr.sum(axis=2)  # (n_samples, n_optical): sum over aero

    opt_mids, opt_edges = _optical_bins_nm(n_opt)

    aero_df = pd.DataFrame(aero_time_bin, columns=[str(m) for m in bin_mids])
    aero_df.insert(0, "Total_conc", aero_df.sum(axis=1))
    aero_df.insert(0, "Datetime", times)
    aero_df = aero_df.dropna(subset=["Datetime"])

    opt_df = pd.DataFrame(opt_time_chan, columns=[str(m) for m in opt_mids])
    opt_df.insert(0, "Total_conc", opt_df.sum(axis=1))
    opt_df.insert(0, "Datetime", times)
    opt_df = opt_df.dropna(subset=["Datetime"])
    optical = _build_aero_2d(opt_df, opt_mids, opt_edges, meta, Aerosol2D)
    optical._meta["instrument"] = "APS (optical)"
    optical._meta["optical_nominal"] = True
    optical._meta["optical_channel_index"] = list(range(1, n_opt + 1))

    # Correlation matrix: time × (optical_mid, aero_mid).
    cols = pd.MultiIndex.from_product([opt_mids, bin_mids], names=["optical", "aero"])
    corr = pd.DataFrame(
        aero_arr.reshape(aero_arr.shape[0], -1),
        index=pd.DatetimeIndex(times),
        columns=cols,
    ).dropna(how="all")

    aps = _build_aero_2d(aero_df, bin_mids, bin_edges, meta, Aerosol3d)
    aps._optical = optical
    aps._correlation = corr
    return aps


def _optical_bins_nm(n_opt):
    """Nominal log-spaced optical bin midpoints/edges over the scattering range.

    The APS reports only side-scatter *channel indices* (their number set by the
    SCR command), not diameters, so these are a nominal mapping over the
    manual's 0.5–20 µm light-scattering range — see ``_APS_OPTICAL_RANGE_NM``.
    """
    lo, hi = _APS_OPTICAL_RANGE_NM
    edges = np.round(np.geomspace(lo, hi, n_opt + 1), 1)
    mids = np.round(np.sqrt(edges[1:] * edges[:-1]), 1)
    return mids, edges


def _build_aero_2d(frame, bin_mids, bin_edges, meta, cls):
    """Construct an Aerosol2D/3d from a prepared (Datetime, Total_conc, bins) df.

    The frame is already plain number (dN, cm⁻³), so no post-construction
    conversion is applied. The APS reports aerodynamic diameter (time-of-flight),
    which is density-independent — so, unlike the ELPI, changing density does NOT
    rescale the size axis; it only affects mass conversion. The AIM text export
    carries no instrument serial number (only the sample-file path), so the
    serial is left blank and the sample name kept as its own metadata field;
    the "Stokes Correction" export flag is kept for transparency.
    """
    sample_file = str(meta.get("Sample File", "")).replace("/", "\\").split("\\")[-1]
    return build_2d(
        frame.copy(),
        cls=cls,
        bin_edges=np.asarray(bin_edges, dtype=float),
        bin_mids=np.asarray(bin_mids, dtype=float),
        instrument="APS",
        serial_number="",
        unit="cm⁻³",
        dtype="dN",
        density=float(meta.get("Density", 1.0) or 1.0),
        extra_meta={
            "sample_file": sample_file,
            "stokes_correction": str(meta.get("Stokes Correction", "")).strip(),
            "sample_time_s": float(meta.get("Sample Time", 0) or 0),
        },
        to_number=False,
        unnormalize=False,
    )
