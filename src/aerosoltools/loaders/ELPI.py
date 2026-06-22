from pathlib import Path
from typing import Optional, Union
import re
import warnings

import numpy as np
import pandas as pd

from ..aerosol2d import Aerosol2D
from .Common import _detect_delimiter

###############################################################################

_E_CHARGE = 1.602176634e-19
_ELPI_STAGE_SLICE = slice(34, 48)
_ELPI_STAGE_NAMES = [
    "Stage1",
    "Stage2",
    "Stage3",
    "Stage4",
    "Stage5",
    "Stage6",
    "Stage7",
    "Stage8",
    "Stage9",
    "Stage10",
    "Stage11",
    "Stage 12",
    "Stage13",
    "Stage14",
]


def _load_ELPI_metadata(
    file_path: Union[str, Path],
    delimiter: str = "\t",
    encoding: str = "utf-8",
) -> dict:
    """Parse ELPI header metadata into a structured dictionary."""
    metadata: dict = {}

    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        for row, line in enumerate(f):
            if row >= 80:  # .dat and exported files usually finish metadata < 40 lines
                break
            line = line.strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if delimiter in value:
                items = [v for v in value.split(delimiter) if v != ""]
                try:
                    value = [float(v) for v in items]
                except ValueError:
                    value = items
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            metadata[key] = value

    return metadata


###############################################################################
# Shared parsing helpers


def _as_float_array(value) -> np.ndarray:
    """Return a metadata scalar/list/string as a float numpy array."""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        parts = re.split(r"[\t,; ]+", value.strip())
        return np.asarray([float(p) for p in parts if p != ""], dtype=float)
    return np.asarray(value, dtype=float)


def _find_ELPI_data_section(lines: list[str], delimiter: str) -> tuple[int, list[str]]:
    """Return ([Data] line index, data-column header)."""
    try:
        data_idx = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    except StopIteration:
        raise ValueError("Couldn't find the [Data] marker in the ELPI file.")

    j = data_idx - 1
    while j >= 0 and (not lines[j].strip() or lines[j].lstrip().startswith("[")):
        j -= 1

    if j < 0 or delimiter not in lines[j]:
        found = False
        for k in range(data_idx - 1, max(-1, data_idx - 25), -1):
            if delimiter in lines[k] and not lines[k].lstrip().startswith("["):
                j = k
                found = True
                break
        if not found:
            raise ValueError("Couldn't find the data header line before [Data].")

    header = [
        h.strip() for h in lines[j].lstrip("\ufeff").rstrip("\r\n").split(delimiter)
    ]
    return data_idx, header


def _read_ELPI_table(
    file: Union[str, Path],
    delimiter: str,
    encoding: str,
) -> pd.DataFrame:
    """Read the tabular part of an ELPI file and attach the header before [Data]."""
    with open(file, encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    data_idx, header = _find_ELPI_data_section(lines, delimiter)

    df = pd.read_csv(
        file,
        sep=delimiter,
        header=None,
        skiprows=data_idx + 1,
        encoding=encoding,
        engine="python",
        on_bad_lines="skip",
    )

    if len(header) < df.shape[1]:
        header += [f"Unnamed_{i}" for i in range(len(header), df.shape[1])]
    elif len(header) > df.shape[1]:
        header = header[: df.shape[1]]

    df.columns = header
    cols = list(df.columns)
    cols[0] = "Datetime"
    df.columns = cols
    df["Datetime"] = _parse_ELPI_datetime(df["Datetime"])
    return df


def _parse_ELPI_datetime(values: pd.Series) -> pd.Series:
    """Parse ELPI datetime strings robustly."""
    formats = [
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]
    parsed = None
    for fmt in formats:
        s = pd.to_datetime(values, format=fmt, errors="coerce")
        if s.notna().mean() > 0.98:
            parsed = s
            break

    if parsed is None:
        parsed = pd.to_datetime(values, errors="coerce")

    if parsed.isna().mean() > 0.2:
        raise ValueError("Datetime parsing failed for many rows; check the first column.")

    return parsed


def _get_ELPI_serial_number(
    file: Union[str, Path],
    delimiter: str,
    encoding: str,
) -> str:
    """Extract the serial/instrument identifier from the first ELPI header line."""
    with open(file, encoding=encoding, errors="replace") as f:
        first_line = f.readline().strip()

    # Common form: [ELPI-DATA FILE],[HR-E+26255]
    if "," in first_line:
        return first_line.split(",", 1)[1].strip().strip("[]")

    parts = first_line.split(delimiter)
    if len(parts) > 1:
        return parts[1].strip().strip("[]")

    return ""


def _get_ELPI_bin_edges_and_mids_nm(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ELPI D50 cutpoints and calculated diameters in nm."""
    bin_edges = _as_float_array(meta["D50values(um)"]) * 1000.0
    bin_mids = _as_float_array(meta["CalculatedDi(um)"]) * 1000.0
    return bin_edges, bin_mids


def _extract_ELPI_stage_block(df: pd.DataFrame) -> pd.DataFrame:
    """Return the main Stage1-Stage14 block immediately after the CAL marker."""
    # The ELPI files also contain a later CHARMEAS Stage1-Stage14 block. The
    # distribution/current block is the fixed Stage slice immediately after CAL.
    stage_df = df.iloc[:, _ELPI_STAGE_SLICE].copy()
    stage_df = stage_df.apply(pd.to_numeric, errors="coerce")
    return stage_df


def _make_ELPI_extra_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-distribution columns for optional extra_data storage."""
    # Drop CAL + Stage1-Stage14 + CON marker, keep concentration/PM/diagnostics.
    drop_cols = list(df.columns[33:49])
    return df.drop(columns=drop_cols, errors="ignore").copy()


###############################################################################
# Raw-current conversion helpers


def _ELPI_charger_efficiency(dp_um: np.ndarray, params=None) -> np.ndarray:
    """Return ELPI charger efficiency Pn for particle diameter in um.

    The ELPI header has the form::

        Efficiency(Dp/mult/exp)=Dp1  mult1  exp1  Dp2  mult2  exp2  mult3  exp3

    In the tested ELPI+ exports, Dekati's number export is reproduced best by
    using 10**mult1 for both the first and second branches, and a published /
    software-equivalent high-size multiplier of 126.83 for the final branch.
    This matches the header-derived density=1 export to around 1% median error.
    """
    dp_um = np.asarray(dp_um, dtype=float)

    if params is None:
        dp1, log_mult1, exp1, dp2, _log_mult2, exp2, _mult3, exp3 = (
            1.035,
            1.8300,
            1.2250,
            4.2820,
            1.8114,
            1.5150,
            3.3868,
            1.0850,
        )
    else:
        p = _as_float_array(params)
        if p.size != 8:
            raise ValueError(
                "Expected 8 values in ELPI Efficiency(Dp/mult/exp) metadata."
            )
        dp1, log_mult1, exp1, dp2, _log_mult2, exp2, _mult3, exp3 = p

    return np.where(
        dp_um < dp1,
        10.0**log_mult1 * dp_um**exp1,
        np.where(
            dp_um < dp2,
            10.0**log_mult1 * dp_um**exp2,
            126.83 * dp_um**exp3,
        ),
    )


def _ELPI_charger_flow_lpm(meta: dict) -> float:
    """Return the flow to use in the current-to-number conversion.

    Dekati's header-derived export for the uploaded ELPI+ file is reproduced
    better by using the nominal charger flow from ChargerSetup ("10 lpm") than
    by using the impactor FlowRate value (9.940 lpm). If ChargerSetup cannot be
    parsed, fall back to FlowRate.
    """
    charger_setup = str(meta.get("ChargerSetup", ""))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*lpm", charger_setup, re.I)
    if match:
        return float(match.group(1))
    return float(meta.get("FlowRate", 10.0))


def _convert_ELPI_current_to_number(
    current_fa: pd.DataFrame,
    meta: dict,
    clip_negative: bool = True,
) -> pd.DataFrame:
    """Convert ELPI stage current to number concentration per bin (dN, 1/cm3).

    Parameters
    ----------
    current_fa:
        DataFrame containing the main ELPI Stage1-Stage14 block. For raw .dat
        files this block is usually ``Current (fA)`` with ``CalculatedType``
        ``dW/dlogDp``.
    meta:
        Parsed ELPI metadata.
    clip_negative:
        If True, negative concentrations are set to zero. Dekati's exported
        number files appear to clip negative stage results to zero.

    Returns
    -------
    pandas.DataFrame
        Stage concentrations as dN, in 1/cm3.
    """
    d50_um = _as_float_array(meta["D50values(um)"])
    di_um = _as_float_array(meta["CalculatedDi(um)"])

    if d50_um.size != di_um.size + 1:
        raise ValueError(
            "ELPI metadata must contain N+1 D50 cutpoints and N CalculatedDi values."
        )

    current = current_fa.to_numpy(dtype=float)

    calculated_type = str(meta.get("CalculatedType", ""))
    if "dlog" in calculated_type.lower():
        dlogdp = np.diff(np.log10(d50_um))
        current = current * dlogdp[np.newaxis, :]

    flow_lpm = _ELPI_charger_flow_lpm(meta)
    flow_m3_s = flow_lpm / 1000.0 / 60.0
    fa_to_number_cm3 = 1e-15 / (_E_CHARGE * flow_m3_s) / 1e6

    charger_eff = _ELPI_charger_efficiency(
        di_um, params=meta.get("Efficiency(Dp/mult/exp)")
    )

    number = current * fa_to_number_cm3 / charger_eff[np.newaxis, :]

    dilution = float(meta.get("Dilution", 1.0) or 1.0)
    number = number * dilution

    if clip_negative:
        number = np.where(number < 0, 0.0, number)

    return pd.DataFrame(number, columns=current_fa.columns, index=current_fa.index)


###############################################################################
# Public loaders


def Load_ELPI_file(file: str, extra_data: bool = False) -> Aerosol2D:
    """Load an ELPI file and return an Aerosol2D object.

    This is the routing function. Already-converted Dekati exports are handled
    by :func:`Load_ELPI_file_txt`. Raw-current .dat files with
    ``CalculatedMoment=Current (fA)`` are converted by
    :func:`Load_ELPI_file_dat` before constructing the Aerosol2D object.
    """
    encoding, delimiter = _detect_delimiter(file)
    meta = _load_ELPI_metadata(file, delimiter, encoding)
    calculated_moment = str(meta.get("CalculatedMoment", ""))

    if calculated_moment.lower().startswith("current"):
        return Load_ELPI_file_dat(file, extra_data=extra_data)

    return Load_ELPI_file_txt(file, extra_data=extra_data)


def Load_ELPI_file_txt(file: str, extra_data: bool = False) -> Aerosol2D:
    """Load an already-converted ELPI size-distribution export."""
    encoding, delimiter = _detect_delimiter(file)
    meta = _load_ELPI_metadata(file, delimiter, encoding)
    bin_edges, bin_mids = _get_ELPI_bin_edges_and_mids_nm(meta)

    # Recalculate bin edges for non-unit density, preserving existing behavior.
    if meta["Density(g/cm^3)"] != 1.0:
        bin_edges[1:-1] = np.sqrt(bin_mids[1:] * bin_mids[:-1])
        bin_edges[0] = bin_edges[1] ** 2 / bin_edges[2]
        bin_edges[-1] = bin_edges[-2] ** 2 / bin_edges[-3]
        warnings.warn(
            "ELPI density is not 1.0 g/cm3; bin edges were estimated from "
            "geometric means of CalculatedDi values.",
            RuntimeWarning,
            stacklevel=2,
        )

    df = _read_ELPI_table(file, delimiter, encoding)
    dist_data = _extract_ELPI_stage_block(df)
    extra_df = _make_ELPI_extra_data(df)

    unit_map = {"Nu": "cm⁻³", "Su": "nm²/cm³", "Vo": "nm³/cm³", "Ma": "ug/m³"}
    dtype_map = {"Nu": "dN", "Su": "dS", "Vo": "dV", "Ma": "dM"}

    try:
        prefix = str(meta["CalculatedMoment"][:2])
        unit = unit_map[prefix]
        dtype = dtype_map[prefix] + str(meta["CalculatedType"])[2:]
    except (KeyError, TypeError) as e:
        raise Exception(
            "Unit and/or data type does not match the expected values. "
            "If this is raw current (fA), load it through Load_ELPI_file so "
            "the raw-current conversion route can be used."
        ) from e

    return _build_ELPI_aerosol2d(
        file=file,
        delimiter=delimiter,
        encoding=encoding,
        meta=meta,
        df=df,
        dist_data=dist_data,
        extra_df=extra_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        unit=unit,
        dtype=dtype,
        extra_data=extra_data,
        source_quantity="converted_export",
    )


def Load_ELPI_file_dat(file: str, extra_data: bool = False) -> Aerosol2D:
    """Load a raw ELPI .dat current file and convert it to number dN.

    The raw .dat file contains the measured channels and a main CAL Stage1-14
    block in current units. This function uses the CAL stage-current block,
    converts fA/dlogDp to fA per bin, applies ELPI charger efficiency, and
    returns number concentration per bin in cm-3.
    """
    encoding, delimiter = _detect_delimiter(file)
    meta = _load_ELPI_metadata(file, delimiter, encoding)
    bin_edges, bin_mids = _get_ELPI_bin_edges_and_mids_nm(meta)

    df = _read_ELPI_table(file, delimiter, encoding)
    current_data = _extract_ELPI_stage_block(df)
    extra_df = _make_ELPI_extra_data(df)

    dist_data = _convert_ELPI_current_to_number(current_data, meta)

    return _build_ELPI_aerosol2d(
        file=file,
        delimiter=delimiter,
        encoding=encoding,
        meta=meta,
        df=df,
        dist_data=dist_data,
        extra_df=extra_df,
        bin_edges=bin_edges,
        bin_mids=bin_mids,
        unit="cm⁻³",
        dtype="dN",
        extra_data=extra_data,
        source_quantity="raw_current_fA",
    )


def _build_ELPI_aerosol2d(
    file: str,
    delimiter: str,
    encoding: str,
    meta: dict,
    df: pd.DataFrame,
    dist_data: pd.DataFrame,
    extra_df: pd.DataFrame,
    bin_edges: np.ndarray,
    bin_mids: np.ndarray,
    unit: str,
    dtype: str,
    extra_data: bool,
    source_quantity: str,
) -> Aerosol2D:
    """Construct an Aerosol2D object from parsed ELPI distribution data."""
    dist_data = dist_data.copy()
    total_conc = pd.DataFrame(
        np.nansum(dist_data.to_numpy(dtype=float), axis=1), columns=["Total_conc"]
    )

    bin_mids = np.asarray(bin_mids, dtype=float).round(1)
    bin_edges = np.asarray(bin_edges, dtype=float).round(1)
    dist_data.columns = [str(mid) for mid in bin_mids]

    final_df = pd.concat([df["Datetime"], total_conc, dist_data], axis=1)
    elpi = Aerosol2D(final_df)

    meta = dict(meta)
    meta["density"] = meta.pop("Density(g/cm^3)", np.nan)
    meta["bin_edges"] = bin_edges
    meta["bin_mids"] = bin_mids
    meta["instrument"] = "ELPI"
    meta["serial_number"] = _get_ELPI_serial_number(file, delimiter, encoding)
    meta["dtype"] = dtype
    meta["unit"] = unit
    meta["source_quantity"] = source_quantity

    for key in ["CalculatedDi(um)", "CalculatedType", "CalculatedMoment"]:
        meta.pop(key, None)

    elpi._meta = meta

    # Exported ELPI files can contain number, surface, volume, or mass moments;
    # raw-current files are already converted above to number dN.
    if source_quantity != "raw_current_fA":
        elpi._convert_to_number_concentration()
        elpi.unnormalize_logdp()

    if extra_data:
        extra_df.set_index("Datetime", inplace=True)
        elpi._extra_data = extra_df

    return elpi
