import re
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..elpi import ELPI
from .support.exceptions import FileFormatError
from .support.reading import read_table, sniff
from .support.units import resolve_extra_columns

###############################################################################

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
        raise FileFormatError("Couldn't find the [Data] marker in the ELPI file.")

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
            raise FileFormatError("Couldn't find the data header line before [Data].")

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
    # Only the header region (up to the "[Data]" marker) is needed to locate the
    # data section, so stop reading there instead of pulling the whole file into
    # memory just to find the marker (the data itself is read by read_csv below).
    lines = []
    with open(file, encoding=encoding, errors="replace") as f:
        for line in f:
            lines.append(line)
            if line.strip() == "[Data]":
                break

    data_idx, header = _find_ELPI_data_section(lines, delimiter)

    df = read_table(
        file,
        delimiter=delimiter,
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
        raise ValueError(
            "Datetime parsing failed for many rows; check the first column."
        )

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
    """Return the ELPI charger conversion vector X (a three-part power law).

    The ELPI+ header stores the charger efficiency as::

        Efficiency(Dp/mult/exp)=Dp1  mult1  exp1  Dp2  mult2  exp2  mult3  exp3

    which defines a three-branch power law in the channel diameter ``Di`` (µm)::

        Di < Dp1         : X = mult1 * Di**exp1
        Dp1 <= Di < Dp2  : X = mult2 * Di**exp2
        Di >= Dp2        : X = mult3 * Di**exp3

    Following the calculation appendix of the *ELPI+ User Manual*, ``X`` is the
    conversion vector from stage current to number concentration: the number
    per bin is ``dN = I / X`` (the caller scales ``X`` by ``FlowRate/10`` to go
    from the 10 lpm calibration flow to the actual sample flow). The header
    multipliers already fold in the elementary charge and the calibration flow,
    so ``X`` has units of fA·cm³ and is used *directly* — no ``10**mult`` and no
    hard-coded high-size constant. This reproduces Dekati's own number export to
    below 0.1 % at unit density across all 14 stages.

    Args:
        dp_um: Channel diameter(s) in micrometres (the geometric-mean/Stokes
            diameter the charger sees).
        params: The eight ``Efficiency(Dp/mult/exp)`` values from the header.
            Defaults to the calibration of the reference ELPI+ file.

    Returns:
        numpy.ndarray: The conversion vector ``X`` (fA·cm³) for each diameter.
    """
    dp_um = np.asarray(dp_um, dtype=float)

    if params is None:
        dp1, mult1, exp1, dp2, mult2, exp2, mult3, exp3 = (
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
        dp1, mult1, exp1, dp2, mult2, exp2, mult3, exp3 = p

    return np.where(
        dp_um < dp1,
        mult1 * dp_um**exp1,
        np.where(
            dp_um < dp2,
            mult2 * dp_um**exp2,
            mult3 * dp_um**exp3,
        ),
    )


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

    # Raw .dat stage currents are stored normalized as dW/dlogDp; multiply by the
    # (aerodynamic) dlogDp to recover the plain per-stage current I in fA.
    calculated_type = str(meta.get("CalculatedType", ""))
    if "dlog" in calculated_type.lower():
        dlogdp = np.diff(np.log10(d50_um))
        current = current * dlogdp[np.newaxis, :]

    # Number per bin: dN = I / X, with X the charger conversion vector (fA·cm³)
    # scaled from the 10 lpm calibration flow to the instrument's actual flow.
    charger_eff = _ELPI_charger_efficiency(
        di_um, params=meta.get("Efficiency(Dp/mult/exp)")
    )
    flow_lpm = float(meta.get("FlowRate", 10.0) or 10.0)
    conversion = charger_eff * (flow_lpm / 10.0)

    number = current / conversion[np.newaxis, :]

    dilution = float(meta.get("Dilution", 1.0) or 1.0)
    number = number * dilution

    if clip_negative:
        number = np.where(number < 0, 0.0, number)

    return pd.DataFrame(number, columns=current_fa.columns, index=current_fa.index)


###############################################################################
# Density recalculation

# Cunningham slip-correction constants for the ELPI's aerodynamic↔Stokes
# diameter conversion. These reproduce Dekati's own published slip-correction
# vector (ELPI+ User Manual, App. A.2 — the ``Cca`` table for the calibrated
# cutpoints) to < 0.1 %. They are specific to this conversion and intentionally
# differ from the generic Allen–Raabe values in
# :mod:`aerosoltools._core.corrections`.
_ELPI_MEAN_FREE_PATH_UM = 0.07305
_ELPI_SLIP_A = 1.1384
_ELPI_SLIP_B = 0.3621
_ELPI_SLIP_C = 1.2157


def _cunningham(dp_um) -> np.ndarray:
    """Cunningham slip correction (Dekati ELPI constants) for diameters in µm."""
    dp_um = np.asarray(dp_um, dtype=float)
    kn = 2.0 * _ELPI_MEAN_FREE_PATH_UM / dp_um
    return 1.0 + kn * (_ELPI_SLIP_A + _ELPI_SLIP_B * np.exp(-_ELPI_SLIP_C / kn))


def _stokes_from_aerodynamic(dpa_um, density: float) -> np.ndarray:
    """Convert aerodynamic diameters (µm) to Stokes diameters at a density.

    The ELPI classifies by aerodynamic behaviour, so each stage's aerodynamic
    diameter is fixed; the geometric (volume-equivalent / Stokes) diameter that
    the size axis reports obeys ``Dpa² · Cc(Dpa) = Dps² · Cc(Dps) · ρ``
    (ELPI+ User Manual, App. A.2). Solved iteratively because the slip
    correction depends on the unknown diameter. In the slip-free limit it
    reduces to the familiar ``D ∝ ρ^(-1/2)``.
    """
    dpa_um = np.asarray(dpa_um, dtype=float)
    target = dpa_um**2 * _cunningham(dpa_um) / density
    d = dpa_um / np.sqrt(density)  # slip-free first guess
    for _ in range(60):
        d = np.sqrt(target / _cunningham(d))
    return d


def recalculate_ELPI_density(elpi, new_density: float) -> bool:
    """Recompute an ELPI dataset's diameters and number for a new density.

    The ELPI reports a *geometric* particle diameter that depends on the assumed
    particle density (it actually classifies by aerodynamic behaviour). Changing
    the density therefore must (1) move the size axis to the diameters the
    particles would have at the new density — otherwise the reported diameters
    are fictitious — and (2) re-attribute the per-bin number concentration,
    because the charger efficiency (which turned current into number) is a
    function of that diameter.

    The diameters are rebuilt the way the Dekati software does it, from the
    *fixed aerodynamic cutpoints* (``D50values``, which never change with
    density) rather than by nudging the current midpoints: each aerodynamic
    cutpoint is converted to its Stokes diameter at the new density
    (:func:`_stokes_from_aerodynamic`), the lowest cutpoint is pinned at the
    fixed 6 nm filter cutpoint (``FilterLow``, which the ELPI does not rescale),
    and the stage midpoints are the geometric means of the resulting cutpoints.
    Because everything is derived from the density-independent cutpoints, the
    size axis is path-independent — changing 0.16 → 1.0 lands exactly on the
    diameters a fresh unit-density load would give. Number is re-attributed
    through the charger efficiency (:func:`_ELPI_charger_efficiency`), which
    depends on that diameter.

    This reproduces the Dekati software's density-dependent size axis to < 0.5 %
    on every stage bar the filter-boundary bin (stage 1 at very low density,
    ~5 %). All inputs are already on the object's metadata (``D50values``,
    ``FilterLow``, ``bin_mids``, charger ``Efficiency`` coefficients and the
    current ``density``), so no separate copy of the header is needed.

    Args:
        elpi: An :class:`~aerosoltools.Aerosol2D` produced by an ELPI loader.
        new_density: The new assumed particle density in g/cm³.

    Returns:
        bool: ``True`` if a recalculation was performed (the object is a
        recognised ELPI), otherwise ``False`` so the caller can fall back to the
        generic density behaviour.

    Notes:
        Operates in place; number is re-attributed only while the data are plain
        (unnormalized) number (``dN``) — the state right after loading. Set the
        density before converting the dtype (e.g. to ``dM``) for the most
        consistent result.
    """
    meta = elpi._meta
    if str(meta.get("instrument", "")).upper() != "ELPI":
        return False
    if "bin_mids" not in meta or "D50values(um)" not in meta:
        return False

    rho_old = float(meta.get("density", np.nan))
    rho_new = float(new_density)
    if not (rho_new > 0):
        return False
    if np.isfinite(rho_old) and np.isclose(rho_new, rho_old):
        meta["density"] = rho_new
        return True

    bin_mids_old = np.asarray(meta["bin_mids"], dtype=float)

    # Rebuild the size axis from the fixed AERODYNAMIC cutpoints (density-
    # independent) rather than from the current midpoints. Convert each cutpoint
    # to its Stokes diameter at the new density, pin the lowest cutpoint at the
    # fixed 6 nm filter cutpoint (the ELPI never rescales it), and take the
    # geometric-mean midpoints. This matches Dekati's own calculation and is
    # path-independent (e.g. 0.16 → 1.0 reproduces a fresh unit-density load).
    aero_cut_um = _as_float_array(meta["D50values(um)"])
    try:
        filter_low_um = float(_as_float_array(meta["FilterLow(um)"]).flat[0])
    except (KeyError, ValueError, IndexError):
        filter_low_um = float(aero_cut_um[0])

    cut_um = _stokes_from_aerodynamic(aero_cut_um, rho_new)
    cut_um[0] = filter_low_um
    mids_um = np.sqrt(cut_um[1:] * cut_um[:-1])

    bin_mids_new = np.round(mids_um * 1000.0, 1)
    bin_edges_new = np.round(cut_um * 1000.0, 1)

    old_headers = [str(x) for x in bin_mids_old]
    new_headers = [str(x) for x in bin_mids_new]
    present = [h for h in old_headers if h in elpi._data.columns]

    # Re-attribute the per-bin number: the charger efficiency depends on the
    # (now density-corrected) diameter, so the same current implies a different
    # number. Applied to both raw-current and converted-export data while the
    # data are plain (unnormalized) number.
    dtype = str(elpi.dtype)
    is_plain_number = ("dN" in dtype) and ("dlogDp" not in dtype)
    if is_plain_number and len(present) == len(old_headers):
        eff_params = meta.get("Efficiency(Dp/mult/exp)")
        eff_old = _ELPI_charger_efficiency(bin_mids_old / 1000.0, params=eff_params)
        eff_new = _ELPI_charger_efficiency(bin_mids_new / 1000.0, params=eff_params)
        factor = np.where(eff_new > 0, eff_old / eff_new, 1.0)
        vals = elpi._data[old_headers].to_numpy(dtype=float) * factor[np.newaxis, :]
        elpi._data[old_headers] = vals
        if "Total_conc" in elpi._data.columns:
            elpi._data["Total_conc"] = np.nansum(vals, axis=1)

    # Relabel the size-bin columns with the new diameters; update the metadata.
    rename = {o: n for o, n in zip(old_headers, new_headers) if o in elpi._data.columns}
    if rename:
        elpi._data.rename(columns=rename, inplace=True)
    meta["bin_mids"] = bin_mids_new
    meta["bin_edges"] = bin_edges_new
    meta["density"] = rho_new
    return True


###############################################################################
# Public loaders


def load_elpi_file(file: str, extra_data: bool = True) -> ELPI:
    """Load an ELPI file and return an Aerosol2D object.

    This is the routing function. Already-converted Dekati exports are handled
    by :func:`load_elpi_file_txt`. Raw-current .dat files with
    ``CalculatedMoment=Current (fA)`` are converted by
    :func:`load_elpi_file_dat` before constructing the Aerosol2D object.
    """
    # Detect + read the header once here to route the file; the chosen sub-loader
    # reuses them (see the encoding/delimiter/meta pass-through) instead of
    # re-sniffing and re-reading the header.
    encoding, delimiter = sniff(file)
    meta = _load_ELPI_metadata(file, delimiter, encoding)
    calculated_moment = str(meta.get("CalculatedMoment", ""))

    if calculated_moment.lower().startswith("current"):
        return load_elpi_file_dat(
            file, extra_data, encoding=encoding, delimiter=delimiter, meta=meta
        )

    return load_elpi_file_txt(
        file, extra_data, encoding=encoding, delimiter=delimiter, meta=meta
    )


def load_elpi_file_txt(
    file: str,
    extra_data: bool = False,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    meta: dict | None = None,
) -> ELPI:
    """Load an already-converted ELPI size-distribution export.

    ``encoding``/``delimiter``/``meta`` are reused when supplied (the router
    :func:`load_elpi_file` passes what it already computed); each is derived from
    the file only when not given, so the file's header is not re-read.
    """
    encoding, delimiter = sniff(file, encoding, delimiter)
    if meta is None:
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
        raise FileFormatError(
            "Unit and/or data type does not match the expected values. "
            "If this is raw current (fA), load it through load_elpi_file so "
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


def load_elpi_file_dat(
    file: str,
    extra_data: bool = False,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    meta: dict | None = None,
) -> ELPI:
    """Load a raw ELPI .dat current file and convert it to number dN.

    The raw .dat file contains the measured channels and a main CAL Stage1-14
    block in current units. This function uses the CAL stage-current block,
    converts fA/dlogDp to fA per bin, applies ELPI charger efficiency, and
    returns number concentration per bin in cm-3.

    ``encoding``/``delimiter``/``meta`` are reused when supplied (the router
    :func:`load_elpi_file` passes what it already computed); each is derived from
    the file only when not given, so the file's header is not re-read.
    """
    encoding, delimiter = sniff(file, encoding, delimiter)
    if meta is None:
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
) -> ELPI:
    """Construct an Aerosol2D object from parsed ELPI distribution data."""
    dist_data = dist_data.copy()
    total_conc = pd.DataFrame(
        np.nansum(dist_data.to_numpy(dtype=float), axis=1), columns=["Total_conc"]
    )

    bin_mids = np.asarray(bin_mids, dtype=float).round(1)
    bin_edges = np.asarray(bin_edges, dtype=float).round(1)
    dist_data.columns = [str(mid) for mid in bin_mids]

    final_df = pd.concat([df["Datetime"], total_conc, dist_data], axis=1)
    elpi = ELPI(final_df)

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
        # ELPI diagnostic headers carry no in-header units, and its several
        # Temperature*/Pressure channels are internal sensors (deliberately not
        # curated as ambient); keep them all, resolving any that do embed a unit.
        extra_df, column_units = resolve_extra_columns(extra_df)
        elpi._extra_data = extra_df
        elpi._meta["column_units"] = column_units

    return elpi
