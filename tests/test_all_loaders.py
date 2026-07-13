import os

import pandas as pd
import pytest

from aerosoltools.loaders import (
    load_aethalometer_file,
    load_aps_file,
    load_cpc_file,
    load_discmini_file,
    load_discmini_raw_file,
    load_elpi_file,
    load_fmps_file,
    load_fourtec_file,
    load_grimm_file,
    load_ns_file,
    load_opcn3_file,
    load_ops_file,
    load_partector_file,
    load_ranger_file,
    load_smps_file,
)
from aerosoltools.loaders.Discmini import (
    _extract_serial_and_firmware,
    _normalize_serial,
)


@pytest.mark.parametrize(
    "loader_func, filename",
    [
        (load_aethalometer_file, "Sample_Aetholometer.csv"),
        (load_aps_file, "Sample_APS_aero.txt"),
        (load_aps_file, "Sample_APS_correlated.txt"),
        (load_cpc_file, "Sample_CPC_Direct.txt"),
        (load_discmini_file, "Sample_Discmini.txt"),
        (load_elpi_file, "Sample_ELPI.txt"),
        (load_elpi_file, "Sample_ELPI2.txt"),
        (load_fmps_file, "Sample_FMPS.txt"),
        (load_fmps_file, "Sample_FMPS2.txt"),
        (load_fourtec_file, "Sample_Fourtec.xlsx"),
        (load_grimm_file, "Sample_Grimm.txt"),
        (load_ns_file, "Sample_NS.csv"),
        (load_opcn3_file, "Sample_OPCN3.txt"),
        (load_ops_file, "Sample_OPS.csv"),
        (load_ops_file, "Sample_OPS2.txt"),
        (load_partector_file, "Sample_Partector.txt"),
        (load_ranger_file, "Sample_Ranger.csv"),
        (load_smps_file, "Sample_SMPS.txt"),
    ],
)
def test_loader_smoke(loader_func, filename):
    test_file = os.path.join(os.path.dirname(__file__), "data", filename)
    assert os.path.exists(test_file), f"Missing test file: {filename}"

    data = loader_func(test_file)
    assert data is not None
    assert hasattr(data, "data"), f"{filename}: missing 'data'"
    assert hasattr(data, "metadata"), f"{filename}: missing 'metadata'"
    assert isinstance(data.data, pd.DataFrame), f"{filename}: data is not DataFrame"


def test_aps_correlated_and_aero_only():
    """APS: correlated -> Aerosol3d (both axes consistent); aero-only -> 2D."""
    from aerosoltools import Aerosol2D, Aerosol3d

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # Aerodynamic-only export loads as a plain 2D distribution.
    aero = load_aps_file(os.path.join(data_dir, "Sample_APS_aero.txt"))
    assert isinstance(aero, Aerosol2D)
    assert not isinstance(aero, Aerosol3d)
    assert len(aero.bin_mids) == 52
    assert aero.unit == "cm⁻³"

    # Correlated export carries both size axes plus their matrix.
    cor = load_aps_file(os.path.join(data_dir, "Sample_APS_correlated.txt"))
    assert isinstance(cor, Aerosol3d)
    assert cor.is_correlated
    assert len(cor.bin_mids) == 52
    assert cor.optical is not None and len(cor.optical.bin_mids) == 16
    # Aerodynamic total, optical total and matrix total agree per sample.
    a0 = float(cor.data["Total_conc"].iloc[0])
    o0 = float(cor.optical.data["Total_conc"].iloc[0])
    m0 = float(cor.correlation.iloc[0].sum())
    assert a0 == pytest.approx(o0, rel=1e-6)
    assert a0 == pytest.approx(m0, rel=1e-6)
    # Time cropping keeps both axes (and the matrix) in sync.
    cor.timecrop(cor.time[3], cor.time[10])
    assert len(cor.data) == len(cor.optical.data) == len(cor.correlation)


_RANGER_STATUS = "Status(f:fail a:aging w:warming-up z:zerocal c:spancal)"


def _write_ranger(path, blocks):
    """Write a synthetic Ranger export with the given (header, rows) blocks."""
    lines = ["Ranger Serial Number:,2605-1000088-A"]
    for header, rows in blocks:
        lines.append(header)
        lines.extend(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_ranger_single_component_returns_single_object():
    """A Cl₂-only file loads as one Gas1D with ppm / Cl₂ metadata."""
    from aerosoltools import Gas1D

    data = load_ranger_file(_data_path("Sample_Ranger.csv"))
    assert isinstance(data, Gas1D)
    assert "Concentration" in data.data.columns
    assert data.metadata["unit"] == "ppm"
    assert data.metadata["dtype"] == "Cl₂"
    assert data.metadata["measurement"] == "Cl₂"
    # 'total_concentration' does not apply to a gas monitor any more.
    with pytest.raises(AttributeError):
        _ = data.total_concentration


def test_ranger_multi_header_splits_by_component(tmp_path):
    """Interleaved heads (PM/Cl₂/NO₂) split into one object per component."""
    from aerosoltools import Aerosol1D, Gas1D

    pm_header = (
        "UTC time,Local time,LocationID,PM1 (ug/m3),PM2.5 (ug/m3),"
        f"PMrsp (ug/m3),PM10 (ug/m3),TSP (ug/m3),{_RANGER_STATUS}"
    )
    cl2_header = f"UTC time,Local time,LocationID,CL2 (PPM),{_RANGER_STATUS}"
    no2_header = f"UTC time,Local time,LocationID,NO2 (PPM),{_RANGER_STATUS}"

    path = tmp_path / "Ranger_multi.csv"
    _write_ranger(
        path,
        [
            (
                pm_header,
                ["2026-05-19 08:42:00,2026-05-19 08:42:00,02,0.2,0.8,1.6,3.1,4.8,"],
            ),
            (
                cl2_header,
                [
                    "2026-06-01 10:53:00,2026-06-01 10:53:00,10,0.12,",
                    "2026-06-01 10:54:00,2026-06-01 10:54:00,10,0.11,",
                ],
            ),
            (no2_header, ["2026-06-02 09:00:00,2026-06-02 09:00:00,10,0.05,"]),
            # A second Cl₂ block must merge into the first Cl₂ object.
            (cl2_header, ["2026-06-03 08:00:00,2026-06-03 08:00:00,10,0.20,"]),
        ],
    )

    result = load_ranger_file(str(path))
    assert isinstance(result, list) and len(result) == 3

    by_measure = {o.metadata["measurement"]: o for o in result}
    assert set(by_measure) == {"PM", "Cl₂", "NO₂"}

    # PM head -> plain multi-channel Aerosol1D with one column per fraction.
    pm = by_measure["PM"]
    assert isinstance(pm, Aerosol1D) and not isinstance(pm, Gas1D)
    for col in ("PM1", "PM2.5", "PMrsp", "PM10", "TSP"):
        assert col in pm.data.columns
    assert pm.metadata["unit"]["PM10"] == "µg/m³"

    # Gas heads -> Gas1D; the two Cl₂ blocks are concatenated (2 + 1 rows).
    cl2 = by_measure["Cl₂"]
    assert isinstance(cl2, Gas1D)
    assert cl2.metadata["unit"] == "ppm"
    assert cl2.concentration.notna().sum() == 3

    assert by_measure["NO₂"].metadata["dtype"] == "NO₂"


def test_ranger_sniffer_identifies_non_chlorine_heads(tmp_path):
    """The sniffer recognizes a Ranger export even with no Cl₂ column."""
    from aerosoltools.gui.loaders import identify_instrument, is_Ranger_file

    pm_header = f"UTC time,Local time,LocationID,PM10 (ug/m3),{_RANGER_STATUS}"
    path = tmp_path / "Ranger_pm.csv"
    _write_ranger(
        path, [(pm_header, ["2026-05-19 08:42:00,2026-05-19 08:42:00,02,3.1,"])]
    )

    assert is_Ranger_file(str(path)) is True
    assert identify_instrument(str(path)) == "Ranger"


def test_aethalometer_per_channel_units():
    """Aethalometer stores per-channel unit/dtype dicts (not a single scalar).

    The AerosolAlt convention is a name→value dict keyed by channel; the BC
    channels are ng/m³ mass concentrations and (when present) the Ångström
    exponent AAE is dimensionless. A scalar unit here would make
    ``summarize_activities`` iterate over the characters of the string.
    """
    import contextlib
    import io

    aeth = load_aethalometer_file(_data_path("Sample_Aetholometer.csv"))
    unit, dtype = aeth.metadata["unit"], aeth.metadata["dtype"]
    assert isinstance(unit, dict) and isinstance(dtype, dict)
    channels = [c for c in aeth.data.columns if c != "All data"]
    assert channels, "expected at least the spectral BC channels"
    for ch in channels:
        if "AAE" in ch:
            assert unit[ch] == "" and dtype[ch] == "AAE"
        else:
            assert unit[ch] == "ng/m³" and dtype[ch] == "dM"

    # With a proper unit dict the per-channel summary runs (it prints a table).
    with contextlib.redirect_stdout(io.StringIO()):
        aeth.summarize_activities()


def _scaled_copy(obj, factor):
    """A deep copy of ``obj`` with every numeric channel multiplied by ``factor``.

    Because the copy shares the original timestamps, ``match='exact'`` aligns
    every sample and the fitted gain recovers ``factor`` (through the origin).
    """
    import pandas as _pd

    ref = obj.copy_self()
    for c in ref._data.columns:
        if _pd.api.types.is_numeric_dtype(ref._data[c]):
            ref._data[c] = ref._data[c] * factor
    return ref


def test_calibrate_against_reference_total():
    """High-level total calibration recovers the gain and does not mutate target."""
    import aerosoltools as at

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 2.0)
    original = target.total_concentration.copy()

    calibrated, model = at.calibrate_against_reference(target, reference)

    assert model.basis == "total"
    assert model.gains[0] == pytest.approx(2.0, rel=1e-3)
    assert model.r_squared[0] == pytest.approx(1.0, abs=1e-3)
    # inplace=False leaves the target untouched and returns a new object.
    assert calibrated is not target
    assert target.total_concentration.equals(original)
    # The calibrated total now matches the reference (2x the original).
    ref_total = reference.total_concentration.dropna()
    cal_total = calibrated.total_concentration.dropna()
    assert cal_total.reindex(ref_total.index).sub(ref_total).abs().max() < 1e-6
    # Metadata records the applied model.
    assert calibrated._meta["calibration"]["basis"] == "total"


def test_calibrate_inplace_mutates():
    """inplace=True corrects the target itself."""
    import aerosoltools as at

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 3.0)
    out, model = at.calibrate_against_reference(target, reference, inplace=True)
    assert out is target
    assert model.gains[0] == pytest.approx(3.0, rel=1e-3)


def test_calibration_model_roundtrip():
    """CalibrationModel serializes and restores identically."""
    from aerosoltools.intercomparison import CalibrationModel, fit_calibration

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 2.0)
    model = fit_calibration(target, reference, include_offset=True)
    restored = CalibrationModel.from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
    assert restored.gains == model.gains


def test_calibration_per_bin():
    """Per-bin calibration fits an independent gain per size bin."""
    from aerosoltools.intercomparison import calibrate_against_reference

    target = load_ops_file(_data_path("Sample_OPS.csv"))
    reference = _scaled_copy(target, 1.5)
    calibrated, model = calibrate_against_reference(target, reference, basis="per_bin")
    assert model.basis == "per_bin"
    assert len(model.gains) == len(target.size_data.columns)
    applied_gains = [g for g, ok in zip(model.gains, model.applied) if ok]
    assert applied_gains, "expected at least one bin to be calibrated"
    for g in applied_gains:
        assert g == pytest.approx(1.5, rel=1e-2)


def test_calibration_clips_negative():
    """clip_negative keeps calibrated concentrations >= 0 (total and per-bin)."""
    from aerosoltools.intercomparison import CalibrationModel

    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    total_model = CalibrationModel(
        basis="total",
        gains=[1.0],
        offsets=[-1e12],
        r_squared=[1.0],
        n_obs=[10],
        applied=[True],
        include_offset=True,
    )
    out = cpc.apply_calibration(total_model)
    assert (out.total_concentration.dropna() >= 0).all()

    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    n = len(ops.size_data.columns)
    bin_model = CalibrationModel(
        basis="per_bin",
        gains=[1.0] * n,
        offsets=[-1e12] * n,
        r_squared=[1.0] * n,
        n_obs=[10] * n,
        applied=[True] * n,
        include_offset=True,
    )
    out2 = ops.apply_calibration(bin_model)
    assert (out2.size_data.to_numpy() >= 0).all()
    assert (out2.total_concentration.dropna() >= 0).all()


def test_calibration_rejects_incompatible():
    """Domain errors for unit / basis / bin-count mismatches."""
    from aerosoltools.intercomparison import CalibrationError, fit_calibration

    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    ops = load_ops_file(_data_path("Sample_OPS.csv"))

    # Per-bin needs both datasets size-resolved.
    with pytest.raises(CalibrationError):
        fit_calibration(cpc, _scaled_copy(cpc, 2.0), basis="per_bin")

    # Unknown basis.
    with pytest.raises(CalibrationError):
        fit_calibration(cpc, _scaled_copy(cpc, 2.0), basis="nonsense")

    # A CPC (number) and an OPS (size-resolved) do not share a unit contract for
    # a per-bin calibration — mismatched bin counts are rejected too.
    with pytest.raises(CalibrationError):
        fit_calibration(ops, cpc, basis="per_bin")


def test_psd_fit_and_evaluator_single_sourced():
    """fit_psd + the public lognormal_modes are the one model the GUI also uses."""
    import numpy as np

    import aerosoltools as at
    from aerosoltools.gui.tabs import _psdfit

    # The GUI overlay evaluator is literally the API function (one implementation).
    assert _psdfit.lognormal_modes is at.lognormal_modes

    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    period = (ops.time[0], ops.time[-1])
    modes, err = ops.fit_psd(period=period, mu=[500], sigma=[2.0], factor=[1000.0])
    triples = list(zip(modes["mu"], modes["sigma"], modes["factor"]))
    total, per_mode = at.lognormal_modes(ops.bin_mids, triples)
    assert len(total) == len(ops.bin_mids)
    assert np.all(np.isfinite(total))
    assert len(per_mode) == len(modes["mu"])


def test_calibration_gui_and_script_agree():
    """The GUI applies the very same model the scripting API produces."""
    import aerosoltools as at
    from aerosoltools.intercomparison import fit_calibration

    target = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    reference = _scaled_copy(target, 2.0)

    model = fit_calibration(target, reference)
    via_method = target.apply_calibration(model)
    via_highlevel, _ = at.calibrate_against_reference(target, reference)

    a = via_method.total_concentration.dropna()
    b = via_highlevel.total_concentration.dropna()
    assert a.sub(b).abs().max() < 1e-9


def test_public_load_file_autodetects():
    """at.load_file auto-detects the instrument and matches the explicit loader."""
    import aerosoltools as at
    from aerosoltools.loaders import InstrumentDetectionError

    auto = at.load_file(_data_path("Sample_OPS.csv"))
    explicit = load_ops_file(_data_path("Sample_OPS.csv"))
    assert type(auto) is type(explicit)
    assert auto.data.shape == explicit.data.shape

    # Explicit instrument name routes to that loader.
    assert type(at.load_file(_data_path("Sample_CPC_Direct.txt"), instrument="CPC"))

    # detect_instrument returns the canonical registry key (or None).
    assert at.detect_instrument(_data_path("Sample_OPS.csv")) == "OPS"
    assert at.detect_instrument(_data_path("Sample_Ranger.csv")) == "Ranger"

    # An unknown instrument name is a domain error, not a KeyError.
    with pytest.raises(InstrumentDetectionError):
        at.load_file(_data_path("Sample_OPS.csv"), instrument="NoSuchDevice")


def test_discmini_serial_normalization():
    """The serial normalizer strips the inconsistent 'SN' prefix."""
    assert _normalize_serial("SN101923") == "101923"
    assert _normalize_serial("101670") == "101670"
    assert _normalize_serial("sn 109172") == "109172"
    assert _normalize_serial(None) is None


def test_discmini_serial_from_processed_and_raw_headers():
    """Serial + firmware parse from both processed and raw header variants."""
    processed = [
        "[testo DiSCmini java tool version 2,1 output file]",
        "[Data recorded with testo DiSCmini SN101923 running firmware 3,42]",
    ]
    serial, fw = _extract_serial_and_firmware(processed)
    assert serial == "101923" and fw == "3.42"

    # Raw header where the serial has no "SN" prefix must still match.
    raw = [
        "nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42",
        "CalData: 101670      6.02   25.98   -2.24    0.45    1.1530348.26    0.96",
    ]
    serial, fw = _extract_serial_and_firmware(raw)
    assert serial == "101670" and fw == "3.42"


def _write_synthetic_raw(path):
    """Write a minimal but realistic raw DiSCmini file for loader testing."""
    lines = [
        "nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42",
        "Filename: TEST.TXT",
        "Averaging Period: 1 sec",
        "Date and Time: 2024.01.01 12:00:00",
        "CalData: SN123456    0.28   30.73   -6.45    1.28    1.1319808.76    0.68",
        " NaCl test",
        "    0.28\t   30.73\t   -6.45\t    1.28\t    1.13\t19808.76\t    0.68\t",
        "Offsets:    -0.75\t   -0.69\t",
        "Sampled:   1000 pC\tC:     10\tW:      1",
        "Time\tDiffusion\tFilter\tTemp\tIdiff\tUcor\tFlow\tBatt\tStatus",
    ]
    # 20 valid measuring rows (Status 8B) with constant currents.
    for t in range(20):
        lines.append(f"{t}\t10.00\t35.00\t27.6\t9.82\t4.19\t1.00\t7.90\t8B")
    # 5 idle rows (Status 88, near-zero currents) that must be excluded.
    for t in range(20, 25):
        lines.append(f"{t}\t0.10\t0.10\t27.6\t0.00\t0.37\t0.37\t7.66\t88")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_discmini_raw_loader_pipeline(tmp_path):
    """Raw loader filters idle rows, averages, and reproduces LDSA exactly."""
    raw = tmp_path / "TEST.TXT"
    _write_synthetic_raw(str(raw))
    # Disable the zero-offset correction here so the exact cal-formula values
    # below are checked without the (separately tested) offset subtraction.
    dm = load_discmini_raw_file(
        str(raw), extra_data=True, period=10, zero_offset_correction=False
    )

    # 20 valid rows / period 10 = 2 output rows (idle rows excluded).
    assert dm.data.shape[0] == 2
    assert dm.serial_number == "123456"  # "SN" stripped
    assert dm._meta.get("firmware") == "3.42"
    assert dm._meta.get("size_inversion") == "cubic_ratio_polynomial"

    # LDSA = cal[6] * (Diff + Filter) = 0.68 * (10 + 35) = 30.6 (exact here).
    assert dm.data["LDSA"].iloc[0] == pytest.approx(30.6, abs=1e-6)
    # Size = cal0 + cal1*R + cal2*R^2 + cal3*R^3 with R = 35/10 = 3.5.
    assert dm.data["Size"].iloc[0] == pytest.approx(83.7, abs=0.1)
    # Idle rows would have dragged the currents toward zero if not excluded.
    assert dm.extra_data["Diffusion"].iloc[0] == pytest.approx(10.0, abs=1e-6)
    assert dm.total_concentration.iloc[0] > 0


def test_discmini_raw_window_gap_handling(tmp_path):
    """Windows with < 8 valid rows are dropped; real time is preserved across gaps."""
    import numpy as np

    header = [
        "nw PERSONAL AEROSOL MONITOR Data written with SW-Ver 3.42",
        "Filename: TEST.TXT",
        "Averaging Period: 1 sec",
        "Date and Time: 2024.01.01 00:00:00",
        "CalData: SN1    0.28   30.73   -6.45    1.28    1.1319808.76    0.68",
        " NaCl",
        "    0.28\t30.73\t-6.45\t1.28\t1.13\t19808.76\t0.68\t",
        "Offsets:  -0.75\t-0.69\t",
        "Sampled: 1 pC\tC: 1\tW: 1",
        "Time\tDiffusion\tFilter\tTemp\tIdiff\tUcor\tFlow\tBatt\tStatus",
    ]
    rows = []
    for t in range(40):
        # Window 2 (t=20..29): only t=20..24 valid (5 rows) -> below the >=8
        # threshold, so it must be dropped. All other windows are full.
        idle = 20 <= t <= 29 and t >= 25
        status = "88" if idle else "8B"
        rows.append(f"{t}\t10.00\t35.00\t27.6\t9.82\t4.19\t1.00\t7.90\t{status}")
    raw = tmp_path / "GAP.TXT"
    with open(raw, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header + rows) + "\n")

    dm = load_discmini_raw_file(str(raw), period=10)
    # Windows 0, 1, 3 kept (window 2 has only 5 valid rows -> dropped).
    assert dm.data.shape[0] == 3
    elapsed = (dm.time - dm.time[0]).total_seconds().to_numpy()
    # Centres at 4.5, 14.5, 34.5 s — the 24.5 window is skipped (70 s jump).
    assert np.allclose(elapsed, [0.0, 10.0, 30.0])


def _data_path(name):
    return os.path.join(os.path.dirname(__file__), "data", name)


def test_discmini_raw_loader_on_sample_file():
    """Load a real (truncated) raw DiSCmini file and sanity-check the output."""
    dm = load_discmini_raw_file(_data_path("Sample_Discmini_raw.txt"), extra_data=True)
    assert dm.serial_number == "101923"  # "SN101923" in the CalData line
    assert dm._meta.get("firmware") == "3.42"
    assert dm._meta.get("size_inversion") == "cubic_ratio_polynomial"
    # 300 valid rows / 10 = 30 averaged output rows.
    assert dm.data.shape[0] == 30
    for col in ("Total_conc", "Size", "LDSA"):
        assert col in dm.data.columns
        assert (dm.data[col] > 0).all()


def test_discmini_raw_matches_processed_size_and_number():
    """Raw-loader Size & Number reproduce the vendor output (< 3% median)."""
    import numpy as np

    dm = load_discmini_raw_file(
        _data_path("Sample_Discmini_raw.txt"), zero_offset_correction=False
    )
    proc = pd.read_csv(
        _data_path("Sample_Discmini_raw_output.txt"),
        sep="\t",
        skiprows=6,
        header=None,
        decimal=",",
        engine="python",
        names=["Time", "Number", "Size", "LDSA", "Filter", "Diff", "z"],
    )
    n = min(len(dm.data), len(proc))
    for mine_col, ven_col in [("Size", "Size"), ("Total_conc", "Number")]:
        mine = dm.data[mine_col].to_numpy()[:n]
        ven = pd.to_numeric(proc[ven_col], errors="coerce").to_numpy()[:n]
        rel = np.abs(mine - ven) / ven
        assert np.median(rel) < 0.03, f"{mine_col} median rel err too high"


def test_discmini_zero_offset_correction_recovers_and_applies():
    """Zero-offset correction reads the idle-period zeros and shifts the currents."""
    from aerosoltools.loaders.Discmini import _parse_raw_header, _zero_offset_series

    raw = _data_path("Sample_Discmini_raw.txt")
    on = load_discmini_raw_file(raw, zero_offset_correction=True)
    off = load_discmini_raw_file(raw, zero_offset_correction=False)
    # The flag is recorded and the correction changes the computed currents.
    assert on._meta.get("zero_offset_correction") is True
    assert off._meta.get("zero_offset_correction") is False
    assert (on.data["LDSA"].to_numpy() != off.data["LDSA"].to_numpy()).any()

    # The offset series is anchored on the header Offsets even without an idle
    # period, and grows a new anchor per hourly idle run when present.
    with open(raw, encoding="latin-1") as fh:
        header = [next(fh) for _ in range(15)]
    meta = _parse_raw_header(header)
    times, off_d, off_f = _zero_offset_series(
        pd.DataFrame(
            {
                "Time": [0],
                "Diffusion": [0.0],
                "Filter": [0.0],
                "Idiff": [9.8],
                "Status": ["8B"],
            }
        ),
        meta["offsets"],
    )
    assert times[0] == 0.0
    assert off_d[0] == pytest.approx(meta["offsets"][0])


def test_nonparticle_classes_exported():
    """The four non-particle classes are part of the public top-level API."""
    import aerosoltools as at

    for name in ("Gas1D", "Aethalometer", "Environmental1D", "Partector"):
        assert hasattr(at, name), f"{name} not exported"
        assert name in at.__all__


def test_aethalometer_class_and_channels():
    """Aethalometer loads as its own class with per-wavelength BCc accessors."""
    from aerosoltools import Aethalometer

    aeth = load_aethalometer_file(_data_path("Sample_Aetholometer.csv"))
    assert isinstance(aeth, Aethalometer)
    # Per-wavelength direct accessors resolve to the matching BCc column.
    assert aeth.ir_bcc.name == "IR BCc"
    assert aeth.uv_bcc.name == "UV BCc"
    # The internal primary hook is the IR-equivalent channel.
    assert aeth._primary.name == "IR BCc"
    # 'total_concentration' does not apply to black carbon.
    with pytest.raises(AttributeError):
        _ = aeth.total_concentration


def test_partector_class_and_ldsa():
    """Partector loads as its own class exposing LDSA and no total concentration."""
    from aerosoltools import Partector

    par = load_partector_file(_data_path("Sample_Partector.txt"))
    assert isinstance(par, Partector)
    assert par.ldsa.name == "LDSA"
    assert par._primary.name == "LDSA"
    assert par.tem_samples is not None
    with pytest.raises(AttributeError):
        _ = par.total_concentration


def test_environmental_class_and_channels():
    """Fourtec loads as Environmental1D with temperature/RH accessors."""
    from aerosoltools import Environmental1D

    ft = load_fourtec_file(_data_path("Sample_Fourtec.xlsx"))
    assert isinstance(ft, Environmental1D)
    assert ft.temperature.name == "Temperature"
    assert ft.rh.name == "RH"
    # Temperature is the default primary channel.
    assert ft._primary.name == "Temperature"
    with pytest.raises(AttributeError):
        _ = ft.total_concentration


def test_nonparticle_summary_uses_selected_channel():
    """AltMixin summaries work on a non-particle channel without total_concentration."""
    import contextlib
    import io

    par = load_partector_file(_data_path("Sample_Partector.txt"))
    with contextlib.redirect_stdout(io.StringIO()):
        summary = par.summarize(parameter="LDSA")
    assert "Mean" in summary.columns
    assert (summary["N datapoints"] >= 0).all()


def test_discmini_ldsa_correction_scales_with_size():
    """The optional LDSA correction raises LDSA and grows with particle size."""
    import numpy as np

    from aerosoltools.loaders.Discmini import _ldsa_size_factor

    # Factor is ~1 for small particles and increases with size (bounded).
    assert _ldsa_size_factor(np.array([20.0]))[0] == pytest.approx(1.0, abs=0.01)
    assert (
        _ldsa_size_factor(np.array([130.0]))[0] > _ldsa_size_factor(np.array([50.0]))[0]
    )
    # Bounded against extrapolation well beyond the fitted range.
    assert _ldsa_size_factor(np.array([1000.0]))[0] <= 1.06

    raw = _data_path("Sample_Discmini_raw.txt")
    on = load_discmini_raw_file(raw, ldsa_correction=True)
    off = load_discmini_raw_file(raw, ldsa_correction=False)
    assert on._meta.get("ldsa_correction") is True
    assert off._meta.get("ldsa_correction") is False
    # The correction is a >= 1 multiplier, so corrected LDSA is never smaller.
    assert (on.data["LDSA"].to_numpy() >= off.data["LDSA"].to_numpy() - 1e-9).all()
    assert (on.data["LDSA"].to_numpy() > off.data["LDSA"].to_numpy()).any()


def test_discmini_processed_has_no_phantom_column_and_only_numeric_series():
    """The trailing-tab 'Unnamed' column is dropped and never offered as a series."""
    from aerosoltools.gui.helpers import plottable_columns

    dm = load_discmini_file(_data_path("Sample_Discmini.txt"), extra_data=True)
    # No phantom/empty column survives the loader.
    assert not any(str(c).startswith("Unnamed") for c in dm.extra_data.columns)
    # Extra currents are coerced to numbers, and the series picker offers them.
    labels = [label for label, _, _ in plottable_columns(dm)]
    assert any("Filter" in x for x in labels)

    # An all-NA column is never offered (would otherwise crash on float(NA)).
    dm.extra_data["Blank"] = pd.array([pd.NA] * len(dm.time), dtype="string")
    labels = [label for label, _, _ in plottable_columns(dm)]
    assert not any("Blank" in x for x in labels)


def test_discmini_raw_matches_processed_ldsa():
    """Raw-loader LDSA reproduces the vendor-processed LDSA (charger physics)."""
    import numpy as np

    # Compare the pure inversion (no zero-offset correction) against the vendor;
    # the offset correction is exercised separately.
    dm = load_discmini_raw_file(
        _data_path("Sample_Discmini_raw.txt"), zero_offset_correction=False
    )
    proc = pd.read_csv(
        _data_path("Sample_Discmini_raw_output.txt"),
        sep="\t",
        skiprows=6,
        header=None,
        decimal=",",
        engine="python",
        names=["Time", "Number", "Size", "LDSA", "Filter", "Diff", "z"],
    )
    proc_ldsa = pd.to_numeric(proc["LDSA"], errors="coerce").to_numpy()
    n = min(len(dm.data), len(proc_ldsa))
    mine = dm.data["LDSA"].to_numpy()[:n]
    ven = proc_ldsa[:n]
    rel = np.abs(mine - ven) / ven
    # LDSA = cal6 * total current reproduces the vendor value to < 1.5%.
    assert np.median(rel) < 0.015


def test_discmini_sniffer_distinguishes_raw_and_processed():
    """The content sniffer routes raw vs processed DiSCmini to the right loader."""
    from aerosoltools.gui.loaders import (
        identify_instrument,
        is_DiSCmini_file,
        is_DiSCmini_raw_file,
    )

    raw = _data_path("Sample_Discmini_raw.txt")
    processed = _data_path("Sample_Discmini.txt")

    assert is_DiSCmini_raw_file(raw) is True
    assert is_DiSCmini_raw_file(processed) is False
    assert is_DiSCmini_file(processed) is True
    assert identify_instrument(raw) == "DiSCmini (raw)"
    assert identify_instrument(processed) == "DiSCmini"


def test_aerosolalt_removed_from_public_api():
    """The legacy AerosolAlt catch-all is gone from the package namespace."""
    import aerosoltools as at

    assert not hasattr(at, "AerosolAlt")
    assert "AerosolAlt" not in at.__all__


def test_discmini_is_own_class_with_accessors():
    """DiSCmini loads as its own particle class with .size/.ldsa accessors."""
    from aerosoltools import Aerosol1D, DiSCmini

    dm = load_discmini_file(_data_path("Sample_Discmini.txt"))
    assert isinstance(dm, DiSCmini) and isinstance(dm, Aerosol1D)
    # Still a particle instrument: total_concentration works.
    assert float(dm.total_concentration.dropna().iloc[0]) > 0
    assert dm.size.name == "Size"
    assert dm.ldsa.name == "LDSA"


def test_dusttrak_class_accessors_and_primary():
    """DustTrak exposes PM accessors and its primary is the total PM mass."""
    from aerosoltools import DustTrak

    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame(
        {
            "Datetime": t,
            "PM1": [1.0] * 4,
            "PM2.5": [2.0] * 4,
            "PM4": [3.0] * 4,
            "PM10": [4.0] * 4,
            "Total": [5.0] * 4,
        }
    )
    d = DustTrak(df)
    # Primary / total_concentration is the "Total" channel, not the first column.
    assert float(d.total_concentration.iloc[0]) == 5.0
    assert float(d._primary.iloc[0]) == 5.0
    assert float(d.pm2_5.iloc[0]) == 2.0
    assert float(d.total.iloc[0]) == 5.0


def test_elpi_is_own_class_and_density_hook_recomputes_diameters():
    """ELPI loads as its own 2D subclass; set_density recomputes the size axis."""
    from aerosoltools import ELPI, Aerosol2D

    elpi = load_elpi_file(_data_path("Sample_ELPI.txt"))
    assert isinstance(elpi, ELPI) and isinstance(elpi, Aerosol2D)
    mids_before = [float(x) for x in elpi.bin_mids]
    elpi.set_density(2.0)
    mids_after = [float(x) for x in elpi.bin_mids]
    # The density-dependent diameters must move (the ELPI hook fired).
    assert mids_before != mids_after
    assert float(elpi.density) == 2.0


def test_metric_unit_registry_converts_within_dimension():
    """The unit registry classifies dimensions and converts within a dimension."""
    from aerosoltools._core.metrics import (
        canonical_unit,
        classify_unit,
        convert_value,
        unit_key,
    )

    assert classify_unit("ng/m³")[0] == "mass"
    assert classify_unit("cm⁻³")[0] == "number"
    # LaTeX and plain LDSA spellings collapse to one key.
    assert unit_key("nm$^{2}$/cm$^{3}$") == unit_key("nm²/cm³")
    # Scale conversions within a dimension.
    assert convert_value(1000.0, "ng/m³", "µg/m³") == pytest.approx(1.0)
    assert convert_value(2.0, "ppm", "ppb") == pytest.approx(2000.0)
    assert canonical_unit("ng/m³") == "µg/m³"
    # Cross-dimension conversion is a no-op (never fabricates numbers).
    assert convert_value(5.0, "cm⁻³", "µg/m³") == 5.0


def test_available_metrics_are_instrument_aware():
    """Each class advertises only the metrics it actually measures."""
    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    assert [m.key for m in cpc.available_metrics()] == ["PNC"]

    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    ops_keys = [m.key for m in ops.available_metrics()]
    assert "PNC" in ops_keys and "MASS" in ops_keys and "PM2.5" in ops_keys

    dm = load_discmini_file(_data_path("Sample_Discmini.txt"))
    assert {m.key for m in dm.available_metrics()} == {"PNC", "Size", "LDSA"}
    assert set(dm._default_metrics()) == {"PNC", "LDSA"}

    par = load_partector_file(_data_path("Sample_Partector.txt"))
    par_keys = {m.key for m in par.available_metrics()}
    assert "LDSA" in par_keys and "TEM" not in par_keys  # bool flag excluded
    assert par._default_metrics() == ["LDSA"]

    aeth = load_aethalometer_file(_data_path("Sample_Aetholometer.csv"))
    assert "IR BCc" in [m.key for m in aeth.available_metrics()]
    assert aeth._default_metrics() == ["IR BCc"]


def test_pnc_metric_is_rejected_for_non_number_instruments():
    """'PNC' works on number instruments and raises on gas/BC/LDSA (the leak fix)."""
    cpc = load_cpc_file(_data_path("Sample_CPC_Direct.txt"))
    ops = load_ops_file(_data_path("Sample_OPS.csv"))
    # No exception, and a number unit.
    assert cpc._get_metric_series("PNC")[1] == "cm⁻³"
    assert ops._get_metric_series("PNC")[1] == "cm⁻³"

    aeth = load_aethalometer_file(_data_path("Sample_Aetholometer.csv"))
    par = load_partector_file(_data_path("Sample_Partector.txt"))
    for obj in (aeth, par):
        with pytest.raises(ValueError):
            obj._get_metric_series("PNC")


def test_gas_metric_keyed_by_species():
    """A gas dataset advertises (and summarises) its species, not a generic key."""
    heads = load_ranger_file(_data_path("Ranger_2605-1000088-A_20260612-0603.csv"))
    gas = next(o for o in heads if type(o).__name__ == "Gas1D")
    keys = [m.key for m in gas.available_metrics()]
    assert keys == [str(gas.dtype)]  # e.g. "Cl₂"
    series, unit = gas._get_metric_series(str(gas.dtype))
    assert unit == gas.unit and len(series) == len(gas.time)
