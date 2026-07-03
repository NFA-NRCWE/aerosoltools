import os

import pandas as pd
import pytest

from aerosoltools.loaders import (
    Load_CPC_file,
    Load_DiSCmini_file,
    Load_DiSCmini_raw_file,
    Load_ELPI_file,
    Load_FMPS_file,
    Load_Fourtec_file,
    Load_Grimm_file,
    Load_NS_file,
    Load_OPCN3_file,
    Load_OPS_file,
    Load_Partector_file,
    Load_SMPS_file,
)
from aerosoltools.loaders.Discmini import (
    _extract_serial_and_firmware,
    _normalize_serial,
)


@pytest.mark.parametrize(
    "loader_func, filename",
    [
        (Load_CPC_file, "Sample_CPC_Direct.txt"),
        (Load_DiSCmini_file, "Sample_Discmini.txt"),
        (Load_ELPI_file, "Sample_ELPI.txt"),
        (Load_ELPI_file, "Sample_ELPI2.txt"),
        (Load_FMPS_file, "Sample_FMPS.txt"),
        (Load_FMPS_file, "Sample_FMPS2.txt"),
        (Load_Fourtec_file, "Sample_Fourtec.xlsx"),
        (Load_Grimm_file, "Sample_Grimm.txt"),
        (Load_NS_file, "Sample_NS.csv"),
        (Load_OPCN3_file, "Sample_OPCN3.txt"),
        (Load_OPS_file, "Sample_OPS.csv"),
        (Load_OPS_file, "Sample_OPS2.txt"),
        (Load_Partector_file, "Sample_Partector.txt"),
        (Load_SMPS_file, "Sample_SMPS.txt"),
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
    dm = Load_DiSCmini_raw_file(str(raw), extra_data=True, period=10)

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

    dm = Load_DiSCmini_raw_file(str(raw), period=10)
    # Windows 0, 1, 3 kept (window 2 has only 5 valid rows -> dropped).
    assert dm.data.shape[0] == 3
    elapsed = (dm.time - dm.time[0]).total_seconds().to_numpy()
    # Centres at 4.5, 14.5, 34.5 s — the 24.5 window is skipped (70 s jump).
    assert np.allclose(elapsed, [0.0, 10.0, 30.0])


def _data_path(name):
    return os.path.join(os.path.dirname(__file__), "data", name)


def test_discmini_raw_loader_on_sample_file():
    """Load a real (truncated) raw DiSCmini file and sanity-check the output."""
    dm = Load_DiSCmini_raw_file(_data_path("Sample_Discmini_raw.txt"), extra_data=True)
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

    dm = Load_DiSCmini_raw_file(_data_path("Sample_Discmini_raw.txt"))
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


def test_discmini_raw_matches_processed_ldsa():
    """Raw-loader LDSA reproduces the vendor-processed LDSA (charger physics)."""
    import numpy as np

    dm = Load_DiSCmini_raw_file(_data_path("Sample_Discmini_raw.txt"))
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
