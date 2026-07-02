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
    assert dm._meta.get("size_inversion") == "provisional"

    # LDSA = cal[6] * (Diff + Filter) = 0.68 * (10 + 35) = 30.6 (exact here).
    assert dm.data["LDSA"].iloc[0] == pytest.approx(30.6, abs=1e-6)
    # Idle rows would have dragged the currents toward zero if not excluded.
    assert dm.extra_data["Diffusion"].iloc[0] == pytest.approx(10.0, abs=1e-6)
    assert dm.total_concentration.iloc[0] > 0
