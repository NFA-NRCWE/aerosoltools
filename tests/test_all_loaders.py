import os

import pandas as pd
import pytest

from aerosoltools.loaders import (
    Load_CPC_file,
    Load_DiSCmini_file,
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
