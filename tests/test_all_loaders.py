import os
import pytest
import pandas as pd
from aerosoltools.loaders import (
    Load_ELPI_file,
    Load_CPC_file,
    Load_OPS_file,
    Load_NS_file,
    Load_DiSCmini_file,
    Load_Partector_file,
)

@pytest.mark.parametrize("loader_func, filename", [
    (Load_ELPI_file,       "sample_elpi.txt"),
    (Load_CPC_file,        "sample_cpc.txt"),
    (Load_OPS_file,        "sample_ops.txt"),
    (Load_NS_file,         "sample_ns.txt"),
    (Load_DiSCmini_file,   "sample_discmini.txt"),
    (Load_Partector_file,  "sample_partector.txt"),
])
def test_loader_smoke(loader_func, filename):
    test_file = os.path.join(os.path.dirname(__file__), "data", filename)
    assert os.path.exists(test_file), f"Missing test file: {filename}"

    data = loader_func(test_file)
    assert data is not None
    assert hasattr(data, "data"), f"{filename}: missing 'data'"
    assert hasattr(data, "metadata"), f"{filename}: missing 'metadata'"
    assert isinstance(data.data, pd.DataFrame), f"{filename}: data is not DataFrame"
