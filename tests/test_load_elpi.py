import os
from aerosoltools.loaders import Load_ELPI_file

def test_load_elpi_smoke():
    # Path to the test data file
    test_file = os.path.join(os.path.dirname(__file__), "data", "sample_elpi.txt")

    # Check file exists
    assert os.path.exists(test_file), "Test ELPI file is missing."

    # Run the loader
    data = Load_ELPI_file(test_file)

    # Basic assertions — adjust to match real object structure
    assert data is not None
    assert hasattr(data, "data"), "Missing 'data' attribute"
    assert hasattr(data, "metadata"), "Missing 'metadata' attribute"
    assert isinstance(data.data, pd.DataFrame)
    assert isinstance(data.metadata, dict)
