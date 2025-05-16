import matplotlib.pyplot as plt
import pandas as pd
from aerosoltools.loaders import Load_ELPI_file
import os

def test_full_elpi_pipeline_with_plotting():
    

    test_file = os.path.join(os.path.dirname(__file__), "data", "sample_elpi.txt")
    data = Load_ELPI_file(test_file)

    activity_periods = {
        "Background": [("2023/09/07 09:06:50", "2023/09/07 09:07:50")],
        "Emission":   [("2023/09/07 09:07:55", "2023/09/07 09:08:30")],
        "Decay":      [("2023/09/07 09:09:00", "2023/09/07 09:10:50")]
    }

    data.mark_activities(activity_periods)
      
    data.convert_to_volume_concentration()
    # Check dtype or column name contains 'dV'
    assert "dV" in str(data.dtype), f"Expected 'dV' in dtype, got: {data.dtype}"
    
    # Check unit
    assert getattr(data, "unit", None) == "nm³/cm³", f"Unexpected unit: {data.unit}"

    fig, axs = plt.subplots(ncols=2, nrows=2)
    data.plot_psd(["Decay", "Emission"], ax=axs[0, 0], normalize=False)
    data.plot_psd(["All data"], ax=axs[0, 0])
    data.plot_psd(normalize=True, ax=axs[1, 0])

    data.normalize_logdp()
    # Check dtype for "dlogDp"
    assert "dlogDp" in str(data.dtype), f"Expected 'dlogDp' in dtype, got: {data.dtype}"
  
    data.plot_timeseries(y_3d=(1, 0), mark_activities=True, ax1=axs[0, 1], ax2=axs[1, 1])

    summary_table = data.summarize()
    
    # Must be a DataFrame
    assert isinstance(summary_table, pd.DataFrame), "summary_table is not a DataFrame"
    
    # Must have a 'Segment' column
    assert "Segment" in summary_table.columns, "'Segment' column missing from summary_table"
    
    # Must include expected segment names
    expected_segments = {"All data", "Background", "Emission", "Decay"}
    found_segments = set(summary_table["Segment"])
    missing = expected_segments - found_segments
    assert not missing, f"Missing expected segments: {missing}"
  
