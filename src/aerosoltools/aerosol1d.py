# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:41:09 2025

@author: B279683
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Union
import copy
from tabulate import tabulate
params = {'legend.fontsize': 15,
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 13,
         'ytick.labelsize': 13,
         'figure.figsize' : (19, 10)}
plt.rcParams.update(params)

# aerosol1d class definition
class Aerosol1D:
    """
     Class for handling 1D aerosol time series measurements.
    
     This class manages timeseries data (e.g., total particle concentration) with a time index.
     It provides utilities for resampling, smoothing, segment marking, and plotting.
    
     Parameters
     ----------
     dataframe : pandas.DataFrame
         A DataFrame containing time-indexed aerosol data. If the index is not a DatetimeIndex,
         the first column will be treated as timestamps and set as the index automatically.
    
     Main Properties
     ---------------
     activities : list of str
         List of activity names that have been marked in the dataset.
     dtype : str
         Data type description of the measurements (e.g., "dN" for particle number concentration).
     instrument : str
         Name or description of the instrument used for the measurements.
     original_data : pandas.DataFrame
         Copy of the raw, original input data before any processing.
     time : pandas.DatetimeIndex
         Time index of the measurements.
     total_concentration : pandas.Series
         Total aerosol concentration over time (assumed to be the first data column).
     unit : str
         Unit of measurement (e.g., "cm$^{-3}$", "µg/m³").
    
     Notes
     -----
     Internal attributes (such as raw data storage) are managed automatically and are not
     intended for direct modification. Users should interact with the object via its public
     properties and methods.
     """
    def __init__(self, dataframe):
        self._meta = {}
        self._extra_data = pd.DataFrame([])
        self._activities = []
        self._activity_periods = {}

        # Automatically handle timestamp column
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            timestamp_col = dataframe.columns[0]
            dataframe[timestamp_col] = pd.to_datetime(dataframe[timestamp_col])
            dataframe.set_index(timestamp_col, inplace=True)
            
        # Ensure there is a meaningful column name
        if dataframe.columns[0] is None or dataframe.columns[0] == 0:
            dataframe.columns = ["Total Concentration"]
            
        self._data = dataframe
        self._raw_data = dataframe.copy()
        self._data["All data"] = True
        self._activities.append("All data")
        self._activity_periods["All data"] = [(self.time.min(), self.time.max())]

    ###########################################################################
    """############################ Properties #############################"""
    ###########################################################################
    
    @property
    def activities(self):
        """
        List of defined activity labels.
    
        Returns
        -------
        list of str
            List of activity names that have been marked in the dataset.
        """
        return self._activities
    
    @property
    def activity_periods(self):
        """
        Dictionary of activity names and their associated time periods.
    
        Returns
        -------
        dict
            Dictionary where keys are activity names and values are lists of (start, end) tuples.
        """
        return self._activity_periods
    
    @property
    def data(self):
        """
        Dataframe with all data, times, and activity columns if marked.
    
        Returns
        -------
        pd.DataFrame
            The full DataFrame.
        """
        return self._data
    
    @property
    def dtype(self):
        """
        Data type description of the measurements.
    
        Returns
        -------
        str
            The type of data (e.g., particle number concentration, mass concentration).
        """
        return self._meta.get("dtype","Uknown dtype")
    @property
    def extra_data(self):
        """
        List of defined activity labels.
    
        Returns
        -------
        list of str
            List of activity names that have been marked in the dataset.
        """
        return self._extra_data
    
    @property
    def instrument(self):
        """
        Instrument used for the measurements.
    
        Returns
        -------
        str
            Name or description of the instrument.
        """
        return self._meta.get("instrument","Uknown instrument")
    
    @property
    def metadata(self):
        """
        Return overiew of meta data

        Returns
        -------
        meta : Dictionary
            Contains; unit, data type, instrument type and serial_number.

        """
        return self._meta
    
    @property
    def original_data(self):
        """
        Unmodified original dataset.
    
        Returns
        -------
        pandas.DataFrame
            Copy of the raw, original data before any processing.
        """
        return self._raw_data
    
    @property
    def serial_number(self):
        """
        Serial number of instrument
    
        Returns
        -------
        str
            
        """
        return self._meta.get("serial_number","Uknown serial number")
    
    @property
    def time(self):
        """
        Timestamps of the dataset.
    
        Returns
        -------
        pandas.DatetimeIndex
            Time index of the measurements.
        """
        return self._data.index
    
    @property
    def total_concentration(self):
        """
        Total concentration measurements.
    
        Returns
        -------
        pandas.Series
            Total concentration data over time.
        """
        if "Total Concentration" in self._data.columns:
            return self._data["Total Concentration"]
        else:
            # Fall back in case something weird happens
            return self._data.iloc[:, 0]
    
    @property
    def unit(self):
        """
        Unit of the measurements.
    
        Returns
        -------
        str
            Unit string (e.g., "#/cm³", "µg/m³").
        """
        return self._meta.get("unit","Uknown unit")
    

    ###########################################################################
    """############################# Functions #############################"""
    ###########################################################################
    
    def copy_self(self):
        """
        Create a deep copy of the current aerosol1d object.

        Returns
        -------
        aerosol1d
            A deep copy of the current instance.
        """
        return copy.deepcopy(self)

    ###########################################################################
        
    def get_activity_data(self, activity_name):
        """
        Extract data corresponding to a specified activity.
    
        Parameters
        ----------
        activity_name : str
            Name of the activity to extract.
    
        Returns
        -------
        pandas.DataFrame
            Subset of the data where the specified activity is active (True).
        """
        if activity_name not in self.activities:
            raise ValueError(f"Activity '{activity_name}' not found in available activities: {self.activities}")
    
        return self._data[self._data[activity_name]].drop(columns=self.activities)

    ###########################################################################
    
    def mark_activities(self, activity_periods):
        """
        Mark activities in the data by adding one boolean column per activity.
    
        Parameters
        ----------
        activity_periods : dict
            Dictionary where keys are activity names (str) and values are 
            (start, end) tuples or list of (start, end) tuples.
    
        Returns
        -------
        None
        """
        for activity, periods in activity_periods.items():
            if activity not in self._data.columns:
                self._data[activity] = False
    
            if isinstance(periods, tuple) and len(periods) == 2:
                periods = [periods]
    
            for start, end in periods:
                mask = (self.time >= pd.Timestamp(start)) & (self.time <= pd.Timestamp(end))
                self._data.loc[mask, activity] = True
    
            # Track activities
            if activity not in self._activities:
                self._activities.append(activity)
    
            # Save periods
            self._activity_periods[activity] = periods
    
    ###########################################################################
            
    def plot_total_conc(self, ax=None, mark_activities=False):
        """
        Plot the total concentration over time.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing Matplotlib Axes object. If None, a new figure and axes are created.
        mark_activities : bool or list of str, optional
            If True, highlights all activity periods **except "All data"**.
            If a list of activity names is provided, only those will be highlighted.
            If False (default), no activities are marked.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object.
        ax : matplotlib.axes.Axes
            The Matplotlib axes object with the plot.
        """
        new_fig_created = False
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_fig_created = True
        else:
            fig = ax.figure
    
        # Plot main data
        ax.plot(self.time, self.total_concentration, linestyle='-')
    
        # Format x-axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    
        ax.set_xlabel("Time")
        if "/" in self.dtype:
            total_conc_dtype = self.dtype.split("/")[0]
            ax.set_ylabel(f"{total_conc_dtype}, {self.unit}")
        else:
            ax.set_ylabel(f"{self.dtype}, {self.unit}")
        ax.grid(True)
    
        # Highlight activities
        if mark_activities and hasattr(self, "_activity_periods"):
            # Exclude "All data" unless explicitly requested
            all_activities = sorted(self._activity_periods.keys())
            color_map = plt.colormaps.get_cmap('gist_ncar')
            activity_colors = {
                activity: color_map(i / max(1, len(all_activities)))
                for i, activity in enumerate(all_activities)
            }
        
            if mark_activities is True:
                selected_activities = [a for a in all_activities if a != "All data"]
            elif isinstance(mark_activities, list):
                selected_activities = [a for a in mark_activities if a in self._activity_periods]
            else:
                selected_activities = []

            for activity in selected_activities:
                color = activity_colors[activity]
                for start, end in self._activity_periods[activity]:
                    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                               color=color, alpha=0.3, label=activity)
            # Clip x-axis to actual data range
            ax.set_xlim(self.time.min(), self.time.max())
            
            # Remove duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys())
    
        if new_fig_created:
            fig.tight_layout()
    
        return fig, ax

    ###########################################################################
    
    def summarize(self, filename=None):
        """
        Summarize total concentration statistics for each defined activity, including 'All data'.
    
        Parameters
        ----------
        filename : str, optional
            Path to an Excel file where the summary will be saved. If None, no file is saved.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing summary statistics.
        """
        rows = []
    
        # Loop through all activities (including "All data")
        for activity in self.activities:
            try:
                subset = self.data[self.data[activity]][self.total_concentration.name]
            except KeyError:
                subset = self.data[self.data[activity]].iloc[:, 0]
    
            if not subset.empty:
                rows.append([
                    activity,
                    subset.min(),
                    subset.max(),
                    subset.mean(),
                    subset.std(),
                    len(subset)
                ])
    
        # Create DataFrame
        summary = pd.DataFrame(rows, columns=["Segment", "Min", "Max", "Mean", "Std", "N datapoints"])
        summary_rounded = summary.round(3)
    
        # Console output
        print("\nSummary of total concentration:\n")
        print(tabulate(summary_rounded, headers="keys", tablefmt="pretty", floatfmt=".3f"))
    
        # Optionally save
        if filename:
            summary_rounded.to_excel(filename, index=False)
            print(f"\nSummary saved to: {filename}")
    
        return summary_rounded

    ###########################################################################
    
    def timecrop(self, start: Optional[Union[str, pd.Timestamp]] = None,
                 end: Optional[Union[str, pd.Timestamp]] = None,
                 inplace: bool = True) -> "aerosol1d":
        """
        Crop the data to a specified time window.
    
        Parameters
        ----------
        start : str or pd.Timestamp, optional
            Start time. If None, cropping starts from the earliest available time.
            If a string, it should have the format "YYYY,MM,DD HH:MM:SS", e.g., "2025,01,04 20:00:00".
        end : str or pd.Timestamp, optional
            End time. If None, cropping ends at the latest available time.
            If a string, it should have the format "YYYY,MM,DD HH:MM:SS", e.g., "2025,01,04 20:00:00".
        inplace : bool, optional
            If True, modifies the current object. If False, returns a new cropped instance.
            Defaults to True.
    
        Returns
        -------
        aerosol1d
            Instance of `aerosol1d`. Either the modified current object (if inplace=True) or a new cropped object (if inplace=False).
        """
        mask = pd.Series(True, index=self.time)
    
        if start is not None:
            start = pd.to_datetime(start)
            mask &= self.time >= start
        if end is not None:
            end = pd.to_datetime(end)
            mask &= self.time <= end
    
        if inplace:
            self._data = self._data.loc[mask]
            return self
        else:
            cropped = self.copy_self()
            cropped._data = self._data.loc[mask]
            return cropped
    
    ###########################################################################
    
    def timerebin(self, freq: str = "s", method: str = "mean", inplace: bool = True):
        """
        Resample the data to a new time frequency using an aggregation function.
    
        Parameters
        ----------
        freq : str, optional
            Resampling frequency. Naming convention is 's', 'min', or 'h' for seconds, minutes and hours
            but these can be combined with integers e.g., '30S', '5min', or '1H'. Default is 's'.
        method : str or function, optional
            Aggregation method to apply e.g., 'mean', 'median', 'sum', 'min', 'max', or a custom function. Default is 'mean'.
        inplace : bool, optional
            If True, modifies the object in place. If False, returns a new rebinned object. Default is True.
    
        Returns
        -------
        aerosol1d
            Instance of `aerosol1d` with rebinned time index.
        """
        numeric_cols = self._data.select_dtypes(exclude="bool").columns
        bool_cols = self._data.select_dtypes(include="bool").columns
    
        rebinned_numeric = self._data[numeric_cols].resample(freq).agg(method)
        rebinned_bool = self._data[bool_cols].resample(freq).max().astype(bool)
    
        rebinned = pd.concat([rebinned_numeric, rebinned_bool], axis=1)
    
        if inplace:
            self._data = rebinned
            return self
        else:
            rebinned_obj = self.copy_self()
            rebinned_obj._data = rebinned
            return rebinned_obj
        
    ###########################################################################  
    
    def timeshift(self, seconds: float = 0, minutes: float = 0, hours: float = 0, inplace: bool = True):
        """
        Shift the time index by a given number of seconds and/or minutes.
    
        Parameters
        ----------
        seconds : float, optional
            Number of seconds to shift. Defaults to 0.
        minutes : float, optional
            Number of minutes to shift. Defaults to 0.
        inplace : bool, optional
            Whether to modify the object in place. Defaults to True.
    
        Returns
        -------
        aerosol1d
            Instance of `aerosol1d` with shifted time index. Either the modified current object (if inplace=True) or a new shifted object (if inplace=False).
        """
        total_seconds = seconds + 60 * minutes + 3600 * hours
        total_shift = pd.to_timedelta(total_seconds, unit='s')
    
        if inplace:
            self._data.index = self._data.index + total_shift
            return self
        else:
            shifted = self.copy_self()
            shifted._data.index = shifted._data.index + total_shift
            return shifted
        
    ###########################################################################
    
    def timesmooth(self, window: int = 5, method: str = "mean", inplace: bool = True):
        """
        Apply rolling window smoothing to the data.
    
        Parameters
        ----------
        window : int, optional
            Size of the moving window (in number of samples). Default is 5.
        method : str, optional
            Aggregation method to use: 'mean', 'median', 'sum', 'min', or 'max'. Default is 'mean'.
        inplace : bool, optional
            If True, modifies the current object. If False, returns a new smoothed instance. Default is True.
    
        Returns
        -------
        aerosol1d
            Instance of `aerosol1d` with smoothed data.
        """
        if method not in ["mean", "median", "sum", "min", "max"]:
            raise ValueError("Invalid method. Choose from 'mean', 'median', 'sum', 'min', 'max'.")
    
        numeric_cols = self._data.select_dtypes(exclude="bool").columns
        bool_cols = self._data.select_dtypes(include="bool").columns
    
        smoothed_numeric = getattr(self._data[numeric_cols].rolling(window=window, center=True, min_periods=1), method)()
        preserved_bool = self._data[bool_cols]  # unchanged
    
        smoothed = pd.concat([smoothed_numeric, preserved_bool], axis=1)
    
        if inplace:
            self._data = smoothed
            return self
        else:
            new_obj = self.copy_self()
            new_obj._data = smoothed
            return new_obj