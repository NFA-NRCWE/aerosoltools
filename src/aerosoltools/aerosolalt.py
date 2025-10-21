import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tabulate import tabulate
from .aerosol1d import Aerosol1D

params = {
    "legend.fontsize": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": (19, 10),
}
plt.rcParams.update(params)


class AerosolAlt(Aerosol1D):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        
        
    
    ###########################################################################
    """############################# Functions #############################"""
    ###########################################################################

    def plot_total_conc(self, parameter=0, ax=None, mark_activities=False):
        """
        Plot the total concentration over time.
    
        Parameters
        ----------
        parameter : int, optional
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
        # Determine the relevenat data based on the chosen parameter
        if type(parameter)==int:
            if parameter>=len(self._raw_data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter=self.data.columns[parameter]
        elif type(parameter)==str:
            pass
        else:
            raise LookupError("Chosen parameter is invalid") 
        
        new_fig_created = False
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_fig_created = True
        else:
            fig = ax.figure
    
        # Plot main data
        ax.plot(self.time, self.data[parameter], linestyle="-")
    
        # Format x-axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    
        ax.set_xlabel("Time")
        
        if type(self.dtype)==str:
            Dtype=self.dtype
        else:  Dtype=self.dtype[parameter]
        
        if type(self.unit)==str:
            Unit=self.unit
        else:  Unit=self.unit[parameter]
        
        if "/" in Dtype:
            total_conc_dtype = Dtype.split("/")[0]
            ax.set_ylabel(f"{total_conc_dtype}, {Unit}")
        else:
            ax.set_ylabel(f"{Dtype}, {Unit}")
        ax.grid(True)
    
        # Highlight activities
        if mark_activities and hasattr(self, "_activity_periods"):
            print("Hello")
            # Exclude "All data" unless explicitly requested
            all_activities = sorted(self._activity_periods.keys())
            color_map = plt.colormaps.get_cmap("gist_ncar")
            activity_colors = {
                activity: color_map(i / max(1, len(all_activities)))
                for i, activity in enumerate(all_activities)
            }
    
            if mark_activities is True:
                selected_activities = [a for a in all_activities if a != "All data"]
            elif isinstance(mark_activities, list):
                selected_activities = [
                    a for a in mark_activities if a in self._activity_periods
                ]
            else:
                selected_activities = []
    
            for activity in selected_activities:
                color = activity_colors[activity]
                first = True
                for start, end in self._activity_periods[activity]:
                    ax.axvspan(
                        pd.Timestamp(start),
                        pd.Timestamp(end),
                        color=color,
                        alpha=0.3,
                        label=activity if first else None,
                        zorder=3,
                    )
                    first = False
            # Clip x-axis to actual data range
            ax.set_xlim(self.time.min(), self.time.max())
            ax.legend()
    
        if new_fig_created:
            fig.tight_layout()
    
        return fig, ax
###########################################################################

    def summarize(self, parameter=0, filename=None):
        """
        Summarize total concentration statistics for each defined activity,
        including 'All data'.
    
        Parameters
        ----------
        filename : str, optional
            Path to an Excel file where the summary will be saved. If None, no file is saved.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing summary statistics.
        """
        # Determine the relevenat data based on the chosen parameter
        if type(parameter)==int:
            if parameter>=len(self._raw_data.columns):
                raise LookupError("Chosen parameter is invalid")
            parameter=self.data.columns[parameter]
        elif type(parameter)==str:
            pass
        else:
            raise LookupError("Chosen parameter is invalid") 
            
            
        rows = []
    
        # Loop through all activities (including "All data")
        for activity in self.activities:
            try:
                subset = self.data[self.data[activity]][self.total_concentration[parameter].name]
            except KeyError:
                subset = self.data[self.data[activity]][parameter]
    
            if not subset.empty:
                rows.append(
                    [
                        activity,
                        subset.min(),
                        subset.max(),
                        subset.mean(),
                        subset.std(),
                        len(subset),
                    ]
                )
    
        # Create DataFrame
        summary = pd.DataFrame(
            rows, columns=["Segment", "Min", "Max", "Mean", "Std", "N datapoints"]
        )
        summary_rounded = summary.round(3)
    
        # Console output
        print("\nSummary of total concentration:\n")
        print(
            tabulate(summary_rounded, headers="keys", tablefmt="pretty", floatfmt=".3f")
        )
    
        # Optionally save
        if filename:
            summary_rounded.to_excel(filename, index=False)
            print(f"\nSummary saved to: {filename}")
    
        return summary_rounded
