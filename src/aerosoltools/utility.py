# -*- coding: utf-8 -*-

import os
from collections import Counter
from typing import List, Union

import pandas as pd
import numpy as np
import datetime as datetime

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import  theilslopes
from scipy.optimize import curve_fit


from ..aerosol1d import Aerosol1D
from ..aerosol2d import Aerosol2D
from ..aerosolalt import AerosolAlt

###############################################################################


def Combine_NS_OPS(NS_raw,OPS_raw,start=None,end=None):
    """
    Function to combine Nanoscan and OPS data. If no start or end time are specified
    then the function will use the first and last datapoints that exist for both
    instruments. 
    It is assumed that the NS and OPS use their standard sizebins, meaning that
    the size bin edges are:
        
    NS: [ 10.  ,  13.45,  17.95,  23.95,  31.95,  42.6 ,  56.8 ,  75.75,
           101.05, 134.75, 179.7 , 239.6 , 319.5 , 420.  ]
    
    OPS: [  300.,   374.,   465.,   579.,   721.,   897.,  1117.,  1391.,
            1732.,  2156.,  2685.,  3343.,  4162.,  5182.,  6451.,  8031.,
           10000.]
    
    Parameters
    ----------
    NS : Aerosol2D
        An Aerosol2D data set of a Nanoscan.
    OPS : Aerosol2D
        An Aerosol2D data set of an OPS.
    start : datetime, optional
        The starting time for combining the two datasets. If not speicifed, a 
        starting point is found as earliest data point of the two datasets. The default is None.
    end : datetime, optional
        The end time for combining the two datasets. If not speicifed, an end
        point is found as earliest data point of the two datasets. The default is None.

    Returns
    -------
    Combined_NS_OPS : numpy.array
        An array of the combined NS and OPS datasets, with newly calculated total
        concentrations. The last bin of the NS has been ignored and the second
        to last has been shortened and its number reduced accordingly.
    New_bin_edges : numpy.array
        Array of new size bin edges in nm.
    Header : list
        Header of all data columns

    """
    
    # Initialize combination by confirming the data type of the two data sets.
    NS=NS_raw.copy_self()
    OPS=OPS_raw.copy_self()
    NS.convert_to_number_concentration()
    NS.unnormalize_logdp()
    OPS.convert_to_number_concentration()
    OPS.unnormalize_logdp()
    
    # Round all the datetime values, so they do not have seconds, in order to ease the alignment process

    NS_time_delta   = NS.time[1]-NS.time[0]
    OPS_time_delta  = OPS.time[1]-OPS.time[0]
    # Determine the slowest frequency for the purpose of aligning the data.
    freq=max(NS_time_delta,OPS_time_delta)
    freq=f"{freq.seconds}s"
    
    # Determine the start time for combining the data either from the specified time
    # or from the first datapoint which exist for both instruments
    if start:
        start = start.replace(second=0)
    else:
        start = min(OPS.time[0],NS.time[0])
    
    # Determine the end time for combining the data either from the specified time
    # or from the last datapoint which exist for both instruments
    if end:
        end = end.replace(second=0)
    else:
        end = max(OPS.time[-1],NS.time[-1])
    
    # Grab the NS data within the specifed start and end times
    NS_time_matched = NS.timerebin(freq,start,end,inplace=False)
    OPS_time_matched = OPS.timerebin(freq,start,end,inplace=False)

    NS_bins    = NS.bin_edges
    OPS_bins   = OPS.bin_edges
    
    # the penultimate NS sizebin has its upper limit reduced from 319.5 to 300, 
    # so the particle number of the relevent bin is reduced by a fraction equal to 
    # the fraction the difference between the previous bin to 300 divided by the 
    # original difference between the bin-edges.
    for i in range(0,len(NS_bins)-1):
        if NS_bins[i+1]>OPS_bins[0]:
            Factor_reduction = (OPS_bins[0]-NS_bins[i])/(NS_bins[i+1]-NS_bins[i]) 
            break
    #Create the combined list of NS_OPS
    NS_OPS_bins =np.append(NS_bins[:i+1],OPS_bins)
    bin_mids    =  np.round(np.array((NS_OPS_bins[:-1] + NS_OPS_bins[1:])/2,dtype=float),1)
    bin_mids[:i]= NS.bin_mids[:i]
    
    # Select the relevant columns
    NS_data     = NS_time_matched.data.drop(columns=NS_time_matched.data.columns[i+2:])
    OPS_data    = OPS_time_matched.data.drop(columns=['Total_conc'])
    NS_columns  = NS_data.columns
    
    NS_data[NS_columns[-1]] = NS_data[NS_columns[-1]].astype("float")*Factor_reduction
   
    # Merge on 'datetime' column, using an outer join to keep all datetime values
    Combined_NS_OPS = pd.concat([NS_data, OPS_data],axis=1)
    NS_OPS_columns = Combined_NS_OPS.columns
    #Define a mask based on where either OPS or NS is off for the purpose of total conc.
    mask=(NS_data[NS_columns[1:]].notna().any(axis=1) ) & (OPS_data[OPS._sizebin_headers].notna().any(axis=1) )
    # Combined_NS_OPS.rename(columns={NS_OPS_columns[1:len(bin_mids)+2]: bin_mids}, inplace=True)
    for s in range(0,len(bin_mids)):
        Combined_NS_OPS.rename( columns={NS_OPS_columns[1:][s]:bin_mids[s].astype(str)},inplace=True)
        
    # Combined_NS_OPS.rename(columns={NS_columns[-1]: bin_mids[i]}, inplace=True)
    NS_OPS_columns = Combined_NS_OPS.columns

    # A new total concentration is calculated based on the combined size bin data
    Combined_NS_OPS['Total_conc']=Combined_NS_OPS[NS_OPS_columns[1:len(NS_OPS_bins)]].sum(axis=1).where(mask, np.nan)
    # Combined_NS_OPS[:,1] = np.round(Combined_NS_OPS[NS_OPS_columns[1:len(NS_OPS)].sum(axis=1).astype("float"),0)
    
    NS_OPS = Aerosol2D(Combined_NS_OPS)
    NS_OPS._activities= NS.activities
    NS_OPS._activity_periods= NS.activity_periods
    NS_OPS._meta["bin_edges"] = NS_OPS_bins
    NS_OPS._meta["bin_mids"] = bin_mids
    NS_OPS._meta["density"] = NS._meta["density"]
    NS_OPS._meta["instrument"] = "NS_OPS"
    NS_OPS._meta["serial_number"] = f"NS: {NS.serial_number},'OPS': {OPS.serial_number}"
    NS_OPS._meta["unit"] = 'cm⁻³'
    NS_OPS._meta["dtype"] = 'dN'
    NS_OPS._raw_extra_data =  pd.concat([NS._raw_extra_data, OPS._raw_extra_data],axis=1)
    return NS_OPS

###############################################################################

def Plot_correlation(X, Y, ax_in=False, intercept=True, uniform_scaling=True, outlier_influence=True):
    """
    Function to plot the correlation between two sets of values, which have been
    aligned so as to have sensible comparison points. 
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions. 
       
    Parameters
    ----------
    X: Numpy.array
        First set of values. 
    Y: Numpy.array
        Second set of values.  
    ax_in : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.
        Usefull for plotting multiple correlations in the same figure.
        
        Example:
        '''
        df: dataframe of values from different instruments with associated label
        instruments: list of instruments used for comparison
        
        fig, axes = plt.subplots(len(instruments)-1, len(instruments)-1)

        # Fill the grid with custom scatter plots, or leave empty for unwanted pairs
        for i in range(len(instruments)-1):  # Loop over instruments 0:-1 for the horizontal axis
            for j in range(i,len(instruments)-1):  # Loop over instruments 1: for the vertical axis
                if i == j+1:  # Skip diagonal (Instrument 2 vs Instrument 2, etc.)
                    axes[j, i].axis('off')
                else:  # Use custom scatter plot function
                    _, _ = UL.cor_plot(df[instruments[i]], df[instruments[j+1]], axes[j, i],intercept=False)
                    if i==0:
                        axes[j, i].set_ylabel(instruments[j+1])
                    if j==len(instruments)-2:
                        axes[j, i].set_xlabel(instruments[i])
         '''                 
    uniform_scaling: boolean, optional
        Boolean that can be turned off so as to not scale axis the max value.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.
        """
        
    #Defining relevant sub-functions for the function to work
    def linear_func(x,A,B=0):
        #Calculates a first order equation.
        return A*x + B
    
    def R2(data,fit):
        # residual sum of squares
        ss_res = np.sum((data - fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        # r-squared
        return round((1 - (ss_res / ss_tot)),3)
    
    #Cleaning up the data and removing rows where either value is nan
    z=np.column_stack((X.copy(),Y.copy())).astype('float64')
    z=z[~np.isnan(z).any(axis=1)]
    z=z[~np.isinf(z).any(axis=1)]
    x=z[:,0]
    y=z[:,1]
    
    if type(x[0])==datetime.datetime:
        x=np.array(mdates.date2num(x[:]))
    
    # This method works well but it's sensitive towards outliers
    if outlier_influence:
        #Apply the fit using curve_fit for a function with or without an intercept.
        if intercept==True:
            parameters, covariance =curve_fit(linear_func,x,y,p0=[1, 1])
            A, B = parameters
            SE = np.sqrt(np.diag(covariance))
            SE_A , SE_B = SE
        
        else:
            parameters, covarience =curve_fit(linear_func,x,y,p0=[1])
            A=parameters[0]
            SE_A=covarience[0][0]
            B=0
            SE_B=0
    else:
        # This method is less sensitive towards 
        A, B, _, _ = theilslopes(y,x)
        
    #Generate the fit and calculate the R^2 value
    fit=linear_func(x,A,B)
    r2=R2(y,fit)

    if uniform_scaling==True:
        factor=max(max(abs(x)),max(abs(y)))
    else: 
        factor=1
    if min(x)<=0:
        x_min=min(x)/factor
    else:
        x_min=0
    if max(x)<=0:
        x_max=0
    else: 
        x_max= z.max()/factor
    
    fit_x=np.linspace(x_min,x_max,20)
    fit_y=fit_x*A+B/factor
    if B==0:
        label=f"y={round(A,2)}$\cdot$x"#"y={:.2f}$\cdot$x+{:.2f}, R$^2$={:.2f}"
    elif B>0:
        label=f"y={round(A,2)}$\cdot$x \n + {round(B,2)}"#"y={:.2f}$\cdot$x+{:.2f}, R$^2$={:.2f}"
        
    else:
        label=f"y={round(A,2)}$\cdot$x \n {round(B,2)}"#"y={:.2f}$\cdot$x+{:.2f}, R$^2$={
    #If no ax is provided, figure and ax is generated here
    if ax_in==False:
        figure, ax = plt.subplots()
        plt.xticks(fontsize=25)  
        plt.yticks(fontsize=25) 
        ax.legend(fontsize=25)
        ax.grid(True)
    else:
        ax = ax_in
    #Plot the 1:1 line
    ax.plot([x_min,x_max],[x_min,x_max],ls="--",c="k",lw=3)
    #Plot the data with scatter plot
    ax.plot(x/factor,y/factor,'bo')
    #Plot the fit with associated uncertainty
    ax.plot(fit_x, fit_y, 'r-',lw=3)#, label=label.format(A, B, r2))
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
    if B==0:
        ax.text(0.05, 0.80, f"r$^2$: {round(r2,2)}", transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
    else:
        ax.text(0.05, 0.65, f"r$^2$: {round(r2,2)}", transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
        
    if outlier_influence:
        ax.fill_between(fit_x, fit_y - ((SE_A*fit_x)**2+(SE_B/factor)**2)**0.5, fit_y + ((SE_A*fit_x)**2+(SE_B/factor)**2)**0.5, alpha=0.33)
    return ax.figure,ax

###############################################################################
def Plot_correlation_df(df, fig_text="", *Plotsettings):
    """
    Function to plot multiple correlation plots together in a n*n grid, where n is the number of instruments minus 1.
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions. 
       
    Parameters
    ----------
    df: dataframe
        dataframe of the instruments structured with an instrument per column,
        with instrument handle equal to 
        
    fig_text: str, optional
        Second set of values.  
        
    *Plotsettings: list of input to 
        Input to the Plot_calibration function:
            intercept=True, uniform_scaling=True, outlier_influence=True
        Call on this
    unifomr_scaling: boolean, optional
        Boolean that can be turned off so as to not scale axis the max value.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    gs : matplotlib.gridspace._subplots.AxesSubplot
        Handles for the gridspace of the plot.
        """
        
    instruments = list(df.columns)
    n=0
    
    if len(instruments)==2:
        fig, ax= Plot_correlation(df[instruments[0]], df[instruments[1]], *Plotsettings)#
        fig.figsize=(18, 18)
        ax.set_ylabel(instruments[1])
        ax.set_xlabel(instruments[0])
    else:
        n = len(instruments) - 1
        
        fig = plt.figure(figsize=(18, 18), constrained_layout=True)
        gs = fig.add_gridspec(n, n)
        
        axes = {}
        # Create only the subplots you need (upper triangle incl. diagonal)
        for i in range(n):
            for j in range(i, n):
                ax = fig.add_subplot(gs[j, i])    # row=j, col=i
                axes[(j, i)] = ax
        
                # Your custom correlation plot
                try:
                    _, _ = Plot_correlation(df[instruments[i]], df[instruments[j+1]], ax, *Plotsettings)#intercept=intercept)
                except Exception:
                    pass
        
                if i == 0:
                    ax.set_ylabel(instruments[j+1])
                if j == n - 1:
                    ax.set_xlabel(instruments[i])
    if len(fig_text)>0:
        # One description box spanning top row, columns 2..n (i.e., gs[0, 1:])
        if n > 1:
            desc_ax = fig.add_subplot(gs[0, 1:])  # uses only empty cells
            desc_ax.axis('off')
            desc_ax.text(
                0.0, 0.5,
                fig_text,
                ha='left', va='center', fontsize=40, wrap=True,
                bbox=dict(boxstyle="round,pad=0.6", fc="whitesmoke", ec="0.5", lw=1),
                transform=desc_ax.transAxes
            )
        else:
            # Fallback if there's only one column in the grid
            ax.text(
                0.5, 0.90,
                fig_text,
                ha='center', va='top',transform=ax.transAxes, fontsize=40
            )
    plt.tight_layout()
    if n==0:
        return fig, ax
    else:
        return fig, gs
