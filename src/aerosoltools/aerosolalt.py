import matplotlib.pyplot as plt
from aerosol1d import Aerosol1D
params = {'legend.fontsize': 15,
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 13,
         'ytick.labelsize': 13,
         'figure.figsize' : (19, 10)}
plt.rcParams.update(params)

class Aerosolalt(Aerosol1D):
    def __init__(self, dataframe):
        super().__init__(dataframe)
