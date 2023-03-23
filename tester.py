# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:14:20 2023

@author: JeremyMoss
"""

from DataLoader import DataLoader
import quasar_functions as qf
import matplotlib.pyplot as plt
data_loader = DataLoader(dropna = False,
                            colours = False,
                            impute_method = 'max')
dataset, datasetname, magnames, mags = data_loader.load_data('milli_x_gleam_fits')

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 9))
fig.tight_layout()
qf.plot_z(dataset['alpha'], dataset['my alpha'], datasetname, ax = ax[0])
qf.plot_delta_z_hist(dataset['alpha'] - dataset['my alpha'], datasetname, model = None, ax = ax[1])
