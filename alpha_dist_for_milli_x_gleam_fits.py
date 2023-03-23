# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:14:20 2023

@author: JeremyMoss
"""

from DataLoader import DataLoader
# import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import quasar_functions as qf
import matplotlib.pyplot as plt

dl = DataLoader(dropna=False,
                colours=False,
                impute_method='max')
dataset, datasetname, magnames, mags = dl.load_data('milli_x_gleam_fits')

df = qf.loaddata('milli_x_gleam_fits', dropna=False,
                 colours=False,
                 impute_method='max')

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
fig.tight_layout()

qf.plot_z(df[0]['alpha'], df[0]['my alpha'], datasetname, ax=[0])
qf.plot_delta_z_hist(
    df[0]['alpha'] - df[0]['my alpha'], datasetname, model=None, ax=ax[1])
