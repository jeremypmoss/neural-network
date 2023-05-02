# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 07:18:22 2023

@author: JeremyMoss
"""

from DataLoader import DataLoader
import quasar_functions as qf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import sys
# sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')

# %% Load data
dl = DataLoader(dropna=False,
                colours=False,
                impute_method=None)

# obj = pd.read_pickle(r'../../data_files/SDSS/SDSS-ML-quasars.pkl')
# obj.to_csv(r'../../data_files/SDSS/SDSS-ML-quasars.csv', sep=',')

dataset1, datasetname1, magnames1, mags1 = dl.load_data('clarke')
dataset2, datasetname2, magnames2, mags2 = dl.load_data('clarke_x_mq')


# %%
sample = dataset1.sample(50000)
qf.sky_distribution(sample['ra'], sample['dec'], datasetname1)

# %%
for dataset, datasetname in zip([dataset1, dataset2], [datasetname1, datasetname2]):
    for band in ['u', 'g', 'r', 'i', 'zmag', 'W1', 'W2']:
        fig, ax = plt.subplots()
        ax.set_title(band, datasetname)
        ax.hist(dataset[band], bins=100, density=True)
        ax.grid(which='both', axis='both')
