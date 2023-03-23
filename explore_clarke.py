# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 07:18:22 2023

@author: JeremyMoss
"""

import pandas as pd
import quasar_functions as qf
from DataLoader import DataLoader
# import sys
# sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')

# %%
dl = DataLoader(dropna=False,
                colours=False,
                impute_method=None)

# obj = pd.read_pickle(r'../data_files/SDSS/SDSS-ML-quasars.pkl')
# obj.to_csv(r'../data_files/SDSS/SDSS-ML-quasars.csv', sep=',')

dataset, datasetname, magnames, mags = dl.load_data('clarke')
