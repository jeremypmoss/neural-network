# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

import matplotlib.pyplot as plt
import time
import pandas as pd
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from tensorflow import keras
from sklearn.model_selection import train_test_split
import quasar_functions as qf
import numpy as np
import sys
from DataLoader import DataLoader
# sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')

start_time = time.time()

# %% Load data
qf.make_test(1000, 10, 0)
dl = DataLoader(dropna=True,
                colours=True,
                impute_method='max')
dataset, datasetname, magnames, mags = dl.load_data('test')
# dataset, datasetname, magnames, mags = qf.loaddata('test',
#                                                    dropna=True,  # to drop NaNs
#                                                    colours=True,  # to compute colours of mags
#                                                    impute_method='max')  # to impute max vals for missing data

# %% Model
hyperparams = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
epochs = 100
opt = 'Nadam'

num_trials = 3
mean_list = []
std_list = []
train_frac = 0.8
X_train, X_test, y_train, y_test = train_test_split(mags,  # features
                                                    # target
                                                    dataset['redshift'],
                                                    train_size=train_frac)

model = qf.build_nn_model(len(mags.columns), hyperparams, loss, metrics, opt)
model.summary()
early_stop = keras.callbacks.EarlyStopping(patience=100)

history = model.fit(X_train, y_train, epochs=epochs,
                    validation_split=1 - train_frac,
                    verbose=0, callbacks=[early_stop,
                                          tfdocs.modeling.EpochDots()])
y_pred = model.predict(X_test)

X_test['z_spec'] = y_test
X_test['z_phot'] = y_pred
X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
print("Model completed in", time.time() - start_time, "seconds")

# %% Plot results

# plot_mse() #https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
# plot_mae()
# qf.plot_deltaz(x, y, datasetname)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot'], datasetname, ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, model, ax=ax[1])
qf.kurt_result(X_test['delta_z'])

print("Script completed in", time.time() - start_time, "seconds")
