# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

import numpy as np
import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import quasar_functions as qf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow_docs.modeling
import tensorflow_docs as tfdocs
import pandas as pd
import time
import matplotlib.pyplot as plt

start_time = time.time()

#%% Load data
dataset, datasetname, magnames, mags = qf.loaddata('test',
                                                   dropna = False,  # to drop NaNs
                                                   colours = False, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data

#%% Model
hyperparams = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
epochs = 100
opt = 'Nadam'

num_trials = 3
mean_list = []
std_list = []
train_frac = 0.8

for i in range(num_trials):
    print('*'*58);print('Run {0} of {1}'.format(i+1, num_trials)); print('*'*58)
    X_train, X_test, y_train, y_test = train_test_split(mags, # features
                                                dataset['redshift'], # target
                                                train_size = train_frac)
    model = qf.build_nn_model(len(mags.columns), hyperparams, loss, metrics, opt)
    model.summary()
    early_stop = keras.callbacks.EarlyStopping(patience=100)
    
    history = model.fit(X_train, y_train, epochs = epochs,
                        validation_split = 1 - train_frac,
                        verbose = 0, callbacks = [early_stop,
                                                  tfdocs.modeling.EpochDots()])
    y_pred = model.predict(X_test)
    
    # Make a new dataset containing the redshifts
    X_test['z_spec'] = y_test
    X_test['z_phot'] = y_pred
    X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
    
    stats = X_test['delta_z'].describe().transpose()
    mean, std = stats['mean'], stats['std']
    mean_list.append(mean)
    std_list.append(std)

# for i in range(num_trials):
    
print("Model completed in", time.time() - start_time, "seconds")
  
#%% Display means and standard deviations
border = '-'*25
separator = '\t\t|\t'
results_list = zip(mean_list, std_list)
print('Means' + separator + 'Std devs')
print(border)
for mean, dev, *_ in results_list:
    print(f"{mean:7f}\t|\t{dev:7f}")
print(border)
print('Average mean = {avg_mean}\nAverage std dev = {avg_std}'.format(
    avg_mean=np.mean(mean_list),
    avg_std=np.mean(std_list)
))

#%% Plot results

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 9))
fig.tight_layout()

# plot_mse() #https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
# plot_mae()
# qf.plot_deltaz(x, y, datasetname)
qf.plot_z(X_test['z_spec'], X_test['z_phot'], datasetname, ax = ax[0])
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, model, ax = ax[1])

qf.kurt_result(X_test['delta_z'])
qf.plot_z_sets(y_train, X_test['z_spec'], datasetname)

print("Script completed in", time.time() - start_time, "seconds")
