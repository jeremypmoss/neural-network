# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

import numpy as np
import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import quasar_functions as qf
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb
import tensorflow_docs.modeling
import tensorflow_docs as tfdocs
import pandas as pd
import time
import matplotlib.pyplot as plt

start_time = time.time()


#%% Load data
dataset, datasetname, magnames, mags = qf.loaddata('test',
                                                   dropna = False,  # to drop NaNs
                                                   colours = True, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data
#%% Model
wandb.init(project = 'nn-KR-GSCV_{}'.format(datasetname))
hyperparams = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
epochs = 100
opt = 'Nadam'

wandb.run.log_code(".")

train_frac = 0.8
X_train, X_test, y_train, y_test = train_test_split(mags, # features
                                                dataset['redshift'], # target
                                                train_size = train_frac)

model = KerasRegressor(qf.build_nn_model(len(mags.columns), hyperparams, loss, metrics, opt))

early_stop = keras.callbacks.EarlyStopping(patience=100)

history = model.fit(X_train, y_train, epochs = epochs,
                    validation_split = 1 - train_frac,
                    verbose = 0,
                    callbacks = [early_stop, tfdocs.modeling.EpochDots(),
                                 WandbMetricsLogger(),
                                 WandbModelCheckpoint("models")])
y_pred = model.predict(X_test)

X_test['z_spec'] = y_test 
X_test['z_phot'] = y_pred
X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
print("Model completed in", (time.time() - start_time), "seconds")

#%% Grid search through the possible hyperparameters
# parameters = {'loss'         : ['squared_error', 'absolute_error', 'huber', 'quantile'],
#               'optimizer'    : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
#               'epochs'       : [10],
#               'batch_size'   : [5, 10, 50]}
#               # what else can I try in here?

# grid = GridSearchCV(estimator = model,
#                     param_grid = parameters,
#                     scoring = 'accuracy',
#                     n_jobs = None, # not -1
#                     refit = 'boolean',
#                     verbose = 0)
# grid_result = grid.fit(X_train, y_train)

# mse_krr = mean_squared_error(y_test, y_pred)
# print(mse_krr)
# print(grid.best_params_)
# print(grid.best_estimator_)

print("Optimisation completed in", (time.time() - start_time), "seconds")
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

print("Script completed in", (time.time() - start_time), "seconds")
