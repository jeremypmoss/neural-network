# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 07:13:31 2023

@author: JeremyMoss
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

"""

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import quasar_functions as qf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import quasar_functions as qf
import matplotlib.pyplot as plt
import time

start_time = time.time()

#%% Load data
dataset, datasetname, magnames, mags = qf.loaddata('deep2',
                                                   dropna = False,  # to drop NaNs
                                                   colours = False, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
kfold_splits = 3

#%% Define and evaluate base model
hyperparams = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
epochs = 100
opt = 'Nadam'

def baseline_model(n = len(mags.columns),
               hyperparameters = [100, 'relu', 100, 'relu', 100, 'relu'],
               loss = 'mae',
               metrics = ['mae'],
               opt = 'Nadam'):
    model = Sequential()
    model.add(Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]))  # number of features
    model.add(Dense(hyperparameters[2], activation=hyperparameters[3]))
    model.add(Dense(hyperparameters[4], activation=hyperparameters[5]))

    model.add(Dense(1)) # 1 output (redshift)

    model.compile(loss=loss, optimizer = opt, metrics = metrics)
    print(model.summary())
    return model

baseline_model = baseline_model()
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=kfold_splits)
base_results = cross_val_score(estimator, X, y, cv=kfold, scoring='accuracy')
print("Baseline model completed in", time.time() - start_time, "seconds")

#%% Evaluate base model with standarised dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
std_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("Standardised model completed in", time.time() - start_time, "seconds")

#%% Evaluate deeper model with standarised dataset
def deeper_model(n = len(mags.columns),
                hyperparameters = [100, 'relu', 100, 'relu', 100, 'relu'],
                loss = 'mae',
                metrics = ['mae'],
                opt = 'Nadam'):
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                            input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
    keras.layers.Dense(1) # 1 output (redshift)
    ])

    model.compile(loss=loss,optimizer = opt, metrics = metrics)
    print(model.summary())
    return model

deeper_model = deeper_model()
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=deeper_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
deep_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("Deep model completed in", time.time() - start_time, "seconds")

#%% Evaluate wider model with standarised dataset
def wider_model(n = len(mags.columns),
                hyperparameters = [200, 'relu', 200, 'relu', 200, 'relu'],
                loss = 'mae',
                metrics = ['mae'],
                opt = 'Nadam'):
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                            input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
    keras.layers.Dense(1) # 1 output (redshift)
    ])

    model.compile(loss=loss, optimizer = opt, metrics = metrics)
    print(model.summary())
    return model

wider_model = wider_model()
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
wide_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
model = wider_model
y_pred = wider_model.fit(X_test)
print("Wide model completed in", time.time() - start_time, "seconds")

#%% Print results
print("Baseline:\t%.2f\t(%.2f) MSE" % (base_results.mean(), base_results.std()))
print("Standardized:\t%.2f\t(%.2f) MSE" % (std_results.mean(), std_results.std()))
print("Deeper:\t%.2f\t(%.2f) MSE" % (deep_results.mean(), deep_results.std()))
print("Wider:\t%.2f\t(%.2f) MSE" % (wide_results.mean(), wide_results.std()))

y_pred = baseline_model.predict(X_test)
X_test['z_spec'] = y_test
X_test['z_phot'] = y_pred
X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 9))
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot'], datasetname, ax = ax[0])
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, baseline_model, ax = ax[1])

print("Script completed in", time.time() - start_time, "seconds")
