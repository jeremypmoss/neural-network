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

#%% Load data
dataset, datasetname, magnames, mags = qf.loaddata('sdssmags',
                                                   dropna = False,  # to drop NaNs
                                                   colours = False, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#%% Define base model
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
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),

    keras.layers.Dense(1) # 1 output (redshift)
    ])

    model.compile(loss=loss,
                  optimizer = opt,
            metrics = metrics)
    print(model.summary())
    return model

#%% Evaluate baseline model
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
base_results = cross_val_score(estimator, X, y, cv=kfold, scoring='neg_mean_squared_error')

#%% Evaluate model with standarised dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
std_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

#%% Deeper network
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

    model.compile(loss=loss,
                  optimizer = opt,
            metrics = metrics)
    print(model.summary())
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=deeper_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
deep_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

#%% Wider network
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

    model.compile(loss=loss,
                  optimizer = opt,
            metrics = metrics)
    print(model.summary())
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
wide_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

#%% Print results
print("Baseline: %.2f (%.2f) MSE" % (base_results.mean(), base_results.std()))
print("Standardized: %.2f (%.2f) MSE" % (std_results.mean(), std_results.std()))
print("Deeper: %.2f (%.2f) MSE" % (deep_results.mean(), deep_results.std()))
print("Wider: %.2f (%.2f) MSE" % (wide_results.mean(), wide_results.std()))
