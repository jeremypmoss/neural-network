# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:02:48 2023

@author: JeremyMoss
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
from sklearn.model_selection import train_test_split
from tensorflow import keras
import quasar_functions as qf
import matplotlib.pyplot as plt
import time

start_time = time.time()

def baseline_model(hyperparameters, n,loss, metrics, opt):
    model = Sequential()
    model.add(Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]))  # number of features
    model.add(Dense(hyperparameters[2], activation=hyperparameters[3]))
    model.add(Dense(hyperparameters[4], activation=hyperparameters[5]))
    model.add(Dense(1)) # 1 output (redshift)

    model.compile(loss=loss, optimizer = opt, metrics = metrics)
    print(model.summary())
    return model

def deeper_model(hyperparameters, n, loss, metrics, opt):
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

def wider_model(hyperparameters, n, loss, metrics, opt):
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

#%% Load data
dataset, datasetname, magnames, mags = qf.loaddata('milli_x_gleam_fits',
                                                   dropna = False,
                                                   colours = False,
                                                   impute_method = 'max')
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
kfold_splits = 2

#%% Define and evaluate base model
regressor_params = {'epochs': 2,
                    'batch_size': 5,
                    'verbose': -1}

model_params = {'n': len(mags.columns),
                'loss': 'mean_squared_error',
                'metrics': ['mae'],
                'opt': 'Nadam'}

hyperparameters = [100, 'relu', 100, 'relu', 100, 'relu']

baseline_model = baseline_model(hyperparameters, **model_params)
regressor = KerasRegressor(model = baseline_model, **regressor_params)
kfold = KFold(n_splits = kfold_splits)
base_results = cross_val_score(regressor, X_train, y_train,
                               cv = kfold, scoring='neg_mean_squared_error')

fit_base = baseline_model.fit(X, y)
y_pred_base = baseline_model.predict(X_test)

print("Baseline model completed in", time.time() - start_time, "seconds")

#%% Evaluate base model with standarised dataset

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
std_results = cross_val_score(pipeline, X_train, y_train,
                              cv=kfold, scoring='neg_mean_squared_error')

print("Standardised model completed in", time.time() - start_time, "seconds")

#%% Evaluate deeper model with standarised dataset

deeper_model = deeper_model(hyperparameters, **model_params)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=deeper_model, **regressor_params)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
deep_results = cross_val_score(pipeline, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')

fit_deep = deeper_model.fit(X, y)
y_pred_deep = deeper_model.predict(X_test)

print("Deep model completed in", time.time() - start_time, "seconds")

#%% Evaluate wider model with standarised dataset
wider_model = wider_model(hyperparameters, **model_params)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, **regressor_params)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
wide_results = cross_val_score(pipeline, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')

fit_wide = wider_model.fit(X, y)
y_pred_wide = wider_model.predict(X_test)

print("Wide model completed in", time.time() - start_time, "seconds")

#%% Display results

print("Baseline:\t%.2f\t(%.2f) MSE" % (base_results.mean(), base_results.std()))
print("Standard:\t%.2f\t(%.2f) MSE" % (std_results.mean(), std_results.std()))
print("Deeper:\t%.2f\t(%.2f) MSE" % (deep_results.mean(), deep_results.std()))
print("Wider:\t%.2f\t(%.2f) MSE" % (wide_results.mean(), wide_results.std()))

X_test['z_spec'] = y_test
X_test['z_phot_base'] = y_pred_base
X_test['z_phot_deep'] = y_pred_deep
X_test['z_phot_wide'] = y_pred_wide

X_test['delta_z_base'] = X_test['z_spec'] - X_test['z_phot_base']
X_test['delta_z_deep'] = X_test['z_spec'] - X_test['z_phot_deep']
X_test['delta_z_wide'] = X_test['z_spec'] - X_test['z_phot_wide']

fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot_base'], datasetname + ' base', ax = ax[0])
qf.plot_delta_z_hist(X_test['delta_z_base'], datasetname + ' base', baseline_model, ax = ax[1])

fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot_deep'], datasetname + ' deep', ax = ax[0])
qf.plot_delta_z_hist(X_test['delta_z_deep'], datasetname + ' deep', baseline_model, ax = ax[1])

fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot_wide'], datasetname + ' wide', ax = ax[0])
qf.plot_delta_z_hist(X_test['delta_z_wide'], datasetname + ' wide', baseline_model, ax = ax[1])

print("Script completed in", time.time() - start_time, "seconds")

#%% Load a test set
# skymap, skymapname, skymagnames, skymags = qf.loaddata('skymapper',
#                                                    dropna = False,
#                                                    colours = False,
#                                                    impute_method = 'max')
# skymap_pred = baseline_model.predict(skymap)
# skymap_pred = deeper_model.predict(skymap)
# skymap_pred = wider_model.predict(skymap)

# qf.plot_one_z_set(skymap_pred, skymapname)
# qf.compare_z(skymap_pred, dataset['redshift'],
#                set1name = skymapname,
#                set2name = datasetname)
