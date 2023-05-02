# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:02:48 2023

@author: JeremyMoss
"""
from sklearn.metrics import median_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
from sklearn.model_selection import train_test_split
from tensorflow import keras
import quasar_functions as qf
import matplotlib.pyplot as plt
import time
from DataLoader import DataLoader

rc_fonts = {
    "text.usetex": True,
    'font.family': 'serif',
    'font.size': 20,
}
plt.rcParams.update(rc_fonts)

start_time = time.time()


# def baseline_model(hyperparameters, n, loss, metrics, opt):
#     model = Sequential()
#     model.add(Dense(hyperparameters[0], activation=hyperparameters[1],  # number of outputs to next layer
#                     input_shape=[n]))  # number of features
#     model.add(Dense(hyperparameters[2], activation=hyperparameters[3]))
#     model.add(Dense(hyperparameters[4], activation=hyperparameters[5]))
#     model.add(Dense(1))  # 1 output (redshift)

#     model.compile(loss=loss, optimizer=opt, metrics=metrics)
#     print(model.summary())
#     return model


def deeper_model(hyperparameters, n, loss, metrics, opt):
    model = keras.Sequential([
        keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1],  # number of outputs to next layer
                           input_shape=[n]),  # number of features
        keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
        keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
        keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
        keras.layers.Dense(1)  # 1 output (redshift)
    ])

    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    print(model.summary())
    return model


def wider_model(hyperparameters, n, loss, metrics, opt):
    model = keras.Sequential([
        keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1],  # number of outputs to next layer
                           input_shape=[n]),  # number of features
        keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
        keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
        keras.layers.Dense(1)  # 1 output (redshift)
    ])

    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    print(model.summary())
    return model


# %% Load data and define train and validation sets
dl = DataLoader(dropna=False,
                colours=False,
                impute_method='max')
dataset, datasetname, magnames, mags = dl.load_data('clarke')

X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kfold_splits = 5

# %% Define and evaluate base model
regressor_params = {'epochs': 5,
                    'batch_size': 5,
                    'verbose': -1}

model_params = {'n': len(mags.columns),
                'hyperparameters': [100, 'relu', 100, 'relu', 100, 'relu'],
                'loss': 'mean_squared_error',
                'metrics': ['mae'],
                'opt': 'Nadam'}

# hyperparameters = [100, 'relu', 100, 'relu', 100, 'relu']

# baseline_model = baseline_model(hyperparameters, **model_params)
baseline_model = qf.build_nn_model(len(mags.columns),
                                   hyperparameters=[100, 'relu',
                                                    100, 'relu', 100, 'relu'],
                                   loss='mean_squared_error',
                                   metrics=['mae'],
                                   opt='Nadam')
regressor = KerasRegressor(model=baseline_model, **regressor_params)
kfold = KFold(n_splits=kfold_splits)
base_results = cross_val_score(regressor, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')

fit_base = baseline_model.fit(X, y)
y_pred_base = baseline_model.predict(X_test)
print("Baseline model completed in", time.time() - start_time, "seconds")
qf.metrics_table(y_test, y_pred_base)

# %% Evaluate base model with standarised dataset

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
std_results = cross_val_score(pipeline, X_train, y_train,
                              cv=kfold, scoring='neg_mean_squared_error')

print("Standardised model completed in", time.time() - start_time, "seconds")

# %% Evaluate deeper model with standarised dataset

deeper_model = deeper_model(**model_params)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(
    model=deeper_model, **regressor_params)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
deep_results = cross_val_score(pipeline, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')

fit_deep = deeper_model.fit(X, y)
y_pred_deep = deeper_model.predict(X_test)
qf.metrics_table(y_test, y_pred_deep)

print("Deep model completed in", time.time() - start_time, "seconds")

# %% Evaluate wider model with standarised dataset
hyperparameters = [500, 'relu', 500, 'relu', 500, 'relu']
wider_model = wider_model(**model_params)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(
    model=wider_model, **regressor_params)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
wide_results = cross_val_score(pipeline, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')

fit_wide = wider_model.fit(X, y)
y_pred_wide = wider_model.predict(X_test)
qf.metrics_table(y_test, y_pred_wide)

print("Wide model completed in", time.time() - start_time, "seconds")

# %% Display results

print("Baseline:\t%.2f\t(%.2f) MSE\t%.2f MAE" % (base_results.mean(),
      base_results.std(), median_absolute_error(y_test, y_pred_base)))
print("Standard:\t%.2f\t(%.2f) MSE" % (std_results.mean(), std_results.std()))
print("Deeper:\t%.2f\t(%.2f) MSE\t%.2f MAE" % (deep_results.mean(),
      deep_results.std(), median_absolute_error(y_test, y_pred_deep)))
print("Wider:\t%.2f\t(%.2f) MSE\t%.2f MAE" % (wide_results.mean(),
      wide_results.std(), median_absolute_error(y_test, y_pred_wide)))

X_test['z_spec'] = y_test
X_test['z_phot_base'] = y_pred_base
X_test['z_phot_deep'] = y_pred_deep
X_test['z_phot_wide'] = y_pred_wide

X_test['delta_z_base'] = X_test['z_spec'] - X_test['z_phot_base']
X_test['delta_z_deep'] = X_test['z_spec'] - X_test['z_phot_deep']
X_test['delta_z_wide'] = X_test['z_spec'] - X_test['z_phot_wide']

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot_base'],
          datasetname + ' base', ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z_base'],
                     datasetname + ' base', baseline_model, ax=ax[1])

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot_deep'],
          datasetname + ' deep', ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z_deep'],
                     datasetname + ' deep', baseline_model, ax=ax[1])

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['z_phot_wide'],
          datasetname + ' wide', ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z_wide'],
                     datasetname + ' wide', baseline_model, ax=ax[1])

# qf.plotResids(X_test['z_phot_base'], r'Base model $z_{\mathrm{phot}}$',
#               )

print(f"Script completed in {time.time() - start_time:.1f} seconds")

# %% Load and predict a test set
skymap, skymapname, skymagnames, skymags = dl.load_data('skymapper_wise')

skymap_pred_base = baseline_model.predict(skymap)
skymap_pred_deep = deeper_model.predict(skymap)
skymap_pred_wide = wider_model.predict(skymap)

# %% Plot predictions for test set
# datasetname = r'GALEX QSOs from SDSS $\times$ WISE'
qf.compare_z(skymap_pred_base, dataset['redshift'],
             set1name='Skymapper base predictions',
             set2name=datasetname,
             yscale='log')

qf.compare_z(skymap_pred_deep, dataset['redshift'],
             set1name='Skymapper deep predictions',
             set2name=datasetname,
             yscale='log')

qf.compare_z(skymap_pred_wide, dataset['redshift'],
             set1name='Skymapper wide predictions',
             set2name=datasetname,
             yscale='log')
