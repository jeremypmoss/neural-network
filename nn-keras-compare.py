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
from sklearn.model_selection import train_test_split
from tensorflow import keras
import quasar_functions as qf
import matplotlib.pyplot as plt
import time
from DataLoader import DataLoader
import numpy as np
from matplotlib.pyplot import cm


start_time = time.time()

rc_fonts = {
    "text.usetex": True,
    'font.family': 'serif',
    'font.size': 20
}

plt.rcParams.update(rc_fonts)


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


# %% Load data and define train and validation sets
dl = DataLoader(dropna=False,
                colours=False,
                impute_method=None)
path = r'../../data_files'
number_of_rows = None
dataset, datasetname, magnames, mags = dl.load_data('mq_x_gleam_nonpeaked_with_z', path,
                                                    number_of_rows=number_of_rows)
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
                'opt': 'Adam'}

baseline_model = qf.build_nn_model(**model_params)
baseline_model.summary()
regressor = KerasRegressor(model=baseline_model, **regressor_params)
kfold = KFold(n_splits=kfold_splits)
base_results = cross_val_score(regressor, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')

fit_base = baseline_model.fit(X, y)
y_pred_base = baseline_model.predict(X_test)
print("Baseline model completed in", time.time() - start_time, "seconds")

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
deeper_model.summary()
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

print("Deep model completed in", time.time() - start_time, "seconds")

# %% Evaluate wider model with standarised dataset
model_params = {'n': len(mags.columns),
                'hyperparameters': [500, 'relu', 500, 'relu', 500, 'relu'],
                'loss': 'mean_squared_error',
                'metrics': ['mae'],
                'opt': 'Nadam'}
wider_model = qf.build_nn_model(**model_params)
wider_model.summary()
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

print("Wide model completed in", time.time() - start_time, "seconds")

# %% Display z_phot against z_spec and z distribution histogram

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

X_test['mean_z_phot'] = X_test[['z_phot_wide',
                                'z_phot_deep', 'z_phot_base']].mean(axis=1)

X_test['delta_z_base'] = X_test['z_spec'] - X_test['z_phot_base']
X_test['delta_z_deep'] = X_test['z_spec'] - X_test['z_phot_deep']
X_test['delta_z_wide'] = X_test['z_spec'] - X_test['z_phot_wide']
X_test['delta_z_mean'] = X_test['z_spec'] - X_test['mean_z_phot']


# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(X_test['z_spec'], X_test['z_phot_base'],
#           datasetname + ' base', ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_base'],
#                       datasetname + ' base', baseline_model, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(X_test['z_spec'], X_test['z_phot_deep'],
#           datasetname + ' deep', ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_deep'],
#                       datasetname + ' deep', baseline_model, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(X_test['z_spec'], X_test['z_phot_wide'],
#           datasetname + ' wide', ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_wide'],
#                       datasetname + ' wide', baseline_model, ax=ax[1])

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['mean_z_phot'],
          datasetname + ' mean', ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z_mean'],
                     datasetname + ' mean', baseline_model, ax=ax[1])

# qf.plotResids(X_test['z_phot6'], r'Base model $z_{\mathrm{phot}}$')

print(f"Script completed in {time.time() - start_time:.1f} seconds")

# %% Load and predict a test set
# Using the clarke catalogue here produces an error.
# new, newname, newmagnames, newmags = dl.load_data('skymapper_wise', path)

# new_pred_base = baseline_model.predict(new)
# new_pred_deep = deeper_model.predict(new)
# new_pred_wide = wider_model.predict(new)

# %% Plot predictions for test set

# set1name = '{0} base predictions'.format(newname)
# fig, ax = plt.subplots()
# # ax.hist(new_pred_base,
# #         bins=100,
# #         density=True,
# #         edgecolor='red',
# #         color='pink',
# #         label='Predictions for {0} (baseline model)'.format(newname))
# ax.hist(dataset['redshift'],
#         bins=100,
#         density=True,
#         edgecolor='blue',
#         color='lightblue',
#         label=r'$z_\mathrm{spec}$ from training data',
#         alpha=0.5)
# ax.hist(y_pred_base,
#         bins=50,
#         density=True,
#         edgecolor='darkgreen',
#         color='g',
#         label=r'$z_\mathrm{phot}$ from training data',
#         alpha=0.5)
# ax.grid(True)
# ax.set_xlabel('Redshift')
# ax.set_yscale('linear')
# ax.set_ylabel('Count')
# ax.set_title('Redshift distributions for\n{0} and {1} (training set)'.format(
#     newname, datasetname))
# ax.legend(loc='upper right')

# qf.compare_z(skymap_pred_deep, dataset['redshift'],
#              set1name='Skymapper deep predictions',
#              set2name=datasetname,
#              yscale='linear')

# set1name = '{0} wide predictions'.format(newname)
# set2name = datasetname
# fig, ax = plt.subplots()
# ax.hist(new_pred_wide,
#         bins=100,
#         density=True,
#         edgecolor='red',
#         color='pink',
#         label='Predictions for {0} (wide model)'.format(newname))
# ax.hist(dataset['redshift'],
#         bins=100,
#         density=True,
#         edgecolor='blue',
#         color='lightblue',
#         label=r'$z_\mathrm{spec}$ from training data',
#         alpha=0.5)
# ax.hist(y_pred_base,
#         bins=50,
#         density=True,
#         edgecolor='darkgreen',
#         color='g',
#         label=r'$z_\mathrm{phot}$ from training data',
#         alpha=0.5)
# ax.grid(True)
# ax.set_xlabel('Redshift')
# ax.set_yscale('linear')
# ax.set_ylabel('Count')
# ax.set_title('Redshift distributions for\n{0} and {1} (training set)'.format(
#     set1name, set2name))
# ax.legend(loc='upper right')

# set1name = 'Skymapper deep predictions'
# set2name = datasetname
# fig, ax = plt.subplots()
# ax.hist(new_pred_deep,
#         bins=100,
#         density=True,
#         edgecolor='red',
#         color='pink',
#         label='Predictions for {0} (deep model)'.format(newname))
# ax.hist(dataset['redshift'],
#         bins=100,
#         density=True,
#         edgecolor='blue',
#         color='lightblue',
#         label=r'$z_\mathrm{spec}$ from training data',
#         alpha=0.5)
# ax.hist(y_pred_base,
#         bins=50,
#         density=True,
#         edgecolor='darkgreen',
#         color='g',
#         label=r'$z_\mathrm{phot}$ from training data',
#         alpha=0.5)
# ax.grid(True)
# ax.set_xlabel('Redshift')
# ax.set_yscale('linear')
# ax.set_ylabel('Count')
# ax.set_title('Redshift distributions for\n{0} and {1} (training set)'.format(
#     set1name, set2name))
# ax.legend(loc='upper right')
# %% All plots on top of each other
n = 4
color = iter(cm.hsv(np.linspace(0, 0.8, n)))
bins = 100
alpha = 0.6
binwidth = 0.05
edges = '0.5'
fig, ax = plt.subplots()
ax.hist(dataset['redshift'],  # spectroscopic redshifts from training set
        bins=np.arange(min(dataset['redshift']), max(
            dataset['redshift']) + binwidth, binwidth),
        density=True,
        color=next(color),
        edgecolor=edges,
        alpha=1,
        label=r'$z_{\mathrm{spec}}$ from '+datasetname)

ax.hist(y_pred_base,  # z_phot derived from baseline model
        bins=np.arange(min(y_pred_base), max(y_pred_base) + binwidth, binwidth),
        density=True,
        color=next(color),
        edgecolor=edges,
        alpha=alpha,
        label=r'$z_{\mathrm{phot}}$ for baseline model')

ax.hist(y_pred_deep,  # z_phot derived from deeper model
        bins=np.arange(min(y_pred_deep), max(y_pred_deep) + binwidth, binwidth),
        density=True,
        color=next(color),
        edgecolor=edges,
        alpha=alpha,
        label=r'$z_{\mathrm{phot}}$ for deeper model')

ax.hist(y_pred_wide,  # z_phot derived from wider model
        bins=np.arange(min(y_pred_wide), max(y_pred_wide) + binwidth, binwidth),
        density=True,
        color=next(color),
        edgecolor=edges,
        alpha=alpha,
        label=r'$z_{\mathrm{phot}}$ for wider model')

ax.grid(True)
ax.set_xlabel('Redshift')
ax.set_yscale('linear')
ax.set_ylabel('Count')
ax.set_title('Redshift distributions of\n{0} and some predicted redshifts'.format(
    r'$z_{\mathrm{spec}}$ from '+datasetname))
ax.legend(loc='upper right')

print('Base model')
qf.metrics_table(y_test, y_pred_base.flatten())
print('Deep model')
qf.metrics_table(y_test, y_pred_deep.flatten())
print('Wide model')
qf.metrics_table(y_test, y_pred_wide.flatten())

# %% Summary plot

sample_sizes = [1000, 2000, 3000, 4000, 5619]
r_squared_list = [0.005, 0.029, 0.004, 0.007, 0.011]
sigma_list = [0.779, 0.744, 0.781, 0.701, 0.741]
fig, ax = plt.subplots()
ax.plot(sample_sizes, r_squared_list, label=r'$r^2$ for mean $z_\mathrm{phot}$')
ax.plot(sample_sizes, sigma_list,
        label=r'$\sigma$ for $z_\mathrm{phot}$ distribution')
ax.set_xlabel('Sample size')
ax.set_title(r'Effect of increasing sample size of $r^2$ and $\sigma$')
ax.grid(True)
ax.legend()
