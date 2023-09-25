# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:02:48 2023

@author: JeremyMoss
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import quasar_functions as qf
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from DataLoader import DataLoader
from matplotlib.pyplot import cm


start_time = time.time()

rc_fonts = {
    "text.usetex": True,
    'font.family': 'serif',
    'font.size': 20
}

plt.rcParams.update(rc_fonts)


def deeper_model_k(X, y, n,
                   hyperparameters=[100, 'relu', 100, 'relu', 100, 'relu'],
                   loss_metric='mean_squared_error',
                   evaluation_metrics=['mae'],
                   opt='Nadam',
                   n_splits=3):
    ''' Define the prediction model. The NN takes magnitudes as input features
    and outputs the redshift. n should be len(train_set.keys())'''

    # Create a KFold instance
    kf = KFold(n_splits=n_splits)

    # Initialize lists to store the evaluation metrics for each fold
    all_loss = []
    all_metrics = []

    # input features X and target values y
    for train_index, val_index in kf.split(X):
        # Split the data into training and validation sets for the current fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Build the model for the current fold
        model = keras.Sequential([
            keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1],  # number of outputs to next layer
                               input_shape=[n]),  # number of features
            keras.layers.Dense(
                hyperparameters[2], activation=hyperparameters[3]),
            keras.layers.Dense(
                hyperparameters[4], activation=hyperparameters[5]),
            keras.layers.Dense(
                hyperparameters[4], activation=hyperparameters[5]),
            keras.layers.Dense(1)  # 1 output (redshift)
        ])

        model.compile(loss=loss_metric, optimizer=opt,
                      metrics=evaluation_metrics)

        # Train the model on the current fold
        # Adjust the number of epochs and batch size as needed
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Evaluate the model on the validation set for the current fold
        loss, metrics = model.evaluate(X_val, y_val)

        # Store the evaluation metrics for the current fold
        all_loss.append(loss)
        all_metrics.append(metrics)

    # Print the average evaluation metrics across all folds
    print('Average loss:', sum(all_loss) / len(all_loss))
    print('Average metrics:', sum(all_metrics) / len(all_metrics))

    return model


def deeper_model(n,
                 hyperparameters=[100, 'relu', 100, 'relu', 100, 'relu'],
                 loss_metric='mean_squared_error',
                 evaluation_metrics=['mae'],
                 opt='Nadam'):
    ''' Define the prediction model. The NN takes magnitudes as input features
    and outputs the redshift. n should be len(train_set.keys())'''
    model = keras.Sequential([
        keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1],  # number of outputs to next layer
                           input_shape=[n]),  # number of features
        keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
        keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
        keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
        keras.layers.Dense(1)  # 1 output (redshift)
    ])

    model.compile(loss=loss_metric, optimizer=opt, metrics=evaluation_metrics)
    print(model.summary())
    return model


# %% Load data and define train and validation sets
dl = DataLoader(dropna=False,
                colours=False,
                impute_method=None)
path = r'../../data_files'
number_of_rows = None
dataset, datasetname, magnames, mags = dl.load_data('milli_x_gleam_fits', path,
                                                    number_of_rows=number_of_rows)
# Filter for non-nan values of my alpha and drop other features
columns_to_drop = ['Rmag', 'Bmag', 'alpha_thin']
filtered_dataset = dataset[columns_to_drop].notnull()

# Drop columns with all NaN values
filtered_dataset = filtered_dataset.dropna(axis=1, how='all')

# mags = mags[mags[column_to_drop].notnull()]
# mags.dropna(axis=1, how='all', inplace=True)

# %% Define train and test sets
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kfold_splits = 20

# %% Define and evaluate base model
regressor_params = {'epochs': 5,
                    'batch_size': 5,
                    'verbose': -1}

model_params = {'n': len(mags.columns),
                'hyperparameters': [100, 'relu', 100, 'relu', 100, 'relu'],
                'loss_metric': 'mean_squared_error',
                'evaluation_metrics': ['mae'],
                'opt': 'Adam',
                # 'X': mags.values,
                # 'y': dataset["redshift"].values
                }

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
                'loss_metric': 'mean_squared_error',
                'evaluation_metrics': ['mae'],
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

print(f"Baseline:\t{base_results.mean():.2f}\t({base_results.std():.2f}) MSE\
      \t{median_absolute_error(y_test, y_pred_base):.2f} MAE")
print(f"Standard:\t{std_results.mean():.2f}\t({std_results.std():.2f}) MSE")
print(f"Deeper:\t{deep_results.mean():.2f}\t({deep_results.std():.2f}) MSE\
      \t{median_absolute_error(y_test, y_pred_deep):.2f} MAE")
print(f"Wider:\t{wide_results.mean():.2f}\t({wide_results.std():.2f}) MSE\
      \t{median_absolute_error(y_test, y_pred_wide):.2f} MAE")


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

sample_sizes = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5619]
r_squared_list = [0.057, 0.005, 0.068, 0.01,
                  0.001, 0.005, 0.029, 0.004, 0.007, 0.011]
sigma_list = [0.789, 0.866, 0.658, 0.818,
              0.743, 0.779, 0.744, 0.781, 0.701, 0.741]
fig, ax = plt.subplots()
ax.plot(sample_sizes, r_squared_list,
        label=r'$r^2$ for mean $z_\mathrm{phot}$', marker='o')
ax.plot(sample_sizes, sigma_list,
        label=r'$\sigma$ for $z_\mathrm{phot}$ distribution', marker='o')
ax.set_xlabel('Sample size')
ax.set_title(r'Effect of increasing sample size of $r^2$ and $\sigma$')
ax.grid(True)
ax.legend()
