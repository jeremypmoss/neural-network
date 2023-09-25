# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import time
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from tensorflow import keras
from sklearn.model_selection import train_test_split
import quasar_functions as qf
from DataLoader import DataLoader
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor
from matplotlib.pyplot import cm

rc_fonts = {
    "text.usetex": False,
    'font.family': 'serif',
    'font.size': 20
}

plt.rcParams.update(rc_fonts)

start_time = time.time()


def mag_dist(band):
    fig, ax = plt.subplots()
    bins = 50
    alpha = 0.5
    density = True

    ax.hist(dataset[band], bins=bins, alpha=alpha,
            density=density, label=datasetname)
    # ax.hist(new[band], bins=bins, alpha=alpha, density=density, label=newname)
    ax.grid(True)
    ax.set_xlabel(band)
    ax.legend()


# %% Load data and define training and validation sets
dl = DataLoader(dropna=False,
                colours=False,
                impute_method='max')
# Specify the number of rows to load
path = r'../../data_files'
number_of_rows = None
dataset, datasetname, magnames, mags = dl.load_data('mq_x_gleam_nonpeaked_with_z',
                                                    path, number_of_rows=number_of_rows)

test_frac = 0.2
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)

# %% Model
model_params = {'n': len(mags.columns),
                'hyperparameters': [100, 'relu', 100, 'relu', 100, 'relu'],
                'loss_metric': 'mean_squared_error',
                'evaluation_metrics': ['mae'],
                'opt': 'Nadam'}

'''
The following section is from nn-keras-compare. It defines the base model in
that script.
This section passes the main model to the KerasRegressor without standardising
the data.
'''
kfold_splits = 5
regressor_params = {'epochs': 5,
                    'batch_size': 5,
                    'verbose': -1}
baseline_model = qf.build_nn_model_old(**model_params)
baseline_regressor = KerasRegressor(model=baseline_model, **regressor_params)
kfold = KFold(n_splits=kfold_splits)

# Perform k-fold cross validation on the baseline model
baseline_results = cross_val_score(baseline_regressor, X_train, y_train,
                                   cv=kfold, scoring='neg_mean_squared_error')

fit_base = baseline_model.fit(X, y)  # fit the baseline model to the target data
# generate z_phots from baseline model
y_pred_base = baseline_model.predict(X_test)
'''
End section
'''
# This defines the main model used for this script
model = qf.build_nn_model_old(**model_params)

history = model.fit(X_train, y_train, epochs=100,
                    validation_split=test_frac,
                    verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=100),
                                          tfdocs.modeling.EpochDots()])
y_pred = model.predict(X_test)

X_test['z_spec'] = y_test
X_test['z_phot'] = y_pred
X_test['baseline_z_phot'] = y_pred_base

X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
X_test['delta_z_base'] = X_test['z_spec'] - X_test['baseline_z_phot']

print("Model completed in", time.time() - start_time, "seconds")

# %% Evaluate models with standardised dataset
'''
This section standardises the data, then passes it to the model within the
KerasRegressor wrapper.

For each feature (column) in the input data, the StandardScaler calculates the
mean and standard deviation. It then subtracts the mean from each feature and
divides the result by the standard deviation. This process ensures that each
feature has a mean of zero and a standard deviation of one.

The 'mlp' model is created using the KerasRegressor class, which wraps a Keras
model and provides a scikit-learn compatible interface for regression tasks.
The actual architecture and configuration of the 'mlp' model is defined
earlier in the code, and the instance of the model is passed to the KerasRegressor
when it is created.
The 'mlp' model is model or baseline_model, with the defined hyperparameters.
The model takes the standardized input features and learns to predict the target
variable based on them through the process of training.
The training is performed using the specified number of epochs and batch size,
which control the number of iterations and the number of samples processed before
updating the model's weights.
The performance of the 'mlp' model is evaluated using the negative mean squared error.
'''
# Main model
regressor = KerasRegressor(model=model, **regressor_params)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
main_results = cross_val_score(pipeline, X_train, y_train,
                               cv=kfold, scoring='neg_mean_squared_error')
print('Cross-validation results:\n' + str(main_results),
      '\nMean =', np.mean(main_results),
      '\nStd  =', np.std(main_results))

# Baseline model
regressor = KerasRegressor(model=baseline_model, **regressor_params)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=kfold_splits)
baseline_results = cross_val_score(pipeline, X_train, y_train,
                                   cv=kfold, scoring='neg_mean_squared_error')
print('Cross-validation results:\n' + str(baseline_results),
      '\nMean =', np.mean(baseline_results),
      '\nStd  =', np.std(baseline_results))

# %% Plot results
y_pred = y_pred.flatten()
y_pred_base = y_pred_base.flatten()

# Plot z_phot against z_spec for the original model
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(y_pred, y_test, datasetname, ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, model, ax=ax[1])

# Plot z_phot against z_spec for the baseline model
# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(y_pred_base, y_test, datasetname, ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_base'], datasetname, model, ax=ax[1])

qf.kurt_result(X_test['delta_z'])
qf.metrics_table(y_test, y_pred, 'Neural Network')

print("Baseline:\t%.2f\t(%.2f) MSE\t%.2f MAE" % (baseline_results.mean(),
      baseline_results.std(), median_absolute_error(y_test, y_pred_base)))
print("Main:\t%.2f\t(%.2f) MSE\t%.2f MAE" % (main_results.mean(),
      main_results.std(), median_absolute_error(y_test, y_pred)))
print(f"Script completed in {time.time() - start_time:.1f} seconds")

# %% Load and predict for a new set
# new, newname, newmagnames, newmags = dl.load_data('skymapper_wise', path,
#                                                   number_of_rows=number_of_rows)

# # Why TF is this producing a shitload of z=-0.825?
# new_pred = model.predict(new)
# new['z_phot'] = new_pred

# %% Plot predictions for new set


# for band in magnames: # Magnitude distributions
#     mag_dist(band)

# set_names = [
#     r'$z_{\mathrm{spec}}$ from ' + datasetname,
#     r'$z_{\mathrm{phot}}$ for main model',
#     # r'$z_{\mathrm{phot}}$ for baseline model',
#     r'$z_{\mathrm{phot}}$ for Skymapper'
# ]

# bins = 50
# alpha = 0.6
# edges = '0.2'

# data = [y_test,
#         y_pred,
#         # y_pred_base,
#         new_pred
#         ]

# n = len(data)
# colors = iter(cm.hsv(np.linspace(0, 0.75, n)))

# fig, ax = plt.subplots()

# for i in range(n):
#     ax.hist(data[i],
#             bins=bins,
#             density=True,
#             color=next(colors),
#             edgecolor=edges,
#             alpha=alpha,
#             label=set_names[i])

# ax.grid(True)
# ax.set_xlabel('Redshift')
# ax.set_yscale('linear')
# ax.set_ylabel('Count')
# ax.set_title(
#     'Redshift distributions of {0}\nand some predicted redshifts'.format(set_names[0]))
# ax.legend(loc='upper right')
