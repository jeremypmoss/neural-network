# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:38:07 2022

@author: JeremyMoss
"""

# %% Imports

import quasar_functions as qf
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import wandb
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')

start_time = time.time()


def plot_feature_importance(feature_importance):
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(np.arange(params["n_estimators"]) + 1,
             reg.train_score_, "b-", label="Training Set Deviance")
    plt.plot(np.arange(params["n_estimators"]) + 1, test_score, "r-",
             label="Test Set Deviance")
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    plt.grid(which='both', axis='both')
    fig.tight_layout()
    plt.show()

    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(mags.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")
    plt.grid(which='both', axis='x')

    result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)

    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=np.array(mags.columns)[sorted_idx],)
    plt.title("Permutation Importance (test set)")
    plt.grid(which='both', axis='x')
    fig.tight_layout()
    plt.show()

    r = permutation_importance(model, X_test, y_test,
                               n_repeats=30,
                               random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{mags.columns[i]:<10}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")


# %% Load training/validation dataset
dataset, datasetname, magnames, mags = qf.loaddata('milli_x_gleam_fits',
                                                   dropna=False,
                                                   colours=False,
                                                   impute_method='max')

X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

qf.training_validation_z_sets(y_train, y_test, datasetname)

# %% Load unknown dataset
skymap, skymapname, skymapmagnames, skymapmags = qf.loaddata('skymapper_wise',
                                                             dropna=False,
                                                             colours=False,
                                                             impute_method='max')

# %% GradientBoostingRegressor
GBR = GradientBoostingRegressor()
parameters = {
    'learning_rate': [0.01, 0.02, 0.03, 0.04],
    'subsample': [0.9, 0.5, 0.2, 0.1],
    'n_estimators': [100, 500, 1000, 1500, 2000],
    'max_depth': [4, 6, 8, 10],
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'criterion': ['friedman_mse', 'squared_error']
}

wandb.init(project='GBR_{0}'.format(datasetname),
           config={'epochs': 4, 'batch_size': 32, 'lr': 'learning_rate'})
wandb.log({'acc': 0.9, 'loss': 0.1})

grid_GBR = GridSearchCV(estimator=GBR, verbose=2,
                        param_grid=parameters, cv=2, n_jobs=-1)
grid_GBR.fit(X_train, y_train)

# Results from Grid Search
# https://blog.paperspace.com/implementing-gradient-boosting-regression-python/
print(" Results from Grid Search")
print("\n The best estimator across ALL searched params:\n",
      grid_GBR.best_estimator_)
print("\n The best score across ALL searched params:\n",
      grid_GBR.best_score_)
print("\n The best parameters across ALL searched params:\n",
      grid_GBR.best_params_)

print("Optimisation completed in", time.time() - start_time, "seconds")

# %% Build and train model
params = {
    "n_estimators": grid_GBR.best_estimator_.n_estimators,
    "max_depth": grid_GBR.best_estimator_.max_depth,
    "min_samples_split": 5,
    "learning_rate": grid_GBR.best_estimator_.learning_rate,
    "loss": 'squared_error',
}

# params = { # these are copied in from the best parameters from the grid search
#     "n_estimators": 2000,
#     "max_depth": 10,
#     "min_samples_split": 5,
#     "learning_rate": 0.01,
#     "loss": 'squared_error',
# }

reg = GradientBoostingRegressor(**params)
model = reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

delta_z = y_test - y_pred

print("Model completed in", time.time() - start_time, "seconds")

# %% Visualise model results
plot_feature_importance(reg.feature_importances_)

# Visualise prediction fit
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(y_test, y_pred, datasetname, ax=ax[0])
qf.plot_delta_z_hist(delta_z, datasetname, model, ax=ax[1])

print("Script completed in", time.time() - start_time, "seconds")

# %% Predict for new dataset

# skymap_pred = reg.predict(skymap)

# skymap['z_pred'] = skymap_pred
# qf.plot_one_z_set(skymap_pred, skymapname)
# qf.compare_z(skymap_pred, dataset['redshift'],
#                set1name = skymapname,
#                set2name = datasetname)
