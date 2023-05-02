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
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import metrics
from DataLoader import DataLoader

rc_fonts = {
    "text.usetex": True,
    'font.family': 'serif',
    'font.size': 20,
}
plt.rcParams.update(rc_fonts)

start_time = time.time()


def plot_feature_importance(feature_importance):
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(np.arange(model_params["n_estimators"]) + 1,
             reg.train_score_, "b-", label="Training Set Deviance")
    plt.plot(np.arange(model_params["n_estimators"]) + 1, test_score, "r-",
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
dl = DataLoader(dropna=False,
                colours=True,
                impute_method='max')

dataset, datasetname, magnames, mags = dl.load_data('sdssmags')

X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

qf.training_validation_z_sets(y_train, y_test, datasetname)

# %% GradientBoostingRegressor
GBR = GradientBoostingRegressor()
grid_params = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
               'subsample': [0.9, 0.5, 0.2, 0.1],
               'n_estimators': [100, 500, 1000, 1500, 2000],
               'max_depth': [4, 6, 8, 10],
               'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
               'criterion': ['friedman_mse', 'squared_error']}

grid_GBR = GridSearchCV(estimator=GBR, verbose=2,
                        param_grid=grid_params, cv=2, n_jobs=-1)
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
model_params = {
    "n_estimators": grid_GBR.best_estimator_.n_estimators,
    "max_depth": grid_GBR.best_estimator_.max_depth,
    "min_samples_split": 5,
    "learning_rate": grid_GBR.best_estimator_.learning_rate,
    "loss": 'squared_error',
}

# model_params = { # these are copied in from the best parameters from the grid search
#     "n_estimators": 2000,
#     "max_depth": 10,
#     "min_samples_split": 5,
#     "learning_rate": 0.01,
#     "loss": 'squared_error',
# }

reg = GradientBoostingRegressor(**model_params)
model = reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((model_params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

delta_z = y_test - y_pred
print(f'Mean absolute error: {metrics.median_absolute_error(y_test, y_pred)}')
print("Model completed in", time.time() - start_time, "seconds")

# %% Visualise model results
plot_feature_importance(reg.feature_importances_)

# %% Visualise prediction fit
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(y_test, y_pred, datasetname, ax=ax[0])
qf.plot_delta_z_hist(delta_z, datasetname, model, ax=ax[1])
qf.metrics_table(y_test, y_pred)

print(f"Script completed in {time.time() - start_time:.1f} seconds")

# %% Predict for new dataset
new, newname, newmagnames, newmags = dl.load_data('galexqso')
new_pred = reg.predict(new)

# %%
new['z_pred'] = new_pred
qf.plot_one_z_set(new_pred, newname)
qf.compare_z(new_pred, dataset['redshift'],
             set1name=newname,
             set2name=datasetname,
             yscale='linear')
