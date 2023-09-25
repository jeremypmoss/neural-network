# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:02:48 2023

@author: JeremyMoss
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quasar_functions as qf
from sklearn.model_selection import train_test_split
from DataLoader import DataLoader
from matplotlib.pyplot import cm


start_time = time.time()

rc_fonts = {
    "text.usetex": False,
    'font.family': 'serif',
    'font.size': 20
}

plt.rcParams.update(rc_fonts)
np.random.seed(42)

# %% Load data and define train and validation sets
dl = DataLoader(dropna=False,
                colours=True,
                impute_method=None)
path = r'../../data_files'
number_of_rows = 3000
dataset, datasetname, magnames, mags = dl.load_data('gleam_x_milliquas', path,
                                                    number_of_rows=number_of_rows)

file_path = "./metrics_list.txt"
metrics_df = pd.read_csv(file_path)

# %% Define training and testing sets
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kfold_splits = 5

model_params = {'X': mags.values,
                'y': dataset["redshift"].values,
                'n': len(mags.columns),
                'num_layers': 3,
                'hyperparameters': [100, 'relu'],
                'loss_metric': 'mean_squared_error',
                'evaluation_metrics': ['mae'],
                'opt': 'Adam'
                }

# Define and train base model
base_model = qf.build_nn_model(**model_params)
# Define and train deep model
model_params['num_layers'] = 5  # change 'num_layers'
deep_model = qf.build_nn_model(**model_params)
# Define and train wide model
model_params['num_layers'] = 3
model_params['hyperparameters'] = [500, 'relu']  # change 'hyperparameters'
wide_model = qf.build_nn_model(**model_params)

# %% Fit the trained models to the test data

# Base model fit
fit_base = base_model.fit(X, y)
y_pred_base = base_model.predict(X_test)

# Deep model fit
fit_deep = deep_model.fit(X, y)
y_pred_deep = deep_model.predict(X_test)

# Wide model fit
fit_wide = wide_model.fit(X, y)
y_pred_wide = wide_model.predict(X_test)

# %% Display z_phot against z_spec and z distribution histogram

X_test = X_test.assign(z_spec=y_test,
                       z_phot_base=y_pred_base,
                       z_phot_deep=y_pred_deep,
                       z_phot_wide=y_pred_wide)

X_test['mean_z_phot'] = X_test[['z_phot_wide',
                                'z_phot_deep', 'z_phot_base']].mean(axis=1)


X_test = X_test.assign(mean_z_phot=X_test[['z_phot_wide', 'z_phot_deep', 'z_phot_base']].mean(axis=1),
                       delta_z_base=X_test['z_spec'] - X_test['z_phot_base'],
                       delta_z_deep=X_test['z_spec'] - X_test['z_phot_deep'],
                       delta_z_wide=X_test['z_spec'] - X_test['z_phot_wide'],
                       delta_z_mean=X_test['z_spec'] - X_test['mean_z_phot'])

# for array in []

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(X_test['z_spec'], X_test['mean_z_phot'],
          datasetname + ' mean', ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z_mean'],
                     datasetname + ' mean', base_model, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(X_test['z_spec'], X_test['z_phot_base'],
#           datasetname + ' base', ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_base'],
#                      datasetname + ' base', base_model, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(X_test['z_spec'], X_test['z_phot_wide'],
#           datasetname + ' wide', ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_wide'],
#                      datasetname + ' wide', base_model, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.tight_layout()
# qf.plot_z(X_test['z_spec'], X_test['z_phot_deep'],
#           datasetname + ' deep', ax=ax[0])
# qf.plot_delta_z_hist(X_test['delta_z_deep'],
#                      datasetname + ' deep', base_model, ax=ax[1])


# %% Print and save statistics for each model
models = [
    # 'Base model', 'Deep model', 'Wide model',
    'Mean of all models']
mean_z_phot = X_test[['z_phot_wide', 'z_phot_deep', 'z_phot_base']].mean(axis=1)
predictions = [
    # y_pred_base.flatten(), y_pred_deep.flatten(),
    # y_pred_wide.flatten(),
    mean_z_phot
]
model_names = [
    # 'Base model', 'Deep model', 'Wide model',
              'Mean of all models']

for model, prediction, name in zip(models, predictions, model_names):
    print(model)
    mean_metrics_list = qf.metrics_table(y_test, prediction, name)
mean_metrics_list["n"] = [number_of_rows]

metrics_df = pd.DataFrame(mean_metrics_list)

# Update metrics file
metrics_df = metrics_df.concat(pd.DataFrame(mean_metrics_list), ignore_index=True)
metrics_df.to_csv(file_path, index=False, sep=',')


# %% Display plots of z_phot distributions for all models

n = 5
color = iter(cm.hsv(np.linspace(0, 0.8, n)))
binwidth = 0.1
alpha = 0.4
edges = '0.1'
fig, ax = plt.subplots()

# Define the data and labels for each histogram
data = [dataset['redshift'],
        # y_pred_base,
        # y_pred_deep,
        # y_pred_wide,
        X_test['mean_z_phot']]
labels = [r'$z_{\mathrm{spec}}$ from ' + datasetname,
          # r'$z_{\mathrm{phot}}$ for baseline model',
          # r'$z_{\mathrm{phot}}$ for deeper model',
          # r'$z_{\mathrm{phot}}$ for wider model',
          r'mean $z_{\mathrm{phot}}$ for all models']

for d, lbl in zip(data, labels):
    ax.hist(d,
            bins=np.arange(min(d), max(d) + binwidth, binwidth),
            density=True,
            color=next(color),
            edgecolor=edges,
            alpha=alpha,
            label=lbl)

ax.grid(True)
ax.set_xlabel('Redshift')
ax.set_yscale('linear')
ax.set_ylabel('Count')
ax.set_title('Redshift distributions of\n{0} and some predicted redshifts'.format(
    r'$z_{\mathrm{spec}}$ from ' + datasetname))
ax.legend(loc='upper right')
