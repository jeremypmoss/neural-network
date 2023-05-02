# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

import numpy as np
# import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import quasar_functions as qf
from DataLoader import DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from sklearn import metrics

import tensorflow_docs.modeling
import tensorflow_docs as tfdocs
import pandas as pd
import time
import matplotlib.pyplot as plt

rc_fonts = {
    "text.usetex": True,
    'font.family': 'serif',
    'font.size': 20,
}
plt.rcParams.update(rc_fonts)

start_time = time.time()

# %% Load data and define train and validation sets
dl = DataLoader(dropna=False,
                colours=False,
                impute_method='max')
dataset, datasetname, magnames, mags = dl.load_data('sdssmags')

test_frac = 0.2
X_train, X_test, y_train, y_test = train_test_split(mags,  # features
                                                    # target
                                                    dataset['redshift'],
                                                    test_size=test_frac)

# %% Model
model_params = {'n': len(mags.columns),
                'hyperparameters': [100, 'relu', 100, 'relu', 100, 'relu'],
                'loss': 'mean_squared_error',
                'metrics': ['mae'],
                'opt': 'Nadam'}

model = KerasRegressor(qf.build_nn_model(**model_params))

early_stop = keras.callbacks.EarlyStopping(patience=100)

history = model.fit(X_train, y_train, epochs=100,
                    validation_split=test_frac,
                    verbose=0,
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
y_pred = model.predict(X_test)

X_test['z_spec'] = y_test
X_test['z_phot'] = y_pred
X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
print("Model completed in", (time.time() - start_time), "seconds")

# qf.grid_search_model(model, X_train, y_train, y_test, y_pred) # grid search

# %% Plot results

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()

qf.plot_z(X_test['z_spec'], X_test['z_phot'], datasetname, ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, model, ax=ax[1])

qf.kurt_result(X_test['delta_z'])
# qf.compare_z(y_train, X_test['z_spec'], datasetname, 'set 2')
qf.metrics_table(y_test, y_pred)

# %% Load and predict for a new set
new, newname, newmagnames, newmags = dl.load_data('skymapper_wise')

new_pred = model.predict(new)
new['z_pred'] = new_pred
# %% Plot predictions for new set
qf.plot_one_z_set(new_pred, newname)
qf.compare_z(new_pred, dataset['redshift'],
             set1name=newname,
             set2name=datasetname,
             yscale='linear')

print(f"Script completed in {time.time() - start_time:.1f} seconds")
