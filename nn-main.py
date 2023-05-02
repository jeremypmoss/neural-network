# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

import matplotlib.pyplot as plt
import time
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from tensorflow import keras
from sklearn.model_selection import train_test_split
import quasar_functions as qf
from DataLoader import DataLoader
import numpy as np

rc_fonts = {
    "text.usetex": True,
    'font.family': 'serif',
    'font.size': 20,
}
plt.rcParams.update(rc_fonts)

start_time = time.time()

# %% Load data and define train and validation sets
dl = DataLoader(dropna=False,
                colours=True,
                impute_method='max')
dataset, datasetname, magnames, mags = dl.load_data('sdssmags')

test_frac = 0.2
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)

# %% Model
model_params = {'n': len(mags.columns),
                'hyperparameters': [100, 'relu', 100, 'relu', 100, 'relu'],
                'loss': 'mean_squared_error',
                'metrics': ['mae'],
                'opt': 'Nadam'}

model = qf.build_nn_model(**model_params)
model.summary()
early_stop = keras.callbacks.EarlyStopping(patience=100)

history = model.fit(X_train, y_train, epochs=100,
                    validation_split=test_frac,
                    verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
y_pred = model.predict(X_test)

X_test['z_spec'] = y_test
X_test['z_phot'] = y_pred
X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
print("Model completed in", time.time() - start_time, "seconds")

# %% Plot results
y_pred = y_pred.flatten()
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
qf.plot_z(y_pred, y_test, datasetname, ax=ax[0])
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, model, ax=ax[1])
qf.kurt_result(X_test['delta_z'])
qf.metrics_table(y_test, y_pred)

print(f"Script completed in {time.time() - start_time:.1f} seconds")

# %% Load and predict for a new set
new, newname, newmagnames, newmags = dl.load_data('skymapper_wise')

new_pred = model.predict(new)
new['z_pred'] = new_pred
# %% Plot predictions for new set
qf.compare_z(new_pred, dataset['redshift'],
             set1name=newname,
             set2name=datasetname,
             yscale='linear')
