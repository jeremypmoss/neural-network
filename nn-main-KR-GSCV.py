# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:02:41 2023

@author: JeremyMoss
"""

import time
import quasar_functions as qf
from DataLoader import DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
import tensorflow_docs.modeling
import tensorflow_docs as tfdocs
import matplotlib.pyplot as plt

# Set matplotlib font settings
rc_fonts = {
    "text.usetex": False,
    'font.family': 'serif',
    'font.size': 20,
}
plt.rcParams.update(rc_fonts)

start_time = time.time()

# %% Load data and define train and validation sets

# Instantiate a DataLoader object to load the data
dl = DataLoader(dropna=False,
                colours=False,
                impute_method=None)

# Specify the number of rows to load
path = r'../../data_files'
number_of_rows = 3000

# Load the dataset into memory
dataset, datasetname, magnames, mags = dl.load_data('mq_x_gleam_nonpeaked_with_z', path,
                                                    number_of_rows=number_of_rows)

# Set the fraction of the dataset to use for testing
test_frac = 0.2

# Split the dataset into training and testing sets
X = mags
y = dataset['redshift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)

# %% Model
# Define the hyperparameters of the neural network model
model_params = {'n': len(mags.columns),
                'hyperparameters': [100, 'relu', 100, 'relu', 100, 'relu'],
                'loss': 'mean_squared_error',
                'metrics': ['mae'],
                'opt': 'Nadam'}

# Instantiate a KerasRegressor object with the specified hyperparameters
model = KerasRegressor(qf.build_nn_model(**model_params))

# Define an EarlyStopping callback to stop training when the validation loss stops improving
early_stop = keras.callbacks.EarlyStopping(patience=100)

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=100,
                    validation_split=test_frac,
                    verbose=0,
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Add the true and predicted redshift values to the testing set
X_test['z_spec'] = y_test
X_test['z_phot'] = y_pred

# Calculate the difference between the true and predicted redshift values
X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']

# Print the time taken to complete the model training and testing
print("Model completed in", (time.time() - start_time), "seconds")

# qf.grid_search_model(model, X_train, y_train, y_test, y_pred) # grid search

# %% Plot results
# Create a figure with two subplots for displaying the results
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()

# Plot the redshift predictions against the true redshift values
qf.plot_z(X_test['z_spec'], X_test['z_phot'], datasetname, ax=ax[0])

# Plot a histogram of the differences between the true and predicted redshift values
qf.plot_delta_z_hist(X_test['delta_z'], datasetname, model, ax=ax[1])

# Calculate and print the kurtosis of the difference between the true and predicted redshift values
qf.kurt_result(X_test['delta_z'])

# qf.compare_z(y_train, X_test['z_spec'], datasetname, 'set 2')
# Calculate and print a table of performance metrics for the model
qf.metrics_table(y_test, y_pred)

# %% Load and predict for a new set

# Load a new dataset to make predictions on
# new, newname, newmagnames, newmags = dl.load_data('skymapper_wise', path,
#                                                   number_of_rows=number_of_rows)

# # Make predictions on the new dataset using the trained model
# new_pred = model.predict(new)

# # Add the predicted redshift values to the new dataset
# new['z_pred'] = new_pred

# %% Plot predictions for new set

# qf.plot_one_z_set(new_pred, newname)
# Plot a comparison of the redshift predictions of the new dataset and the original dataset
# qf.compare_z(new_pred, dataset['redshift'],
#              set1name=newname,
#              set2name=datasetname,
#              yscale='log')
