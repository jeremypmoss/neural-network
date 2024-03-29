# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 08:30:36 2023

@author: JeremyMoss
"""
import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error
from tensorflow import keras

import tensorflow_docs.modeling
import tensorflow_docs as tfdocs
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


start_time = time.time()

def make_test_df(n_galaxies=10, n_mags=5, seed=0, file_name = 'test_dataset.csv'):
    # Make a toy dataset
    if seed:
        seed = seed
    else:
        seed = np.random.seed()
    np.random.seed(seed)
    data = np.random.uniform(10, 20, (n_galaxies,n_mags))
    try:
        data[np.diag_indices(n_mags)] = np.nan
    except IndexError:
        print('Cannot generate dataset: n_galaxies ({0}) must be >= n_mags ({1})'.format(n_galaxies, n_mags))
    np.random.shuffle(data)
    
    magnames = [f'mag{i}' for i in range(1, n_mags + 1)]
    
    df = pd.DataFrame(data, columns=magnames)
    df.insert(0, 'Name', [f'Galaxy {i}' for i in range(1, n_galaxies + 1)])

    # Generate redshift, RA and dec
    df['redshift'] = np.random.uniform(0.01, 5, n_galaxies) # generate redshift col
    df['RAJ2000'] = np.random.uniform(8, 8.1, n_galaxies)   # generate RA col
    df['DEJ2000'] = np.random.uniform(5, 5.1, n_galaxies)   # generate dec col

    # Move RA and dec to positions 1 and 2
    df.insert(1, 'RAJ2000', df.pop('RAJ2000'))
    df.insert(2, 'DEJ2000', df.pop('DEJ2000'))

    # Save as file
    path = 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code'
    df.to_csv(path + file_name, index = False)

def loaddata(name, colours = False, impute_method = None, cols = None,
             dropna = True, number_of_rows = 'all'):
    # Load a dataset
    path = ''
    df = pd.read_csv(path + 'test_dataset.csv',
                     sep = ',', index_col = False, header = 0)
    
    datasetname = 'Test dataset'
    print('Colours cannot be computed for the test frame')
    magnames = df.columns[3:-1]
    
    mgf = df[magnames]
    df = df.where(df != -999, np.nan)
    mgf = mgf.where(mgf != -999, np.nan)
    
    # Inspect structure of missing data; requires dropna = False in qf.loaddata()
    if dropna:
        df = df.dropna(axis = 0, how = 'any')
        mgf = mgf.dropna(axis = 0, how = 'any')
        print('NaNs have been dropped from the original data.')
    else: pass

    if impute_method == 'max':
        df = df.fillna(df.max()) # using max() assumes missing data are due to detection limit
        mgf = mgf.fillna(mgf.max())
        print('Missing values have been imputed with the maximum for each column.')
    elif impute_method == 'mean':
        impute_mean = SimpleImputer(missing_values = np.nan,
                                    strategy = 'mean')
        # impute_mean.fit(df)
        impute_mean.fit(mgf)
        # impute_mean.transform(df)
        mgf = impute_mean.transform(mgf) # converts to np.array
        mgf = pd.DataFrame(mgf, columns = magnames) # back to DataFrame
        print('Missing values have been imputed with the mean for each column.')
    
    return df, datasetname, magnames, mgf

def build_nn_model(n, hyperparameters, loss, metrics, opt):
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
    keras.layers.Dense(1)]) # 1 output (redshift)

    model.compile(loss=loss,
                  optimizer = opt,
                  metrics = metrics)
    print(model.summary())
    return model

#%% Load data
make_test_df(100, 20, 0)
dataset, datasetname, magnames, mags = loaddata('test',
                                                   dropna = False,  # to drop NaNs
                                                   colours = False, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data

#%% Train and fit a model

hyperparams = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
epochs = 100
opt = 'Nadam'

train_frac = 0.8

X_train, X_test, y_train, y_test = train_test_split(mags, # features
                                                dataset['redshift'], # target
                                                train_size = train_frac)

model = KerasRegressor(build_nn_model(len(mags.columns), hyperparams, loss, metrics, opt))

early_stop = keras.callbacks.EarlyStopping(patience=100)

history = model.fit(X_train, y_train, epochs = epochs,
                    validation_split = 1 - train_frac,
                    verbose = 0,
                    callbacks = [early_stop, tfdocs.modeling.EpochDots()])
y_pred = model.predict(X_test)

print("Model completed in", (time.time() - start_time)/60, "minutes")

#%% Grid search through the possible hyperparameters
parameters = {'loss'         : ['squared_error', 'absolute_error', 'huber', 'quantile'],
              'optimizer'    : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
              'epochs'       : [10, 50, 100],
              'batch_size'   : [5, 10, 50]
              # what else can I try in here?
}

grid = GridSearchCV(estimator = model,
                    param_grid = parameters,
                    scoring = 'accuracy',
                    n_jobs = -1, # use all processors
                    refit = 'boolean',
                    verbose = 4)
grid_result = grid.fit(X_train, y_train)

mse_krr = mean_squared_error(y_test, y_pred)
print(mse_krr)
print(grid.best_params_)
print(grid.best_estimator_)

print("Optimisation completed in", (time.time() - start_time)/60, "minutes")

plt.scatter(y_test, y_pred)
