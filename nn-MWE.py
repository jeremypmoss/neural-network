# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 08:30:36 2023

@author: JeremyMoss
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow_docs.modeling
import tensorflow_docs as tfdocs
import pandas as pd
import time
import matplotlib.pyplot as plt

start_time = time.time()

def make_test_df(n_galaxies=10, n_mags=5, seed=0, file_name = 'test_dataset.csv'):
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
    path = ''
    df.to_csv(path + file_name, index = False)

def loaddata(name, colours = False, impute_method = None, cols = None,
             dropna = True, number_of_rows = 'all'):
    path = ''
    df = pd.read_csv(path + 'test_dataset.csv',
                     sep = ',', index_col = False, header = 0)
    
    datasetname = 'Test dataset'
    print('Colours cannot be computed for the test frame')
    magnames = df.columns[3:-1]
    
    mgf = df[magnames]
    df = df.where(df != -999, np.nan)
    mgf = mgf.where(mgf != -999, np.nan)
    
    return df, datasetname, magnames, mgf

def build_nn_model(n, hyperparameters, loss, metrics, opt):
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),

    keras.layers.Dense(1) # 1 output (redshift)
    ])

    model.compile(loss=loss,
                  optimizer = opt,
            metrics = metrics)
    return model

#%% Load data
make_test_df(10, 5, 0)
dataset, datasetname, magnames, mags = loaddata('test',
                                                   dropna = False,  # to drop NaNs
                                                   colours = False, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data

#%% Main body

hyperparams = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
epochs = 100
opt = 'Nadam'

num_trials = 3
mean_list = []
std_list = []
train_frac = 0.8

for i in range(num_trials):
    print('*'*58);print('Run {0} of {1}'.format(i+1, num_trials)); print('*'*58)
    X_train, X_test, y_train, y_test = train_test_split(mags, # features
                                                dataset['redshift'], # target
                                                train_size = train_frac)
    model = build_nn_model(len(mags.columns), hyperparams, loss, metrics, opt)
    model.summary()
    early_stop = keras.callbacks.EarlyStopping(patience=100)
    
    history = model.fit(X_train, y_train, epochs = epochs,
                        validation_split = 1 - train_frac,
                        verbose = 0, callbacks = [early_stop,
                                                  tfdocs.modeling.EpochDots()])
    y_pred = model.predict(X_test)
    
    # Record the redshift predictions in the test set
    X_test['z_spec'] = y_test
    X_test['z_phot'] = y_pred
    X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot']
    
    stats = X_test['delta_z'].describe().transpose()
    mean, std = stats['mean'], stats['std']
    mean_list.append(mean)
    std_list.append(std)

# for i in range(num_trials):
    
print("Model completed in", time.time() - start_time, "seconds")
  
#%% Display means and standard deviations
border = '-'*25
separator = '\t\t|\t'
results_list = zip(mean_list, std_list)
print('Means' + separator + 'Std devs')
print(border)
for mean, dev, *_ in results_list:
    print(f"{mean:7f}\t|\t{dev:7f}")
print(border)
print('Average mean = {avg_mean}\nAverage std dev = {avg_std}'.format(
    avg_mean=np.mean(mean_list),
    avg_std=np.mean(std_list)))
