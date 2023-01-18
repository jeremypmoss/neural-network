# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:00:42 2020

@author: Jeremy Moss
https://stackoverflow.com/questions/58939031/how-to-predict-new-data-with-a-
trained-neural-network-tensorflow-2-0-regression
"""

#%% Imports

import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import seaborn as sns
# import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.model_selection import GridSearchCV
# from tensorflow.keras import layers
from scipy.stats import gaussian_kde, kurtosis, kurtosistest
from sklearn.preprocessing import MinMaxScaler

import shap # https://www.analyticsvidhya.com/blog/2019/11/shapley-value-machine-learning-interpretability-game-theory
font = {'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)

#%% Definitions

def plot_mae():
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")
    plt.title('Mean Absolute Error evolution for %s'%datasetname)
    plt.ylabel(r'$\Delta z$')
    plt.show()

def plot_mse():
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mse")
    plt.title('Mean Standard Error evolution for %s'%datasetname)
    plt.ylabel(r'$\Delta z$')
    plt.show()
    
def build_model(n, hyperparameters, loss, metrics, opt, **kwargs):
    ''' Define the prediction model. The NN takes magnitudes as input features
    and outputs the redshift. n should be len(train_set.keys())'''
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),
    keras.layers.Dense(1) # 1 output (redshift)
    ])
    
    # if optim == "Adam":
    #     model.compile(loss=loss,
    #             optimizer = keras.optimizers.Adam(0.001),
    #             metrics = metrics)
    
    # if optim == "RMSprop":
    model.compile(loss=loss,
                  optimizer = opt,
            # optimizer = keras.optimizers.RMSprop(0.001),
            metrics = metrics)
    return model
#%% Load and preprocess data

import quasar_functions as qf

# qf.make_test_df(n_galaxies = 1000, n_mags = 15, seed = 0)
dataset, datasetname, magnames, mags = qf.loaddata('sdss12_x_ukidss',
                                                   dropna = False,  # to drop NaNs
                                                   colours = False, # to compute colours of mags
                                                   impute_method = 'max') # to impute max vals for missing data

#%% Split dataset into training and validation sets

### Copy this section to the experiment log in Notion #########################
features = mags
# number of neurons and activation function for each of the three layers:
hyperparameters = [100, 'relu', 100, 'relu', 100, 'relu']
loss = 'mae'
metrics = ['mae']
N = 100
opt = 'Nadam'

train_frac = 0.8

print('Features used as predictors:')
print(list(mags.columns))

#%% Normalise the data
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
# column_names = list(features.columns)
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(features)
# features = pd.DataFrame(scaled_data, columns = column_names)

#%% Build the model
num_trials = 3
mean_list = []
std_list = []
# TODO add predicted z for each run and plot the mean of these

for i in range(num_trials):
    print('*'*58);print('Run {0} of {1}'.format(i+1, num_trials)); print('*'*58)
    # can I replace this with the sets from GBR?
    X = train_dataset = dataset.sample(frac = train_frac, random_state = None)
    valid_dataset = dataset.drop(train_dataset.index)
    y = train_labels = train_dataset.loc[:, 'redshift'].values
    # print('Training set (first 5 rows): \n', train_dataset.head())
    # print('Validation set (first 5 rows): \n', valid_dataset.head())
    
    model = build_model(len(features.columns), hyperparameters, loss, metrics, opt)
        
    model.summary()
    early_stop = keras.callbacks.EarlyStopping(patience=100)
    
    # train_dataset = train_dataset[magnames]
    train_dataset = train_dataset[features.columns]
    valid_dataset_for_predictions = valid_dataset[features.columns]
    print(train_dataset)
    print(valid_dataset)
    # print(valid_dataset_for_predictions)

    history = model.fit(train_dataset, train_labels, epochs = N,
                        validation_split = 1 - train_frac,
                        verbose = 0, callbacks = [early_stop,
                                                 tfdocs.modeling.EpochDots()])
    
    # Model testing
    valid_predictions = model.predict(valid_dataset_for_predictions)
    # valid_predictions = model.predict(valid_dataset)
    valid_dataset['Predicted z'] = valid_predictions # add predictions column to dataset
    zpred_list = pd.DataFrame(valid_predictions, columns = ['Predicted z'+str(i+1)])
    # zpred_list['next zpred'] = pd.DataFrame(valid_predictions, columns = ['z'+str(i)])
    # zpred_list['Predicted z'+str(i)] = valid_predictions
    
    print("\nPredicted\n")
    print(valid_dataset)
    
    # Visualise the model's training progress using the stats in the history object
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail)
    zpred_list['zpred '+str(i)] = valid_predictions
    print('Spectroscopic z stats: \n', valid_dataset['redshift'].describe())
    print('Predicted z stats: \n', valid_dataset['Predicted z'].describe())
    valid_dataset['Delta z'] = valid_dataset['Predicted z'] - valid_dataset['redshift']
    print('Delta z stats: \n', valid_dataset['Delta z'].describe())
    
    stats = valid_dataset['Delta z'].describe().transpose()
    mean, std = stats['mean'], stats['std']
    mean_list.append(mean)
    std_list.append(std)
    
    print('Mean = {0}, std = {1}'.format(mean, std))

#%% Display means and standard deviations
# border = '-'*25
# separator = '\t\t|\t'
qf.print_two_lists(mean_list, std_list)
# print('Means' + separator + 'Std devs')
# print(border)
# for i in range(len(results_list)):
#     print('{0: 7f}\t|\t{1:7f}'.format(results_list[i][0], results_list[i][1]))
# print(border)
# print('Average mean = {0}\nAverage std dev = {1}'.format(np.mean(mean_list), np.mean(std_list)))
#%% Grid search - Tune Dropout Regularization
# define the grid search parameters
# weight_constraint = [1.0, 2.0, 3.0, 4.0, 5.0]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# param_grid = dict(model__dropout_rate=dropout_rate, model__weight_constraint=weight_constraint)
# #param_grid = dict(model__dropout_rate=dropout_rate)
# grid = GridSearchCV(estimator=model, scoring = 'accuracy', param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X, Y)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
    
#%% Plot results
import quasar_functions as qf # reload so this cell can be run separately

# dataset = scaler.inverse_transform(dataset)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 9))
fig.tight_layout()

# plot_mse() #https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
# plot_mae()
# qf.plot_deltaz(x, y, datasetname)
qf.plot_z(valid_dataset['redshift'], valid_dataset['Predicted z'], datasetname, ax = ax[0])
qf.plot_delta_z_hist(valid_dataset['Delta z'], datasetname, model, ax = ax[1])
# qf.plot_z_boxplot(dataset, x, y, datasetname, True, ax = None)
# qf.plot_z_boxplot(dataset, x, y, datasetname, False, ax = None)

qf.kurt_result(valid_dataset['Delta z'])
qf.plot_z_sets(train_labels, valid_dataset['redshift'], datasetname)

#%% Shapley vizualiser (not working)
'''
explainer = shap.DeepExplainer(model, train_dataset)
shap_values = explainer.shap_values(valid_predictions)

shap.summary_plot(shap_values, train_dataset['redshift'])

#%% Compare with SDSS12

# fig, ax = plt.subplots()
# x_var = valid_dataset['redshift']
# y_var = valid_dataset['Predicted z']
# xy = np.vstack([x_var,y_var])
# z = gaussian_kde(xy)(xy)
# ax.scatter(sdss12_compare[0]['redshift'], sdss12_compare[0]['zph'],
#             alpha = 0.1,
#             color = 'r',
#             label = 'zph',
#             marker = '.')
# ax.scatter(sdss12_compare[0]['redshift'], sdss12_compare[0]['<zph>'],
#             alpha = 0.1,
#             color = 'pink',
#             label ='<zph>',
#             marker = '.')
# ax.scatter(valid_dataset['redshift'], valid_dataset['Predicted z'],
#             c = z,
#             alpha = 0.3,
#             marker = '*',
#             label = 'Neural network predictions')
# ax.legend(loc = 'upper right')
# xpoints = ypoints = ax.get_xlim()
# ax.plot(xpoints, ypoints, linestyle='-', color='r', lw=3, scalex=False, scaley=False)
# ax.grid('both')
# ax.set_xlabel('Spectroscopic redshift')
# ax.set_ylabel('Photometric redshift')
# ax.set_title('Comparison of Neural Network photometric redshift predictions using\n{0}\nto those of {1}'.format(datasetname, sdss12_compare[1]))
# ax.set_ylim(0, 2.6)
'''