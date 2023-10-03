import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from DataLoader import DataLoader
import quasar_functions as qf

# Set up the style for plotting
rc_fonts = {
    "text.usetex": False,
    'font.family': 'serif',
    'font.size': 20,
}

# Merge the dictionaries
rc_params = {'figure.autolayout': True, **rc_fonts}

# Update the matplotlib rcParams
plt.rcParams.update(rc_params)

#%% Load data using DataLoader
class_params = {'dropna': False,
                'colours': False,
                'impute_method': 100.0} # None, 'max', 'mean' or float
load_params = {'name': 'test',
               'path': r'../../data_files',
               'number_of_rows': None,
               'binning': True,
               'selected_bin': None}

dl = DataLoader(**class_params)
dataset, datasetname, magnames, mags = dl.load_data(**load_params)

# Split the data into training and testing sets
X = mags
y = dataset['redshift']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('Training data shape:', X_train.shape)
print('Testing data shape:', X_test.shape)
print(y_train.shape, y_test.shape)

apply_neural_network = True

#%% Standardize all datasets and the target
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Apply PCA to the necessary datasets

from sklearn.model_selection import GridSearchCV

# Create a PCA instance without specifying the number of components
pca = PCA()
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

# Define a range of percentages of explained variance to consider
# For example, you can consider 95%, 98%, 99%, or any other desired value
variance_thresholds = [0.95, 0.98, 0.99]

# Create a parameter grid for GridSearchCV to search through
param_grid = {'n_components': [None] + [int(X_train.shape[1] * threshold) for threshold in variance_thresholds]}

# Perform a grid search with cross-validation to find the best number of components
grid_search = GridSearchCV(estimator=pca, param_grid=param_grid, cv=cv)
grid_search.fit(X_train_scaled)

# Get the best number of components
best_n_components = grid_search.best_params_['n_components']
# best_n_components = 2

# Apply PCA with the best number of components to the data
pca = PCA(n_components=best_n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check explained variance
print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')

# Now, best_n_components contains the optimal number of components
print(f'Optimal number of components: {best_n_components}')

# n_components = 4
pca = PCA(n_components=best_n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#%% Compare regression models

regr = LinearRegression(); regr_name = "Linear Regression"
# regr = Ridge(alpha=1.0); regr_name = "Ridge Regression"
# regr = Lasso(alpha=1.0); regr_name = "Lasso Regression"
# regr = ElasticNet(alpha=1.0, l1_ratio=0.5); regr_name = "ElasticNet"
# regr = SVR(kernel='linear', C=1.0); regr_name = "SVR"
# regr = RandomForestRegressor(n_estimators=100); regr_name = "Random Forest Regression"
# regr = GradientBoostingRegressor(n_estimators=100); regr_name = "Gradient Boosting Regression"

# Define the cross-validation strategy
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

mse = []

# Calculate MSE using cross-validation
score = -model_selection.cross_val_score(regr,
                                         X_train,
                                         y_train, cv=cv,
                                         scoring='neg_mean_squared_error').mean()
mse.append(score)
print(f"Mean Squared Error (MSE) from Cross-Validation ({regr_name}): {mse}")

n_components_to_try = np.arange(1, 25)

for i in n_components_to_try:
    score = -model_selection.cross_val_score(regr,
                                             X_train_pca[:, :i],
                                             y_train, cv=cv,
                                             scoring='neg_mean_squared_error').mean()
    mse.append(score)

# Plot cross-validation results
x = mse
x_ticks = list(range(len(x))) # create a list of integers for the x-axis ticks
n_components_labels = np.append(n_components_to_try, n_components_to_try[-1] + 1)

plt.scatter(x_ticks, x, marker='o', color='b')
plt.plot(x_ticks, x, color='r')
plt.xticks(x_ticks, labels=n_components_labels)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title(f'''Improvement in Mean Squared Error for {regr_name}
on {datasetname} with increasing principal components''')
plt.grid(True)

plt.show()

# Calculate percentage of variation explained
cum_percent = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(f'Cumulative percentage variation explained by each successive PC in {regr_name}: {cum_percent}')

# Scale the PCA-transformed training and testing data
X_train_pca_scaled = scaler.fit_transform(X_train_pca)
X_test_pca_scaled = scaler.transform(X_test_pca)

# Fit the regression model on reduced data
regr.fit(X_train_pca_scaled[:, :best_n_components], y_train)

# Calculate RMSE
pred = regr.predict(X_test_pca_scaled)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f'Root Mean Squared Error (RMSE) for {regr_name}: {rmse}')

# Visualize the predicted vs. actual redshift
X_test['z_spec'] = y_test
X_test[f'z_phot_{regr_name}'] = pred
X_test[f'delta_z_{regr_name}'] = X_test['z_spec'] - X_test[f'z_phot_{regr_name}']

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
title = f'Predicted redshift comparison\nfrom PCA and {regr_name}\nfor {datasetname}'
qf.plot_z(y_test, pred, datasetname, ax=ax[0], title=title)
qf.plot_delta_z_hist(X_test[f'delta_z_{regr_name}'], datasetname, ax=ax[1])

print('===== Results summary =====')
print(f'{regr_name}, {datasetname}, n = {y.shape[0]}')
print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')
print(f'Optimal number of components: {best_n_components}')
print(f"Mean Squared Error (MSE) from Cross-Validation ({regr_name}): {mse}")
print(f'Root Mean Squared Error (RMSE) for {regr_name}: {rmse}')
print(f'Cumulative percentage variation explained by each successive PC in {regr_name}: {cum_percent}')
print(qf.metrics_table(np.array(y_test), pred, regr_name))

#%% Apply neural network to the standardised, PCA'd datasets

if apply_neural_network:
    # Set X and y for the neural network
    X = X_train_pca_scaled  # Input features (standardised PCA-transformed training data)
    y = np.array(y_train)   # Target variable (redshift)

    # Model parameters
    model_params = {
        'X': X,
        'y': y,
        'n': X.shape[1],                   # Number of input features (number of PCA components)
        'num_layers': 3,                   # Number of hidden layers in your neural network
        'hyperparameters': [100, 'relu'],  # Number of neurons and activation function for hidden layers
        'loss_metric': 'mean_squared_error',  # Loss function for training
        'evaluation_metrics': ['mae'],        # Evaluation metrics during training
        'opt': 'Adam',                        # Optimizer
        'kf': cv                              # Cross-validation settings
    }

    # Build neural network model
    model = qf.build_nn_model(**model_params)

    # Make predictions on the test data
    y_pred = model.predict(X_test_pca_scaled)  # Use standardized PCA-transformed test data

    # Visualise the fit for the neural network
    print(qf.metrics_table(y_test, y_pred.flatten(), 'Neural network'))
    y_test = np.array(y_test)
    X_test['z_phot_NN'] = y_pred

    X_test['delta_z'] = X_test['z_spec'] - X_test['z_phot_NN']
#%% Visualize the predicted vs. spectroscopic redshift
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.tight_layout()
    title = f'Predicted redshift comparison from Neural Network\nfor {datasetname}'
    qf.plot_z(y_test, y_pred.flatten(), datasetname, ax=ax[0], title=title)
    qf.plot_delta_z_hist(X_test['delta_z'], datasetname, ax=ax[1])
