### 06/16/23
# use filtered augmented data, scaling = .2

from src.utils import *
from src.optimize import *
import numpy as np
import pandas as pd
from scipy import stats
import os
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb


input_path = 'Data/ML_features'

with open(f'{input_path}/augmentset.pkl', 'rb') as f:
    data_dict_loaded = pkl.load(f)

X_features = data_dict_loaded['X_features']
y_true = data_dict_loaded['y_true']

X_features_aug = data_dict_loaded['X_features_aug']
y_true_aug = data_dict_loaded['y_true_aug']


def para_search_with_aug(seed, X_features, X_features_aug, y_true, y_true_aug):
    # Define the search space 
    rf_param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True]
    }

    xgb_param_dist = {
        'n_estimators': [50, 100, 200, 250],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }

    # Stack the original and augmented data
    stacked_X = np.vstack((X_features, X_features_aug))
    stacked_y = np.concatenate((y_true, y_true_aug))

    # Create models
    rf = RandomForestRegressor(random_state=seed)
    xgb_model = xgb.XGBRegressor(random_state=seed)

    # Perform Random Search with cross-validation for Random Forest
    rf_random = RandomizedSearchCV(estimator=rf, 
                                   param_distributions=rf_param_dist, 
                                   n_iter=50, 
                                   cv=10, 
                                   random_state=seed, 
                                   n_jobs=-1)
    rf_random.fit(stacked_X, stacked_y)

    best_rf = rf_random.best_estimator_

    # Perform Random Search with cross-validation for XGBoost
    xgb_random = RandomizedSearchCV(estimator=xgb_model, 
                                     param_distributions=xgb_param_dist, 
                                     n_iter=100, 
                                     cv=10, 
                                     random_state=seed, 
                                     n_jobs=-1)
    xgb_random.fit(stacked_X, stacked_y)

    best_xgb = xgb_random.best_estimator_

    return rf_random.best_params_, xgb_random.best_params_


def avg_performance_with_aug(rf_best, xgb_best, X_features, X_features_aug, y_true, y_true_aug):
    # Random seeds to get average performance
    random_seeds = [42, 123, 456, 789, 1011]
    n_fold = 10  # Number of folds for cross-validation

    rf_corr_list = []
    rf_rmse_list = []
    xgb_corr_list = []
    xgb_rmse_list = []

    # Stack the original and augmented data
    stacked_X = np.vstack((X_features, X_features_aug))
    stacked_y = np.concatenate((y_true, y_true_aug))

    # Get the number of original samples
    n_original_samples = X_features.shape[0]

    # Evaluate the models with different random seeds
    for seed in random_seeds:
        np.random.seed(seed)

        # Create the KFold object for cross-validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

        rf_corr_fold = []
        rf_rmse_fold = []
        xgb_corr_fold = []
        xgb_rmse_fold = []

        # Perform cross-validation
        for train_index, test_index in kf.split(stacked_X):
            X_train, X_test = stacked_X[train_index], stacked_X[test_index]
            y_train, y_test = stacked_y[train_index], stacked_y[test_index]

            # Get the original test indices
            original_test_index = test_index[test_index < n_original_samples]

            # Create and train the Random Forest model
            rf_model = RandomForestRegressor(**rf_best, random_state=seed)
            rf_model.fit(X_train, y_train)

            # Create and train the XGBoost model
            xgb_model = xgb.XGBRegressor(**xgb_best, random_state=seed)
            xgb_model.fit(X_train, y_train)

            # Evaluate the models on the original testing data
            rf_pred = rf_model.predict(X_features[original_test_index])
            rf_corr = np.corrcoef(rf_pred, y_true[original_test_index])[0, 1]
            rf_rmse = np.sqrt(mean_squared_error(y_true[original_test_index], rf_pred))

            xgb_pred = xgb_model.predict(X_features[original_test_index])
            xgb_corr = np.corrcoef(xgb_pred, y_true[original_test_index])[0, 1]
            xgb_rmse = np.sqrt(mean_squared_error(y_true[original_test_index], xgb_pred))

            rf_corr_fold.append(rf_corr)
            rf_rmse_fold.append(rf_rmse)
            xgb_corr_fold.append(xgb_corr)
            xgb_rmse_fold.append(xgb_rmse)

        # Calculate the average performance across all folds
        rf_corr_avg = np.mean(rf_corr_fold)
        rf_rmse_avg = np.mean(rf_rmse_fold)
        xgb_corr_avg = np.mean(xgb_corr_fold)
        xgb_rmse_avg = np.mean(xgb_rmse_fold)

        rf_corr_list.append(rf_corr_avg)
        rf_rmse_list.append(rf_rmse_avg)
        xgb_corr_list.append(xgb_corr_avg)
        xgb_rmse_list.append(xgb_rmse_avg)

    print("Random Forest Average Performance:")
    print("R mean:", np.mean(rf_corr_list))
    print("R std:", np.std(rf_corr_list))
    print("RMSE mean:", np.mean(rf_rmse_list))
    print("RMSE std:", np.std(rf_rmse_list))
    print()
    print("XGBoost Average Performance:")
    print("R mean:", np.mean(xgb_corr_list))
    print("R std:", np.std(xgb_corr_list))
    print("RMSE mean:", np.mean(xgb_rmse_list))
    print("RMSE std:", np.std(xgb_rmse_list))

# Main script
if __name__ == "__main__":
    # Load your data (X_features, X_features_aug, y_true, y_true_aug) here

    seed = 314159

    # Find the best hyperparameters using the augmented data
    rf_best, xgb_best = para_search_with_aug(seed, X_features, X_features_aug, y_true, y_true_aug)

    print("Best Random Forest Hyperparameters:")
    print(rf_best)
    print()
    print("Best XGBoost Hyperparameters:")
    print(xgb_best)
    print()

    # Evaluate the average performance using the best hyperparameters and augmented data
    avg_performance_with_aug(rf_best, xgb_best, X_features, X_features_aug, y_true, y_true_aug)

