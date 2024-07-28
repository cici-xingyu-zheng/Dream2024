
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator, MetaEstimatorMixin
import numpy as np
import xgboost as xgb

import sys
sys.path.append("/Users/xinzheng/Desktop/Desktop/DreamRF")
from src.utils import *


class MatchedKFold(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0] // 2
        base_kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in base_kf.split(np.arange(n_samples)):
            train_index_matched = np.concatenate([2*train_index, 2*train_index+1])
            test_index_matched = np.concatenate([2*test_index, 2*test_index+1])
            yield train_index_matched, test_index_matched

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def para_search(seed, X, y_true):
    # Define the search space 
    rf_param_dist = {
        'n_estimators': [200, 300, 400, 500, 600, 700, 800],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    xgb_param_dist = {
        'n_estimators': [200, 300, 400, 500, 600, 700, 800],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }
    
    # Create models
    rf = RandomForestRegressor(random_state=seed)
    xgb_model = xgb.XGBRegressor(random_state=seed)

    # Create the custom MatchedKFold splitter
    matched_cv = MatchedKFold(n_splits=10, shuffle=True, random_state=seed)

    # Perform Random Search with cross-validation for Random Forest
    rf_random = RandomizedSearchCV(estimator=rf, 
                                param_distributions=rf_param_dist, 
                                n_iter=50, 
                                cv=matched_cv, 
                                random_state=seed, 
                                n_jobs=-1)
    rf_random.fit(X, y_true)

    best_rf = rf_random.best_estimator_

    # Perform Random Search with cross-validation for XGBoost
    xgb_random = RandomizedSearchCV(estimator=xgb_model, 
                                    param_distributions=xgb_param_dist, 
                                    n_iter=100, 
                                    cv=matched_cv, 
                                    random_state=seed, 
                                    n_jobs=-1)
    xgb_random.fit(X, y_true)

    best_xgb = xgb_random.best_estimator_

    # Evaluate 
    rf_pred = best_rf.predict(X)
    rf_corr = np.corrcoef(rf_pred, y_true)[0, 1]
    rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))

    xgb_pred = best_xgb.predict(X)
    xgb_corr = np.corrcoef(xgb_pred, y_true)[0, 1]
    xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_pred))

    print("Best Random Forest model:")
    print("Hyperparameters:", rf_random.best_params_)
    print("Correlation:", rf_corr)
    print("RMSE:", rf_rmse)
    print()
    print("Best XGBoost model:")
    print("Hyperparameters:", xgb_random.best_params_)
    print("Correlation:", xgb_corr)
    print("RMSE:", xgb_rmse)

    return rf_random.best_params_, xgb_random.best_params_


def avg_rf_best(rf_best, X_features, y_true):
    # Random seeds to get average performance
    random_seeds = [42, 123, 456, 789, 1011]
    n_fold = 10  # Number of folds for cross-validation

    rf_corr_list = []
    rf_rmse_list = []

    # Evaluate the models with different random seeds
    for seed in random_seeds:
        np.random.seed(seed)
        
        # Create the Random Forest model with the best hyperparameters
        rf_model = RandomForestRegressor(**rf_best, random_state=seed)
        
        # Create indices for the original samples (before duplication)
        original_indices = np.arange(X_features.shape[0] // 2)
        
        # Create the KFold object for cross-validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        
        rf_corr_fold = []
        rf_rmse_fold = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(original_indices):
            # Convert original indices to the coupled indices
            train_index_coupled = np.concatenate([2*train_index, 2*train_index+1])
            test_index_coupled = np.concatenate([2*test_index, 2*test_index+1])
            
            X_train, X_test = X_features[train_index_coupled], X_features[test_index_coupled]
            y_train, y_test = y_true[train_index_coupled], y_true[test_index_coupled]
            
            # Train the model
            rf_model.fit(X_train, y_train)
            
            # Evaluate the model on the testing fold
            rf_pred = rf_model.predict(X_test)
            rf_corr = np.corrcoef(rf_pred, y_test)[0, 1]
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            rf_corr_fold.append(rf_corr)
            rf_rmse_fold.append(rf_rmse)
        
        # Calculate the average performance across all folds
        rf_corr_avg = np.mean(rf_corr_fold)
        rf_rmse_avg = np.mean(rf_rmse_fold)
        
        rf_corr_list.append(rf_corr_avg)
        rf_rmse_list.append(rf_rmse_avg)

    print("RandomForest Average Performance:")
    print("R mean:", np.mean(rf_corr_list))
    print("R std:", np.std(rf_corr_list))
    print("RMSE mean:", np.mean(rf_rmse_list))
    print("RMSE std:", np.std(rf_rmse_list))

    return np.mean(rf_corr_list), np.mean(rf_rmse_list)

def avg_rgb_best(rbg_best, X_features, y_true):
    # Random seeds to get average performance
    random_seeds = [42, 123, 456, 789, 1011]
    n_fold = 10  # Number of folds for cross-validation

    xgb_corr_list = []
    xgb_rmse_list = []

    # Evaluate the models with different random seeds
    for seed in random_seeds:
        np.random.seed(seed)
        
        # Create the XGBoost model with the best hyperparameters
        xgb_model = xgb.XGBRegressor(**rbg_best, random_state=seed)
        
        # Create indices for the original samples (before duplication)
        original_indices = np.arange(X_features.shape[0] // 2)
        
        # Create the KFold object for cross-validation
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        
        xgb_corr_fold = []
        xgb_rmse_fold = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(original_indices):
            # Convert original indices to the coupled indices
            train_index_coupled = np.concatenate([2*train_index, 2*train_index+1])
            test_index_coupled = np.concatenate([2*test_index, 2*test_index+1])
            
            X_train, X_test = X_features[train_index_coupled], X_features[test_index_coupled]
            y_train, y_test = y_true[train_index_coupled], y_true[test_index_coupled]
            
            # Train the model
            xgb_model.fit(X_train, y_train)
            
            # Evaluate the model on the testing fold
            xgb_pred = xgb_model.predict(X_test)
            xgb_corr = np.corrcoef(xgb_pred, y_test)[0, 1]
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            
            xgb_corr_fold.append(xgb_corr)
            xgb_rmse_fold.append(xgb_rmse)
        
        # Calculate the average performance across all folds
        xgb_corr_avg = np.mean(xgb_corr_fold)
        xgb_rmse_avg = np.mean(xgb_rmse_fold)
        
        xgb_corr_list.append(xgb_corr_avg)
        xgb_rmse_list.append(xgb_rmse_avg)

    print("XGBoost Average Performance:")
    print("R mean:", np.mean(xgb_corr_list))
    print("R std:", np.std(xgb_corr_list))
    print("RMSE mean:", np.mean(xgb_rmse_list))
    print("RMSE std:", np.std(xgb_rmse_list))

    return np.mean(xgb_corr_list), np.mean(xgb_rmse_list)