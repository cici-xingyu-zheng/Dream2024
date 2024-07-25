import os
import sys


from src.train_test import *
from src.utils import *
from src.optimize import *

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge 
import xgboost as xgb
from scipy.stats import pearsonr


sns.set_style('ticks')

input_path = 'Data/'

features_file_1 = 'featureSelection/selection_cleanMordredDescriptors.csv'
features_file_2 =  'deepnose_features.npy'
CID_file = 'molecules_train_cid.npy'

# Read all copies, before and after correction; before was also downloaded from Dropbox.
mixture_file = 'Mixure_Definitions_Training_set_UPD2.csv' 
training_task_file = 'TrainingData_mixturedist.csv'

# Mordred features
features_1 = pd.read_csv(os.path.join(input_path, features_file_1), index_col= 0)

features_2 = np.load(os.path.join(input_path, features_file_2))

features_CIDs = np.load(os.path.join(input_path, CID_file))
# Training dataframe
training_set = pd.read_csv(os.path.join(input_path, training_task_file))

# Mapping helper files
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))


feature_file_3 = 'Fingerprints/Morgan_Fingerprints_Frequency_Size50.csv'
features_3 = pd.read_csv(os.path.join(input_path, feature_file_3), index_col= 0)
features_file_4 =  'leffingwell_features_96.npy'
features_4 = np.load(os.path.join(input_path, features_file_4))


scaler = StandardScaler(with_mean=True, with_std=True)

# standardize Mordred
features_1_np = scaler.fit_transform(features_1)
features_1 = pd.DataFrame(features_1_np, columns=features_1.columns, index=features_1.index)


# log standardize deepnose
scaler = StandardScaler(with_mean=True, with_std=True)
epsilon = 1e-8 
features_2 = scaler.fit_transform(np.log(features_2 + epsilon))

# Map CID to features:
# Dense
CID2features_deepnose=  {CID: features_2[i] for i, CID in enumerate(features_CIDs)}
CID2features_mordred =  {CID: features_1.loc[CID].tolist() for CID in features_CIDs}

# Sparse
CID2features_morgan =  {CID: features_3.loc[CID].tolist() for CID in features_CIDs}
CID2features_leffingwell = {CID: features_4[i] for i, CID in enumerate(features_CIDs)}

# Make X_feature and y
features_list = [CID2features_mordred, CID2features_deepnose]
features_list_sparse = [CID2features_morgan, CID2features_leffingwell]

X_dense, y_true = stacking_X_features(features_list, "avg")
X_sparse, _ = stacking_X_features(features_list_sparse, "sum")

X_dense_new, y_test_true = stacking_X_test_features(features_list,  X_dense, "avg")
X_sparse_new, _ = stacking_X_test_features(features_list_sparse,  X_sparse, "sum")


n_folds = 10
seed = 314159

# obtained from individual optimization
best_rf_dense = {'n_estimators': 500, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
best_rf_sparse = {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.5, 'max_depth': 30, 'bootstrap': True}

def residual_ensemble_cv(X_dense, X_sparse, y, base_model_dense, base_model_sparse, n_folds=10):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=314159)
    
    dense_preds = np.zeros(len(y))
    sparse_preds = np.zeros(len(y))
    combined_preds = np.zeros(len(y))
    
    for train_index, val_index in kf.split(X_dense):
        X_dense_train, X_dense_val = X_dense[train_index], X_dense[val_index]
        X_sparse_train, X_sparse_val = X_sparse[train_index], X_sparse[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Train and predict with dense model
        base_model_dense.fit(X_dense_train, y_train)
        dense_preds[val_index] = base_model_dense.predict(X_dense_val)
        
        # Calculate residuals
        train_residuals = y_train - base_model_dense.predict(X_dense_train)
        
        # Train sparse model on residuals
        base_model_sparse.fit(X_sparse_train, train_residuals)
        sparse_preds[val_index] = base_model_sparse.predict(X_sparse_val)
        
        # Combined prediction
        combined_preds[val_index] = dense_preds[val_index] + sparse_preds[val_index]
    
    # Evaluate models
    dense_rmse = np.sqrt(mean_squared_error(y, dense_preds))
    dense_corr, _ = pearsonr(y, dense_preds)
    sparse_rmse = np.sqrt(mean_squared_error(y-dense_preds, sparse_preds))
    sparse_corr, _ = pearsonr(y, sparse_preds)
    combined_rmse = np.sqrt(mean_squared_error(y, combined_preds))
    combined_corr, _ = pearsonr(y, combined_preds)
    
    return {
        'performance': {
            'dense_model': {'RMSE': dense_rmse, 'Correlation': dense_corr},
            'sparse_model (residuals)': {'RMSE': sparse_rmse, 'Correlation': sparse_corr},
            'combined_model': {'RMSE': combined_rmse, 'Correlation': combined_corr}
        }
    }

base_model_dense = RandomForestRegressor(**best_rf_dense, random_state=314159)
base_model_sparse = RandomForestRegressor(**best_rf_sparse, random_state=314159)

cv_results_residual = residual_ensemble_cv(X_dense, X_sparse, y_true, base_model_dense, base_model_sparse)

print("Cross-validation Performance (Residual Approach):")
print("Dense Model Performance:", cv_results_residual['performance']['dense_model'])
print("Sparse Model Performance (on residuals):", cv_results_residual['performance']['sparse_model (residuals)'])
print("Combined Model Performance:", cv_results_residual['performance']['combined_model'])

def train_final_residual_models(X_dense, X_sparse, y, base_model_dense_class, base_model_sparse_class, n_models=10):
    final_models = []
    
    for seed in range(n_models):
        base_model_dense = base_model_dense_class(**best_rf_dense, random_state=seed)
        base_model_sparse = base_model_sparse_class(**best_rf_sparse, random_state=seed)
        
        # Train dense model
        final_base_model_dense = base_model_dense.fit(X_dense, y)
        
        # Calculate residuals
        dense_predictions = final_base_model_dense.predict(X_dense)
        residuals = y - dense_predictions
        
        # Train sparse model on residuals
        final_base_model_sparse = base_model_sparse.fit(X_sparse, residuals)
        
        final_models.append((final_base_model_dense, final_base_model_sparse))
    
    return final_models

def predict_residual_ensemble(X_dense_new, X_sparse_new, final_models):
    dense_predictions = []
    sparse_predictions = []
    combined_predictions = []
    
    for dense_model, sparse_model in final_models:
        dense_pred = dense_model.predict(X_dense_new)
        sparse_pred = sparse_model.predict(X_sparse_new)
        
        dense_predictions.append(dense_pred)
        sparse_predictions.append(sparse_pred)
        
        combined_pred = dense_pred + sparse_pred
        combined_predictions.append(combined_pred)
    
    mean_dense_pred = np.mean(dense_predictions, axis=0)
    mean_sparse_pred = np.mean(sparse_predictions, axis=0)
    mean_combined_pred = np.mean(combined_predictions, axis=0)
    
    return {
        'dense_prediction': mean_dense_pred,
        'sparse_prediction': mean_sparse_pred,
        'combined_prediction': mean_combined_pred
    }

final_models = train_final_residual_models(X_dense, X_sparse, y_true, RandomForestRegressor, RandomForestRegressor)

# Make predictions on new data
predictions = predict_residual_ensemble(X_dense_new, X_sparse_new, final_models)

# Access predictions
dense_preds = predictions['dense_prediction']
sparse_preds = predictions['sparse_prediction']
combined_preds = predictions['combined_prediction']


# Evaluate models
dense_rmse = np.sqrt(mean_squared_error(y_test_true, dense_preds))
dense_corr, _ = pearsonr(y_test_true, dense_preds)
sparse_rmse = np.sqrt(mean_squared_error(y_test_true-dense_preds, sparse_preds))
sparse_corr, _ = pearsonr(y_test_true-dense_preds, sparse_preds)
combined_rmse = np.sqrt(mean_squared_error(y_test_true, combined_preds))
combined_corr, _ = pearsonr(y_test_true, combined_preds)

performance = {
    'performance': {
        'dense_model': {'RMSE': dense_rmse, 'Correlation': dense_corr},
        'sparse_model (residuals)': {'RMSE': sparse_rmse, 'Correlation': sparse_corr},
        'combined_model': {'RMSE': combined_rmse, 'Correlation': combined_corr}
    }
}

print('Leaderboard Performance:')
print("Dense Model Performance:", performance['performance']['dense_model'])
print("Sparse Model Performance (on residuals):", performance['performance']['sparse_model (residuals)'])
print("Combined Model Performance:", performance['performance']['combined_model'])


# ------------- above are all just one trail -----------------------

def optimize_residual_model(X_dense, X_sparse, y, best_rf_dense, n_folds=10, seed=314159):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    residuals = np.zeros(len(y))
    
    # Calculate residuals using cross-validation
    for train_index, val_index in kf.split(X_dense):
        X_dense_train, X_dense_val = X_dense[train_index], X_dense[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Train dense model
        base_model_dense = RandomForestRegressor(**best_rf_dense, random_state=seed)
        base_model_dense.fit(X_dense_train, y_train)
        
        # Calculate residuals for validation set
        dense_preds = base_model_dense.predict(X_dense_val)
        residuals[val_index] = y_val - dense_preds
    
    # Now optimize sparse model on these residuals
    best_rf_sparse, _ = para_search(seed, X_sparse, residuals)
    
    return best_rf_sparse

# Use the existing best_rf_dense, but train for sparse
for seed in [0, 1, 2]:
    print()
    print('Start evaluating using optimized model...')
    print('Round', seed + 1)
    best_rf_sparse = optimize_residual_model(X_dense, X_sparse, y_true, best_rf_dense, seed = seed)

    # Use the optimized models, test on random seed (used to test on 5)
    base_model_dense = RandomForestRegressor(**best_rf_dense, random_state=314159)
    base_model_sparse = RandomForestRegressor(**best_rf_sparse, random_state=314159)

    cv_results_residual = residual_ensemble_cv(X_dense, X_sparse, y_true, base_model_dense, base_model_sparse)

    print("Cross-validation Performance (Residual Approach):")
    print("Dense Model Performance:", cv_results_residual['performance']['dense_model'])
    print("Sparse Model Performance (on residuals):", cv_results_residual['performance']['sparse_model (residuals)'])
    print("Combined Model Performance:", cv_results_residual['performance']['combined_model'])


    final_models = train_final_residual_models(X_dense, X_sparse, y_true, RandomForestRegressor, RandomForestRegressor)

    # Make predictions on new data
    predictions = predict_residual_ensemble(X_dense_new, X_sparse_new, final_models)

    # Access predictions
    dense_preds = predictions['dense_prediction']
    sparse_preds = predictions['sparse_prediction']
    combined_preds = predictions['combined_prediction']


    # Evaluate models
    dense_rmse = np.sqrt(mean_squared_error(y_test_true, dense_preds))
    dense_corr, _ = pearsonr(y_test_true, dense_preds)
    sparse_rmse = np.sqrt(mean_squared_error(y_test_true-dense_preds, sparse_preds))
    sparse_corr, _ = pearsonr(y_test_true-dense_preds, sparse_preds)
    combined_rmse = np.sqrt(mean_squared_error(y_test_true, combined_preds))
    combined_corr, _ = pearsonr(y_test_true, combined_preds)

    performance = {
        'performance': {
            'dense_model': {'RMSE': dense_rmse, 'Correlation': dense_corr},
            'sparse_model (residuals)': {'RMSE': sparse_rmse, 'Correlation': sparse_corr},
            'combined_model': {'RMSE': combined_rmse, 'Correlation': combined_corr}
        }
    }

    print('Leaderboard Performance:')
    print("Dense Model Performance:", performance['performance']['dense_model'])
    print("Sparse Model Performance (on residuals):", performance['performance']['sparse_model (residuals)'])
    print("Combined Model Performance:", performance['performance']['combined_model'])

