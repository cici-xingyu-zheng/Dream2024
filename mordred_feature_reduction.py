import os
import sys


from src.utils import *

from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

input_path = 'Data'

feature_file = 'Mordred_features_in-house.csv'
CID_file = 'molecules_train_cid.npy'
mixture_file = 'Mixure_Definitions_Training_set.csv' 
training_task_file = 'TrainingData_mixturedist.csv'

# Mordred descriptors
features = pd.read_csv(os.path.join(input_path, feature_file), index_col= 0)

# All CIDs
features_CIDs = np.load(os.path.join(input_path, CID_file))

# Training dataframe
training_set = pd.read_csv(os.path.join(input_path, training_task_file))

# Mapping helper files
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))

# Get descriptors names
Mordred_des = features.columns.tolist()

# Standardize Mordred
scaler = StandardScaler(with_mean=True, with_std=True)
features = scaler.fit_transform(features)

# Convert DataFrame to a numpy array
features_array = features

# Create an imputer object with mean strategy, can change later!!!
imputer = SimpleImputer(strategy='mean')
# Impute missing values
imputed_features = imputer.fit_transform(features_array)


# Make Dataframe
CID2features_mordred =  {CID: imputed_features[i] for i, CID in enumerate(features_CIDs)}
X_m, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features_mordred, method = 'avg')

# Convert the input pairs to a suitable format for training
X_pairs_m = np.array([(np.concatenate((x1, x2))) for x1, x2 in X_m])
X_features = X_pairs_m
feature_dim = X_features.shape[1]

y_true = np.array(y)

# Feature names
feature_names = [f'{m}_1' for m in Mordred_des] +  [f'{m}_2' for m in Mordred_des]
name2index = dict(zip(feature_names, list(range(len(feature_names)))))


# Start the loop:
n_folds = 10
seed = 314159
num_removal = 271828
iter = 0

while num_removal != 0:
    print(f'Starting {iter + 1}th round reduction! \n')
    print(f'Number of features used now: {feature_dim}')

    rf_pred_list = []
    y_true_list = []
    low_importance_features = []
    kf_importances = []

    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(X_features):
        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]
        
        # Train the Random Forest regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(X_train, y_train)
        
        rf_pred = rf.predict(X_test)
        rf_pred_list.extend(rf_pred)
        y_true_list.extend(y_test)
 
        # Get the feature importances
        importances = rf.feature_importances_
        kf_importances.append(importances)
        # Find the indices of features with importance < 0.00010
        low_importance_indices_fold = np.where(importances < 0.000100)[0]

        # Get the names of the low importance features
        low_importance_features_fold = [feature_names[i] for i in low_importance_indices_fold]
        # Append the low importance features for this fold to the list
        low_importance_features.append(low_importance_features_fold)

    # Calculate the correlation and R^2 for Random Forest
    rf_corr = np.corrcoef(rf_pred_list, y_true_list)[0, 1]
    rf_rmse = np.sqrt(mean_squared_error(np.array(y_true_list), np.array(rf_pred_list)))

    print(f"Random Forest - R: {rf_corr:.3f}")
    print(f"Random Forest - RMSE: {rf_rmse:.3f}")
    print()

    # Find the features that appear in all the low importance lists
    consistent_low_importance_features = set.intersection(*map(set, low_importance_features))

    # If the Descriptor shows up for both molecules, we select it:
    feature_candidates = [c[:-2] for c in consistent_low_importance_features]
    element_counts = Counter(feature_candidates)
    descriptors_to_remove = [element for element, count in element_counts.items() if count == 2]

    features_to_remove_1 = [f'{d}_1' for d in descriptors_to_remove]
    features_to_remove_2 = [f'{d}_2' for d in descriptors_to_remove]

    features_to_remove = features_to_remove_1 + features_to_remove_2
    idx_to_remove = [name2index[f] for f in features_to_remove]
    num_removal = len(features_to_remove)

    print(f'Remove {num_removal} of features;')

    feature_dim = feature_dim - len(features_to_remove)
    print(f'Left with {feature_dim} features.')
    print()

    # Update feature matrix:
    X_features = np.delete(X_features, idx_to_remove, axis=1)
    feature_names = [feature for feature in feature_names if feature not in features_to_remove]
    name2index = dict(zip(feature_names, list(range(len(feature_names)))))

    iter += 1
    np.save(f'Output/feature_reduction/feature_names_iter_{iter}.npy', feature_names)
