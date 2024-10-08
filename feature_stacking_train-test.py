from src.utils import *
from src.optimize import *
from src.train_test import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


import os


input_path = 'Data/'


## Group 1
feature_file_1a = 'Mordred_reduced_features_50.npy'
features_1a = np.load(os.path.join(input_path, feature_file_1a))
scaler_1a = StandardScaler(with_mean=True, with_std=True)
features_1a = scaler_1a.fit_transform(features_1a)
CID2features_1a =  {CID: features_1a[i] for i, CID in enumerate(features_CIDs)}

feature_file_1b = 'Mordred_reduced_features_96.npy'
features_1b = np.load(os.path.join(input_path, feature_file_1b))
scaler_1b = StandardScaler(with_mean=True, with_std=True)
features_1b = scaler_1b.fit_transform(features_1b)
CID2features_1b =  {CID: features_1b[i] for i, CID in enumerate(features_CIDs)}

feature_file_1c = 'Mordred_reduced_features_162.npy'
features_1c = np.load(os.path.join(input_path, feature_file_1c))
scaler_1c = StandardScaler(with_mean=True, with_std=True)
features_1c= scaler_1c.fit_transform(features_1c)
CID2features_1c =  {CID: features_1c[i] for i, CID in enumerate(features_CIDs)}

feature_file_2 = 'deepnose_features.npy'
features_2 = np.load(os.path.join(input_path, feature_file_2))
epsilon = 1e-8 
scaler_2 = StandardScaler(with_mean=True, with_std=True)
features_2 = scaler_2.fit_transform(np.log(features_2 + epsilon))
CID2features_2 =  {CID: features_2[i] for i, CID in enumerate(features_CIDs)}


## Group 2
sparse_feature_file_1a100 = f'Fingerprints/Morgan_Fingerprints_Frequency_Size100.csv'
sparse_features_1a100 = pd.read_csv(os.path.join(input_path, sparse_feature_file_1a100), index_col= 0)
sparse_1a100 =  {CID: sparse_features_1a100.loc[CID].tolist() for CID in features_CIDs}

sparse_feature_file_1a50 = f'Fingerprints/Morgan_Fingerprints_Frequency_Size50.csv'
sparse_features_1a50 = pd.read_csv(os.path.join(input_path, sparse_feature_file_1a50), index_col= 0)
sparse_1a50 =  {CID: sparse_features_1a50.loc[CID].tolist() for CID in features_CIDs}

sparse_feature_file_1b = f'Fingerprints/TopologicalTorsions_Fingerprints_Frequency_Size50.csv'
sparse_features_1b = pd.read_csv(os.path.join(input_path, sparse_feature_file_1b), index_col= 0)
sparse_1b =  {CID: sparse_features_1b.loc[CID].tolist() for CID in features_CIDs}

sparse_feature_file_1c = f'Fingerprints/AtomPairs_Fingerprints_Frequency_Size50.csv'
sparse_features_1c = pd.read_csv(os.path.join(input_path, sparse_feature_file_1c), index_col= 0)
sparse_1c =  {CID: sparse_features_1c.loc[CID].tolist() for CID in features_CIDs}

sparse_features_file_2 =  'leffingwell_features.npy'
sparse_features_2 = np.load(os.path.join(input_path, sparse_features_file_2))
sparse_2 = {CID: sparse_features_2[i] for i, CID in enumerate(features_CIDs)}


## Group 3
selected_feature_file_1a = 'featureSelection/selection_cleanDragonDescriptors.csv'
feature_1a = pd.read_csv(os.path.join(input_path, selected_feature_file_1a), index_col= 0)
scaler = StandardScaler(with_mean=True, with_std=True)
features_np = scaler.fit_transform(feature_1a)
feature_1a = pd.DataFrame(features_np, columns=feature_1a.columns, index=feature_1a.index)
CID2features_selected_1a = {CID: np.array(feature_1a.loc[CID].tolist()) if CID in feature_1a.index else np.full(len(feature_1a.columns), np.nan) for CID in features_CIDs}

selected_feature_file_1b =  'featureSelection/selection_cleanMordredDescriptors.csv'
feature_1b = pd.read_csv(os.path.join(input_path, selected_feature_file_1b), index_col= 0)
scaler = StandardScaler(with_mean=True, with_std=True)
features_np = scaler.fit_transform(feature_1b)
feature_1b = pd.DataFrame(features_np, columns=feature_1b.columns, index=feature_1b.index)
CID2features_selected_1b = {CID: np.array(feature_1b.loc[CID].tolist()) if CID in feature_1b.index else np.full(len(feature_1b.columns), np.nan) for CID in features_CIDs}

selected_feature_file_1c =  'featureSelection/selection_cleanMordredDescriptorsNormalized.csv'
feature_1c = pd.read_csv(os.path.join(input_path, selected_feature_file_1c), index_col= 0)
scaler = StandardScaler(with_mean=True, with_std=True)
features_np = scaler.fit_transform(feature_1c)
feature_1c = pd.DataFrame(features_np, columns=feature_1c.columns, index=feature_1c.index)
CID2features_selected_1c = {CID: np.array(feature_1c.loc[CID].tolist()) if CID in feature_1c.index else np.full(len(feature_1c.columns), np.nan) for CID in features_CIDs}


CID2features_list_m9 = [CID2features_1a, CID2features_2] 
CID2features_list_m10 = [CID2features_1a, CID2features_2] 
CID2features_list_m11 = [CID2features_1b, CID2features_2] 
CID2features_list_m12 = [CID2features_1b, CID2features_2] 
CID2features_list_m13 = [CID2features_1c, CID2features_2] 

CID2features_list_m20 = [sparse_1c, sparse_2] 
CID2features_list_m21 = [sparse_1a50, sparse_2] 
CID2features_list_m22 = [sparse_1a50, sparse_2] 
CID2features_list_m23 = [sparse_1a100, sparse_2] 
CID2features_list_m24 = [sparse_1b, sparse_2] 
CID2features_list_m25 = [sparse_1a50, sparse_1b, sparse_2] 

CID2features_list_m26 = [CID2features_selected_1a, CID2features_2] 
CID2features_list_m27 = [CID2features_selected_1b, CID2features_2] 
CID2features_list_m28 = [CID2features_selected_1b, CID2features_2] 
CID2features_list_m29 = [CID2features_selected_1c, CID2features_2] 
CID2features_list_m30 = [CID2features_selected_1c, CID2features_2] 

CID2features_list_m31 = [CID2features_selected_1a, CID2features_selected_1b, CID2features_2] 
CID2features_list_m32 = [CID2features_selected_1a, CID2features_selected_1b, CID2features_2] 

CID2features_list_m33 = [sparse_1a50]
CID2features_list_m34 = [sparse_1a50]

param_m9 = {'n_estimators': 300, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}
param_m10 = {'subsample': 0.7, 'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.01, 'colsample_bytree': 0.5}
param_m11 = {'n_estimators': 250, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
param_m12 = {'subsample': 0.5, 'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.01, 'colsample_bytree': 0.5}
param_m13 = {'n_estimators': 500, 'min_samples_split': 5, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}

param_m20 = {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
param_m21 = {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.5, 'max_depth': 30, 'bootstrap': True}
param_m22 = {'subsample': 1.0, 'reg_lambda': 0, 'reg_alpha': 0, 'n_estimators': 200, 'min_child_weight': 5, 'max_depth': 9, 'learning_rate': 0.01, 'colsample_bytree': 0.5}
param_m23 = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
param_m24 = {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
param_m25 = {'subsample': 0.7, 'reg_lambda': 0.1, 'reg_alpha': 0, 'n_estimators': 300, 'min_child_weight': 3, 'max_depth': 9, 'learning_rate': 0.01, 'colsample_bytree': 1.0}

param_m26 = {'n_estimators': 500, 'min_samples_split': 5, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
param_m27 = {'n_estimators': 500, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
param_m28 = {'subsample': 0.7, 'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.01, 'colsample_bytree': 0.5}
param_m29 = {'n_estimators': 250, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}
param_m30 = {'subsample': 0.7, 'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.01, 'colsample_bytree': 0.5}

param_m31 = {'n_estimators': 500, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}
param_m32 = {'subsample': 0.7, 'n_estimators': 300, 'max_depth': 9, 'learning_rate': 0.01, 'colsample_bytree': 0.5}

param_m33 = {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 0.5, 'max_depth': 20, 'bootstrap': False}
param_m34 = {'subsample': 1.0, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 400, 'min_child_weight': 5, 'max_depth': 6, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

model_specs = {
    9:{'CID2features_list':CID2features_list_m9, 'param':param_m9, 'method':'avg', 'model':'rf'}, 
    10:{'CID2features_list':CID2features_list_m10, 'param':param_m10, 'method':'avg', 'model':'xgb'}, 
    11:{'CID2features_list':CID2features_list_m11, 'param':param_m11, 'method':'avg', 'model':'rf'}, 
    12:{'CID2features_list':CID2features_list_m12, 'param':param_m12, 'method':'avg', 'model':'xgb'}, 
    13:{'CID2features_list':CID2features_list_m13, 'param':param_m13, 'method':'avg', 'model':'rf'}, 

    20:{'CID2features_list':CID2features_list_m20, 'param':param_m20, 'method':'sum', 'model':'rf'}, 
    21:{'CID2features_list':CID2features_list_m21, 'param':param_m21, 'method':'sum', 'model':'rf'}, 
    22:{'CID2features_list':CID2features_list_m22, 'param':param_m22, 'method':'sum', 'model':'xgb'}, 
    23:{'CID2features_list':CID2features_list_m23, 'param':param_m23, 'method':'sum', 'model':'rf'}, 
    24:{'CID2features_list':CID2features_list_m24, 'param':param_m24, 'method':'sum', 'model':'rf'}, 
    25:{'CID2features_list':CID2features_list_m25, 'param':param_m25, 'method':'sum', 'model':'xgb'},

    26:{'CID2features_list':CID2features_list_m26, 'param':param_m26, 'method':'avg', 'model':'rf'}, 
    27:{'CID2features_list':CID2features_list_m27, 'param':param_m27, 'method':'avg', 'model':'rf'}, 
    28:{'CID2features_list':CID2features_list_m28, 'param':param_m28, 'method':'avg', 'model':'xgb'}, 
    29:{'CID2features_list':CID2features_list_m29, 'param':param_m29, 'method':'avg', 'model':'rf'}, 
    30:{'CID2features_list':CID2features_list_m30, 'param':param_m30, 'method':'avg', 'model':'xgb'},

    31:{'CID2features_list':CID2features_list_m31, 'param':param_m31, 'method':'avg', 'model':'rf'}, 
    32:{'CID2features_list':CID2features_list_m32, 'param':param_m32, 'method':'avg', 'model':'xgb'}, 
}

# model_specs = {
#         33:{'CID2features_list':CID2features_list_m33, 'param':param_m33, 'method':'sum', 'model':'rf'}, 
#         34:{'CID2features_list':CID2features_list_m34, 'param':param_m34, 'method':'sum', 'model':'xgb'}, 
# }


# # Initialize a list to store results for each model
# results_list = []

# for key in model_specs.keys():
#     print('_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_')
#     print(f'Testing Model: M{key} \n')
#     CID2features_list = model_specs[key]['CID2features_list']
#     param = model_specs[key]['param']
#     method = model_specs[key]['method']
#     model_type = model_specs[key]['model']
 
#     X_features, y_true = stacking_X_features(CID2features_list, method)
#     models = ensemble_models(X_features, y_true, param, type=model_type, num_models=10)
#     X_test, y_test_true = stacking_X_test_features(CID2features_list, X_features, method)
#     y_train_pred_avg = pred_mean(models, X_features)
#     y_test_pred_avg = pred_mean(models, X_test)

#     train_corr = np.corrcoef(y_train_pred_avg, y_true)[0, 1]
#     train_rmse = np.sqrt(mean_squared_error(np.array(y_true), y_train_pred_avg))

#     bootstrap_results = bootstrap_metrics_small_sample(y_test_true, y_test_pred_avg)

#     original_corr = np.corrcoef(y_test_pred_avg, y_test_true)[0, 1]
#     original_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred_avg))

#     # Print results (you can keep or remove this part)
#     print(f"Bootstrapped Metrics (with 95% CI):")
#     print(f"Correlation: {bootstrap_results['corr']['mean']:.4f} (95% CI: {bootstrap_results['corr']['ci'][0]:.4f} - {bootstrap_results['corr']['ci'][1]:.4f})")
#     print(f"RMSE: {bootstrap_results['rmse']['mean']:.4f} (95% CI: {bootstrap_results['rmse']['ci'][0]:.4f} - {bootstrap_results['rmse']['ci'][1]:.4f})")
#     print("\nOriginal Metrics:")
#     print(f"Correlation: {original_corr:.4f}")
#     print(f"RMSE: {original_rmse:.4f}")
#     print()

#     # Collect results
#     results_list.append({
#         'Model': f'M{key}',
#         'Train_Corr': train_corr,
#         'Train_RMSE': train_rmse,
#         'Test_Corr': original_corr,
#         'Test_RMSE': original_rmse,
#         'Bootstrap_Corr_Mean': bootstrap_results['corr']['mean'],
#         'Bootstrap_Corr_CI_Lower': bootstrap_results['corr']['ci'][0],
#         'Bootstrap_Corr_CI_Upper': bootstrap_results['corr']['ci'][1],
#         'Bootstrap_RMSE_Mean': bootstrap_results['rmse']['mean'],
#         'Bootstrap_RMSE_CI_Lower': bootstrap_results['rmse']['ci'][0],
#         'Bootstrap_RMSE_CI_Upper': bootstrap_results['rmse']['ci'][1]
#     })

#     print('y_true:')
#     print(y_test_true)
#     y_preds.append(y_test_true)

#     print('y_pred:')
#     print(y_test_pred_avg)
#     y_preds.append(y_test_pred_avg)

# # Create DataFrame from results
# performance_df = pd.DataFrame(results_list)

# # Save DataFrame to CSV
# performance_df.to_csv("Output/performance_base-models.csv", index=False)

# print("Performance results saved to Output/performance_base-models.csv")

# Initialize empty dictionaries to store predictions
predictions_test = {}
predictions_train = {}

y_test_true_stored = None
y_train_true_stored = None

for key in model_specs.keys():
    print(f'Testing Model: M{key} \n')
    CID2features_list = model_specs[key]['CID2features_list']
    param = model_specs[key]['param']
    method = model_specs[key]['method']
    model_type = model_specs[key]['model']
 
    X_features, y_true = stacking_X_features(CID2features_list, method)
    models = ensemble_models(X_features, y_true, param, type=model_type, num_models=10)
    X_test, y_test_true = stacking_X_test_features(CID2features_list, X_features, method)
    y_test_pred_avg = pred_mean(models, X_test)
    y_train_pred_avg = pred_mean(models, X_features)

    # Store y_test_true only once
    if y_test_true_stored is None:
        y_test_true_stored = y_test_true
    
    if y_train_true_stored is None:
        y_train_true_stored = y_true
    
    # Store predictions for this model
    predictions_test[f'M{key}'] = y_test_pred_avg
    predictions_train[f'M{key}'] = y_train_pred_avg

# Create the dataframe
y_pred_test_df = pd.DataFrame({'y_test_true': y_test_true_stored})
y_pred_train_df = pd.DataFrame({'y_true': y_train_true_stored})

# Add predictions for each model
for key, pred in predictions_test.items():
    y_pred_test_df[key] = pred

for key, pred in predictions_train.items():
    y_pred_train_df[key] = pred

# Save the dataframe to CSV
y_pred_train_df.to_csv('Performance/y_pred_training.csv', index=False)
print("Predictions saved to Performance/y_pred_training.csv")

y_pred_test_df.to_csv('Performance/y_pred_leaderboard.csv', index=False)
print("Predictions saved to Performance/y_pred_leaderboard.csv")

