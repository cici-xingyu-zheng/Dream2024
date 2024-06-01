from src.utils import *
from src.optimize import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb


input_path = 'Data'

feature_file = 'deepnose_features.npy'
CID_file = 'molecules_train_cid.npy'
mixture_file = 'Mixure_Definitions_Training_set.csv'
training_task_file = 'TrainingData_mixturedist.csv'

features = np.load(os.path.join(input_path, feature_file))
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))
training_set = pd.read_csv(os.path.join(input_path, training_task_file))
features_CIDs = np.load(os.path.join(input_path, CID_file))



scaler = StandardScaler(with_mean=True, with_std=True)
sdt_features = scaler.fit_transform(features)
CID2sdtfeatures =  {CID: sdt_features[i] for i, CID in enumerate(features_CIDs)}

scaler = StandardScaler(with_mean=True, with_std=True)
epsilon = 1e-8 
log_features = scaler.fit_transform(np.log(features + epsilon))
CID2logfeatures =  {CID: log_features[i] for i, CID in enumerate(features_CIDs)}


# X, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = 'log', beta = 5)
X_sdt_avg, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2sdtfeatures, method = 'avg', beta = None)
X_sdt_log, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2sdtfeatures, method = 'log', beta = 5)
X_log_avg, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2logfeatures, method = 'avg', beta = None)

X_diff1 = np.array([x1 - x2 for x1, x2 in X_sdt_avg])
X_diff2= np.array([np.divide(x1, x2) for x1, x2 in X_sdt_log])
X_diff3 = np.array([x1 - x2 for x1, x2 in X_log_avg])

y_true = np.array(y)

distances_1 = [get_euclidean_distance(m[0], m[1]) for m in X_sdt_avg]
similarities_1 = [get_cosine_similarity(m[0], m[1]) for m in X_sdt_avg]
angles_1 = [get_cosine_angle(m[0], m[1]) for m in X_sdt_avg]

distances_2 = [get_euclidean_distance(m[0], m[1]) for m in X_sdt_log]
similarities_2 = [get_cosine_similarity(m[0], m[1]) for m in X_sdt_log]
angles_2 = [get_cosine_angle(m[0], m[1]) for m in X_sdt_log]

distances_3 = [get_euclidean_distance(m[0], m[1]) for m in X_log_avg]
similarities_3 = [get_cosine_similarity(m[0], m[1]) for m in X_log_avg]
angles_3 = [get_cosine_angle(m[0], m[1]) for m in X_log_avg]


# Mixture related summary features:
shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]

# Dataset info
datasets = training_set['Dataset'].to_numpy()
encoder = OneHotEncoder()
data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
data_arr = data_arr.toarray()

# Add features:
X_features = np.hstack((X_diff1, X_diff2, X_diff3,
                        np.array([distances_1, similarities_1, angles_1, distances_2, similarities_2, angles_2, distances_3, similarities_3, angles_3]).reshape(500, 9), 
                        np.array(shared_monos).reshape(500, 1), 
                        np.array(diff_monos).reshape(500, 1), 
                        np.array(num_mixtures), 
                        data_arr))

seeds = list(range(3))
for seed in seeds: 
    print(f"Random search for best hyperparams: round {seed +1} \n")
    rf_best,rbg_best = para_search(seed, X_features, y_true)
    print()
    rf_out = avg_rf_best(rf_best, X_features, y_true)
    print()
    rbg_out = avg_rgb_best(rbg_best, X_features, y_true)
    print()

