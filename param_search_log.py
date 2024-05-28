from utils import *
from optimize import *
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


scaler = StandardScaler(with_mean=True, with_std=True)
features = scaler.fit_transform(features)
features_CIDs = np.load(os.path.join(input_path, CID_file))
CID2features =  {CID: features[i] for i, CID in enumerate(features_CIDs)}

betas = [.1,.2,.5, 1, 2, 5, 10, 20, 50, 100]

print('Varying beta..')

for beta in betas:
    print(f'beta = {beta} \n')
    X, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = 'log', beta = beta)

    # Convert the input pairs to a suitable format for training
    X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
    y_true = np.array(y)

    X_pair1 = X_pairs[:, :96] 
    X_pair2 = X_pairs[:, 96:] 

    # Embedding related summary features:
    distances = [get_euclidean_distance(m[0], m[1]) for m in X]
    similarities = [get_cosine_similarity(m[0], m[1]) for m in X]
    angles = [get_cosine_angle(m[0], m[1]) for m in X]

    # Mixture related summary features:
    shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
    diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]

    # Dataset info
    datasets = training_set['Dataset'].to_numpy()
    encoder = OneHotEncoder()
    data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
    data_arr = data_arr.toarray()

    # Add features:
    X_features = np.hstack((X_pairs, np.array(distances).reshape(500, 1), 
                            np.array(similarities).reshape(500, 1), 
                            np.array(angles).reshape(500, 1), 
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

    print()
    print('----------------------------------------------')
