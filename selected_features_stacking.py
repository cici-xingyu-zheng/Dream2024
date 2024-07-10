import os
import sys

from src.utils import *
from src.optimize import *

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import xgboost as xgb

selected_choices = [ 'selection_cleanDragonDescriptors.csv', 
                     'selection_cleanMordredDescriptors.csv',
                     'selection_cleanMordredDescriptorsNormalized.csv'
                    ]
for feature_choice in selected_choices:
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('Try Out Selected Features:', feature_choice)
    print()
    input_path = 'Data/'

    features_file_1 = f'featureSelection/{feature_choice}'
    features_file_2 =  'deepnose_features.npy'
    CID_file = 'molecules_train_cid.npy'

    # Read all copies, before and after correction; before was also downloaded from Dropbox.
    mixture_file = 'Mixure_Definitions_Training_set_UPD2.csv' 

    training_task_file = 'TrainingData_mixturedist.csv'

    features = pd.read_csv(os.path.join(input_path, features_file_1), index_col= 0)
    features_2 = np.load(os.path.join(input_path, features_file_2))

    features_CIDs = np.load(os.path.join(input_path, CID_file))

    # Training dataframe
    training_set = pd.read_csv(os.path.join(input_path, training_task_file))

    # Mapping helper files
    mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))

    scaler = StandardScaler(with_mean=True, with_std=True)

    # standardize Mordred
    features_np = scaler.fit_transform(features)
    features = pd.DataFrame(features_np, columns=features.columns, index=features.index)

    # log standardize deepnose
    epsilon = 1e-8 
    features_2 = scaler.fit_transform(np.log(features_2 + epsilon))

    # Map CID to 96 dim features:
    CID2features = {CID: np.array(features.loc[CID].tolist()) if CID in features.index else np.full(len(features.columns), np.nan) for CID in features_CIDs}
    CID2features_deepnose=  {CID: features_2[i] for i, CID in enumerate(features_CIDs)}

    X_m, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = 'avg')
    X_d, _, _, _ = format_Xy(training_set,  mixtures_IDs, CID2features_deepnose, method = 'avg')

    # Convert the input pairs to a suitable format for training
    X_pairs_m = np.array([(np.concatenate((x1, x2))) for x1, x2 in X_m])
    X_pairs_d = np.array([(np.concatenate((x1, x2))) for x1, x2 in X_d])

    y_true = np.array(y)

    distances_m = [get_euclidean_distance(m[0], m[1]) for m in X_m]
    similarities_m = [get_cosine_similarity(m[0], m[1]) for m in X_m]
    angles_m = [get_cosine_angle(m[0], m[1]) for m in X_m] 

    distances_d = [get_euclidean_distance(m[0], m[1]) for m in X_d]
    similarities_d = [get_cosine_similarity(m[0], m[1]) for m in X_d]
    angles_d = [get_cosine_angle(m[0], m[1]) for m in X_d] 

    shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
    diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]

    datasets = training_set['Dataset'].to_numpy()
    encoder = OneHotEncoder()
    data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
    data_arr = data_arr.toarray()

    ### add all information above
    X_features = np.hstack( (X_pairs_d, X_pairs_m,
                            np.array(distances_m).reshape(500, 1), 
                            np.array(similarities_m).reshape(500, 1), 
                            np.array(angles_m).reshape(500, 1), 
                            np.array(distances_d).reshape(500, 1), 
                            np.array(similarities_d).reshape(500, 1), 
                            np.array(angles_d).reshape(500, 1), 
                            np.array(shared_monos).reshape(500, 1), 
                            np.array(diff_monos).reshape(500, 1), 
                            np.array(num_mixtures).reshape(500,2), 
                            data_arr))

    n_folds = 10

    seeds = [0, 1, 2] # save time
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