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

sizes = [50, 100]
versions = ['AtomPairs', 'Morgan', 'TopologicalTorsions']
input_path = 'Data/'
methods = ['sum']

# update on 07/23/24

for version in versions:
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('Version of physical chemical fingerprint:', version)
    print()
    for method in methods:
        print('+_+_+_+_+_+_+_+_+_+_')
        print('Method:', method)
        print()
        for size in sizes:
            print('Used dimension:', size)
            print()
            feature_file = f'Fingerprints/{version}_Fingerprints_Frequency_Size{size}.csv'
            features_file_2 =  'leffingwell_features_96.npy'
            CID_file = 'molecules_train_cid.npy'

            mixture_file = 'Mixure_Definitions_Training_set_UPD2.csv' 
            training_task_file = 'TrainingData_mixturedist.csv'

            # Mordred features
            features_1 = pd.read_csv(os.path.join(input_path, feature_file), index_col= 0)
            features_2 = np.load(os.path.join(input_path, features_file_2))

            features_CIDs = np.load(os.path.join(input_path, CID_file))

            # Training dataframe
            training_set = pd.read_csv(os.path.join(input_path, training_task_file))

            # Mapping helper files
            mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))

            CID2features_morgan =  {CID: features_1.loc[CID].tolist() for CID in features_CIDs}
            CID2features_leffingwell = {CID: features_2[i] for i, CID in enumerate(features_CIDs)}
            
            X_m, y, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features_morgan, method = f'{method}')
            X_l, _, _, _ = format_Xy(training_set,  mixtures_IDs, CID2features_leffingwell, method = f'{method}')

            # Convert the input pairs to a suitable format for training
            X_pairs_m = np.array([(np.concatenate((x1, x2))) for x1, x2 in X_m])
            X_pairs_l = np.array([(np.concatenate((x1, x2))) for x1, x2 in X_l])

            y_true = np.array(y)

            distances_m = [get_euclidean_distance(m[0], m[1]) for m in X_m]
            similarities_m = [get_cosine_similarity(m[0], m[1]) for m in X_m]
            angles_m = [get_cosine_angle(m[0], m[1]) for m in X_m] 

            distances_l = [get_euclidean_distance(m[0], m[1]) for m in X_l]
            similarities_l = [get_cosine_similarity(m[0], m[1]) for m in X_l]
            angles_l = [get_cosine_angle(m[0], m[1]) for m in X_l] 

            shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
            diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]

            datasets = training_set['Dataset'].to_numpy()
            encoder = OneHotEncoder()
            data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
            data_arr = data_arr.toarray()

            ### add all information above
            X_features = np.hstack( (X_pairs_l, X_pairs_m,
                                    np.array(distances_m).reshape(500, 1), 
                                    np.array(similarities_m).reshape(500, 1), 
                                    np.array(angles_m).reshape(500, 1), 
                                    np.array(distances_l).reshape(500, 1), 
                                    np.array(similarities_l).reshape(500, 1), 
                                    np.array(angles_l).reshape(500, 1), 
                                    np.array(shared_monos).reshape(500, 1), 
                                    np.array(diff_monos).reshape(500, 1), 
                                    np.array(num_mixtures).reshape(500,2), 
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
