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

import xgboost as xgb

input_path = 'Data'

feature_file = 'deepnose_features.npy'
CID_file = 'molecules_train_cid.npy'

mixture_file = 'Mixure_Definitions_Training_set.csv'
intensity_file = 'Mixure_Definitions_Intensity_Training_set.csv'
training_task_file = 'TrainingData_mixturedist.csv'

# Deepnose features
features = np.load(os.path.join(input_path, feature_file))
# Training dataframe
training_set = pd.read_csv(os.path.join(input_path, training_task_file))

# Mapping helper files
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))
mixtures_intensities = pd.read_csv(os.path.join(input_path, intensity_file))


# scales = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

# scales_2 = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

scales_3 = [2, 6]

print('Varying the scaling for non Ravia data..')

for scaling_constant in scales_3:

    print(f'scale = {scaling_constant} \n')

    # Define the scaling constant
    scaling_constant = scaling_constant  # Adjust this value as needed

    # Get the list of column names containing "CID"
    cid_columns = [col for col in mixtures_intensities.columns if 'CID' in col]

    # Create a mask to identify rows where "Dataset" is in ['Snitz 1', 'Snitz 2', 'Bushdid']
    mask = mixtures_intensities['Dataset'].isin(['Snitz 1', 'Snitz 2', 'Bushdid'])

    # Scale the values of "CID" columns for the selected rows
    mixtures_intensities.loc[mask, cid_columns] *= scaling_constant

    features_CIDs = np.load(os.path.join(input_path, CID_file))

    epsilon = 1e-8

    scaler = StandardScaler(with_mean=True, with_std=True)
    features = scaler.fit_transform(np.log(features + epsilon))

    # Map CID to 96 dim features:
    CID2features =  {CID: features[i] for i, CID in enumerate(features_CIDs)}

    # formatting X y 
    X = []
    y = []
    num_monos = []
    CIDs_all = []

    for _, row in training_set.iterrows():
        mixture1 = combine_molecules_intensity_weighed(label=row['Mixture 1'], dataset=row['Dataset'],
                                                mixtures_IDs=mixtures_IDs, CID2features=CID2features,
                                                mixtures_intensities= mixtures_intensities)
        mixture2 = combine_molecules_intensity_weighed(label=row['Mixture 2'], dataset=row['Dataset'],
                                                mixtures_IDs=mixtures_IDs, CID2features=CID2features,
                                                mixtures_intensities= mixtures_intensities)
        X.append((mixture1, mixture2))
        y.append(row['Experimental Values'])

    # construct features:
    _, _, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = 'avg')

    X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
    y_true = np.array(y)

    distances = [get_euclidean_distance(m[0], m[1]) for m in X]
    similarities = [get_cosine_similarity(m[0], m[1]) for m in X]
    angles = [get_cosine_angle(m[0], m[1]) for m in X]

    shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
    diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]

    datasets = training_set['Dataset'].to_numpy()
    encoder = OneHotEncoder()
    data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
    data_arr = data_arr.toarray()

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