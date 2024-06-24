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

def intensity_function(x, alpha_int = 1.3, beta_int = 0.07):
    return 1 / (1 + np.exp(-(x - alpha_int)/beta_int))

def combine_molecules_intensity_weighed(label, dataset, mixtures_IDs, CID2features, mixtures_intensities):
    # Grab the unique data row:
    row = mixtures_IDs[(mixtures_IDs['Mixture Label'] == label) & (mixtures_IDs['Dataset'] == dataset)]
    # The intensity of that data row:
    intesnity_row = mixtures_intensities[(mixtures_intensities['Mixture Label'] == label) & (mixtures_intensities['Dataset'] == dataset)]
    
    non_zero_CIDs = row.loc[:, row.columns.str.contains('CID')].loc[:, (row != 0).any(axis=0)]
    non_zero_intensities = intesnity_row.loc[:, intesnity_row.columns.str.contains('CID')].loc[:, (intesnity_row != 0).any(axis=0)]
    if len(non_zero_CIDs) != 1:
        print('Not a Unique pointer!!!')
    CIDs = non_zero_CIDs.iloc[0].tolist()
    intensities = non_zero_intensities.iloc[0].tolist()
    CID2intensity = dict(zip(CIDs, intensities))

    molecule_embeddings = []
    # Create feature matrix for all number of mono odor molecules in the mixture:
    for CID in CIDs:
        molecule_embeddings.append(np.array(CID2features[CID])*intensity_function(CID2intensity[CID]/100))

    # Combine by sum across molecules:
    mixture_embedding = np.nansum(molecule_embeddings, axis=0)
    
    return mixture_embedding

for dim in [50, 162]:
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('Try out Modred projection dim: ', dim)
    print
    input_path = 'Data'

    feature_file = f'Mordred_reduced_features_{dim}.npy'
    features_file_2 =  'deepnose_features.npy'
    CID_file = 'molecules_train_cid.npy'

    # Read all copies, before and after correction; before was also downloaded from Dropbox.
    mixture_file = 'Mixure_Definitions_Training_set.csv' 
    intensity_file = 'Mixure_Definitions_Intensity_Training_set.csv'
    training_task_file = 'TrainingData_mixturedist.csv'

    # Mordred features
    features_1 = np.load(os.path.join(input_path, feature_file))
    features_2 = np.load(os.path.join(input_path, features_file_2))

    features_CIDs = np.load(os.path.join(input_path, CID_file))

    # Training dataframe
    training_set = pd.read_csv(os.path.join(input_path, training_task_file))

    # Mapping helper files
    mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))
    mixtures_intensities = pd.read_csv(os.path.join(input_path, intensity_file))

    # Define the scaling constant
    scaling_constant = 2  # Adjust this value as needed
    # Get the list of column names containing "CID"
    cid_columns = [col for col in mixtures_intensities.columns if 'CID' in col]
    # Create a mask to identify rows where "Dataset" is in ['Snitz 1', 'Snitz 2', 'Bushdid']
    mask = mixtures_intensities['Dataset'].isin(['Snitz 1', 'Snitz 2', 'Bushdid'])
    # Scale the values of "CID" columns for the selected rows
    mixtures_intensities.loc[mask, cid_columns] *= scaling_constant

    scaler = StandardScaler(with_mean=True, with_std=True)

    # standardize Mordred
    features_1 = scaler.fit_transform(features_1)
    # log standardize deepnose
    epsilon = 1e-8 
    features_2 = scaler.fit_transform(np.log(features_2 + epsilon))

    # Create an imputer object with mean strategy, can change later!!!
    imputer = SimpleImputer(strategy='mean')
    # Impute missing values
    features_1 = imputer.fit_transform(features_1)

    features = np.hstack((features_1, features_2))
    CID2features =  {CID: features[i] for i, CID in enumerate(features_CIDs)}

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


    _, _, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = 'avg')

    # Convert the input pairs to a suitable format for training
    X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
    y_true = np.array(y)

    distances_1 = [get_euclidean_distance(m[0][:96], m[1][:96]) for m in X]
    similarities_1 = [get_cosine_similarity(m[0][:96], m[1][:96]) for m in X]
    angles_1 = [get_cosine_angle(m[0][:96], m[1][:96]) for m in X]

    distances_2 = [get_euclidean_distance(m[0][96:], m[1][96:]) for m in X]
    similarities_2 = [get_cosine_similarity(m[0][96:], m[1][96:]) for m in X]
    angles_2 = [get_cosine_angle(m[0][96:], m[1][96:]) for m in X]

    shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
    diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]

    datasets = training_set['Dataset'].to_numpy()
    encoder = OneHotEncoder()
    data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
    data_arr = data_arr.toarray()

    ### add all information above
    X_features = np.hstack((X_pairs, 
                            np.array(distances_1).reshape(500, 1), 
                            np.array(similarities_1).reshape(500, 1), 
                            np.array(angles_1).reshape(500, 1), 
                            np.array(distances_2).reshape(500, 1), 
                            np.array(similarities_2).reshape(500, 1), 
                            np.array(angles_2).reshape(500, 1), 
                            np.array(shared_monos).reshape(500, 1), 
                            np.array(diff_monos).reshape(500, 1), 
                            np.array(num_mixtures), 
                            data_arr))


    n_folds = 10

    seeds = [1, 2, 3] # save time
    for seed in seeds: 
        print(f"Random search for best hyperparams: round {seed} \n")
        rf_best,rbg_best = para_search(seed, X_features, y_true)
        print()
        rf_out = avg_rf_best(rf_best, X_features, y_true)
        print()
        rbg_out = avg_rgb_best(rbg_best, X_features, y_true)
        print()

    print()
    print('----------------------------------------------')