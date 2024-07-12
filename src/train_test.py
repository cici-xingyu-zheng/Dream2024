from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os

from src.utils import *


# Below we add functions that streamline feature stacking and testing.

input_path = 'Data/'

CID_file = 'molecules_train_cid.npy'
mixture_file = 'Mixure_Definitions_Training_set_UPD2.csv' 

training_task_file = 'TrainingData_mixturedist.csv'
test_task_file = 'Test/Data/LeaderboardData_mixturedist.csv'

features_CIDs = np.load(os.path.join(input_path, CID_file))
# Mapping helper files
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))


# Training dataframe
training_set = pd.read_csv(os.path.join(input_path, training_task_file))
# test dataframe
test_set = pd.read_csv(test_task_file)


def stacking_X_features(CID2features_list, method):

    stacks = []
    
    for CID2features in CID2features_list:

        X, y_true, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = method)
        X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
        
        distances= [get_euclidean_distance(m[0], m[1]) for m in X]
        similarities = [get_cosine_similarity(m[0], m[1]) for m in X]
        angles = [get_cosine_angle(m[0], m[1]) for m in X] 
        
        stack = np.hstack( (X_pairs,
                        np.array(distances).reshape(500, 1), 
                        np.array(similarities).reshape(500, 1), 
                        np.array(angles).reshape(500, 1)))
        stacks.append(stack)
    

    shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
    diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]
    
    datasets = training_set['Dataset'].to_numpy()
    # Returns the uniques in the order of appearance
    desired_order = training_set['Dataset'].unique().tolist() 
    encoder = OneHotEncoder(categories=[desired_order])
    data_arr = encoder.fit_transform(datasets.reshape(-1, 1))
    data_arr = data_arr.toarray()

    engineered_stack = np.hstack(
                        (np.array(shared_monos).reshape(500, 1), 
                        np.array(diff_monos).reshape(500, 1), 
                        np.array(num_mixtures).reshape(500,2), 
                        data_arr))
    
    stacks.append(engineered_stack)

    concatenated_array = np.hstack(stacks)
    X_features = np.stack(concatenated_array)
        
    return X_features, np.array(y_true)

def ensemble_models(X_features, y_true, param_best, type = 'rf', num_models = 10):
    models = []
    for i in range(num_models):
        if type == 'rf': 
            model = RandomForestRegressor(**param_best, random_state=i)
            model.fit(X_features, y_true)
        elif type == 'rgb':
            model = xgb.XGBRegressor(**param_best, random_state=i)
            model.fit(X_features, y_true)
        models.append(model)
    return models

def stacking_X_test_features(CID2features_list, X_train, method):

    stacks = []
    
    for CID2features in CID2features_list:

        X, y_true, num_mixtures, all_pairs_CIDs = format_Xy(test_set,  mixtures_IDs, CID2features, method = method)
        X_pairs = np.array([(np.concatenate((x1, x2))) for x1, x2 in X])
        
        distances= [get_euclidean_distance(m[0], m[1]) for m in X]
        similarities = [get_cosine_similarity(m[0], m[1]) for m in X]
        angles = [get_cosine_angle(m[0], m[1]) for m in X] 
        
        stack = np.hstack( (X_pairs,
                        np.array(distances).reshape(46, 1), 
                        np.array(similarities).reshape(46, 1), 
                        np.array(angles).reshape(46, 1)))
        stacks.append(stack)

    shared_monos = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in all_pairs_CIDs]
    diff_monos = [ len( set(pair[0]).difference(set(pair[1]))) for pair in all_pairs_CIDs]
    
    data_arr = np.full((len(test_set), 4), np.nan) 

    engineered_stack = np.hstack(
                        (np.array(shared_monos).reshape(46, 1), 
                        np.array(diff_monos).reshape(46, 1), 
                        np.array(num_mixtures).reshape(46,2), 
                        data_arr))
    
    stacks.append(engineered_stack)

    concatenated_array = np.hstack(stacks)
    X_features = np.stack(concatenated_array)
    
    # Create a KNNImputer object
    imputer = KNNImputer(n_neighbors=5)

    # Fit the imputer on the training data
    imputer.fit(X_train)

    # Transform the test data
    X_test = imputer.transform(X_features)

    return X_test, np.array(y_true)

def pred_mean(models, X_test):

    y_pred_list = []
    for model in models: 
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)

    y_pred_avg = np.mean(y_pred_list, axis=0)

    return y_pred_avg
