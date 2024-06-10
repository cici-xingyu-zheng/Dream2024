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

sns.set_style('ticks')

input_path = 'Data'

feature_file = 'deepnose_features.npy'
CID_file = 'molecules_train_cid.npy'

mixture_file = 'Mixure_Definitions_Training_set.csv'
intensity_file = 'Mixure_Definitions_Intensity_Training_set.csv'
training_task_file = 'TrainingData_mixturedist.csv'

features = np.load(os.path.join(input_path, feature_file))
training_set = pd.read_csv(os.path.join(input_path, training_task_file))
mixtures_IDs = pd.read_csv(os.path.join(input_path, mixture_file))
molecule_intensities = pd.read_csv(os.path.join(input_path, intensity_file))

features_CIDs = np.load(os.path.join(input_path, CID_file))

extended_features = np.load(os.path.join(input_path, 'Extra/deepnose_features_extRavia.npy'))
extended_training_set = pd.read_csv(os.path.join(input_path, 'Extra/extended_training_set.csv'))
extended_mixture_IDs = pd.read_csv(os.path.join(input_path, 'Extra/extended_mixture_IDs.csv'))
extended_molecule_intensities = pd.read_csv(os.path.join(input_path, 'Extra/extended_molecule_intensites.csv'))
extended_features_CIDs = np.load(os.path.join(input_path, 'Extra/extended_ravia_cid.npy'))

scaler = StandardScaler(with_mean=True, with_std=True)
epsilon = 1e-8
features = scaler.fit_transform(np.log(features + epsilon))
CID2features =  {CID: features[i] for i, CID in enumerate(features_CIDs)}
extended_features = scaler.fit_transform(np.log(extended_features + epsilon))
extended_CID2features =  {CID: extended_features[i] for i, CID in enumerate(extended_features_CIDs)}

for key, value in extended_CID2features.items():
        if key not in CID2features:
            CID2features[key] = value


# Define the scaling constant
scaling_constant = 2 # Adjust this value as needed

# Get the list of column names containing "CID"
cid_columns = [col for col in molecule_intensities.columns if 'CID' in col]

# Create a mask to identify rows where "Dataset" is in ['Snitz 1', 'Snitz 2', 'Bushdid']
mask = molecule_intensities['Dataset'].isin(['Snitz 1', 'Snitz 2', 'Bushdid'])

# Scale the values of "CID" columns for the selected rows
molecule_intensities.loc[mask, cid_columns] *= scaling_constant

X = []
y = []
num_monos = []
CIDs_all = []

for _, row in training_set.iterrows():
    mixture1 = combine_molecules_intensity_weighed(label=row['Mixture 1'], dataset=row['Dataset'],
                                            mixtures_IDs=mixtures_IDs, CID2features=CID2features,
                                            mixtures_intensities= molecule_intensities)
    mixture2 = combine_molecules_intensity_weighed(label=row['Mixture 2'], dataset=row['Dataset'],
                                            mixtures_IDs=mixtures_IDs, CID2features=CID2features,
                                            mixtures_intensities= molecule_intensities)
    X.append((mixture1, mixture2))
    y.append(row['Experimental Values'])

_, _, num_mixtures, all_pairs_CIDs = format_Xy(training_set,  mixtures_IDs, CID2features, method = 'avg')

# Convert the input pairs to a suitable format for training
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
### add all information above
X_features = np.hstack((X_pairs, np.array(distances).reshape(500, 1), 
                        np.array(similarities).reshape(500, 1), 
                        np.array(angles).reshape(500, 1), 
                        np.array(shared_monos).reshape(500, 1), 
                        np.array(diff_monos).reshape(500, 1), 
                        np.array(num_mixtures), 
                        data_arr,
                        np.zeros((500, 2)) # this is given the addition of the new dataset// changed to 2 after filtiering out exp1
                        ))

extended_training_set = extended_training_set[extended_training_set['Dataset']!= 'Exp1']

X_extended = []
y_extended = []
num_monos = []
CIDs_all = []

for _, row in extended_training_set.iterrows():
    mixture1 = combine_molecules_intensity_weighed(label=row['Mixture 1'], dataset=row['Dataset'],
                                            mixtures_IDs=extended_mixture_IDs, CID2features=CID2features,
                                            mixtures_intensities= extended_molecule_intensities)
    mixture2 = combine_molecules_intensity_weighed(label=row['Mixture 2'], dataset=row['Dataset'],
                                            mixtures_IDs=extended_mixture_IDs, CID2features=CID2features,
                                            mixtures_intensities= extended_molecule_intensities)
    X_extended.append((mixture1, mixture2))
    y_extended.append(row['Experimental Values']/100)


X_extended = np.array([(np.concatenate((x1, x2))) for (x1, x2) in X_extended])
_, _, extra_num_mixtures, extend_pairs_CIDs = format_Xy(extended_training_set,  extended_mixture_IDs, CID2features, method = 'avg')

y_true_extended = np.array(y_extended)

distances_e = [get_euclidean_distance(m[0], m[1]) for m in X_extended]
similarities_e = [get_cosine_similarity(m[0], m[1]) for m in X_extended]
angles_e = [get_cosine_angle(m[0], m[1]) for m in X_extended]
shared_monos_e = [ len( set(pair[0]).intersection(set(pair[1]))) for pair in extend_pairs_CIDs]
diff_monos_e = [ len( set(pair[0]).difference(set(pair[1]))) for pair in extend_pairs_CIDs]
datasets_e = extended_training_set['Dataset'].to_numpy()
encoder = OneHotEncoder()
data_arr_e = encoder.fit_transform(datasets_e.reshape(-1, 1))
data_arr_e = data_arr_e.toarray()
### add all information above

X_features_extended = np.hstack((X_extended, np.array(distances_e).reshape(220, 1), 
                        np.array(similarities_e).reshape(220, 1), 
                        np.array(angles_e).reshape(220, 1), 
                        np.array(shared_monos_e).reshape(220, 1), 
                        np.array(diff_monos_e).reshape(220, 1), 
                        np.array(extra_num_mixtures), 
                        np.zeros((220, 4)), # this is given the addition of the new dataset
                        data_arr_e
                        ))

### not done bc the current optimization code cannot handle augmented features yet.