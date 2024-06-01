import numpy as np
import math


def get_individual_embedding(label, dataset, mixtures_IDs, CID2features):
    # Grab the unique data row:
    row = mixtures_IDs[(mixtures_IDs['Mixture Label'] == label) & (mixtures_IDs['Dataset'] == dataset)]
    non_zero_CIDs = row.loc[:, row.columns.str.contains('CID')].loc[:, (row != 0).any(axis=0)]
    if len(non_zero_CIDs) != 1:
        print('Not a Unique pointer!!!')
    CIDs = non_zero_CIDs.iloc[0].tolist()
    molecule_embeddings = []
    # Create feature matrix for all number of mono odor molecules in the mixture:
    for CID in CIDs:
        molecule_embeddings.append(CID2features[CID])
    
    return np.array(molecule_embeddings), CIDs

def combine_molecules(label, dataset, mixtures_IDs, CID2features, method='avg', beta=None):
    """
    Return mixture embedding vector, and summary stats
    """
    molecule_embeddings, CIDs = get_individual_embedding(label, dataset, mixtures_IDs, CID2features)
    num_mono = molecule_embeddings.shape[0]

    if method == 'avg':
        mixture_embedding = molecule_embeddings.mean(axis=0)
    elif method == 'sum':
        mixture_embedding = molecule_embeddings.sum(axis=0)
    elif method == 'max':
        mixture_embedding = molecule_embeddings.max(axis=0)
    elif method == 'log':
        if beta is None:
            exp_embeddings = np.exp(molecule_embeddings)
            summed = np.sum(exp_embeddings, axis=0)
            mixture_embedding = np.log(summed)
        else:
            mixture_embedding = log_sum_exp_beta(molecule_embeddings, beta)
    elif method == 'geometric':
        product = np.prod(molecule_embeddings, axis=0)
        mixture_embedding = np.power(product, 1 / molecule_embeddings.shape[0])
    else:
        raise ValueError(f"Invalid method: {method}")

    return mixture_embedding, (num_mono, CIDs)

def log_sum_exp_beta(molecule_embeddings, beta):
    exp_embeddings = np.exp(beta * molecule_embeddings)
    summed = np.sum(exp_embeddings, axis=0)
    mixture_embedding = (1 / beta) * np.log(summed)
    return mixture_embedding


def format_Xy(training_set, mixtures_IDs, CID2features, method='log', beta=None):
    X = []
    y = []
    num_monos = []
    CIDs_all = []

    for _, row in training_set.iterrows():
        mixture1, summary1 = combine_molecules(label=row['Mixture 1'], dataset=row['Dataset'],
                                               mixtures_IDs=mixtures_IDs, CID2features=CID2features,
                                               method=method, beta=beta)
        mixture2, summary2 = combine_molecules(label=row['Mixture 2'], dataset=row['Dataset'],
                                               mixtures_IDs=mixtures_IDs, CID2features=CID2features,
                                               method=method, beta=beta)
        X.append((mixture1, mixture2))
        y.append(row['Experimental Values'])
        num_monos.append([summary1[0], summary2[0]])
        CIDs_all.append([summary1[1], summary2[1]])

    return X, y, num_monos, CIDs_all


def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def get_euclidean_distance(mixture_1, mixture_2):

    sum_squared_diff = np.sum((mixture_1 - mixture_2) ** 2)

    return np.sqrt(sum_squared_diff)

def get_cosine_similarity(mixture_1, mixture_2):

    # Calculate the dot product
    dot_product = np.dot(mixture_1, mixture_2)
    
    # Calculate the magnitudes
    magnitude_1 = np.sqrt(np.sum(mixture_1 ** 2))
    magnitude_2 = np.sqrt(np.sum(mixture_2 ** 2))
    
    # Calculate the cosine similarity
    similarity = dot_product / (magnitude_1 * magnitude_2)

    # Clip the similarity value to the valid range [-1, 1]; to prevent numerical error
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return similarity


def get_cosine_angle(mixture_1, mixture_2):
    cosyne_sim = get_cosine_similarity(mixture_1, mixture_2)

    return  math.acos(cosyne_sim)


