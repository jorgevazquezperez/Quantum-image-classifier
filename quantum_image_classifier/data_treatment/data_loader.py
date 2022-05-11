from optparse import OptionError
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

import os
dirname = os.path.dirname(__file__)
train = os.path.join(dirname, '../../data/mnist_train.csv')
test = os.path.join(dirname, '../../data/mnist_test.csv')

def get_MNIST(n_components, reduction: str = "PCA") -> tuple:
    """
    Function to get the MNIST dataset and perform a PCA to it
    """
    train_csv = pd.read_csv(train)
    train_y = np.array(train_csv['label'])
    train_X = np.array(train_csv.drop("label",axis=1))

    test_csv = pd.read_csv(test)
    test_y = np.array(test_csv['label'])
    test_X = np.array(test_csv.drop("label",axis=1))

    if reduction == "PCA":
        # We apply PCA to each the training and the test dataset
        train_X = do_pca(n_components, train_X)
        test_X = do_pca(n_components, test_X)
    elif reduction == "AE":
        train_X = do_AE(n_components, train_X)
        test_X = do_AE(n_components, train_X)
    else:
        raise OptionError()

    return train_X, train_y, test_X, test_y

def do_pca(n_components: int, data: np.ndarray) -> np.ndarray:
    """
    Function to perform PCA to a given data.

    Args:
        n_components: number of components to which we reduce the data
        data: the info we want to apply PCA to
    """
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = np.array(pca.fit_transform(X))
    return X_pca

def do_AE(n_components: int, data: np.ndarray) -> np.ndarray:
    # Shape of input and latent variable
    n_input = 28*28

    # Encoder structure
    n_encoder1 = 500
    n_encoder2 = 300

    n_latent = n_components

    # Decoder structure
    n_decoder2 = 300
    n_decoder1 = 500

    reg = MLPRegressor(hidden_layer_sizes = (n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1), 
                   activation = 'tanh', 
                   solver = 'adam', 
                   learning_rate_init = 0.0001, 
                   max_iter = 20, 
                   tol = 0.0000001, 
                   verbose = True)
    reg.fit(data, data)

    data = np.asmatrix(data)
    
    encoder1 = data*reg.coefs_[0] + reg.intercepts_[0]
    encoder1 = (np.exp(encoder1) - np.exp(-encoder1))/(np.exp(encoder1) + np.exp(-encoder1))
    
    encoder2 = encoder1*reg.coefs_[1] + reg.intercepts_[1]
    encoder2 = (np.exp(encoder2) - np.exp(-encoder2))/(np.exp(encoder2) + np.exp(-encoder2))
    
    latent = encoder2*reg.coefs_[2] + reg.intercepts_[2]
    latent = (np.exp(latent) - np.exp(-latent))/(np.exp(latent) + np.exp(-latent))
    
    return np.asarray(latent)