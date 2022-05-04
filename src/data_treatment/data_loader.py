import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
dirname = os.path.dirname(__file__)
train = os.path.join(dirname, '../../data/mnist_train.csv')
test = os.path.join(dirname, '../../data/mnist_test.csv')

def get_MNIST() -> None:
    """
    Function to get the MNIST dataset and perform a PCA to it
    """
    train_csv = pd.read_csv(train)
    train_y = train_csv['label']
    train_X = train_csv.drop("label",axis=1)

    test_csv = pd.read_csv(test)
    test_y = test_csv['label']
    test_X = test_csv.drop("label",axis=1)

    train_X = do_pca(8, train_X)
    test_X = do_pca(8, test_X)

    return train_X, train_y, test_X, test_y

def do_pca(n_components: int, data: np.ndarray) -> None:
    """
    Function to perform PCA to a given data.

    Args:
        n_components: number of components to which we reduce the data
        data: the info we want to apply PCA to
    """
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

if(__name__ == "__main__"):
    get_MNIST()