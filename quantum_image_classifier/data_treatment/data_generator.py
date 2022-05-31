from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generate_synthetic_data(n_dim: int, n_clusters: int, n_samples: int) -> tuple:
    """
    Function to generate synthetic data to test the algorithms.

    Return:
        tuple: containing the set of training and test values, with their associated labels
    """

    X, y = make_blobs(n_samples=n_samples, 
                    n_features=n_dim, 
                    centers=n_clusters)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    return train_X, train_y, test_X, test_y