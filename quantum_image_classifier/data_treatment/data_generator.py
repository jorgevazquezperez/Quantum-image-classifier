from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def generate_synthetic_data(n_dim: int, n_clusters: int, n_samples: int) -> tuple:
    """
    Function to generate synthetic data to test the algorithms.

    Return:
        tuple: containing the set of training and test values, with their associated labels
    """
    random_state = 42

    X, y = make_blobs(n_samples=n_samples, 
                    n_features=n_dim, 
                    centers=n_clusters, 
                    random_state=random_state)

    breakpoint = n_samples * 3 // 4

    # (training, test)
    return X[:breakpoint], y[:breakpoint], X[breakpoint:], y[breakpoint:]