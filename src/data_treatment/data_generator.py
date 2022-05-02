from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def generate_synthetic_data(n_dim: int, n_clusters: int):
    n_samples = 250
    random_state = 42

    X, y = make_blobs(n_samples=n_samples, 
                    n_features=n_dim, 
                    centers=n_clusters, 
                    random_state=random_state)
    

    # fig=plt.figure(figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    return X, y