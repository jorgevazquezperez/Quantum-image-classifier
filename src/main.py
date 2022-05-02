import numpy as np
from classifier_algorithms.NearestCentroid import NearestCentroid
from encoding.Encoding import Encoding
from data_treatment.data_generator import generate_synthetic_data

n_dim = 8
n_clusters = 4
X, y = generate_synthetic_data(n_dim, n_clusters)
nearest_centroid = NearestCentroid(X[:200], y[:200], n_dim)
labels_predicted = nearest_centroid.predict(X[200:], y[200:])

print(y[200:])
print(np.array(labels_predicted))