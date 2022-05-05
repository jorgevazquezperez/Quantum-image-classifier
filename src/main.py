import numpy as np
from classifier_algorithms.nearest_centroid import NearestCentroid
from data_treatment.data_generator import generate_synthetic_data
from data_treatment.data_loader import get_MNIST

n_dim = 8
n_clusters = 4
#X, y = generate_synthetic_data(n_dim, n_clusters)
train_X, train_y, test_X, test_y = get_MNIST(8)
nearest_centroid = NearestCentroid(train_X[:200], train_y[:200], n_dim)
labels_predicted = nearest_centroid.predict(test_X[:50])

print(test_y[:50])
print(labels_predicted)
