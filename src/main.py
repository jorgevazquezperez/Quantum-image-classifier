import numpy as np
from classifier_algorithms.NearestCentroid import NearestCentroid
from encoding.Encoding import Encoding

#x = np.random.rand(1,8).flatten()
#y = np.random.rand(1,8).flatten()

x = np.array([1, 2, 0, 0, 3, 4, 1, 1])


y = np.array([0, 1, 0, 1, 2, 0, 0, 2])

betas = []
#nearest_centroid = NearestCentroid(x, y)
Encoding._recursive_compute_beta2(x, betas)
#quantum_dist = nearest_centroid.quantum_distance(x, y)
#classical_dist = nearest_centroid.classical_distance(x, y)  

#print(nearest_centroid.circ)
#print("Quantum dist: " + str(quantum_dist))
#print("Classical dist: " + str(classical_dist))
