import numpy as np
from classifier_algorithms.NearestCentroid import NearestCentroid
from encoding.Encoding import Encoding

a = np.array(np.random.randint(5, size=8))
b = np.array(np.random.randint(5, size=8))


x = np.array([4, 2, 4, 4, 0, 4, 4, 2])
y = np.array([3, 1, 1, 2, 2, 2, 1, 0])

print(a, b)
print(x, y)

betas = []
nearest_centroid = NearestCentroid(x, y)
quantum_dist = nearest_centroid.quantum_distance(x, y)
classical_dist = nearest_centroid.classical_distance(x, y)  


print(nearest_centroid.circ)
print("Quantum dist: " + str(quantum_dist))
print("Classical dist: " + str(classical_dist))