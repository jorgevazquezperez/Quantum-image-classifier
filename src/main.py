import numpy as np
from classifier_algorithms.NearestCentroid import NearestCentroid
from encoding.Encoding import Encoding

x = np.array(np.random.randint(5, size=8))
y = np.array(np.random.randint(5, size=8))


#x = np.array([3, 2, 3, 0, 1, 2, 2, 4])
#y = np.array([3, 1, 2, 0, 1, 0, 0, 1])

nearest_centroid = NearestCentroid(x, y)
quantum_dist = nearest_centroid.quantum_distance(x, y)
classical_dist = nearest_centroid.classical_distance(x, y)  

print(nearest_centroid.circ)
print(quantum_dist, classical_dist)
