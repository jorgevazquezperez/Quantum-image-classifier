from turtle import circle
import numpy as np
from qiskit import QuantumCircuit
from classifier_algorithms.NearestCentroid import NearestCentroid
from encoding.Amplitude_Encoding import Amplitude_Encoding
from encoding.DC_Amplitude_Encoding import DC_Amplitude_Encoding
from gates.iRBS import iRBS

x = np.repeat(1, 8)
y = np.repeat(1, 8)

nearest_centroid = NearestCentroid(x, y)
nearest_centroid.quantum_distance(x, y)