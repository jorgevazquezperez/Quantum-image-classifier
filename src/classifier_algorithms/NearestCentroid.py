import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram
from encoding.DC_Amplitude_Encoding import DC_Amplitude_Encoding

class NearestCentroid:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_dim: int):

        self.n_dim = n_dim
        self.centroids = self._calc_centroids(X, y)
        self.circ_centroids = {}
        for label, centroid in self.centroids.items():
            qregisters = QuantumRegister(n_dim, "q")
            cregisters = ClassicalRegister(1, "c")
            circ_centroid = QuantumCircuit(qregisters, cregisters)

            circ_centroid.x(qregisters[0])
            circ_centroid.compose(DC_Amplitude_Encoding(centroid).circ, inplace=True)
            self.circ_centroids[label] = circ_centroid
            print(circ_centroid)

        """
        circ_x = DC_Amplitude_Encoding(x).circ
        circ_y = DC_Amplitude_Encoding(y, inverse=True).circ.inverse()

        qregisters = QuantumRegister(len(x), "q")
        cregisters = ClassicalRegister(1, "c")
        self.circ = QuantumCircuit(qregisters, cregisters)

        self.circ.x(qregisters[0])
        self.circ = self.circ.compose(circ_x)
        self.circ = self.circ.compose(circ_y)

        self.circ.measure(qregisters[0], cregisters[0])
        """

    @property
    def circ(self):
        return self._circ
    
    @circ.setter
    def circ(self, circuit: QuantumCircuit):
        self._circ = circuit

    def _calc_centroids(self, X, y):
        centroids = dict.fromkeys(y)
        for label in centroids:
            centroids[label] = np.repeat(float(0), self.n_dim)
        for x, label in zip(X, y):
            centroids[label] += x / len(X)
        return centroids

    def predict(self, X: np.ndarray, y: np.ndarray):
        labels = []
        for x in X:
            dist = {}
            circ_x = DC_Amplitude_Encoding(x, inverse=True).circ.inverse()
            for label, circ_centroid in self.circ_centroids.items():
                circ_aux = circ_centroid.compose(circ_x)
                circ_aux.measure([0], [0])
                dist[label] = self.quantum_distance(circ_aux, self.centroids[label], x)
            labels.append(min(dist, key=dist.get))
        return labels
    
    def quantum_distance(self, circuit, x, y, repetitions=1000):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        inner = self._quantum_inner(circuit, repetitions)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner)
        return dist

    def classical_distance(self, x, y):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * np.dot(x / norm_x, y / norm_y))
        return np.dot(x / norm_x, y / norm_y)

    def _quantum_inner(self, circuit, repetitions=1000):
        simulator = Aer.get_backend('aer_simulator')
        result = simulator.run(circuit, shots= repetitions).result()
        counts = result.get_counts(circuit)

        if '1' in counts.keys():
            z = counts['1'] / repetitions
        else:
            z = 0
        return np.sqrt(z)