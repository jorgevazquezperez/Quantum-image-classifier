import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from encoding.unary_loader import UnaryLoader


class NearestCentroid:
    """Nearest Centroid classifier.

    The Nearest Centroid is a classification algorithm that assigns each observation a label
    based on which centroid of the different clusters available in the data is the closest.

    For doing this, we load the classical data to quantum info using a Unary Loader [1] and calculate
    an inner product of the vectors. We use this inner product to estimate the distance between
    the observation with each centroid to, only then, select the closest to assign a label to it.

    ** References: **
    [1] Sonika Johri, Shantanu Debnath, Avinash Mocherla, Alexandros SINGK, Anupam Prakash, Jungsang Kim 
    and Iordanis Kerenidis et al.,
    `Nearest centroid classification on a trapped ion quantum computer 
    <https://www.nature.com/articles/s41534-021-00456-5>`

    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_dim: int) -> None:
        """

        Args:
            X: Training data values of dimension k
            y: Training data labels of dimension k
            n_dim: Dimension of the training dataset, i.e., k
        """
        self.n_dim = n_dim
        self.centroids = self._calc_centroids(X, y)

        self.circ_centroids = {}
        for label, centroid in self.centroids.items():
            qregisters = QuantumRegister(n_dim, "q")
            cregisters = ClassicalRegister(1, "c")
            circ_centroid = QuantumCircuit(qregisters, cregisters)

            circ_centroid.x(qregisters[0])
            circ_centroid.compose(UnaryLoader(centroid).circuit, inplace=True)
            self.circ_centroids[label] = circ_centroid

    def _calc_centroids(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Calculate the centroids of the clusters classically

        Args:
            X: Training data values of dimension k
            y: Training data labels of dimension k
        Returns:
            dict: with the centroid point associated to each label
        """
        centroids = dict.fromkeys(y)
        for label in centroids:
            centroids[label] = np.repeat(float(0), self.n_dim)
        for x, label in zip(X, y):
            centroids[label] += x / len(X)
        return centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the label of a batch of observations

        Args:
            X: Test data values of dimension l
            y: Test data labels of dimension l
        Returns:
            nd.ndarray: with the label of each observation in the batch
        """
        labels = []
        for x in X:
            dist = {}
            for label, centroid in self.centroids.items():
                dist[label] = self.quantum_distance(
                    self.circ_centroids[label], centroid, x)
            labels.append(min(dist, key=dist.get))
        return np.array(labels)

    def quantum_distance(self, circ_centroid: QuantumCircuit, centroid: np.ndarray, y: np.ndarray, repetitions: int = 1000) -> float:
        """
        Calculates the distance between x (centroid) and y using circuit to calculate the inner product. The distance is the
        euclidean distance, represented by the formula:

        dist = sqrt(||x||**2 + ||y||**2 - 2*||x||*||y||*inner)

        where inner = <x|y>, i.e., inner product between x and y.

        Args:
            circ_centroid: Unary loader of the centroid
            centroid: First vector to perform the distance
            y: First vector to perform the distance
        Returns:
            float: distance between x and y
        """
        norm_x = np.linalg.norm(centroid)
        norm_y = np.linalg.norm(y)
        inner = self._quantum_inner(circ_centroid, y, repetitions)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner)
        return dist

    def _quantum_inner(self, circ_centroid: QuantumCircuit, y: np.ndarray, repetitions: int = 1000) -> float:
        """
        Inner product calculation of x (centroid) and y.

        Args:
            circ_centroid: Unary loader of the centroid
            y: First vector to perform the distance
        Returns:
            float: inner product between a centroid and y
        """ 
        circ_y = UnaryLoader(y, inverse=True).circuit.inverse()
        circuit = circ_centroid.compose(circ_y)
        circuit.measure([0], [0])
        simulator = Aer.get_backend('aer_simulator')
        result = simulator.run(circuit, shots=repetitions).result()
        counts = result.get_counts(circuit)

        if '1' in counts.keys():
            z = counts['1'] / repetitions
        else:
            z = 0
        return np.sqrt(z)
