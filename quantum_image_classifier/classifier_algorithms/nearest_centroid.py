from typing import Union
import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from ..encoding import UnaryLoader

from qat.interop.qiskit import qiskit_to_qlm


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

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, n_dim: int = None) -> None:
        """

        Args:
            X: Training data values of dimension k or None
            y: Training data labels of dimension k or None
            n_dim: Dimension of the training dataset, i.e., k or None
        Raise: 
            AttributeError: In case some attributes have values and others are empty
        """
        if (X is not None) and (y is not None) and (n_dim is not None):
            self.train_X = X
            self.train_y = y
            self.n_dim = n_dim
        elif (X is None) and (y is None) and (n_dim is None):
            pass
        else:
            raise AttributeError("Either all attributes are empty or have the accurate values.")
    
    def _calc_circ_centroids(self) -> None:
        """ Calculate centroids' circuits """
        self.centroids = self._calc_centroids()

        self.circ_centroids = {}
        for label, centroid in self.centroids.items():
            # After we calculate each centroid, we create its respective circuit
            circ_centroid = UnaryLoader(centroid).circuit
            self.circ_centroids[label] = circ_centroid
    
    def _calc_centroids(self) -> dict:
        """ 
        Calculate the centroids of the clusters classically

        Returns:
            dict: with the centroid point associated to each label
        """
        centroids = dict.fromkeys(self.train_y)
        for label in centroids:
            centroids[label] = np.repeat(float(0), self.n_dim)
        for x, label in zip(self.train_X, self.train_y):
            centroids[label] += x / len(self.train_X)
        return centroids

    def _quantum_distance(self, circ_centroid: QuantumCircuit, centroid: np.ndarray, y: np.ndarray, repetitions: int = 1000) -> float:
        """
        Calculates the distance between x (centroid) and y using a quantum circuit to calculate the inner product. 
        The distance is the euclidean distance, represented by the formula:

        dist = sqrt(||x||**2 + ||y||**2 - 2*||x||*||y||*inner)

        where inner = <x|y>, i.e., inner product between x and y.

        Args:
            circ_centroid: Unary loader of the centroid
            centroid: First vector to perform the distance
            y: Second vector to perform the distance
            repetitions: Times we simulate the circuit
        Returns:
            float: distance between x and y
        """
        norm_x = np.linalg.norm(centroid)
        norm_y = np.linalg.norm(y)
        inner = self._quantum_inner(circ_centroid, y, repetitions)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner)
        return dist
    
    def _classical_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the distance between x (centroid) and y using the classical version. The distance is the
        euclidean distance, represented by the formula:

        dist = sqrt(||x||**2 + ||y||**2 - 2*||x||*||y||*inner)

        where inner = <x|y>, i.e., inner product between x and y.

        Args:
            x: First vector to perform the distance
            y: Second vector to perform the distance
        Returns:
            float: distance between x and y
        """

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * np.dot(x / norm_x, y / norm_y))
        return dist

    def _quantum_inner(self, circ_centroid: QuantumCircuit, x: np.ndarray, repetitions: int = 2000) -> float:
        """
        Inner product calculation of x (centroid) and y.

        Args:
            circ_centroid: Unary loader of the centroid
            x: First vector to perform the distance
            repetitions: Times we simulate the circuit
        Returns:
            float: inner product between a centroid and y
        """ 
        # We execute the circuit a defined times of repetitions
        circ_x = UnaryLoader(x, inverse=True).circuit
        circuit = circ_centroid.compose(circ_x)
        circuit.measure([0], [0])

        qlm_circuit = qiskit_to_qlm(circuit)
        print(qlm_circuit)

        simulator = Aer.get_backend('aer_simulator')
        result = simulator.run(circuit, shots=repetitions).result()
        counts = result.get_counts(circuit)

        if '1' in counts.keys():
            z = counts['1'] / repetitions
        else:
            z = 0

        return np.sqrt(z)

    def fit(self, X: np.ndarray = None, y: np.ndarray = None, n_dim: int = None) -> None:
        """
        Calculates the centroids of the training data classically.

        Args:
            X: Train data values of dimension l
            y: Train data labels of dimension l 
            n_dim: Dimension of each value of the training set
        Raise:
            AttributeError: if the training set is not initialize and it is not passed as argument
            to the function
        """
        if (X is not None) and (y is not None):
            self.train_X = X
            self.train_y = y
            self.n_dim = n_dim
            self._calc_circ_centroids()
        elif (X is None) and (y is None):
            if (self.train_X is not None) and (self.train_y is not None):
                self._calc_circ_centroids()
            else:
                raise AttributeError("You need to initialize the dataset.")

    def predict(self, X: np.ndarray, test_distance: bool = False) -> Union[tuple,np.ndarray]:
        """
        Predicts the label of a batch of observations and, if test_distance == True
        returns the difference between classical and quantum distance (the error of the 
        quantum distance), along with the average distance calculated.

        Args:
            X: Test data values of dimension l
            test_distance: boolean which determines wheter to return the average distance
            calculated along with the error of the quantum distance.
        Returns:
            nd.ndarray: with the label of each observation in the batch
            tuple: with the label of each observation, the average distance calculated and
            the error of the quantum distance
        """
        if self.circ_centroids is not None:
            labels = []
            difference = []
            distance = []
            for x in X:
                dist = {}
                # For every x in the test set X we calculate the distance to each centroid and select the minimum 
                for label, centroid in self.centroids.items():
                    dist[label] = self._quantum_distance(
                        self.circ_centroids[label], centroid, x)
                    difference.append(dist[label] - self._classical_distance(centroid, x))
                    distance.append(dist[label])
                labels.append(min(dist, key=dist.get))
            if test_distance ==  True:
                return np.array(labels), difference, distance
            else:
                return np.array(labels)
        else:
            raise AttributeError("Training set cannot be empty to predict.")

    