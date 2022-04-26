import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram

from encoding.DC_Amplitude_Encoding import DC_Amplitude_Encoding

class NearestCentroid:


    def __init__(self, x: np.array, y: np.array):
        circ_x = DC_Amplitude_Encoding(x).circ
        circ_y = DC_Amplitude_Encoding(y).circ.reverse_ops()

        qregisters = QuantumRegister(len(x), "q")
        cregisters = ClassicalRegister(1, "c")
        self.circ = QuantumCircuit(qregisters, cregisters)

        self.circ.x(qregisters[0])
        self.circ = self.circ.compose(circ_x)
        self.circ = self.circ.compose(circ_y)

        self.circ.measure(qregisters[0], cregisters[0])

    @property
    def circ(self):
        return self._circ
    
    @circ.setter
    def circ(self, circuit: QuantumCircuit):
        self._circ = circuit
    
    def quantum_distance(self, x, y, repetitions=1000):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        inner = self._quantum_inner(x, y, repetitions)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner)
        return dist

    def classical_distance(self, x, y):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * np.dot(x / norm_x, y / norm_y))
        return dist

    def _quantum_inner(self, x, y, repetitions=1000):
        simulator = Aer.get_backend('aer_simulator')
        self.circ = transpile(self.circ, simulator)
        result = simulator.run(self.circ, shots= repetitions).result()
        counts = result.get_counts(self.circ)
        if '1' in counts.keys():
            z = counts['1'] / repetitions
        else:
            z = 0
        return z