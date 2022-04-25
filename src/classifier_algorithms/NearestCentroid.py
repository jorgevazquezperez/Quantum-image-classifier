import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, transpile

from encoding.DC_Amplitude_Encoding import DC_Amplitude_Encoding

class NearestCentroid:


    def __init__(self, x: np.array, y: np.array):
        circ_x = DC_Amplitude_Encoding(x).get_circuit()
        circ_y = DC_Amplitude_Encoding(y).get_circuit().reverse_ops()

        qregisters = QuantumRegister(len(x), "q")
        cregisters = ClassicalRegister(1, "c")
        self.circ = QuantumCircuit(qregisters, cregisters)

        self.circ.x(qregisters[0])
        self.circ = self.circ.compose(circ_x)
        self.circ = self.circ.compose(circ_y)

        self.circ.measure(qregisters[0], cregisters[0])

        print(self.circ)
    
    def quantum_distance(self, x, y, repetitions=1000):

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        inner = self._quantum_inner(x, y, repetitions)
        dist = np.sqrt(norm_x**2 + norm_y**2 - 2 * norm_x * norm_y * inner)
        return dist

    """
    def _quantum_inner(x, y, repetitions=1000):
        qubits = cirq.LineQubit.range(len(x))
        gates_x = cirq.decompose(VectorLoader(x)(*qubits))
        gates_y = cirq.decompose(VectorLoader(y, x_flip=False)(*qubits)**-1)
        circuit = cirq.Circuit(gates_x + gates_y, cirq.measure(qubits[0]))
        simulator = cirq.Simulator()
        measurements = simulator.run(circuit, repetitions=repetitions)
        total = measurements.histogram(key='0')[1]
        inner = np.sqrt(total / repetitions)
        return inner
    """
    

    def _quantum_inner(self, x, y, repetitions=1000):
        backend = BasicAer.get_backend('qasm_simulator')
        #circ = transpile(self.circ, backend)
        job = backend.run(self.circ)
        counts = job.result().get_counts(0)
        print(counts)
        
    
    def _quantum_distance(self, x, y, repetitions=1000):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        dist = np.sqrt(norm_x**2 + norm_y**2 -
                    2 * norm_x * norm_y * self.quantum_inner(x, y))
        return dist