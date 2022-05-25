import cmath, math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister

from .encoding import Encoding


class PhaseEncoding(Encoding):

    def __init__(self, input_vector: np.ndarray):
        input_vector = (input_vector.astype(float) / 255)* math.pi
        self._generate_circuit(input_vector)
        

        super().__init__("Phase encoding")
        

    def _generate_circuit(self, input_vector: np.ndarray):
        self.num_qubits = math.ceil(math.log2(len(input_vector)))

        self.quantum_data = QuantumRegister(self.num_qubits)
        self.circuit = QuantumCircuit(self.quantum_data)

        initial_state = np.array([], dtype=np.complex_)
        for k in range(len(input_vector)):
            x_k = complex(0, input_vector[k])
            initial_state = np.append(initial_state, cmath.exp(x_k))

        norm = 0
        for elem in initial_state:
            norm += elem * np.conj(elem)
        initial_state = initial_state / math.sqrt(norm)

        for _ in range(2**self.num_qubits - len(input_vector)):
            initial_state = np.append(initial_state, 0)
        
        self.circuit.initialize(initial_state, self.quantum_data)