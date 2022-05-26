import numpy as np

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer

from ..encoding import PhaseEncoding

class QuantumAE():
    def __init__(self, num_qubits: int, n_components: int):
        self.num_qubits = num_qubits
        self.n_trash = num_qubits - n_components
        self._build_autoencoder(self.n_trash)

    def _build_autoencoder(self):
        
        self._build_circuit()
        self._build_SWAPTest(self.n_trash)
        
        self.qregisters = QuantumRegister(self.num_qubits + self.n_trash + 1, "q")
        self.autoencoder = QuantumCircuit(self.qregisters, name="Autoencoder")
        self.autoencoder.compose(self.circuit, qubits=self.qregisters[self.n_trash + 1:], inplace = True)
        self.autoencoder.compose(self.swap_test, qubits=self.qregisters[:2*self.n_trash + 1], inplace=True)

        print(self.autoencoder)


    def _build_circuit(self):
        N_PARAMETERS = self.num_qubits*4 + self.num_qubits*(self.num_qubits - 1)
        disc_weights_2 = ParameterVector('θ_d2', N_PARAMETERS)

        self.circuit = QuantumCircuit(self.num_qubits, name="Main circuit")
        self.circuit.barrier()

        weight_counter = 0

        for i in range(self.num_qubits):
            self.circuit.rx(disc_weights_2[weight_counter], i)
            self.circuit.rz(disc_weights_2[weight_counter+1], i)
            weight_counter += 2

        iterator = list(range(self.num_qubits-1, -1, -1))
        for i in iterator:
            iterator_j = iterator.copy()
            iterator_j.remove(i)
            for j in iterator_j:
                self.circuit.crx(disc_weights_2[weight_counter], i, j)
                weight_counter += 1

        for i in range(self.num_qubits):
            self.circuit.rx(disc_weights_2[weight_counter], i)
            self.circuit.rz(disc_weights_2[weight_counter+1], i)
            weight_counter += 2
        
    def _build_SWAPTest(self, n_trash):
        qregisters = QuantumRegister(2*n_trash + 1, "q")
        self.swap_test = QuantumCircuit(qregisters, name="SWAP test")
        self.swap_test.barrier()

        self.swap_test.h(qregisters[0])
        for i in range(n_trash):
            self.swap_test.cswap(qregisters[0], qregisters[i+1], qregisters[n_trash+i+1])
        self.swap_test.h(qregisters[0])

    def fit(self, dataset: np.ndarray = None) -> None:
        
        for input_vector in dataset:
            state = PhaseEncoding(input_vector)

            cregister = ClassicalRegister(1, "c")
            circ_aux = QuantumCircuit(self.qregisters, cregister)
            circ_aux.compose(state.circuit, qubits=self.qregisters[self.n_trash + 1:], inplace=True)
            circ_aux.compose(self.autoencoder, qubits=self.qregisters[self.n_trash + 1:], inplace=True)

            circ_aux.measure(self.qregisters[0], cregister[0])
            simulator = Aer.get_backend('aer_simulator')
            result = simulator.run(circ_aux, shots=1000).result()
            counts = result.get_counts(circ_aux)

            if '1' in counts.keys():
                z = counts['1'] / 1000
            else:
                z = 0
            return np.sqrt(z)
