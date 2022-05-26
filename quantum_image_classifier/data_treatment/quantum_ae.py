from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister

class QuantumAE():
    def __init__(self, num_qubits: int, n_components: int):
        self.num_qubits = num_qubits
        self._build_autoencoder(self.num_qubits - n_components)

    def _build_autoencoder(self, n_trash: int):
        
        self._build_circuit()
        self._build_SWAPTest(n_trash)
        
        qregisters = QuantumRegister(self.num_qubits + n_trash + 1, "q")
        self.autoencoder = QuantumCircuit(qregisters, name="Autoencoder")
        self.autoencoder.compose(self.circuit, qubits=qregisters[n_trash + 1:], inplace = True)
        self.autoencoder.compose(self.swap_test, qubits=qregisters[:2*n_trash + 1], inplace=True)

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

    def fit():
        pass

