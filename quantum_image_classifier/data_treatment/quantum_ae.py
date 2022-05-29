import numpy as np

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble, transpile
from qiskit.algorithms.optimizers import ADAM

from ..encoding import PhaseEncoding


class QuantumAE():
    def __init__(self, num_qubits: int, n_components: int):
        self.num_qubits = num_qubits
        self.n_trash = num_qubits - n_components
        self._build_autoencoder()

    def _build_autoencoder(self):

        self._build_circuit()
        self._build_SWAPTest(self.n_trash)

        self.qregisters = QuantumRegister(
            self.num_qubits + self.n_trash + 1, "q")
        self.autoencoder = QuantumCircuit(self.qregisters, name="Autoencoder")
        self.autoencoder.compose(
            self.circuit, qubits=self.qregisters[self.n_trash + 1:], inplace=True)
        self.autoencoder.compose(
            self.swap_test, qubits=self.qregisters[:2*self.n_trash + 1], inplace=True)

    def _build_circuit(self):
        N_PARAMETERS = self.num_qubits*4 + \
            self.num_qubits*(self.num_qubits - 1)
        self.params = ParameterVector('θ', N_PARAMETERS)

        self.circuit = QuantumCircuit(self.num_qubits, name="Main circuit")
        self.circuit.barrier()

        weight_counter = 0

        for i in range(self.num_qubits):
            self.circuit.rx(self.params[weight_counter], i)
            self.circuit.rz(self.params[weight_counter+1], i)
            weight_counter += 2

        iterator = list(range(self.num_qubits-1, -1, -1))
        for i in iterator:
            iterator_j = iterator.copy()
            iterator_j.remove(i)
            for j in iterator_j:
                self.circuit.crx(self.params[weight_counter], i, j)
                weight_counter += 1

        for i in range(self.num_qubits):
            self.circuit.rx(self.params[weight_counter], i)
            self.circuit.rz(self.params[weight_counter+1], i)
            weight_counter += 2

    def _build_SWAPTest(self, n_trash):
        qregisters = QuantumRegister(2*n_trash + 1, "q")
        self.swap_test = QuantumCircuit(qregisters, name="SWAP test")
        self.swap_test.barrier()

        self.swap_test.h(qregisters[0])
        for i in range(n_trash):
            self.swap_test.cswap(
                qregisters[0], qregisters[i+1], qregisters[n_trash+i+1])
        self.swap_test.h(qregisters[0])

    def fit(self, dataset: np.ndarray = None) -> None:

        opt = ADAM(lr=0.1, maxiter=10)

        init_ae_params = np.random.uniform(low=0,
                                           high=np.pi,
                                           size=(len(self.params),))
        self.i = 0
        def loss(params):
            expectation = 0
            dict = {}
            for param, value in zip(self.params, params):
                dict[param] = value
            for input_vector in dataset:
                state = PhaseEncoding(input_vector)

                cregister = ClassicalRegister(1, "c")
                circ_aux = QuantumCircuit(self.qregisters, cregister)
                circ_aux.compose(                  
                                state.circuit, qubits=self.qregisters[self.n_trash + 1:], inplace=True)
                circ_aux.compose(self.autoencoder,
                                 qubits=self.qregisters, inplace=True)
                circ_aux.measure(self.qregisters[0], cregister[0])

                simulator = Aer.get_backend('aer_simulator')
                t_qc = transpile(circ_aux,
                                simulator)

                qobj = assemble(t_qc,
                                shots=1024,
                                parameter_binds = [dict])
                job = simulator.run(qobj)
                result = job.result().get_counts()

                counts = np.array(list(result.values()))
                states = np.array(list(result.keys())).astype(float)
                
                # Compute probabilities for each state
                probabilities = counts / 1024
                # Get state expectation
                expectation += np.sum(states * probabilities)
            print(expectation / len(dataset))
            return expectation / len(dataset)

        result_params = opt.minimize(loss, init_ae_params)
        print(loss(result_params))
