import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from encoding.encoding import Encoding
from gates.RBS import RBS


class UnaryLoader(Encoding):

    def __init__(self, input_vector, inverse=False):
        self.num_qubits = int(len(input_vector))
        self.quantum_data = QuantumRegister(self.num_qubits)
        self.circ = QuantumCircuit(self.quantum_data)
        newx = np.copy(input_vector)
        betas = []
        self._beta_calc(newx, betas)
        self._dc_generate_circuit(betas)
        super().__init__("Divide and Conquer Amplitude Encoding")

    def _dc_generate_circuit(self, betas):
        logarithm = int(np.log2(self.num_qubits))

        for i in range(logarithm):
            for k in range(2**i):
                w = self.num_qubits // 2**i
                self.circ.unitary(RBS(betas[2**i + k - 1]).rbs, [self.quantum_data[k*w],
                                  self.quantum_data[k*w + w // 2]], label="RBS({})".format(np.around(betas[2**i + k - 1], 3)))

    @property
    def circ(self):
        return self._circ

    @circ.setter
    def circ(self, circuit: QuantumCircuit):
        self._circ = circuit
