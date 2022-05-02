import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from encoding.Encoding import Encoding
from encoding.bin_tree import bin_tree
from gates.iRBS import iRBS
from gates.RBS import RBS

class DC_Amplitude_Encoding(Encoding):
    tree = None
    
    def __init__(self, input_vector, inverse = False):
            self.num_qubits = int(len(input_vector))
            self.quantum_data = QuantumRegister(self.num_qubits)
            self.circ = QuantumCircuit(self.quantum_data)
            newx = np.copy(input_vector)
            betas = []
            #betas = self.get_angles(newx)
            self._beta_calc(newx, betas)
            self._dc_generate_circuit(betas)
            super().__init__("Divide and Conquer Amplitude Encoding")

    def _dc_generate_circuit(self, betas):
        logarithm = int(np.log2(self.num_qubits))

        for i in range(logarithm):
            for k in range(2**i):
                w = self.num_qubits // 2**i
                #self.qcircuit.append(iRBS(beta[i]), [self.quantum_data[k*i], self.quantum_data[k*i + k // 2]], [])
                self.circ.unitary(RBS(betas[2**i + k - 1]).rbs, [self.quantum_data[k*w], self.quantum_data[k*w + w // 2]], label="RBS({})".format(np.around(betas[2**i + k -1], 3)))
        
    @property
    def circ(self):
        return self._circ
    
    @circ.setter
    def circ(self, circuit: QuantumCircuit):
        self._circ = circuit