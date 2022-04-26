import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from encoding.Encoding import Encoding
from encoding.bin_tree import bin_tree
from gates.iRBS import iRBS
from gates.RBS import RBS

class DC_Amplitude_Encoding(Encoding):
    tree = None
    
    def __init__(self, input_vector):
            self.num_qubits = int(len(input_vector))
            self.quantum_data = QuantumRegister(self.num_qubits)
            self.qcircuit = QuantumCircuit(self.quantum_data)
            newx = np.copy(input_vector)
            betas = []
            self._recursive_compute_beta2(newx, betas)
            self._dc_generate_circuit(betas)
            super().__init__("Divide and Conquer Amplitude Encoding")

    def _dc_generate_circuit(self, betas):
        k = 0
        """
        for angles in betas:
            linear_angles = linear_angles + angles
            for angle in angles:
                self.qcircuit.append(iRBS(angle), [self.quantum_data[k], self.quantum_data[(k+1)%self.num_qubits]], [])
                k += 1
        """
        for beta in betas:
            k = self.num_qubits // len(beta)
            for i in range(len(beta)):
                #self.qcircuit.append(iRBS(beta[i]), [self.quantum_data[k*i], self.quantum_data[k*i + k // 2]], [])
                self.qcircuit.unitary(RBS(beta[i]).rbs, [self.quantum_data[k*i], self.quantum_data[k*i + k // 2]], label="RBS")  
        
    
    def get_circuit(self):
        return self.qcircuit