from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from encoding.Encoding import Encoding

class Amplitude_Encoding(Encoding):
    def __init__(self, features):
        self.num_qubits = int(np.log2(len(features)))
        self.quantum_data = QuantumRegister(self.num_qubits)
        self.qcircuit = QuantumCircuit(self.quantum_data)
        newx = np.copy(features)
        betas1 = []
        betas2 = []
        self._recursive_compute_beta1(newx, betas1)
        self._recursive_compute_beta2(newx, betas2)
        print(betas1)
        print(betas2)

        #self._generate_circuit(betas)
        super().__init__("Amplitude Encoding")

    def _generate_circuit(self, betas):
        numberof_controls = 0  # number of controls
        control_bits = []
        for angles in betas:
            if numberof_controls == 0:
                self.qcircuit.ry(angles[0], self.quantum_data[self.num_qubits-1])
                numberof_controls += 1
                control_bits.append(self.quantum_data[self.num_qubits-1])
            else:
                for k, angle in enumerate(reversed(angles)):
                    self._index(k, self.qcircuit, control_bits, numberof_controls)

                    self.qcircuit.mcry(angle,
                                control_bits,
                                self.quantum_data[self.num_qubits - 1 - numberof_controls],
                                None,
                                mode='noancilla')

                    self._index(k, self.qcircuit, control_bits, numberof_controls)
                control_bits.append(self.quantum_data[self.num_qubits - 1 - numberof_controls])
                numberof_controls += 1

    def get_circuit(self):
        return self.qcircuit