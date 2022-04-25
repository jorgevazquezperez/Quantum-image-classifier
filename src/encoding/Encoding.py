import numpy as np

class Encoding:
    qcircuit = None
    quantum_data = None
    classical_data = None
    num_qubits = None
    output_qubits = []
    name = None

    def __init__(self, name: str):
        self.name = name
    
    def print_circuit(self):
        print(self.qcircuit)

    def _recursive_compute_beta1(self, input_vector, betas):
        if len(input_vector) > 1:
            new_x = []
            beta = []
            for k in range(0, len(input_vector), 2):
                norm = np.sqrt(input_vector[k] ** 2 + input_vector[k + 1] ** 2)
                new_x.append(norm)
                if norm == 0:
                    beta.append(0)
                else:
                    if input_vector[k] < 0:
                        beta.append(2 * np.pi - 2 * np.arcsin(input_vector[k + 1] / norm)) ## testing
                    else:
                        beta.append(2 * np.arcsin(input_vector[k + 1] / norm))
            self._recursive_compute_beta1(new_x, betas)
            betas.append(beta)
    
    def _recursive_compute_beta2(self, input_vector, betas):
        if len(input_vector) > 1:
            new_x = []
            beta = []
            for k in range(0, len(input_vector), 2):
                norm = np.sqrt(input_vector[k] ** 2 + input_vector[k + 1] ** 2)
                new_x.append(norm)
                if norm == 0:
                    beta.append(0)
                else:
                    if input_vector[k] < 0:
                        beta.append(2 * np.pi - np.arccos(input_vector[k + 1] / norm)) ## testing
                    else:
                        beta.append(np.arccos(input_vector[k + 1] / norm))
            self._recursive_compute_beta2(new_x, betas)
            betas.append(beta)

    def _index(self, k, circuit, control_qubits, numberof_controls):
        binary_index = '{:0{}b}'.format(k, numberof_controls)
        for j, qbit in enumerate(control_qubits):
            if binary_index[j] == '1':
                circuit.x(qbit)
