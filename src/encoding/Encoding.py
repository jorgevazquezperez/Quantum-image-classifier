from abc import abstractmethod
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
    
    @abstractmethod
    def _recursive_compute_beta2(input_vector, betas):
        d_half = len(input_vector)//2
        rigth_half = input_vector[d_half:]

        r = []
        for i in range(len(rigth_half)):
            r_elem = np.sqrt(input_vector[2*i]**2 + input_vector[2*i+1]**2)
            r.append(r_elem)

        for i in range(d_half-2, -1, -1):
            r_elem = np.sqrt(r[i+1]**2 + r[i]**2)
            r.insert(0, r_elem)
        
        betas = []
        for i in range(len(input_vector)-1):
            if i < d_half - 1:
                print(i)
                betas.append(np.arccos(r[2*i+1] / r[i]))
            else:
                if input_vector[i - d_half + 2] >= 0:
                    if r[i] != 0:
                        betas.append(np.arccos(input_vector[i - d_half + 1] / r[i]))
                    else:
                        betas.append(0)
                else:
                    if r[i] != 0:
                        betas.append(2*np.pi - np.arccos(input_vector[i - d_half + 1] / r[i]))
                    else:
                        betas.append(0)
        print(betas)

    def _index(self, k, circuit, control_qubits, numberof_controls):
        binary_index = '{:0{}b}'.format(k, numberof_controls)
        for j, qbit in enumerate(control_qubits):
            if binary_index[j] == '1':
                circuit.x(qbit)
