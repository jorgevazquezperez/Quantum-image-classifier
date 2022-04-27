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
    
    def _beta_calc(self, input_vector, betas):
        d_half = len(input_vector)//2
        rigth_half = input_vector[d_half:]

        r = []
        for i in range(len(rigth_half)):
            r_elem = np.sqrt(input_vector[2*i]**2 + input_vector[2*i+1]**2)
            r.append(r_elem)

        for i in range(d_half-2, -1, -1):
            r_elem = np.sqrt(r[i+1]**2 + r[i]**2)
            r.insert(0, r_elem)

        for i in range(len(input_vector)-1):
            if i < d_half - 1:
                betas.append(np.arccos(r[2*i+1] / r[i]))
            else:
                if input_vector[i - d_half + 2] >= 0:
                    if r[i] != 0 :
                        betas.append(np.arccos(input_vector[(i - d_half + 1)*2] / r[i]))
                    else:
                        betas.append(1)
                else:
                    if r[i] != 0:
                        betas.append(2*np.pi - np.arccos(input_vector[(i - d_half + 1)*2] / r[i]))
                    else:
                        betas.append(1)
