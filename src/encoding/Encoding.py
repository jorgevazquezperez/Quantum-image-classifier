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
                if r[i] != 0:
                    betas.append(np.arccos(r[2*i+1] / r[i]))
                else:
                    betas.append(1)
            else:
                if input_vector[i - d_half + 3] >= 0:
                    if r[i] != 0 :
                        print(input_vector[(i - d_half + 1)*2] / r[i])
                        betas.append(np.arccos(input_vector[(i - d_half + 1)*2] / r[i]))
                    else:
                        betas.append(1)
                else:
                    if r[i] != 0:
                        betas.append(2*np.pi - np.arccos(input_vector[(i - d_half + 1)*2] / r[i]))
                    else:
                        betas.append(1)

    def get_angles(self, x):
        # convert to array
        x = np.array(x)
        shape = x.shape
        if len(shape) == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[1] == 1:
            x = x.T

        # get recursively the angles
        def angles(y, wire):
            d = y.shape[-1]
            if d == 2:
                thetas = np.arccos(y[:, 0] / np.linalg.norm(y, 2, 1))
                signs = (y[:, 1] > 0.).astype(int)
                thetas = signs * thetas + (1. - signs) * (2. * np.pi - thetas)
                thetas = np.expand_dims(thetas, 1)
                wires = [(wire, wire + 1)]
                return thetas, wires
            else:
                thetas = np.arccos(
                    np.linalg.norm(y[:, :d // 2], 2, 1, True) /
                    np.linalg.norm(y, 2, 1, True))
                thetas_l, wires_l = angles(y[:, :d // 2], wire)
                thetas_r, wires_r = angles(y[:, d // 2:], wire + d // 2)
                thetas = np.concatenate([thetas, thetas_l, thetas_r], axis=1)
                wires = [(wire, wire + d // 2)] + wires_l + wires_r
            return thetas, wires

        # result
        thetas, wires = angles(x, 0)

        # remove nan and one dims
        thetas = np.nan_to_num(thetas, nan=0)
        thetas = thetas.squeeze()

        return thetas
