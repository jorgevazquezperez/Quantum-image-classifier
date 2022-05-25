import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from .encoding import Encoding
from ..gates import RBS


class UnaryLoader(Encoding):
    """
    Encoding of classical data into quantum info based on a unary basis, i.e., |10..0>, 
    |01..0>, ..., |00..1>.

    With this encoding we obtain d qubits, d-1 parametrized two qubits gates (plus a cnot
    in the first qubit) and a log d depth of the circuit.

    With this codification for two data points, if we invert one of them, connect it to the
    other and measure the probability of reading |1> in the first qubit, we obtain the inner 
    product of the two data points normalized.
    """

    def __init__(self, input_vector: np.ndarray, inverse: bool = False) -> None:
        """

        Args:
            input_vector: Vector of k dimensions
            inverse: Boolean that determines whether we should invert the circuit or not
                    (in order to obtain the inner product)
        """
        self.num_qubits = int(len(input_vector))
        self.quantum_data = QuantumRegister(self.num_qubits)
        self.circuit = QuantumCircuit(self.quantum_data)
        newx = np.copy(input_vector)

        betas = []
        self._theta_calc(newx, betas)
        self._generate_circuit(betas)

        super().__init__("Unary encoding")

    def _generate_circuit(self, betas: np.ndarray) -> None:
        """
        Function to generate the circuit with the info of the real data.

        Args:
            betas: Vector of k-1 dimensions with the angles for the gates 
        """
        logarithm = int(np.log2(self.num_qubits))

        # Generation a cicuit with a depth of log d, being d the number of qubits, using the RBS gate 
        for i in range(logarithm):
            for k in range(2**i):
                w = self.num_qubits // 2**i
                self.circuit.unitary(RBS(betas[2**i + k - 1]).rbs, [self.quantum_data[k*w],
                                                                    self.quantum_data[k*w + w // 2]], label="RBS({})".format(np.around(betas[2**i + k - 1], 3)))

    def _theta_calc(self, input_vector: np.ndarray, betas: np.ndarray) -> None:
        """
        Function to calculate the angles for the gates of the circuit. We follow the procedure of the paper [1] in
        the section Methods.


        ** References: **
        [1] Sonika Johri, Shantanu Debnath, Avinash Mocherla, Alexandros SINGK, Anupam Prakash, Jungsang Kim 
        and Iordanis Kerenidis et al.,
        `Nearest centroid classification on a trapped ion quantum computer 
        <https://www.nature.com/articles/s41534-021-00456-5>`
        Args:
            input_vector: Vector of k dimensions to be encoded.
        """
        d_half = len(input_vector)//2
        rigth_half = input_vector[d_half:]

        # r = (r_1, ..., r_(d-1))
        r = []

        # Calculations for r_(d/2), ..., r_(d-1)
        for i in range(len(rigth_half)):
            r_elem = np.sqrt(input_vector[2*i]**2 + input_vector[2*i+1]**2)
            r.append(r_elem)

        # Calculations for r_1, ..., r_(d/2)
        for i in range(d_half-2, -1, -1):
            r_elem = np.sqrt(r[i+1]**2 + r[i]**2)
            r.insert(0, r_elem)

        # Calculation of the θ's
        for i in range(len(input_vector)-1):

            # Calculations for θ_1, ..., θ_(d/2)
            if i < d_half - 1:
                if r[i] != 0:
                    betas.append(np.arccos(r[2*i+1] / r[i]))
                else:
                    betas.append(1)

            # Claculations for θ_(d/2), ..., θ_d
            else:
                if input_vector[(i - d_half + 1)*2 + 1] >= 0:
                    if r[i] != 0:
                        betas.append(
                            np.arccos(input_vector[(i - d_half + 1)*2] / r[i]))
                    else:
                        betas.append(1)
                else:
                    if r[i] != 0:
                        betas.append(
                            2*np.pi - np.arccos(input_vector[(i - d_half + 1)*2] / r[i]))
                    else:
                        betas.append(1)
