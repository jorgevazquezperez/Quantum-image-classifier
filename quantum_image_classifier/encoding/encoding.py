import numpy as np

from qiskit import QuantumCircuit

class Encoding:
    """
    Base class for any encoding of classical data into quantum data.
    """
    def __init__(self, name: str) -> None:
        """
        Args:
            name: Name of the encoding.
        """
        self.name = name

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @circuit.setter
    def circuit(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit


    def _generate_circuit(self, input_vector: np.ndarray):
        """
        Function to generate the circuit with the info of the real data.

        Args:
            input_vector: Real data
        """
        raise NotImplementedError()
