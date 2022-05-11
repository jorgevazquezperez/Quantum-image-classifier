import numpy as np


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

    def _generate_circuit(self, input_vector: np.ndarray):
        """
        Function to generate the circuit with the info of the real data.

        Args:
            input_vector: Real data
        """
        raise NotImplementedError()
