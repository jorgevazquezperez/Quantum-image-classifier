import numpy as np
import qiskit.quantum_info as qi

class RBS:
    """
    Implementation of a RBS gate as an operator.
    """
    def __init__(self, theta: float) -> None:
        """Create new RBS operation."""
        a = np.cos(theta)
        b = np.sin(theta)

        # Matrix definition of the gate
        self.rbs = qi.Operator([ [1, 0,  0, 0],
                                 [0, a,  b, 0],
                                 [0, -b, a, 0],
                                 [0, 0,  0, 1]])
        
    @property
    def rbs(self):
        return self._rbs
    
    @rbs.setter
    def rbs(self, value):
        self._rbs = value