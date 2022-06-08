import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, CRZGate, CXGate
from qiskit.circuit.parameterexpression import ParameterValueType
from typing import Optional


class iRBS():
    """
    Implementation of a iRBS gate as a circuit.
    """
    def __init__(self, angle: float) -> None:
        """Create new RBS operation."""
        self.theta = Parameter("θ")

        q = QuantumRegister(2)
        self._irbs = QuantumCircuit(q, name="iRBS")
        self._irbs.rz(np.pi/2, q[1])
        self._irbs.cx(q[1], q[0])
        self._irbs.rz(-np.pi/2, q[0])
        self._irbs.ry(np.pi/2 - 2*self.theta, q[1])
        self._irbs.cx(q[0], q[1])
        self._irbs.ry(2*self.theta - np.pi/2, q[1])
        self._irbs.cx(q[1], q[0])
        self._irbs.rz(-np.pi/2, q[0])
        self._irbs = self._irbs.bind_parameters({self.theta: angle})


    @property
    def irbs(self):
        return self._irbs
    
    @irbs.setter
    def irbs(self, value):
        self._irbs = value