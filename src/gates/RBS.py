from msilib.schema import Property
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.circuit.library import CRYGate, CRZGate, CXGate
from qiskit.circuit.parameterexpression import ParameterValueType
from typing import Optional


class RBS:

    def __init__(self, theta):
        """Create new RBS operation."""
        a = np.cos(theta)
        b = np.sin(theta)
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