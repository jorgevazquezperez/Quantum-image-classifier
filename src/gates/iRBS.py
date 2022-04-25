import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import CRYGate, CRZGate, CXGate
from qiskit.circuit.parameterexpression import ParameterValueType
from typing import Optional


class iRBS(Gate):
    r"""The SWAP gate.
    This is a symmetric and Clifford gate.
    **Circuit symbol:**
    .. parsed-literal::
        q_0: ─X─
              │
        q_1: ─X─
    **Matrix Representation:**
    .. math::
        SWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
    The gate is equivalent to a state swap and is a classical logic gate.
    .. math::
        |a, b\rangle \rightarrow |b, a\rangle
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new iRBS gate."""
        super().__init__("iRBS", 2, [theta])

    def _define(self):
        # pylint: disable=cyclic-import

        #                 ┌───┐┌──────────────┐                     ┌───┐┌──────────┐
        # q_0: ───────────┤ X ├┤   Rz(-π/2)   ├──■──────────────────┤ X ├┤ Rz(-π/2) ├
        #      ┌─────────┐└─┬─┘├──────────────┤┌─┴─┐┌──────────────┐└─┬─┘└──────────┘
        # q_1: ┤ Rz(π/2) ├──■──┤ Ry(π/2 - 2ϴ) ├┤ X ├┤ Ry(2ϴ - π/2) ├──■──────────────
        #      └─────────┘     └──────────────┘└───┘└──────────────┘

        q = QuantumRegister(2, "q")
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (CRZGate(np.pi / 2), [q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (CRZGate(-np.pi / 2), [q[0]], []),
            (CRYGate(np.pi / 2 - 2*theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (CRYGate(2*theta - np.pi / 2 ), [q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (CRZGate(-np.pi / 2), [q[0]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def get_iRBS(self):
        return self.definition