from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
import qiskit

import torch
from torch.autograd import Variable

import pennylane as qml

from ..encoding import PhaseEncoding
from .autoencoder_circuit import AutoencoderCircuit

class QuantumAE2():
    def __init__(self, num_qubits: int, n_components: int):
        self.num_qubits = num_qubits
        self.n_trash = num_qubits - n_components

        simulator = qiskit.Aer.get_backend('aer_simulator')
        N_PARAMETERS = num_qubits*4 + \
            num_qubits*(num_qubits - 1)

        self.params = np.random.uniform(0, np.pi, (N_PARAMETERS,))

    def fit(self, dataset: np.ndarray = None) -> None:

        self.states = []
        for point in dataset:
            self.states.append(PhaseEncoding(point))

        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev, interface="torch")
        def circuit(params):

            weight_counter = 0

            for i in range(self.num_qubits):
                qml.RX(params[weight_counter], i)
                qml.RZ(params[weight_counter+1], i)
                weight_counter += 2

            iterator = list(range(self.num_qubits-1, -1, -1))
            for i in iterator:
                iterator_j = iterator.copy()
                iterator_j.remove(i)
                for j in iterator_j:
                    qml.CRX(self.params[weight_counter], [i, j])
                    weight_counter += 1

            for i in range(self.num_qubits):
                qml.RX(self.params[weight_counter], i)
                qml.RZ(self.params[weight_counter+1], i)
                weight_counter += 2
            
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        fig, ax = qml.draw_mpl(circuit, decimals=2)(self.params)
        plt.show()


        

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        

