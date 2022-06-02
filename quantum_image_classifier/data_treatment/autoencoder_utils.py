
import numpy as np
import qiskit

import torch
from torch.autograd import Function
import torch.nn as nn
from .autoencoder_circuit import AutoencoderCircuit

class Net(nn.Module):
    def __init__(self, circuit: AutoencoderCircuit):
        super(Net, self).__init__()
        self.thetas = nn.ParameterList()
        for _ in range(len(circuit.params)):
            self.thetas.append(nn.Parameter(torch.tensor([np.random.uniform(0, np.pi)])))
        self.hybrid = Hybrid(circuit, self.thetas, np.pi / 2)

    def forward(self, x):
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)            

class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, thetas, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0], thetas)
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, thetas, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, thetas, expectation_z = ctx.saved_tensors

        thetas_list = np.array(thetas.tolist())
        
        shift_right = thetas_list + np.ones(thetas_list.shape) * ctx.shift
        shift_left = thetas_list - np.ones(thetas_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(thetas_list)):
            expectation_right = ctx.quantum_circuit.run(input[0], shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(input[0], shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, circuit, thetas, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = circuit
        self.thetas = thetas
        self.shift = shift
        
    def forward(self, input):
        result = HybridFunction.apply(input, self.thetas, self.quantum_circuit, self.shift)
        return result