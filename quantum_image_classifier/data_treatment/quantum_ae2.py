import numpy as np

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble, transpile
from qiskit.algorithms.optimizers import ADAM

from scipy.optimize import basinhopping

from ..encoding import PhaseEncoding


class QuantumAE2():
    def __init__(self, num_qubits: int, n_components: int):
        self.num_qubits = num_qubits
        self.n_trash = num_qubits - n_components
        self._build_autoencoder()

    def _build_autoencoder(self):
        N_PARAMETERS = self.num_qubits*4 + \
            self.num_qubits*(self.num_qubits - 1)
        self.params = ParameterVector('θ', N_PARAMETERS)

        self.cregisters = ClassicalRegister(self.n_trash, "c")
        self.qregisters = QuantumRegister(self.num_qubits, "q")
        self.circuit = QuantumCircuit(self.cregisters, self.qregisters)
        self.circuit.barrier()

        weight_counter = 0

        for i in range(self.num_qubits):
            self.circuit.rx(self.params[weight_counter], i)
            self.circuit.rz(self.params[weight_counter+1], i)
            weight_counter += 2

        iterator = list(range(self.num_qubits-1, -1, -1))
        for i in iterator:
            iterator_j = iterator.copy()
            iterator_j.remove(i)
            for j in iterator_j:
                self.circuit.crx(self.params[weight_counter], i, j)
                weight_counter += 1

        for i in range(self.num_qubits):
            self.circuit.rx(self.params[weight_counter], i)
            self.circuit.rz(self.params[weight_counter+1], i)
            weight_counter += 2

        for i in range(self.n_trash):
            self.circuit.measure(self.qregisters[i], self.cregisters[i])
        

    def fit(self, dataset: np.ndarray = None) -> None:

        opt = ADAM(lr=0.1, maxiter=200)

        pt = np.random.uniform(low=0,
                               high=4*np.pi,
                               size=(len(self.params),))

        self.states = []
        for point in dataset:
            self.states.append(PhaseEncoding(point))
        
        def loss(params):
            expectation = 0
            dict = {}
            for param, value in zip(self.params, params):
                dict[param] = value
            for state in self.states:
                circ_aux = QuantumCircuit(self.qregisters, self.cregisters)
                circ_aux.compose(
                    state.circuit, qubits=self.qregisters, inplace=True)
                circ_aux.compose(self.circuit,
                                 qubits=self.qregisters, clbits=self.cregisters, inplace=True)

                simulator = Aer.get_backend('aer_simulator')
                t_qc = transpile(circ_aux,
                                 simulator)

                qobj = assemble(t_qc,
                                shots=1024,
                                parameter_binds=[dict])
                job = simulator.run(qobj)
                result = job.result().get_counts()

                counts = result["00"]
                expectation += 1 - counts / 1024

            return expectation / len(self.states)
        
        print(loss(pt))
        
        
        """
        # perform the basin hopping search
        result = basinhopping(loss, pt, stepsize=0.06, niter=20, disp=True)
        # summarize the result
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        # evaluate solution
        solution = result['x']
        evaluation = loss(solution)
        print('Solution: f(%s) = %.5f' % (solution, evaluation))"""

        result = opt.minimize(loss, pt)
        # summarize the result
        print('Total Evaluations: %d' % result.nfev)
        # evaluate solution
        solution = result.x
        evaluation = result.fun
        print('Solution: f(%s) = %.5f' % (solution, evaluation))
