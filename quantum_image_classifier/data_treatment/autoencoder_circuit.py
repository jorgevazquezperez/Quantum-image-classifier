import pennylane as qml

class AutoencoderCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, n_trash, params):
        # --- Circuit definition ---
        self.n_qubits = n_qubits
        self.params = params
        self.n_trash = n_trash

        dev = qml.device("default.qubit", wires=n_qubits)
        
            

        
    