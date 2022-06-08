import os
from collections import Counter
import warnings
from uuid import uuid4
import numpy as np

from qiskit.providers import BackendV1, JobV1, Options
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.qobj import QobjExperimentHeader

#QLM imports
from qat.interop.qiskit.converters import qiskit_to_qlm
from qat.interop.qiskit.converters import job_to_qiskit_circuit
from qat.comm.shared.ttypes import Job
from qat.comm.shared.ttypes import Result as QlmRes
from qat.core.qpu.qpu import QPUHandler, get_registers
from qat.core.bits import DefaultRegister
from qat.core import Batch
from qat.core.wrappers.result import aggregate_data
from qat.core.wrappers.result import Result as WResult, BatchResult, Sample
from qat.qlmaas.result import AsyncResult


def to_string(state, nbqbits):
    """
    Converts a state into a string.

    Args:
        state: Int representing the quantum state written in decimal
        nbqbits: Number of qubits of the quantum state

    Returns:
        String of the quantum state in binary form
    """
    state_str = bin(state)[2:]
    state_str = "0" * (nbqbits - len(state_str)) + state_str
    return state_str


def generate_qlm_result(qiskit_result):
    """
    Generates a QLM Result from a Qiskit result.

    Args:
        qiskit_result: The qiskit Result object to convert

    Returns:
        A QLM Result object built from the data in qiskit_result
    """

    nbshots = qiskit_result.results[0].shots
    try:
        counts = [result.data.counts for result in qiskit_result.results]
    except AttributeError:
        print("No measures, so the result is empty")
        return QlmRes(raw_data=[])
    counts = [{int(k, 16): v for k, v in count.items()} for count in counts]
    ret = QlmRes(raw_data=[])
    for state, freq in counts[0].items():
        if not isinstance(state, int):
            print(f"State is {type(state)}")
        ret.raw_data.append(
            Sample(state=state,
                   probability=freq / nbshots,
                   err=np.sqrt(
                       freq / nbshots * (1. - freq / nbshots) / (nbshots - 1))
                   if nbshots > 1 else None)
        )
    return ret


def generate_qlm_list_results(qiskit_result):
    """
    Generates a QLM Result from a qiskit result.

    Args:
        qiskit_result: The qiskit.Result object to convert

    Returns:
        A QLM Result object built from the data in qiskit_result
    """

    nbshots = qiskit_result.results[0].shots
    try:
        counts = [result.data.counts for result in qiskit_result.results]
    except AttributeError:
        print("No measures, so the result is empty")
        return QlmRes(raw_data=[])
    counts = [{int(k, 16): v for k, v in count.items()} for count in counts]
    ret_list = []
    for count in counts:
        ret = QlmRes(raw_data=[])
        for state, freq in count.items():
            if not isinstance(state, int):
                print("State is {type(state)}")
            ret.raw_data.append(
                Sample(state=state,
                       probability=freq / nbshots,
                       err=np.sqrt(
                           freq / nbshots * (1. - freq / nbshots) / (nbshots - 1))
                       if nbshots > 1 else None)
            )
        ret_list.append(ret)
    return ret_list

def _generate_experiment_result(qlm_result, metadata):
    """
    Generates a Qiskit experiment result.

    Args:
        qlm_result: qat.core.wrappers.Result object which data is aggregated
        metadata: metadata of the circuit of the experiment

    Returns:
        An ExperimentResult structure.
    """
    samples = [hex(s.state.state) for s in qlm_result.raw_data]
    #samples = [hex(int(bin(s.state.state)[-1:1:-1],2)) for s in qlm_result.raw_data]
    #print(samples)
    counts = dict(Counter(samples))
    #print(counts)
    data = ExperimentResultData.from_dict({"counts": counts})
    return ExperimentResult(
        shots=len(qlm_result.raw_data),
        success=True,
        data=data,
        header=QobjExperimentHeader(memory_slots=metadata.num_qubits,metadata=metadata.metadata,name=metadata.name),
    )


def _qlm_to_qiskit_result(
        backend_name,
        backend_version,
        qobj_id,
        job_id,
        success,
        qlm_results,
        metadata,
        qobj_header):
    """
    Tranform a QLM result into a Qiskit result structure.

    Args:
        backend_name:
        backend_version:
        qobj_id:
        job_id:
        success:
        qlm_results: List of qat.core.wrappers.Result objects
        metadata: List of the circuit's metadata
        qobj_header: user input that will be in the Result

    Returns:
        A qiskit Result structure.
    """
    return Result(
        backend_name=backend_name,
        backend_version=backend_version,
        qobj_id=qobj_id,
        job_id=job_id,
        success=success,
        results=[
            _generate_experiment_result(result, md)
            for result, md in zip(qlm_results, metadata)
        ],
        header=qobj_header,
    )


class QLMJob(JobV1):
    """
    QLM Job implement the required JobV1 interface of Qiskit with a
    small twist: everything is computed synchronously (meaning that the
    job is stored at submit and computed at result).
    """
    def __init__(self, *args, **kwargs):
        self._results = None
        super().__init__(*args, **kwargs)

    def set_results(self, qlm_result, qobj_id, metadata, qobj_header):
        """
        Sets the results of the Job.

        Args:
            qlm_result: :class:`~qat.core.wrappers.result.Result` object
            qobj_id: Identifier of the initial Qobj structure
            metadata: List of the circuit's metadata
            qobj_header: user input that will be in the Result
        """
        self._results = _qlm_to_qiskit_result(
            self._backend._configuration.backend_name,
            self._backend._configuration.backend_version,
            qobj_id,
            self._job_id,
            True,
            qlm_result,
            metadata,
            qobj_header,
        )

    def status(self):
        pass

    def cancel(self):
        pass

    def submit(self):
        pass

    def result(self):
        return self._results

_QLM_GATE_NAMES = [
    "i",
    "id",
    "iden",
    "u",
    "u0",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cx",
    "cy",
    "cz",
    "ch",
    "crz",
    "cu1",
    "cu3",
    "swap",
    "ccx",
    "cswap",
    "r",
]

_QLM_GATES = [{"name": "FOO", "parameters": [], "qasm_def": "BAR"}]

_QLM_PARAMS = {
    "backend_name": "QiskitConnector",  # Name of the back end
    "backend_version": "0.0.1",  # Version of the back end
    "n_qubits": 100,  # Nb qbits
    "basis_gates": _QLM_GATE_NAMES,  # We accept all the gates of Qiskit
    "gates": _QLM_GATES,  # They don't even use it for their simulators, so...
    "local": True,  # Its a local backend
    "simulator": True,  # Its a simulator. Is it though?
    "conditional": True,  # We support conditionals
    "open_pulse": False,  # We do not support open Pulse
    "memory": False,  # We do not support Memory (wth?)
    "max_shots": 409600,
    "coupling_map": None,
}  # Max shots is 4096 (required :/)


class NoQpuAttached(Exception):
    """
    Exception raised in QPUToBackend.run() when there is not qpu attached to it
    """


_QLM_BACKEND = BackendConfiguration.from_dict(_QLM_PARAMS)


class CESGAQPUToBackend(BackendV1):
    """
    Basic connector implementing a Qiskit Backend, plugable on a QLM QPU.

    Parameters:
        qpu: :class:`~qat.core.qpu.QPUHandler` object
        configuration: BackendConfiguration object, leave default value for
                standard uses
        provider: Provider responsible for this backend
    """

    @classmethod
    def _default_options(cls):
        return Options(shots=0, qobj_id=str(uuid4()), qobj_header={},
                       parameter_binds={})

    def __init__(self, qpu=None, configuration=_QLM_BACKEND, provider=None):
        """
        Args:
            qpu: :class:`~qat.core.qpu.QPUHandler` object
            configuration: BackendConfiguration object, leave default value for
                    standard uses
            provider: Provider responsible for this backend
        """
        super().__init__(configuration, provider)
        self.id_counter = 0
        self._qpu = qpu

    def set_qpu(self, qpu):
        """
        Sets the QLM QPU that this backend is supposed to use.

        Args:
            qpu: QLM QPU object
        """
        self._qpu = qpu



    def run(self, run_input, **kwargs):
        """ Convert all the circuits inside qobj into a Batch of
            QLM jobs before sending them into a QLM qpu.

        Args:
            run_input (list or QuantumCircuit or Schedule)
            kwargs: any option that can replace the default ones

        Returns:
            Returns a :class:`~qat.interop.qiskit.QLMJob` object containing
            the results of the QLM qpu execution after being converted into
            Qiskit results
        """
        if self._qpu is None:
            raise NoQpuAttached("No qpu attached to the QLM connector.")
        circuits = run_input if isinstance(run_input, list) else [run_input]
        #circuits_metadata = [circuit.metadata for circuit in circuits]

        for kwarg in kwargs:
            if not hasattr(self.options, kwarg):
                raise ValueError(f"'{kwarg}' parameter not supported")
        nbshots = kwargs.get('shots', self.options.shots)
        qobj_id = kwargs.get('qobj_id', self.options.qobj_id)
        qobj_header = kwargs.get('qobj_header', self.options.qobj_header)
        # TODO: use parameter_binds for constructing the job
        # this involves not only iterating on the experiments
        # but also iterating on the parameter sets so provided

        qlm_task = Batch(jobs=[])
        for circuit in circuits:
            qlm_circuit = qiskit_to_qlm(circuit.reverse_bits())
            job = qlm_circuit.to_job(aggregate_data=False)
            job.nbshots = nbshots
            job.qubits = list(range(0, qlm_circuit.nbqbits))
            qlm_task.jobs.append(job)

        #Submitting and return results (AGT)
        aresults = self._qpu.submit(qlm_task)
        
        if type(aresults) == AsyncResult:
            #print(type(aresults))
            results=aresults.join()
            
        else:
            results=aresults
        
        #print(results)
        for res in results:
            if res.raw_data != None:
                for sample in res.raw_data:
                    sample.intermediate_measures = None
            #res = aggregate_data(res)
            #print(res.lsb_first)
            #res.lsb_first=True
        #print(results)
        # Creating a job that will contain the results
        job = QLMJob(self, str(self.id_counter))
        self.id_counter += 1
        #print(circuits_metadata)
        #job.set_results(results, qobj_id, circuits, circuits_metadata, qobj_header)
        job.set_results(results, qobj_id, circuits, qobj_header)
        return job