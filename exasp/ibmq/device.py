""" Module for managing IBM Quantum backends and running quantum circuits """
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler, QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
import numpy as np

class Device:
    """ Class to manage IBM Quantum backends and run circuits """
    def __init__(self,backend,noiseless,default_shots=1024,optimization_level=3,hardware=False,seed=None):
        """Initialise the device with backend and parameters.
        
            Args:
                backend:            Name of the IBM Quantum backend to use
                noiseless:          Boolean indicating whether to use noiseless simulation
                default_shots:      Number of shots for sampling [default=1024]
                optimization_level: Optimization level for transpiling circuits [default=3]
                hardware:           Boolean indicating whether to run on actual hardware [default=False]
                seed:               Random seed for sampling [default=None]
        """
        self.qbackend = QiskitRuntimeService(channel='ibm_quantum_platform').backend(backend)
        if(hardware):
            self.backend = self.qbackend
        else:
            self.backend = AerSimulator(method='statevector') if noiseless else AerSimulator.from_backend(self.qbackend)

        # Pass manager for transpiling circuits
        self.pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=self.backend)

        # Estimator for calculating expectation values
        self.estimator = (StatevectorEstimator(default_precision=1/np.sqrt(default_shots),seed=seed) if noiseless else 
                          Estimator(self.backend, options={"default_shots": default_shots}))

        # Sampler for evaluating circuit outcomes
        self.sampler   = (StatevectorSampler(default_shots=default_shots,seed=seed) if noiseless else 
                          Sampler(self.backend, options={"default_shots": default_shots}) )

    def run_circuit(self,circuit,operator):
        """ Run a circuit on the backend to measure expectation values of operator
            Args:
                circuit:  QuantumCircuit to run
                operator: SparsePauliOp or other operator to measure
            Returns:
                Result of the job execution, which contains expectation values and standard deviations
        """
        # Optimize the circuit using the passmanager 
        isa_circuit = self.pm.run(circuit)
        # Run the circuit with the Hamiltonian
        job = self.estimator.run([(isa_circuit, operator.apply_layout(isa_circuit.layout))])
        return job.result()

    def run_sampler(self,circuit):
        """ Run a circuit on the backend using the sampler primitive
            Args:
                circuit: QuantumCircuit to run
            Returns:
                Result of the job execution, which contains counts
        """

        circuit_list = [circuit] if (type(circuit) == QuantumCircuit) else circuit
        # Optimize the circuit using the passmanager 
        isa_circuit = map(self.pm.run, circuit_list)
        # Run the circuit with the sampler
        job = self.sampler.run(isa_circuit)
        isa_circuit = map(self.pm.run, circuit_list)
        return job.result()

    @property
    def num_qubits(self):
        """ Return the number of qubits in the simulation """
        return self.backend.num_qubits
