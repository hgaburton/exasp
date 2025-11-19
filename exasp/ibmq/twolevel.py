"""" Module to implement a two-qubit two-level Hamiltonian circuit"""
import numpy as np
from .measurement import measurement_outcome, add_measurements
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
    
class TwoLevelCircuit:
    """ Class to implement a two-qubit two-level Hamiltonian
    """ 
    def __init__(self,eps,t,mu,repeat=1):
        """ Initialise two-level Hamiltonian
        
            Args:
                eps   : Energy difference between the two levels
                t     : Coupling between the two levels
                mu    : Transition dipole moment
                repeat: Number of times to repeat the circuit
        """
        # Number of times to repeat the circuit
        self.repeat = repeat
        # Store parameters
        self.eps = eps
        self.t = t
        # Average energy of states
        self.eZ  = - eps
        self.tX  = t
        # Dipole self-energy scaling
        self.ddI = mu*mu
        # Transition dipole coupling
        self.mu  = mu
        self.reset()

    def reset(self):
        """ Reset the circuit to the initial state |ep> """
        self.circuit = QuantumCircuit(2*self.repeat)
        th = np.arctan2(-self.t,self.eps)
        # Loop over circuit repeats
        for i in range(self.repeat):
            self.circuit.ry(th,2*i)

    def create_photon(self):
        """ Create a photon in the circuit by applying an X gate on the third qubit """
        # Loop over circuit repeats
        for i in range(self.repeat):
            self.circuit.x(2*i+1)
    
    def time_evolve(self,dt,wk,lk):
        """ Perform a time evolution step  using Trotter-Suzuki decomposition.

            Args:
                dt: time step for the evolution
                wk: value of omega at this step
                lk: value of lambda at this step
        """
        # Coupling Hamiltonian
        fac = - lk * self.mu * np.sqrt(0.5*wk)    
        for i in range(self.repeat):
            # Evolve electronic part
            self.circuit.rz(2*dt*self.eZ,2*i)
            self.circuit.rx(2*dt*self.tX,2*i)
            # Photon Hamiltonian
            self.circuit.rz(-dt*wk,2*i+1)
            # photon-electron dipole coupling
            self.circuit.rxx(2*fac*dt,2*i,2*i+1)

    def postselect_energy(self, dev):
        """ Compute the energy, excited-state fidelity, and photon number 
            before and after post-selecting for the zero photon state.

            Args:
                dev: Device object to run the circuit on
            Returns:
                Et   : Total energy before post-selection
                E    : Total energy after post-selection
                pnum : Photon number after post-selection
                wt   : Excited-state fidelity before post-selection
                w    : Excited-state fidelity after post-selection"""
        # make a copy of the circuit
        qc_ZI = self.circuit.copy()
        qc_ZI.measure_all()
        qc_XI = self.circuit.copy()
        for i in range(self.repeat):
            qc_XI.ry(-0.5*np.pi,2*i)
        qc_XI.measure_all()
        results = dev.run_sampler([qc_ZI,qc_XI])
        res_ZI = results[0].data.meas
        res_XI = results[1].data.meas

        # Containers for measurement outcomes
        meas_IZ = measurement_outcome()
        meas_ZI = measurement_outcome()
        meas_XI = measurement_outcome()
        ps_meas_ZI = measurement_outcome()
        ps_meas_XI = measurement_outcome()

        # Loop over outcomes
        for k, ncount in res_ZI.get_counts().items():
            for i in range(self.repeat):
                # Extract the bits for this repeat
                ki = k[2*i:2*(i+1)]
                ZI = 1 if (ki[1] == '0') else -1
                # Total energy contriubtion
                meas_ZI.add_outcome(self.eZ*ZI, ncount)
                # Post-selection measurement
                if(ki[0] == '0'): ps_meas_ZI.add_outcome(self.eZ*ZI, ncount)
                # Measure photon number
                pvalue = 1 if (ki[0] == '1') else 0
                meas_IZ.add_outcome(pvalue, ncount)

        # Also need to measure in X basis to get coupling term
        for k, ncount in res_XI.get_counts().items():
            for i in range(self.repeat):
                # Extract the bits for this repeat
                ki = k[2*i:2*(i+1)] 
                XI = np.prod([1 if x == '0' else -1 for x in ki[1:]])
                # Total energy contributions
                meas_XI.add_outcome(self.tX*XI, ncount)
                # Post-selection measurement
                if(ki[0] == '0'): ps_meas_XI.add_outcome(self.tX*XI, ncount)
                # Measure photon number
                pvalue= 1 if (ki[0] == '1') else 0
                meas_IZ.add_outcome(pvalue, ncount)

        # Total energy
        Et = add_measurements([meas_ZI, meas_XI])
        # Total post-selection energy
        E  = add_measurements([ps_meas_ZI, ps_meas_XI])
        # Get total photon number
        pn = (meas_IZ.value()[0], meas_IZ.value()[1])
        return Et, E, pn
