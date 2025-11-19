"""Hubbard Dimer circuit implementation for both state-vector and quantum circuit approaches."""
import numpy as np
from .measurement import measurement_outcome, add_measurements
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

class HubbardSimulate:
    """Class to perform operations for the Hubbard Dimer circuit. 

       This version performs a state-vector implementation using dense matrices to 
       allow for debugging of the circuit implementation.
    """
    def __init__(self,U,t,d,trotter=True):
        """ Initialise the Hubbard Dimer circuit parameters.
            Args:
                U: Hubbard interaction strength
                t: Hopping parameter
                d: Dipole moment
                trotter: Control if H time evolution is trotterised (default=True)
        """
        self.U = U
        self.t = t
        self.d = d
        self.trotter = trotter
        self.reset()

    def initialise(self,tamp):
        """ Initialise the state vector as electronic ground state using tUPS ansatz result
            Args:
                tamp: (t1,t2) vector of amplitudes for tUPS ground state
        """
        k2 = SparsePauliOp.from_list([('IYX',1j), ('IXY', 1j)]).to_matrix().real
        k1 = SparsePauliOp.from_list([('IIY',-1j),('IYI',-1j)]).to_matrix().real
        self.vec = expm(tamp[1] * k2) @ self.vec
        self.vec = expm(tamp[0] * k1) @ self.vec
        return

    def reset(self):
        """ Reset the state vector to initial reference state"""
        self.vec = np.eye(8,M=1) + 0j
        return

    def create_photon(self):
        """ Create a photon in the circuit by applying an X gate on the third qubit """
        x0 = SparsePauliOp.from_list([('XII',1.0)]).to_matrix().real
        self.vec = x0 @ self.vec
        return
    
    def Hm(self,l):
        """ Generate Hamiltonian as a SparsePauliOp for the Hubbard dimer
            Args:
                l: value of lambda coupling
            Returns:
                SparsePauliOp representing the Hubbard dimer Hamiltonian
        """
        return SparsePauliOp.from_list([
            ("III",0.5*self.U + l*l*self.d*self.d),("IZZ",0.5*self.U + l*l*self.d*self.d),
            ('IIX',-self.t),('IXI',-self.t)
        ]).to_matrix().real
    
    def Hp(self,w):
        """ Photon Hamiltonian for the Hubbard dimer.
            Args:
                w: value of omega for this Hamiltonian
            Returns:
                SparsePauliOp representing the photon Hamiltonian
        """
        return SparsePauliOp.from_list([("III",0.5*w),("ZII",-0.5*w)]).to_matrix().real
  
    def Hint(self,w,l):
        """ Electron-photon interaction
            Args:
                w: value of omega for this Hamiltonian
                l: value of lambda coupling
            Returns:
                SparsePauliOp representing the electron-photon interaction
        """
        fac = -self.d * l * np.sqrt(0.5 * w)
        return SparsePauliOp.from_list([("XIZ",fac),("XZI",fac)]).to_matrix().real
    
    def Hfull(self,w,l):
        """ Full Hamiltonian for the photon-electron system
            Args:
                w: value of omega for this Hamiltonian
                l: value of lambda coupling
            Returns:
                SparsePauliOp representing the full Hamiltonian
        """
        return self.Hm(l) + self.Hp(w) + self.Hint(w,l)

    def time_evolve(self,dt,wk,lk):
        """ Perform a time evolution step in the Hubbard dimer circuit.
            Args:
                dt: time step for the evolution
                wk: value of omega at this step
                lk: value of lambda at this step
        """
        # Coupling strength times dipole moment
        ldk = lk * self.d        
        # One-body molecular Hamiltonian
        h1 = SparsePauliOp.from_list([('IXI',-self.t), ('IIX', -self.t)]).to_matrix().real
        h2 = SparsePauliOp.from_list([('IZZ', 0.5*self.U + ldk*ldk)]).to_matrix().real
        # Photon Hamiltonian
        h3 = SparsePauliOp.from_list([('ZII',-0.5*wk)]).to_matrix().real
        # Coupling Hamiltonian
        pe_fac = - ldk * np.sqrt(0.5*wk)
        h4 = SparsePauliOp.from_list([('XIZ', pe_fac), ('XZI', pe_fac)]).to_matrix().real
    
        if(self.trotter):
            self.vec = expm(-1j * dt * h1) @ self.vec
            self.vec = expm(-1j * dt * h2) @ self.vec
            self.vec = expm(-1j * dt * h3) @ self.vec
            self.vec = expm(-1j * dt * h4) @ self.vec
        else:
            self.vec = expm(-1j * dt * (h1 + h2 + h3 + h4)) @ self.vec  

class HubbardCircuit:
    """Class to perform operations for the Hubbard Dimer circuit 

       This implementation uses Qiskit circuit operations and is suitable for porting
       to a real quantum backend.
    """
    def __init__(self,U,t,d,repeat=1):
        """ Initialise the Hubbard Dimer circuit parameters.
            Args:
                U: Hubbard interaction strength
                t: Hopping parameter
                d: Dipole moment
                repeat: Number of circuit repeats
        """
        self.U = U
        self.t = t
        self.d = d
        self.repeat = repeat
        self.reset()

    def initialise(self,tamp):
        """ Initialise the circuit using a tUPS parameterisation with parameters tamp
            Args:
                tamp: tuple of (tsingle, tdouble) for the tUPS state preparation
        """
        if len(tamp) != 2:
            raise ValueError("tamp must be a tuple of (tsingle, tdouble)")
        for i in range(self.repeat):
            # Get qubit indices
            q0 = i*3+0
            q1 = i*3+1
            q2 = i*3+2
            # Apply the double excitation as Rxy Rxy
            self.circuit.s(q0)
            self.circuit.rxx(2*tamp[1],q0,q1)
            self.circuit.sdg(q0)
            self.circuit.s(q1)
            self.circuit.rxx(2*tamp[1],q0,q1)
            self.circuit.sdg(q1)
            # Apply single qubit operators
            self.circuit.ry(2*tamp[0],q0)
            self.circuit.ry(2*tamp[0],q1)
        return

    def reset(self):
        """ Reset the circuit to the initial state |ee,p> """
        self.circuit = QuantumCircuit(3*self.repeat)

    def create_photon(self):
        """ Create a photon in the circuit by applying an X gate on the third qubit """
        for i in range(self.repeat):
            self.circuit.x(i*3+2)
    
    def time_evolve(self,dt,wk,lk):
        """ Perform a time evolution step in the Hubbard dimer circuit.
            Args:
                dt: time step for the evolution
                wk: value of omega at this step
                lk: value of lambda at this step
        """
        # Coupling strength times dipole moment
        ldk = lk * self.d        
        # Coupling Hamiltonian
        pe_fac = ldk * np.sqrt(0.5*wk)
        for i in range(self.repeat):
            # Get qubit indices
            q0 = i*3+0
            q1 = i*3+1
            q2 = i*3+2
            # One-body molecular Hamiltonian
            self.circuit.rx(-2*self.t*dt,q0)
            self.circuit.rx(-2*self.t*dt,q1)
            # Two-body terms including dipole self interaction
            self.circuit.rzz(dt*(self.U+2*ldk*ldk),q0,q1)
            # Photon Hamiltonian
            self.circuit.rz(-wk*dt,q2)
            # photon-electron dipole coupling
            self.circuit.rzx(-2*pe_fac*dt,q0,q2)
            self.circuit.rzx(-2*pe_fac*dt,q1,q2)
    
    def postselect_energy(self, dev):
        """Compute the electronic energy following post-selection to states with 0 photon.

           This performs explicit measurements for the electronic Hamiltonian and only includes
           measurement outcomes with no photon. Therefore, it filters out states with the 
           incorrect photon number, where the excitation is incomplete.

           Args:
               dev: ibmq.Device() that interfaces with the Qiskit backend for sampling
                    [see ibmq/device.py]
        """
        # Circuit to measure in ZZZ basis
        qc_III = self.circuit.copy()
        qc_III.measure_all()
        # Circuit to measure in ZXX basis
        qc_IXX = self.circuit.copy()
        for i in range(self.repeat):
            qc_IXX.ry(-0.5*np.pi,3*i+0)
            qc_IXX.ry(-0.5*np.pi,3*i+1)
        qc_IXX.measure_all()

        # Run sampler to get results
        results = dev.run_sampler([qc_III,qc_IXX])
        res_III = results[0].data.meas
        res_IXX = results[1].data.meas

        # Containers for measurement outcomes
        meas_ZII = measurement_outcome()
        meas_IIX = measurement_outcome()
        meas_IXI = measurement_outcome()
        meas_IZZ = measurement_outcome()
        ps_meas_IIX = measurement_outcome()
        ps_meas_IXI = measurement_outcome()
        ps_meas_IZZ = measurement_outcome()

        # Compute expectation values for IXI and IIX contributions (1-body)
        for k, ncount in res_IXX.get_counts().items():
            for i in range(self.repeat):
                # Extract the bits for this repeat
                ki = k[3*i:3*(i+1)]
                # Compute contribution
                IIX = 1 if ki[2]=='0' else -1
                IXI = 1 if ki[1]=='0' else -1
                # Total energy contribution
                meas_IIX.add_outcome(-self.t*IIX, ncount)
                meas_IXI.add_outcome(-self.t*IXI, ncount)
                # Post-selection measurement
                if(ki[0] == '0'):
                    ps_meas_IIX.add_outcome(-self.t*IIX, ncount)
                    ps_meas_IXI.add_outcome(-self.t*IIX, ncount)
                # Measure photon number
                pvalue = 1 if (ki[0] == '1') else 0
                meas_ZII.add_outcome(pvalue, ncount)

        # Compute expectation values for IZZ energy contribution (2-body)
        for k, ncount in res_III.get_counts().items():
            for i in range(self.repeat):
                # Extract the bits for this repeat
                ki = k[3*i:3*(i+1)]
                # Compute contribution
                IZZ = np.prod([1 if x == '0' else -1 for x in ki[1:]]) 
                # Total energy contribution
                meas_IZZ.add_outcome(0.5*self.U*IZZ, ncount)
                # Post-selection measurement
                if(ki[0] == '0'): ps_meas_IZZ.add_outcome(0.5*self.U*IZZ, ncount)
                # Measure photon number
                pvalue = 1 if (ki[0] == '1') else 0
                meas_ZII.add_outcome(pvalue, ncount)

        # Total energy
        Et = add_measurements([meas_IIX, meas_IXI, meas_IZZ], 0.5*self.U)
        # Total post-selection energy
        E  = add_measurements([ps_meas_IIX, ps_meas_IXI, ps_meas_IZZ], 0.5*self.U)
        # Get total photon number
        pn = (meas_ZII.value()[0], meas_ZII.value()[1])
        return Et, E, pn 
