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
        k2 = SparsePauliOp.from_list([('IIYX',1j), ('IIXY', 1j)]).to_matrix().real
        k1 = SparsePauliOp.from_list([('IIIY',-1j),('IIYI',-1j)]).to_matrix().real
        self.vec = expm(tamp[1] * k2) @ self.vec
        self.vec = expm(tamp[0] * k1) @ self.vec
        return

    def reset(self):
        """ Reset the state vector to initial reference state"""
        self.vec = np.eye(16,M=1) + 0j
        return

    def create_photon(self, nphoton):
        """ Create n photons in the circuit by applying apppropiate X gates """
        # NOTE: had to switch the order of the bits to match the circuit
        # implementation, so 0b10 is first qubit and 0b01 is second qubit
        if nphoton < 0 or nphoton > 3:
            raise ValueError("nphoton must be between 0 and 3")
        first_char = 'X' if nphoton & 0b01 else 'I'
        second_char = 'X' if nphoton & 0b10 else 'I'
        x0 = SparsePauliOp.from_list([(first_char + second_char + 'II', 1.0)]).to_matrix().real
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
            ("IIII",0.5*self.U + l*l*self.d*self.d),("IIZZ",0.5*self.U + l*l*self.d*self.d),
            ('IIIX',-self.t),('IIXI',-self.t)
        ]).to_matrix().real
    
    def Hp(self,w):
        """ Photon Hamiltonian for the Hubbard dimer.
            Args:
                w: value of omega for this Hamiltonian
            Returns:
                SparsePauliOp representing the photon Hamiltonian
        """
        return SparsePauliOp.from_list([("IIII",1.5*w),("IZII",-w),("ZIII",-0.5*w)]).to_matrix().real
  
    def Hint(self,w,l):
        """ Electron-photon interaction
            Args:
                w: value of omega for this Hamiltonian
                l: value of lambda coupling
            Returns:
                SparsePauliOp representing the electron-photon interaction
        """
        fac = -self.d * l * np.sqrt(0.5 * w)
        return SparsePauliOp.from_list([
            ("XIZI",fac),("XIIZ",fac),("XXZI",0.5*fac),("XXIZ",0.5*fac),
            ("YYZI",0.5*fac),("YYIZ",0.5*fac)
        ]).to_matrix().real
    
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
        h1 = SparsePauliOp.from_list([('IIXI',-self.t), ('IIIX', -self.t)]).to_matrix().real
        # Two-body molecular Hamiltonian
        h2 = SparsePauliOp.from_list([('IIZZ', 0.5*self.U + ldk*ldk)]).to_matrix().real
        # Photon Hamiltonian
        h3 = SparsePauliOp.from_list([('ZIII',-0.5*wk), ('IZII',-wk)]).to_matrix().real
        # Coupling Hamiltonian
        pe_fac = - ldk * np.sqrt(0.5*wk)
        h4 = SparsePauliOp.from_list([('XIZI', pe_fac), ('XIIZ', pe_fac)]).to_matrix().real
        h5 = SparsePauliOp.from_list([('XXZI', 0.5*pe_fac), ('XXIZ', 0.5*pe_fac)]).to_matrix().real
        h6 = SparsePauliOp.from_list([('YYZI', 0.5*pe_fac), ('YYIZ', 0.5*pe_fac)]).to_matrix().real
    
        if(self.trotter):
            self.vec = expm(-1j * dt * h1) @ self.vec
            self.vec = expm(-1j * dt * h2) @ self.vec
            self.vec = expm(-1j * dt * h3) @ self.vec
            self.vec = expm(-1j * dt * h4) @ self.vec
            self.vec = expm(-1j * dt * h5) @ self.vec
            self.vec = expm(-1j * dt * h6) @ self.vec
        else:
            self.vec = expm(-1j * dt * (h1 + h2 + h3 + h4 + h5 + h6)) @ self.vec  

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
            q0 = i*4+0
            q1 = i*4+1
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
        """ Reset the circuit to the initial state |ee,pp> """
        self.circuit = QuantumCircuit(4*self.repeat)
        # could generalise self.nq in initialisation.

    def create_photon(self, nphoton):
        """ Create n photons in the circuit by applying apppropiate X gates 
           Args:
               nphoton: number of photons to create (0, 1, 2, or 3)
        """
        if nphoton < 0 or nphoton > 3:
            raise ValueError("nphoton must be between 0 and 3")
        for i in range(self.repeat):
            q2 = i*4+2
            q3 = i*4+3
            if nphoton & 0b10:
                self.circuit.x(q2)
            if nphoton & 0b01:
                self.circuit.x(q3)
    
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
            q0 = i*4+0
            q1 = i*4+1
            q2 = i*4+2
            q3 = i*4+3

            # Split into functions so we can change order easily
            # Since defined inside, already has access to self.t (etc) so can use directly
            def apply_h1():
                # One-body molecular Hamiltonian
                # XIII and IXII
                self.circuit.rx(-2*self.t*dt,q0)
                self.circuit.rx(-2*self.t*dt,q1)
                return

            def apply_h2():
                # Two-body molecular Hamiltonian
                # ZZII
                self.circuit.cx(q0,q1)
                self.circuit.rz(dt*(self.U+2*ldk*ldk),q1)
                self.circuit.cx(q0,q1)
                return

            def apply_h3(): 
                # Photon Hamiltonian
                # IIZI and IZII
                self.circuit.rz(-2*wk*dt,q2)
                self.circuit.rz(-wk*dt,q3)
                return

            def apply_h4():
                # ZIIX and IZIX
                self.circuit.h(q3)
                self.circuit.cx(q3,q0)
                self.circuit.cx(q3,q1)
                self.circuit.rz(-2*pe_fac*dt,q0)
                self.circuit.rz(-2*pe_fac*dt,q1)
                self.circuit.cx(q3,q1)
                self.circuit.cx(q3,q0)
                self.circuit.h(q3)
                return

            def apply_h5():
                # ZIXX and IZXX
                self.circuit.h(q2)  
                self.circuit.h(q3)
                self.circuit.cx(q2,q3)
                self.circuit.cx(q3,q0)
                self.circuit.cx(q3,q1)
                self.circuit.rz(-pe_fac*dt,q0)
                self.circuit.rz(-pe_fac*dt,q1)
                self.circuit.cx(q3,q1)
                self.circuit.cx(q3,q0)
                self.circuit.cx(q2,q3)
                self.circuit.h(q2)
                self.circuit.h(q3)
                return
                
            def apply_h6():
                # ZIYY and IZYY
                self.circuit.sdg(q2)
                self.circuit.sdg(q3)
                self.circuit.h(q2)
                self.circuit.h(q3)
                self.circuit.cx(q2,q3)
                self.circuit.cx(q3,q0)
                self.circuit.cx(q3,q1)
                self.circuit.rz(-pe_fac*dt,q0)
                self.circuit.rz(-pe_fac*dt,q1)
                self.circuit.cx(q3,q1)
                self.circuit.cx(q3,q0)
                self.circuit.cx(q2,q3)
                self.circuit.h(q2)
                self.circuit.h(q3)
                self.circuit.s(q2)
                self.circuit.s(q3)
                return

            apply_h1()
            apply_h3() # these two can run in parallel
            apply_h2()
            apply_h4() 
            apply_h5()  
            apply_h6()
    
    def postselect_energy(self, dev):
        """Compute the electronic energy following post-selection to states with 0 photon.

           This performs explicit measurements for the electronic Hamiltonian and only includes
           measurement outcomes with no photon. Therefore, it filters out states with the 
           incorrect photon number, where the excitation is incomplete.

           Args:
               dev: ibmq.Device() that interfaces with the Qiskit backend for sampling
                    [see ibmq/device.py]
        """
        # Circuit to measure in ZZZZ basis
        qc_IIII = self.circuit.copy()
        qc_IIII.measure_all()
        # Circuit to measure in ZZXX basis
        qc_IIXX = self.circuit.copy()
        for i in range(self.repeat):
            qc_IIXX.ry(-0.5*np.pi,4*i+0)
            qc_IIXX.ry(-0.5*np.pi,4*i+1)
        qc_IIXX.measure_all()

        # Run sampler to get results
        results = dev.run_sampler([qc_IIII,qc_IIXX])
        res_IIII = results[0].data.meas
        res_IIXX = results[1].data.meas

        # Containers for measurement outcomes
        meas_ZII = measurement_outcome() # come back to, will be diff for 2 photons
        meas_IIIX = measurement_outcome()
        meas_IIXI = measurement_outcome()
        meas_IIZZ = measurement_outcome()
        ps_meas_IIIX = measurement_outcome()
        ps_meas_IIXI = measurement_outcome()
        ps_meas_IIZZ = measurement_outcome()

        # Compute expectation values for IIXI and IIIX contributions (1-body)
        for k, ncount in res_IIXX.get_counts().items():
            for i in range(self.repeat):
                # Extract the bits for this repeat
                ki = k[4*i:4*(i+1)]
                # Compute contribution
                IIIX = 1 if ki[3]=='0' else -1
                IIXI = 1 if ki[2]=='0' else -1
                # Total energy contribution
                meas_IIIX.add_outcome(-self.t*IIIX, ncount)
                meas_IIXI.add_outcome(-self.t*IIXI, ncount)
                # Post-selection measurement
                if(ki[0] == '0' and ki[1] == '0'):
                    ps_meas_IIIX.add_outcome(-self.t*IIIX, ncount)
                    ps_meas_IIXI.add_outcome(-self.t*IIXI, ncount)
                # Measure photon number
                if (ki[0] == '0' and ki[1] == '0'):
                    pvalue = 0
                elif (ki[0] == '1' and ki[1] == '0'):
                    pvalue = 1
                elif (ki[0] == '0' and ki[1] == '1'):
                    pvalue = 2
                else:
                    pvalue = 3
                meas_ZII.add_outcome(pvalue, ncount)

        # Compute expectation values for IZZ energy contribution (2-body)
        for k, ncount in res_IIII.get_counts().items():
            for i in range(self.repeat):
                # Extract the bits for this repeat
                ki = k[4*i:4*(i+1)]
                # Compute contribution
                IIZZ = np.prod([1 if x == '0' else -1 for x in ki[2:]]) 
                # Total energy contribution
                meas_IIZZ.add_outcome(0.5*self.U*IIZZ, ncount)
                # Post-selection measurement
                if(ki[0] == '0' and ki[1] == '0'):
                    ps_meas_IIZZ.add_outcome(0.5*self.U*IIZZ, ncount)
                # Measure photon number
                if (ki[0] == '0' and ki[1] == '0'):
                    pvalue = 0
                elif (ki[0] == '1' and ki[1] == '0'):
                    pvalue = 1
                elif (ki[0] == '0' and ki[1] == '1'):
                    pvalue = 2
                else:
                    pvalue = 3
                meas_ZII.add_outcome(pvalue, ncount)

        # Total energy
        Et = add_measurements([meas_IIIX, meas_IIXI, meas_IIZZ], 0.5*self.U)
        # Total post-selection energy
        E  = add_measurements([ps_meas_IIIX, ps_meas_IIXI, ps_meas_IIZZ], 0.5*self.U)
        # Get total photon number
        pn = (meas_ZII.value()[0], meas_ZII.value()[1])
        return Et, E, pn 
