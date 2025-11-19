#!/usr/bin/env python
import numpy as np
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import PolynomialTensor, FermionicOp
from qiskit import transpile
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit_algorithms import TimeEvolutionProblem
from qiskit_algorithms import TrotterQRTE
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.circuit import QuantumCircuit

def get_2nd_quant_hamiltonian(mo_ints, nbasis, eps):
    h1a = mo_ints.oei_matrix(True)
    h2aa_init = mo_ints.tei_array(True, False)
    h2aa = np.ndarray(h2aa_init.shape)
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    h2aa[i,j,k,l] = h2aa_init[i,j,l,k]
    h1b = None
    h2bb = None
    h2ba = None
    hamil = ElectronicEnergy.from_raw_integrals(h1a, h2aa, h1b, h2bb, h2ba, auto_index_order=False)
    hamil.electronic_integrals.alpha += PolynomialTensor({"": mo_ints.scalar_potential()})
    d1a = mo_ints.dipole_matrix(True)
    d1a = np.einsum('x,xpq->pq', eps, d1a)
    d2aa = None
    dipole_hamil = ElectronicEnergy.from_raw_integrals(d1a, d2aa, h1b, h2bb, h2ba,
                                                       auto_index_order=False)
    return hamil, dipole_hamil

def get_qubit_hamiltonian(hmat, dip_hmat):
    jw_mapper = JordanWignerMapper()
    hmat_q = jw_mapper.map(hmat.second_q_op())
    dip_hmat_q = jw_mapper.map(dip_hmat.second_q_op())
    return hmat_q, dip_hmat_q

def coupling_hamil_terms(hmat, dip_hmat, dse = True):
    phot_hmat = SparsePauliOp(['I', 'Z'], coeffs=[0.5, - 0.5])
    phot_x = SparsePauliOp('X')
    Hm_c = (SparsePauliOp('I').tensor(hmat)).simplify()
    mol_ident = 'I'*hmat.num_qubits
    Hp_c = phot_hmat.tensor(SparsePauliOp(mol_ident)).simplify()
    Hi_c = phot_x.tensor(dip_hmat).simplify()
    if dse:
        Hd_c = SparsePauliOp('I').tensor(dip_hmat.dot(dip_hmat)).simplify()
    else:
        Hd_c = 0
    return Hm_c, Hp_c, Hi_c, Hd_c

def coupling_hamil_terms_3qub_hub(U, delta):
    hmat_q = SparsePauliOp(['II', 'ZZ', 'XI', 'IX'], coeffs=[U/2, U/2, -1, -1])
    hmat_e = SparsePauliOp(['III', 'ZZI', 'XII', 'IXI'], coeffs=[U/2, U/2, -1, -1])
    hmat_p = SparsePauliOp(['III', 'IIZ'], coeffs = [0.5,-0.5])
    hmat_imu = SparsePauliOp(['IZX', 'ZIX'], coeffs = [-delta, -delta])
    hmat_d = SparsePauliOp(['III', 'ZZI'], coeffs = [2*delta**2, 2*delta**2])
    return hmat_q, hmat_e, hmat_p, hmat_imu, hmat_d

def get_coupled_hamil(h,w,l):
    hamiltonian = h[0] + w * h[1] + l * np.sqrt(0.5 * w) * h[2] + 0.5 * l ** 2 * h[3]
    return hamiltonian.simplify()

def trotter_evolve(Hk, vk, dt, circuit_file=None,order=1):
    if order not in (1,2):
        raise ValueError(f"Invalid Trotter order {order} requested")
    pf = SuzukiTrotter(2) if order==2 else LieTrotter()
    problem = TimeEvolutionProblem(Hk, initial_state=vk, time=dt)
    trotter = TrotterQRTE(product_formula=pf,num_timesteps=1)
    result = trotter.evolve(problem)
    if circuit_file is not None:
        print('Trotter step operations: ', result.evolved_state.decompose(reps=2).decompose("disentangler_dg").decompose(
       "multiplex1_reverse_dg").count_ops())
        print('Trotter step depth ', result.evolved_state.decompose(reps=2).decompose("disentangler_dg").decompose(
       "multiplex1_reverse_dg").depth())
        result.evolved_state.decompose(reps=2).decompose("disentangler_dg").decompose(
     "multiplex1_reverse_dg").draw(output='mpl', filename = circuit_file) 
    vk = Statevector(result.evolved_state)
    return vk

def  trotter_evolve_noisy_dm(path, nsteps, nqubits, hamil_terms, dt, vk, order=1):
    if order not in (1,2):
        raise ValueError(f"Invalid Trotter order {order} requested")
    pf = SuzukiTrotter(2) if order==2 else LieTrotter()
    intermediate_state = Statevector(np.zeros(2**nqubits))
    full_trotter = QuantumCircuit(nqubits)
    device_backend = FakeManilaV2()
    noise_model = NoiseModel.from_backend(device_backend)
    noisy_simulator = AerSimulator(method = "density_matrix",
        noise_model = noise_model)
    Eks = []
    Pks = []
    for i, k in enumerate(np.linspace(0,1,nsteps+1)):
        wk, lk = path.w(k), path.l(k)
        Hk = get_coupled_hamil(hamil_terms, wk, lk)
        if i == 0:
            problem = TimeEvolutionProblem(Hk, initial_state=vk, time=dt)
        else:
            problem = TimeEvolutionProblem(Hk, initial_state=intermediate_state, time=dt)
        trotter = TrotterQRTE(product_formula=pf,num_timesteps=1)
        result = trotter.evolve(problem)
        tcirc= transpile(result.evolved_state)
        full_trotter.append(tcirc)
        result_noise = noisy_simulator.run(full_trotter).result()
        Ek = np.trace(np.dot(result_noise.data()["density_matrix"], Hk.to_matrix(sparse = True)))
        Pk = np.trace(np.dot(result_noise.data()["density_matrix"], hamil_terms[1].to_matrix(sparse=True)))
        Eks.append(Ek)
        Pks.append(Pk)
    return Eks, Pks

class tUPSAdaptor:
    def __init__(self, nmo):
        # Save the number of spatial orbitals
        self.nmo = nmo
        # Save number of qubits
        self.nqubit = 2 * self.nmo
        # Jordan-Wigner mapping
        self.jw = JordanWignerMapper()
        # Dimension of the Fock space
        self.dim = 2 ** self.nqubit

    def write_hamiltonian(self, hmat_q):
        # Convert to a dense matrix
        hmat = self.jw.map(hmat_q.second_q_op()).to_matrix().real
        # Save to file
        hmat.tofile("ham")
        print("\nHamiltonian matrix saved to 'ham'...")

    def write_reference(self,nalfa,nbeta,perfect_pairing=False):
        vk = Statevector(np.eye(self.dim)[:,0])
        if perfect_pairing:
            # Loop over alfa creation
            for i in range(nalfa):
                pa = FermionicOp({f'+_{2*i}': 1.0}, num_spin_orbitals=self.nqubit)
                vk = vk.evolve(self.jw.map(pa))
            # Loop over beta creation
            for j in range(nbeta):
                pb = FermionicOp({f'+_{2*j+self.nmo}': 1.0}, num_spin_orbitals=self.nqubit)
                vk = vk.evolve(self.jw.map(pb))
        else:
            # Loop over alfa creation
            for i in range(nalfa):
                pa = FermionicOp({f'+_{i}': 1.0}, num_spin_orbitals=self.nqubit)
                vk = vk.evolve(self.jw.map(pa))
            # Loop over beta creation
            for j in range(nbeta):
                pb = FermionicOp({f'+_{j+self.nmo}': 1.0}, num_spin_orbitals=self.nqubit)
                vk = vk.evolve(self.jw.map(pb))

        # save to file
        ref_vec = vk.data.real
        if perfect_pairing:
            np.savetxt("pp.ref", ref_vec, fmt="% 20.16f")
        else:
            np.savetxt("ref", ref_vec, fmt="% 20.16f")


    def write_operators(self):
        # Loop over single excitations and double
        self.opmat = np.zeros((self.dim,self.dim,2*(self.nmo-1)))
        self.ops_info = []
        count = 0

        for i in range(1, int(np.ceil(self.nmo/2))):
            # indices for the excitation
            pa, qa = 2*i, 2*i-1
            pb, qb = pa + self.nmo, qa + self.nmo
            if qa >= 0:
                # Single excitation
                op1 = FermionicOp({f'+_{pa} -_{qa}': 1.0, f'+_{pb} -_{qb}': 1.0}, num_spin_orbitals=self.nqubit)
                op1 -= op1.adjoint()
                self.opmat[:,:,count] = self.jw.map(op1).to_matrix().real
                count += 1
                # Save the operator info
                self.ops_info.append(f'op{count:<3d} : S[{pa},{qa}]')
                # Double excitation
                op2 = FermionicOp({f'+_{pa} +_{pb} -_{qb} -_{qa}': 2.0}, num_spin_orbitals=self.nqubit)
                op2 -= op2.adjoint()
                self.opmat[:,:,count] = self.jw.map(op2).to_matrix().real
                count += 1
                # Save the operator info
                self.ops_info.append(f'op{count:<3d} : D[{pa},{qa}]')

        for i in range(0, int(np.floor(self.nmo/2))):
            # indices for the excitation
            pa, qa = 2*i+1, 2*i
            pb, qb = pa + self.nmo, qa + self.nmo
            if pa < self.nmo:
                # Single excitation
                op1 = FermionicOp({f'+_{pa} -_{qa}': 1.0, f'+_{pb} -_{qb}': 1.0}, num_spin_orbitals=self.nqubit)
                op1 -= op1.adjoint()
                self.opmat[:,:,count] = self.jw.map(op1).to_matrix().real
                count += 1
                # Save the operator info
                self.ops_info.append(f'op{count:<3d} : S[{pa},{qa}]')
                # Double excitation
                op2 = FermionicOp({f'+_{pa} +_{pb} -_{qb} -_{qa}': 2.0}, num_spin_orbitals=self.nqubit)
                op2 -= op2.adjoint()
                self.opmat[:,:,count] = self.jw.map(op2).to_matrix().real
                count += 1
                # Save the operator info
                self.ops_info.append(f'op{count:<3d} : D[{pa},{qa}]')

        # Save to file
        np.savetxt('opmat',self.opmat.flatten(order='F'), fmt="%d")

        with open('ops_info', 'w') as f:
            for i,item in enumerate(self.ops_info):
                f.write(f"{item:s}\n")
        print(f"dim = {self.dim}, nops = {len(self.ops_info)}")
