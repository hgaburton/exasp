#!/usr/bin/env python
import sys
from argparse import ArgumentParser
from datetime import datetime
import math
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply
from utils.system import HubbardSystem
from utils.path import SinNPath
import utils.quantum as quantum
from copy import deepcopy

np.set_printoptions(linewidth=1000,suppress=True,precision=6)

if __name__=="__main__":
    parser = ArgumentParser(description="Perform time evolution for excited-state adiabatic state preparation in Hubbard lattice")
    parser.add_argument("-U",dest="U",type=float,help="Hubbard U parameter [default=None]")
    parser.add_argument("-nelec",dest="nelec",type=str,help="Number of electrons in (nalfa,nbeta) format [default=None]")
    parser.add_argument("-dim",dest="dim",default='1,1,1',type=str,help="Hubbard lattice dimensions [default=1,1,1]")
    parser.add_argument("-lmax",dest="lmax",default=1.0,type=float,help="Coupling strengths [default=1.0]")
    parser.add_argument("-wmax",dest="wmax",default=1.0,type=float,help="Maximum value of omega [default=1.0]")
    parser.add_argument("-pol",dest="eps",default='0,0,1',type=str,help="Polarisation vector in (x,y,z) format [default=0,0,1]")
    parser.add_argument("-dt",dest="dt",default=0.1,type=float,help="Size of timesteps [default=0.1]")
    parser.add_argument("-T",dest="T",default=101,type=float,help="Total time evolution [default=101]")
    parser.add_argument("--nphoton",dest="nphoton",default=1,type=int,help="Maximum number of allowed photons [default=1]")
    parser.add_argument("--init",dest="initial_state",default='0,1',type=str,help="Initial state 'n,m' where n is the electronic state and m is the photon number [default=0,0]")
    parser.add_argument("-o",dest="order",default=3,type=float,help="Power order of the sin function for lambda path [default=3]")
    parser.add_argument("--site",dest="site",default=False,action='store_true',help="This flag sets the calculation to work in the site basis [default=False]")
    parser.add_argument("-f",dest="nfrozen",default=0,type=int,help="Number of frozen core orbitals. [default=0]")
    parser.add_argument("--trotter",dest="trotter",action='store_true',help="Whether to perform trotterised time evolution. [default=False]")
    parser.add_argument("--noisy",dest="noisy",action='store_true',help="Whether to perform trotterised time evolution with a noisy backend. [default=False]")
    parser.add_argument("-no_dse",dest="dse",default=True,action='store_false',help="Turn off the dipole self-interaction [default=True]")
    parser.add_argument("-ref",dest="ref",default="",help="File containing initial reference wavefunction [default=]")
    parser.add_argument("--electronic-energy",dest="electronic_energy",default=False,action='store_true',help="Only compute electronic energy [default=False]")
    args = parser.parse_args()

    # Control if Trotter approximation used for time evolution step
    trotter = args.trotter
    # Control if noisy simulation to be considered
    noisy = args.noisy

    print('##########################################################')
    if trotter:
        print('# Performing trotterised time evolution for excited-state ASP  ')
    else:
        print('# Performing exact time evolution for excited-state ASP  ')
    print(datetime.now().strftime("# Today: %d %b %Y at %H:%M:%S"))
    print('#', args)
    print('##########################################################')

    # Get polarisation vector
    assert len(args.eps.split(',')) == 3
    eps = list(map(float,args.eps.split(',')))
    assert len(args.initial_state.split(',')) == 2
    (init_e,init_p) = map(int,args.initial_state.split(','))

    # Read in and process information about the Hubbard lattice
    assert(len(args.dim.split(','))==3)
    dim = list(map(int,args.dim.split(',')))
    assert(len(args.nelec.split(','))==2)
    ne  = list(map(int,args.nelec.split(',')))

    # Setup the HubbardSystem
    if(args.site):
        mo_coeff = np.identity(dim[0]*dim[1]*dim[2])
        system = HubbardSystem(args.U,1,args.nphoton,eps,ne,dim,mo_coeff=mo_coeff,dse=args.dse)
    else:
        system = HubbardSystem(args.U,1,args.nphoton,eps,ne,dim,dse=args.dse)

    # Set up the path definition
    path = SinNPath(args.wmax,args.lmax,args.order)

    # Initial state
    einit0, vinit = system.solve_molecular_hamiltonian()
    # Get Hamiltonian and dipole in qubit form
    hmat, dip_hmat = quantum.get_2nd_quant_hamiltonian(system.mo_ints, system.ints.nmo()-system.nfrozen, eps)
    hmat_q, dip_hmat_q = quantum.get_qubit_hamiltonian(hmat, dip_hmat)
    # Find Fock space solution corresponding to molecular ground state
    einit, vinit = np.linalg.eigh(hmat_q)
    init_e0 = init_e
    init_es = {x:-1 for x in range(len(einit0))}
    for i, e in enumerate(einit):
        if math.isclose(e,einit0[init_e0]):
            init_e=i
        for j in range(len(einit0)):
            if math.isclose(e,einit0[j]) and init_es[j] == -1:
                init_es[j] = i
                break
            
    # Setup propagation parameters
    T=args.T
    dt = args.dt
    n= int(T/dt)

    nqubit = dim[0]*dim[1]*dim[2]*2+1
    # Print details of the calculation
    print()
    print('# Parameters:')
    print('# --------------------------------------------------------')
    print(f"#  Number of photons    = {args.nphoton: 10d}")
    print(f"#  Total time evolution = {args.T: 10.6e}")
    print(f"#  Number of steps      = {n: 10d}")
    print(f"#  Time step            = {dt: 10.6e}")
    print(f'#  Initial state        = |e{init_e},p{init_p}>')
    print(f'#  Polarisation vector  = {eps}')
    print(f'#  lmax                 = {args.lmax}')
    print(f'#  wmax                 = {args.wmax}')
    print(f'#  order                = {args.order}')
    print(f'#  Dipole self-inter.   = {args.dse}')
    print(f'#  Hubbard U parameter  = {args.U}')
    print(f'#  Number of electrons  = {args.nelec}')
    print(f'#  Dimensions           = {args.dim}')
    print(f'#  Frozen core orbitals = {args.nfrozen}')
    print(f'#  Trotterised          = {trotter}')
    print(f'#  Noisy simulation     = {noisy}')
    print('# --------------------------------------------------------')

    with open("total.txt","w") as f:
        f.write('# --------------------------------------------------------\n')
        f.write(f'#  {"t_k":^14s}   {"<E(t_k)>":^18s} {"<Pn(t_k)>":^12s}   State fidelity -->\n')
        f.write('# --------------------------------------------------------\n')

    with open("projected.txt","w") as f:
        f.write('# --------------------------------------------------------\n')
        f.write(f'#  {"t_k":^14s}   {"<E(t_k)>":^18s} {"<Pn(t_k)>":^12s}   State fidelity -->\n')
        f.write('# --------------------------------------------------------\n')
    sys.stdout.flush()


    hamil_terms = quantum.coupling_hamil_terms(hmat_q, dip_hmat_q, args.dse)

    # Get the vectors to project against
    v = {}
    for j in range(len(einit0)):
        v[j] = np.kron(np.array([1,0]), vinit[:,init_es[j]])

    if(args.ref == ''):
        # Initialise from init_e target
        vk = np.kron(np.array([0,1]), vinit[:,init_e])
    else:
        # Initialise from file
        vk = np.kron(np.array([0,1]), np.genfromtxt(args.ref)+0j)

    # Get the initial target statevector
    if trotter:
        vk = Statevector(vk)

    if args.electronic_energy:
        H_elec = hamil_terms[0].to_matrix(sparse=True)

    if trotter and noisy:
        Eks, Pks = quantum.trotter_evolve_noisy_dm(path, n, system.ints.nmo()*2, hamil_terms, dt, vk)
        for i,k in enumerate(np.linspace(0,1,n+1)):
            print(f'{k: 8.6e}   {Eks[i]: 16.10f}   {Pks[i]: 10.6e}')
        sys.stdout.flush()
    else:
        for i,k in enumerate(np.linspace(0,1,n+1)):
            # Get values of w and l
            wk, lk = path.w(k), path.l(k)
            
            # Perform time evolution step
            if trotter:
                # Hamiltonian
                Hk = quantum.get_coupled_hamil(hamil_terms, wk, lk)
                # Time propagation
                vk = quantum.trotter_evolve(Hk, vk, dt)
                Hk = Hk.to_matrix(sparse = True)
            else:
                # Hamiltonian
                Hk = quantum.get_coupled_hamil(hamil_terms, wk,lk).to_matrix(sparse=True)
                # Time propagation
                vk = expm_multiply(-1j * dt * Hk, vk)

            # Compute instantaneous energy
            if args.electronic_energy:
                Ek = np.dot(np.conj(vk), H_elec @ vk).real
            else:
                Ek = np.dot(np.conj(vk), Hk @ vk).real

            # Compute instantaneous photon number
            Pk = np.dot(np.conj(vk), hamil_terms[1].to_matrix(sparse=True) @ vk).real

            # Compute fidelity with different states
            f = {}
            for j in range(len(einit0)):
                f[j] = np.abs(np.dot(v[j], vk))**2

            # Print output
            with open("total.txt","a") as outF:
                outF.write(f'  {k: 8.6e}   {Ek: 16.10f}   {Pk: 10.6e}')
                for j in range(len(einit0)):
                    outF.write(f'{f[j]: 10.6e}')
                outF.write('\n')
                
            # Make a statevector copy to perform projection
            vk_copy = deepcopy(vk)
            if trotter: vk_copy = vk_copy.data

            # Perform the projection
            for l in range(len(vk_copy)):
                if l & 2**(nqubit-1):
                    vk_copy[l] = 0
            
            norm = np.sqrt(sum(abs(x)**2 for x in vk_copy))
            if norm != 0:
                vk_copy /= norm

            # Compute instantaneous energy
            if args.electronic_energy:
                pEk = np.nan if norm==0 else np.dot(np.conj(vk_copy), H_elec @ vk_copy).real
            else:
                pEk = np.nan if norm==0 else np.dot(np.conj(vk_copy), Hk @ vk_copy).real

            # Compute instantaneous photon number
            pPk = np.dot(np.conj(vk_copy), hamil_terms[1].to_matrix(sparse=True) @ vk_copy).real

            # Compute fidelity with different states
            pf = {}
            for j in range(len(einit0)):
                pf[j] = np.nan if norm==0 else np.abs(np.dot(v[j], vk_copy))**2
            
            with open("projected.txt","a") as outF:
                outF.write(f'  {k: 8.6e}   {pEk: 16.10f}   {pPk: 10.6e}')
                for j in range(len(einit0)):
                    outF.write(f'{pf[j]: 10.6e}')
                outF.write('\n')
