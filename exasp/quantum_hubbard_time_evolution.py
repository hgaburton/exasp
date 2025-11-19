#!/usr/bin/env python
import sys
from argparse import ArgumentParser
from datetime import datetime
import math
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply
from system import HubbardSystem
from path import SinNPath
import quantum
from copy import deepcopy

np.set_printoptions(linewidth=1000,suppress=True,precision=6)

parser = ArgumentParser()
parser.add_argument("-U", dest="U", type=float,
                    help="Hubbard U parameter")
parser.add_argument("-nelec", dest="nelec", type=str, 
                    help = "Number of electrons in (nalfa,nbeta) format")
parser.add_argument("-dim", dest="dim", default='1,1,1', type=str,
                    help = "Hubbard lattice dimensions")
parser.add_argument("-lmax", "--lambda_max", dest="lmax", default=1.0, type=float,
                    help = "Coupling strengths")
parser.add_argument("-wmax", "--omega_max", dest="wmax", default=1.0, type=float,
                    help = "Maximum value of omega")
parser.add_argument("-pol", "--polarisation", dest="eps", default='0,0,1', type=str,
                    help = "Polarisation vector in (x,y,z) format")
parser.add_argument("-dt", "--timestep", dest="dt", default=0.1, type=float,
                    help = "Size of timesteps")
parser.add_argument("-T", dest="T", default=101, type=float,
                    help = "Total time evolution")
parser.add_argument('--nphoton', dest='nphoton', default=1, type=int,
                    help = "Maximum number of allowed photons")
parser.add_argument('--init', dest='initial_state', default='0,0', type=str,
                    help = "Initial state 'n,m' where n is the electronic state and m is the photon number")
parser.add_argument("-o", "--order", dest="order", default=3, type=float,
                    help = "Power order of the sin function for lambda path")
parser.add_argument("--site", dest="site", default=False, action='store_true',
                    help = "This flag sets the calculation to work in the site basis")
parser.add_argument("-f", "--nfrozen", dest="nfrozen", default=0, type=int,
                    help = "Number of frozen core orbitals.")
parser.add_argument("--trotter", dest="trotter", action='store_true',
                    help = "Whether to perform trotterised time evolution.")
parser.add_argument("--noisy", dest="noisy", action='store_true',
                    help = "Whether to perform trotterised time evolution with a noisy backend.")
parser.add_argument("-no_dse", dest="dse", default=True, action='store_false',
                    help = "Turn off the dipole self-interaction")
parser.add_argument("-ref", dest="ref", default="", 
                    help = "File containing initial reference wavefunction")
parser.add_argument("--electronic-energy", dest="electronic_energy", default=False, action='store_true',
                    help = "Only compute electronic energy")
parser.add_argument("--project", dest="project", default=False, action='store_true',
                    help = "Project out photon at the end")
args = parser.parse_args()

trotter = args.trotter
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
# NOTE: The ordering of the cartesian components is Z/X/Y, which matches the
#       ordering of spherical harmonics in the ORCA format.
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

print()
print('# --------------------------------------------------------')
print(f'#  {"t_k":^14s}   {"<E(t_k)>":^18s} {"<Pn(t_k)>":^12s}   State fidelity -->')
print('# --------------------------------------------------------')
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

print(trotter, noisy)
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
        # Instantaneous energy
        if args.electronic_energy:
            Ek = np.dot(np.conj(vk), H_elec @ vk).real
        else:
            Ek = np.dot(np.conj(vk), Hk @ vk).real
        # Instantaneous photon number
        Pk = np.dot(np.conj(vk), hamil_terms[1].to_matrix(sparse=True) @ vk).real
        # Fidelity with different states
        f = {}
        for j in range(len(einit0)):
            f[j] = np.abs(np.dot(v[j], vk))**2
        print(f'  {k: 8.6e}   {Ek: 16.10f}   {Pk: 10.6e}', end='')
        for j in range(len(einit0)):
           print(f'{f[j]: 10.6e}', end='')
        if args.project:
            vk_copy = deepcopy(vk)
            if trotter:
                vk_copy = vk_copy.data
            for l in range(len(vk_copy)):
                if l & 2**(nqubit-1):
                    vk_copy[l] = 0
            norm = np.sqrt(sum(abs(x)**2 for x in vk_copy))
            vk_copy /= norm
            if args.electronic_energy:
                Ek = np.dot(np.conj(vk_copy), H_elec @ vk_copy).real
            else:
                Ek = np.dot(np.conj(vk_copy), Hk @ vk_copy).real
            # Instantaneous photon number
            Pk = np.dot(np.conj(vk_copy), hamil_terms[1].to_matrix(sparse=True) @ vk_copy).real
            # Fidelity with different states
            f = {}
            for j in range(len(einit0)):
                f[j] = np.abs(np.dot(v[j], vk_copy))**2
            print(f' {norm: 10.6e}   {Ek: 16.10f}   {Pk: 10.6e}', end='')
            for j in range(len(einit0)):
               print(f'{f[j]: 10.6e}', end='')
        print()
        sys.stdout.flush()
print('# --------------------------------------------------------')
