#!/usr/bin/env python
import os
#os.environ["OMP_NUM_THREADS"] = "1" 
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
from argparse import ArgumentParser
from datetime import datetime
import math
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.sparse import kron, eye
from system import MolecularSystem
from path import SinNPath
import quantum

np.set_printoptions(linewidth=1000,suppress=True,precision=6)

parser = ArgumentParser()
parser.add_argument("-mol", "--xyzfile", dest="xyzfile",
                    help="Molecule .xyz file")
parser.add_argument("-b", "--basis", dest="basis", default='sto-3g', type=str,
                    help="Basis set")
parser.add_argument("-lmax", "--lambda_max", dest="lmax", default=1.0, type=float,
                    help = "Coupling strengths")
parser.add_argument("-wmax", "--omega_max", dest="wmax", default=1.0, type=float,
                    help = "Maximum value of omega")
parser.add_argument("-pol", "--polarisation", dest="eps", default='0,0,1', type=str,
                    help = "Polarisation vector in (x,y,z) format")
parser.add_argument("-n", "--nstep", dest="nstep", default=101, type=int,
                    help = "Number timesteps")
parser.add_argument("-T", dest="T", default=101, type=float,
                    help = "Total time evolution")
parser.add_argument('--nphoton', dest='nphoton', default=1, type=int,
                    help = "Maximum number of allowed photons")
parser.add_argument('--init', dest='initial_state', default='0,0', type=str,
                    help = "Initial state 'n,m' where n is the electronic state and m is the photon number")
parser.add_argument("-o", "--order", dest="order", default=3, type=float,
                    help = "Power order of the sin function for lambda path")
parser.add_argument("-f", "--nfrozen", dest="nfrozen", default=0, type=int,
                    help = "Number of frozen core orbitals.")
parser.add_argument("--trotter", dest="trotter", action='store_true',
                    help = "Whether to perform trotterised time evolution.")
args = parser.parse_args()

trotter = args.trotter

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

# Setup the EXASP system
system = MolecularSystem(args.xyzfile,args.basis,args.nphoton,eps,nfrozen=args.nfrozen)
# Set up the path definition
path = SinNPath(args.wmax,args.lmax,args.order)

# Print the molecules
print(' Molecule [angstrom]:')
print(' --------------------------------------------------------')
with open(args.xyzfile) as f:
    for l in f.readlines():
        print("  ",l,end='')

# Initial state
einit0, vinit0 = system.solve_molecular_hamiltonian()
hmat, dip_hmat = quantum.get_2nd_quant_hamiltonian(system.mo_ints, system.ints.nmo()-system.nfrozen, eps)
hmat_q, dip_hmat_q = quantum.get_qubit_hamiltonian(hmat, dip_hmat)
spmat_hmat_q = hmat_q.to_matrix(sparse=True)
einit, vinit = eigsh(spmat_hmat_q,k=20)
print(einit)
print(einit0)
dim = vinit.shape[0]
#einit, vinit = np.linalg.eigh(hmat_q)
#print(einit)
init_e0 = init_e
init_es = {0:-1, 1:-1, 2:-1, 3:-1}
for i, e in enumerate(einit):
     if math.isclose(e,einit0[init_e0]):
        init_e=i
     for j in range(4):
        if math.isclose(e,einit0[j]):
            init_es[j] = i
print(init_e)
print(init_es)
vk = np.kron(np.array([0,1]), vinit[:,init_e])

T=args.T
n=args.nstep
dt=T / n

# Print details of the calculation
print()
print(f' Parameters:')
print(' --------------------------------------------------------')
print(f"  Number of photons    = {args.nphoton: 10d}")
print(f"  Total time evolution = {args.T: 10.6e}")
print(f"  Number of steps      = {args.nstep: 10d}")
print(f"  Time step            = {dt: 10.6e}")
print(f'  Initial state        = |e{init_e},p{init_p}>')
print(f'  Polarisation vector  = {eps}')
print(f'  lmax                 = {args.lmax}')
print(f'  wmax                 = {args.wmax}')
print(f'  order                = {args.order}')

print()
print(' --------------------------------------------------------')
print(f'  {"t_k":^14s}   {"<E(t_k)>":^18s}   {"<pE(t_k)>":^18s}   {"<Pn(t_k)>":^12s}   State fidelity -->')
print(' --------------------------------------------------------')
sys.stdout.flush()

hamil_terms = quantum.coupling_hamil_terms(hmat_q, dip_hmat_q)

v0 = np.kron(np.array([1,0]), vinit[:,init_es[0]])
v1 = np.kron(np.array([1,0]), vinit[:,init_es[1]])
v2 = np.kron(np.array([1,0]), vinit[:,init_es[2]])
v3 = np.kron(np.array([1,0]), vinit[:,init_es[3]])

if trotter:
    vk = Statevector(vk)
for i,k in enumerate(np.linspace(0,1,n+1)):
    # Get values of w and l
    wk, lk = path.w(k), path.l(k)
    if trotter:
        # Hamiltonian
        Hk = quantum.get_coupled_hamil(hamil_terms, wk,lk)
        # Time propagation
        vk = quantum.trotter_evolve(Hk, vk, dt)
        Hk = Hk.to_matrix(sparse = True)
    else:
        # Hamiltonian
        Hk = quantum.get_coupled_hamil(hamil_terms, wk,lk).to_matrix(sparse=True)
        # Time propagation
        vk = expm_multiply(-1j * dt * Hk, vk)
    # Instantaneous energy
    Ek = np.dot(np.conj(vk), Hk @ vk).real
    # Project to zero photon state
    vproj = vk.copy()
    vproj[dim:] = 0
    norm_proj = np.dot(np.conj(vproj), vproj).real
    # Instantaneous photon number
    Pk = np.dot(np.conj(vk), hamil_terms[1].to_matrix(sparse=True) @ vk).real
    # Fidelity with different states
    f0 = np.abs(np.dot(v0, vk))**2
    f1 = np.abs(np.dot(v1, vk))**2
    f2 = np.abs(np.dot(v2, vk))**2
    f3 = np.abs(np.dot(v3, vk))**2
    # Calculate projected energy
    pEk = np.nan if (norm_proj==0) else np.real(np.dot(np.conj(vproj), Hk @ vproj)) / norm_proj
    # Projected fidelity with different states
    pf0 = np.nan if (norm_proj==0) else np.abs(np.dot(v0, vproj))**2 / norm_proj 
    pf1 = np.nan if (norm_proj==0) else np.abs(np.dot(v1, vproj))**2 / norm_proj
    pf2 = np.nan if (norm_proj==0) else np.abs(np.dot(v2, vproj))**2 / norm_proj
    pf3 = np.nan if (norm_proj==0) else np.abs(np.dot(v3, vproj))**2 / norm_proj
    print(f'  {k: 8.6e}   {Ek: 16.10f}   {pEk: 16.10f}   {Pk: 10.6e}   {f0: 10.6e}   {pf0: 10.6e}   {f1: 10.6e}   {pf1: 10.6e}  {f2: 10.6e}   {pf2: 10.6e}   {f3: 10.6e}   {pf3: 10.6e}')
    sys.stdout.flush()
print(' --------------------------------------------------------')
