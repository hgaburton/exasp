#!/usr/bin/env python
import sys
from argparse import ArgumentParser
from datetime import datetime
import math
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply, eigsh
from system import HubbardSystem
from path import SinNPath
import quantum

np.set_printoptions(linewidth=1000,suppress=True,precision=6)

parser = ArgumentParser()
parser.add_argument("-U", dest="U", type=float,
                    help="Hubbard U parameter")
parser.add_argument("-nelec", dest="nelec", type=str, 
                    help = "Number of electrons in (nalfa,nbeta) format")
parser.add_argument("-dim", dest="dim", default='1,1,1', type=str,
                    help = "Hubbard lattice dimensions")
parser.add_argument("--site", dest="site", default=False, action='store_true',
                    help = "This flag sets the calculation to work in the site basis")
parser.add_argument("-f", "--nfrozen", dest="nfrozen", default=0, type=int,
                    help = "Number of frozen core orbitals.")
parser.add_argument("-pp", "--perfect_pairing", dest="pp", default=False, action='store_true',
                    help = "Use a perfect pairing reference state")
args = parser.parse_args()

print('##########################################################')
print('# Generating tUPS input files for Hubbard lattice         ')
print(datetime.now().strftime("# Today: %d %b %Y at %H:%M:%S"))
print('#', args)
print('##########################################################')

# Get polarisation vector
eps = [0,0,0]

# Read in and process information about the Hubbard lattice
assert(len(args.dim.split(','))==3)
dim = list(map(int,args.dim.split(',')))
assert(len(args.nelec.split(','))==2)
ne  = list(map(int,args.nelec.split(',')))

# Setup the HubbardSystem
nsite = dim[0]*dim[1]*dim[2]
if(args.site):
    mo_coeff = np.identity(nsite)
    system = HubbardSystem(args.U,1,1,[0,0,0],ne,dim,mo_coeff=mo_coeff)
else:
    system = HubbardSystem(args.U,1,1,[0,0,0],ne,dim)


# Initial state
einit0, vinit = system.solve_molecular_hamiltonian()
np.savetxt("ham.eigval", einit0, fmt="% 20.16f")
hmat, dip_hmat = quantum.get_2nd_quant_hamiltonian(system.mo_ints, system.ints.nmo()-system.nfrozen, eps)

# Save for tUPS
tUPS = quantum.tUPSAdaptor(nsite)
tUPS.write_hamiltonian(hmat)
tUPS.write_operators()
tUPS.write_reference(ne[0],ne[1],perfect_pairing=args.pp)
