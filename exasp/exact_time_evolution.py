#!/usr/bin/env python
import sys
import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply
from utils.system import MolecularSystem
from utils.path import SinNPath
from datetime import datetime
np.set_printoptions(linewidth=1000,suppress=True,precision=6)
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Perform exact time evolution along an excited-state adiabatic state preparation path for a molecular system.")
    parser.add_argument("-mol",dest="xyzfile",help="Molecule .xyz file")
    parser.add_argument("-b",dest="basis",default='sto-3g',type=str,help="Basis set [default=sto-3g]")
    parser.add_argument("-lmax",dest="lmax",default=1.0,type=float,help="Coupling strengths maximum value [default=1.0]")
    parser.add_argument("-wmax",dest="wmax",default=1.0,type=float,help="Maximum value of omega [default=1.0]")
    parser.add_argument("-pol","--polarisation",dest="eps",default='0,0,1',type=str,help="Polarisation vector in (x,y,z) format [default=0,0,1]")
    parser.add_argument("-n",dest="nstep",default=101,type=int,help="Number timesteps [default=101]")
    parser.add_argument("-T",dest="T",default=101,type=float,help="Total time evolution [default=101]")
    parser.add_argument('--nphoton',dest='nphoton',default=1,type=int,help="Maximum number of allowed photons [default=1]")
    parser.add_argument('--init',dest='initial_state',default='0,0',type=str,help="Initial state 'n,m' where n is the electronic state and m is the photon number [default=0,0]")
    parser.add_argument("-o",dest="order",default=3,type=float,help="Power order of the sin function for lambda path [default=3]")
    parser.add_argument("-f",dest="nfrozen",default=0,type=int,help="Number of frozen core orbitals [default=0]")
    args = parser.parse_args()

    print(f'##########################################################')
    print(f'# Performing exact time evolution for excited-state ASP  ')
    print(datetime.now().strftime("# Today: %d %b %Y at %H:%M:%S"))
    print('#', args)
    print(f'##########################################################')

    # Get polarisation vector
    assert(len(args.eps.split(',')) == 3)
    eps = list(map(float,args.eps.split(',')))
    assert(len(args.initial_state.split(',')) == 2)
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
    einit, vinit = system.solve_molecular_hamiltonian()
    vk = np.kron(np.eye(args.nphoton+1,1,k=-init_p,dtype=complex).flatten(), vinit[:,init_e])

    # Time evolution
    T = args.T
    n = args.nstep
    dt = T / n

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
    print(f'  Basis set            = {args.basis}')
    print(f'  Frozen core orbitals = {args.nfrozen}')

    print()
    print(' --------------------------------------------------------')
    print(f'  {"t_k":^14s}   {"<E(t_k)>":^18s} {"<Pn(t_k)>":^12s}   State fidelity -->')
    print(' --------------------------------------------------------')
    sys.stdout.flush()

    v0 = np.kron(np.eye(args.nphoton+1,1).flatten(), vinit[:,0])
    v1 = np.kron(np.eye(args.nphoton+1,1).flatten(), vinit[:,1])
    v2 = np.kron(np.eye(args.nphoton+1,1).flatten(), vinit[:,2])
    v3 = np.kron(np.eye(args.nphoton+1,1).flatten(), vinit[:,3])

    for i,k in enumerate(np.linspace(0,1,n+1)):
        # Get values of w and l
        wk, lk = path.w(k), path.l(k)
        # Hamiltonian
        Hk = system.H(wk,lk)
        # Time propagation
        vk = expm_multiply(-1j * dt * Hk, vk) 
        # Instantaneous energy
        Ek = np.dot(np.conj(vk), Hk @ vk).real
        # Instantaneous photon number
        Pk = np.dot(np.conj(vk), system.Hp_c @ vk).real
        # Fidelity with different states
        f0 = np.abs(np.dot(v0, vk))**2 
        f1 = np.abs(np.dot(v1, vk))**2 
        f2 = np.abs(np.dot(v2, vk))**2 
        f3 = np.abs(np.dot(v3, vk))**2 
        print(f'  {k: 8.6e}   {Ek: 16.10f}   {Pk: 10.6e}   {f0: 10.6e}   {f1: 10.6e}   {f2: 10.6e}   {f3: 10.6e}')
        sys.stdout.flush()
    print(' --------------------------------------------------------')
