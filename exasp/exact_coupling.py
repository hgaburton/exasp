#!/usr/bin/env python
import sys
import numpy as np
from scipy.sparse.linalg import eigsh
from exasp.utils.system import MolecularSystem
from exasp.utils.path import SinNPath
from datetime import datetime
np.set_printoptions(linewidth=1000,suppress=True,precision=6)
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Compute exact eigenstates along the excited-state adiabatic path")
    parser.add_argument("-mol","--xyzfile",dest="xyzfile",help="Molecule .xyz file")
    parser.add_argument("-b","--basis",dest="basis",default='sto-3g',type=str,help="Basis set [default=sto-3g]")
    parser.add_argument("-lmax",dest="lmax",default=1.0,type=float,help="Coupling strengths")
    parser.add_argument("-wmax",dest="wmax",default=1.0,type=float,help="Maximum value of omega")
    parser.add_argument("-pol",dest="eps",default='0,0,1',type=str,help="Polarisation vector in (x,y,z) format. [default=0,0,1]")
    parser.add_argument("-n",dest="ngrid",default=101,type=int,help="Number of grid points to consider [default-101]")
    parser.add_argument('--nphoton',dest='nphoton',default=1,type=int,help="Maximum number of allowed photons [default=1]")
    parser.add_argument("-o",dest="order",default=3,type=float,help="Power order of the sin function for lambda path [default=3]")
    args = parser.parse_args()

    print(f'##########################################################')
    print(f'# Computing exact eigenstates for excited-state ASP  ')
    print(datetime.now().strftime("# Today: %d %b %Y at %H:%M:%S"))
    print('#', args)
    print(f'##########################################################')

    # Get polarisation vector
    assert(len(args.eps.split(',')) == 3)
    eps = list(map(float,args.eps.split(',')))

    # Setup the EXASP system
    system = MolecularSystem(args.xyzfile,args.basis,args.nphoton,eps)

    # Set up the path definition
    path = SinNPath(args.wmax,args.lmax,args.order)

    # Print the molecules
    print(' Molecule [angstrom]:')
    print(' --------------------------------------------------------')
    with open(args.xyzfile) as f:
        for l in f.readlines():
            print("  ",l,end='')

    # Print details of the calculation
    print()
    print(f' Input parameters:')
    print(' --------------------------------------------------------')
    print(f"  Molecule XYZ file    = {args.xyzfile}")
    print(f"  Basis set            = {args.basis}")
    print(f"  Number of photons    = {args.nphoton: 10d}")
    print(f"  Grid size            = {args.ngrid: 10d}")
    print(f'  Polarisation vector  = {eps}')
    print(f'  lmax                 = {args.lmax}')
    print(f'  wmax                 = {args.wmax}')
    print(f'  order                = {args.order}')

    print(' --------------------------------------------------------')
    for k in np.linspace(0,1,args.ngrid):
        # Get values of w and l
        wk, lk = path.w(k), path.l(k)
        # Hamiltonian
        Hk = system.H(wk,lk)
        dim = Hk.shape[0]
        # Eigensolution
        if(dim <= 20):
            Ek, V = np.linalg.eigh(Hk.toarray())
        else:
            Ek, V = eigsh(Hk,k=20)
        # Print result
        print(f'{k: 10.6f}   ',end="")
        print((len(Ek[:20])*'{: 14.8f} ').format(*Ek[:20]),end='\n')
        sys.stdout.flush()
    print(' --------------------------------------------------------')
