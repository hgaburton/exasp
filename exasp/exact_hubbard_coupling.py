#!/usr/bin/env python
import sys
import numpy as np
from utils.system import HubbardSystem
from utils.path import SinNPath
from datetime import datetime
np.set_printoptions(linewidth=1000,suppress=True,precision=6)
from argparse import ArgumentParser

if __name__=="__main__":
    parser = ArgumentParser(description="Compute eigenstates along an excited-state adiabatic state preparation path for a Hubbard lattice.")
    parser.add_argument("-U",dest="U",type=float,help="Hubbard U parameter")
    parser.add_argument("-nelec",dest="nelec",type=str,help="Number of electrons in (nalfa,nbeta) format")
    parser.add_argument("-lmax",dest="lmax",default=1.0,type=float,help="Coupling strengths maximum value [default=1.0]")
    parser.add_argument("-wmax",dest="wmax",default=1.0,type=float,help="Maximum value of omega [default=1.0]")
    parser.add_argument("-dim",dest="dim",default='1,1,1',type=str,help = "Hubbard lattice dimensions [nx,ny,nz] format. Default is 1x1x1")
    parser.add_argument("-pol",dest="eps",default='0,0,1',type=str,help = "Polarisation vector in (x,y,z) format. [default='0,0,1']")
    parser.add_argument("-n",dest="ngrid",default=101,type=int,help="Number of grid points to consider [default=101]")
    parser.add_argument('--nphoton',dest='nphoton',default=1,type=int,help = "Maximum number of allowed photons [default=1]")
    parser.add_argument("--site",dest="site",action='store_true',default=False,help="This flag sets the calculation to work in the site basis [default is MO basis]")
    parser.add_argument("-o",dest="order",default=3,type=float,help="Power order of the sin function for lambda path [default=3]")
    parser.add_argument("-no_dse",dest="dse",default=True,action='store_false',help = "Turn off the dipole self-interaction")
    args = parser.parse_args()

    print(f'##########################################################')
    print(f'# Compute exact eigenstates for EXASP in Hubbard lattice ')
    print(datetime.now().strftime("# Today: %d %b %Y at %H:%M:%S"))
    print('#', args)
    print(f'##########################################################')

    # Get polarisation vector
    assert(len(args.eps.split(',')) == 3)
    eps = list(map(float,args.eps.split(',')))

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

    # Print details of the calculation
    print()
    print(f'# Parameters:')
    print(f'# --------------------------------------------------------')
    print(f'#  Hubbard U            = {args.U: 10.6f}')
    print(f'#  Number of electrons  = {ne[0]:d} alpha, {ne[1]:d} beta')
    print(f'#  Lattice dimensions   = {dim[0]:d}x{dim[1]:d}x{dim[2]:d}')
    print(f'#  Basis set            = {"site" if args.site else "MO"}')
    print(f"#  Number of photons    = {args.nphoton: 10d}")
    print(f"#  Grid size            = {args.ngrid: 10d}")
    print(f'#  Polarisation vector  = {eps}')
    print(f'#  lmax                 = {args.lmax}')
    print(f'#  wmax                 = {args.wmax}')
    print(f'#  order                = {args.order}')
    print(f'#  Dipole self-inter.   = {args.dse}')
    print(f'# --------------------------------------------------------')

    # Solve system Hamiltonian along the path
    for k in np.linspace(0,1,args.ngrid):
        # Get values of w and l
        wk, lk = path.w(k), path.l(k)
        # Hamiltonian
        Hk = system.H(wk,lk)
        # Eigensolution
        Ek, V = np.linalg.eigh(Hk.toarray())
        # Print result
        print(f'{k: 10.6f}   ',end="")
        print((len(Ek[:60])*'{: 14.8f} ').format(*Ek[:60]),end='\n')
        sys.stdout.flush()
    print(f'# --------------------------------------------------------')
