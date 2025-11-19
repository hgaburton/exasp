#!/usr/bin/env python
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import utils.quantum as quantum
from utils.system import HubbardSystem
np.set_printoptions(linewidth=1000,suppress=True,precision=6)

if __name__=="__main__":
    parser = ArgumentParser(description="Generate tUPS input files for Hubbard lattice")
    parser.add_argument("-U", dest="U", type=float, default=1.0, help="Hubbard U parameter [default=1.0]")
    parser.add_argument("-nelec", dest="nelec", type=str, default="1,1", help="Number of electrons in (nalfa,nbeta) format [default=1,1]")
    parser.add_argument("-dim", dest="dim", type=str, default="1,1,1", help="Hubbard lattice dimensions [default=1,1,1]")
    parser.add_argument("--site", dest="site", action="store_true", default=False, help="This flag sets the calculation to work in the site basis [default=False]")
    parser.add_argument("-f", dest="nfrozen", type=int, default=0, help="Number of frozen core orbitals [default=0]")
    parser.add_argument("-pp", dest="pp", action="store_true", default=False, help="Use a perfect pairing reference state [default=False]")
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