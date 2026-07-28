""" Script to perform Hubbard dimer simulation using IBM Quantum backends"""
from utils.path import SinNPath
from ibmq.hubbard_threelevel import HubbardCircuit
from ibmq.device import Device
import numpy as np
from argparse import ArgumentParser
from datetime import datetime   
import sys

if __name__ == "__main__":
    # Read arguments
    parser = ArgumentParser(description="Hubbard dimer, three-level simulation using IBM Quantum backends")
    parser.add_argument("-noiseless", action='store_true', help="Perform noiseless simulation [default=False]")
    parser.add_argument("-hardware", action='store_true', help="Perform simulation on QPU [default=False]")
    parser.add_argument("-nshot", default=1024, type=int, help="Number of shots for sampling [default=1024]")
    parser.add_argument("-seed", default=None, type=int, help="Random seed for sampling [default=None]")
    parser.add_argument("-fname", default="path.txt", type=str, help="Name for path output file [default=path.txt]")
    parser.add_argument("-backend", default="ibm_kingston", type=str, help="Choice of Qiskit QPU backend [default=ibm_kingston]")
    parser.add_argument("-T", default=20, type=float, help="Total evolution time [default=20]")
    parser.add_argument("-dt", default=0.1, type=float, help="Time step [default=0.1]")
    parser.add_argument("-wmax", default=5.5, type=float, help="Maximum photon frequency [default=5.5]")
    parser.add_argument("-lmax", default=1.4, type=float, help="Maximum coupling strength [default=1.4]")
    parser.add_argument("-U", default=4.0, type=float, help="Hubbard interaction strength [default=4.0]")
    parser.add_argument("-t", default=1.0, type=float, help="Hopping parameter [default=1.0]")
    parser.add_argument("-measure_path", action='store_true', help="Measure the path during simulation [default=False]")
    parser.add_argument("-repeat", default=1, type=int, help="Number of three-level systems per device [default=1, max=num_qubits/4]") 
    parser.add_argument("-tups", default="opt", type=str, help="File containing initial tUPS parameters [default=opt]")
    parser.add_argument("-nphoton", default=2, type=int, help="Number of photons to create in the circuit [default=2]") 
    args = parser.parse_args()

    # Define the device
    dev = Device(backend=args.backend, noiseless=args.noiseless, hardware=args.hardware,
                 default_shots=args.nshot, optimization_level=3, seed=args.seed)
    if args.repeat < 0:
        nrepeat = dev.num_qubits//4
    else:
        nrepeat =min(args.repeat, dev.num_qubits//4)

    # Set numpy random seed
    np.random.seed(args.seed)
    # Define the path we're going to follow
    path = SinNPath(wmax=args.wmax,lmax=args.lmax,order=3)
    # Total time and timestep
    nstep = int(np.ceil(args.T / args.dt))
    imax = nstep

    # Print information about path and steps with # formatting
    print(f"#-----------------------------------------------------------")
    print(f"# Hubbard dimer three-level simulation using IBM Quantum backend     ")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#-----------------------------------------------------------")
    print(f"# Total evolution time     = {args.T}")
    print(f"# Time step                = {args.dt}")
    print(f"# Number of steps          = {nstep}")
    print(f"# Maximum photon frequency = {args.wmax}")
    print(f"# Maximum photon coupling  = {args.lmax}")
    print(f"# Number of shots          = {args.nshot}")
    print(f"# Noiseless                = {args.noiseless}")
    print(f"# Hardware                 = {args.hardware}")
    print(f"# Measure path             = {args.measure_path}")
    print(f"# Number of repeats        = {nrepeat}")
    print(f"# {path}") 
    print(f"#-----------------------------------------------------------")
    print(f"# Hubbard dimer three-level parameters:")
    print(f"# U = {args.U}, t = {args.t}, mu = {0.5}")  # Example parameters for ThreeLevelCircuit
    print(f"#-----------------------------------------------------------")
    sys.stdout.flush()

    with open(args.fname,'w') as outF:
        outF.write(f'#      s        <E>_total      <E>_postselect       <N>_photon        Var(E)_total   Var(E)_postselect   Var(N)_photon\n')

    istart = 0 if (args.measure_path) else nstep
    for imax in range(istart,nstep+1):
        # Define the system
        system = HubbardCircuit(U=args.U,t=args.t,d=0.5,repeat=nrepeat)
        system.initialise(np.genfromtxt('opt'))
        system.create_photon(nphoton=args.nphoton)

        # Loop over steps up to imax
        for it, sk in enumerate(np.linspace(0,imax/nstep,imax+1)):
            # Geth the current parameters and take time step
            wk = path.w(sk)
            lk = path.l(sk)
            system.time_evolve(args.dt,wk,lk)
      
        Et, E, pn = system.postselect_energy(dev)
        with open(args.fname,'a') as outF:
            outF.write(f'{sk: 10.6f} {Et[0]: 16.10f}  {E[0]: 16.10f}  {pn[0]: 16.10f}   {Et[1]: 16.10f}  {E[1]: 16.10f}  {pn[1]: 16.10f}\n')  
            
        print()
        print(f"#-----------------------------------------------------------")
        print(f"# Final circuit output after {imax:5d} steps:               ")
        print(f"#-----------------------------------------------------------")
        print(f"#   Electron total energy      = {Et[0]: 16.10f}  ({Et[1]:8f})")
        print(f"#   Postselect total energy    = {E[0]: 16.10f}  ({E[1]:8f})")
        print(f"#                                                           ")
        print(f"#   Final photon number        = {pn[0]: 16.10f}  ({pn[1]:8f})")
        print(f"#                                                           ")
        print(f"#   Thermal state energy       = {args.U/2: 16.10f}     ")
        print(f"#   Thermal state photon num   = {1.5: 16.10f}              ")
        print(f"#-----------------------------------------------------------")
        sys.stdout.flush()
