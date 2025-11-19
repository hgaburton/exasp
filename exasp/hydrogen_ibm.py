from path import SinNPath
from ibmq.hydrogen import HydrogenCircuit
from ibmq.device import Device
import numpy as np
import sys
from argparse import ArgumentParser

if __name__ == "__main__":
        # Read arguments
    parser = ArgumentParser()
    parser.add_argument("-noiseless", action='store_true', help="Perform noiseless simulation")
    parser.add_argument("-hardware", action='store_true', help="Perform simulation on QPU")
    parser.add_argument("-nshot", default=1024, type=int, help="Number of shots for sampling")
    parser.add_argument("-seed", default=None, type=int, help="Random seed for sampling")
    parser.add_argument("-fname", default="path.txt", type=str, help="Name for path output file")
    parser.add_argument("-backend", default="ibm_torino", type=str, help="Choice of Qiskit QPU backend")
    parser.add_argument("-T", default=20, type=float, help="Total evolution time")
    parser.add_argument("-dt", default=0.1, type=float, help="Time step")
    args = parser.parse_args()

    # Define the device
    dev = Device(backend=args.backend, noiseless=args.noiseless, hardware=args.hardware,
                 default_shots=args.nshot, optimization_level=3, seed=args.seed)
    np.set_printoptions(linewidth=1000)

    # Define the path we're going to follow
    path = SinNPath(wmax=1,lmax=0.1,order=3)
    # Total time and timestep
    nstep = int(np.ceil(args.T / args.dt))
    imax = nstep
    print(f"Number of steps: {nstep}")

    # Define the system
    system = HydrogenCircuit()
    system.create_photon()

    # Loop over steps up to imax
    for it, sk in enumerate(np.linspace(0,imax/nstep,imax+1)):
        # Geth the current parameters and take time step
        wk = path.w(sk)
        lk = path.l(sk)
        system.time_evolve(args.dt,wk,lk)

        # Periodically evaluate circuit to get a path for visualisation
        #if(1): #it % 10 == 0):
        #res1 = dev.run_circuit(system.circuit, system.Hfull(wk,lk))
        #res2 = dev.run_circuit(system.circuit, system.Hm(0))
        #print(f"{sk:10.6f} {res1[0].data.evs:16.10f} {res2[0].data.evs:16.10f}")
        #sys.stdout.flush()
  
    res1 = dev.run_circuit(system.circuit, system.Hfull(wk,lk))
    res2 = dev.run_circuit(system.circuit, system.Hm(0))
    Et, E, pn, wt, w = system.postselect_energy(dev)

    print()
    print(f"#-----------------------------------------------------------")
    print(f"# Final circuit output after {imax:5d} steps:               ")
    print(f"#-----------------------------------------------------------")
    print(f"#   Electron total energy      = {Et: 16.10f}               ")
    print(f"#   Postselect total energy    = {E: 16.10f}                ")
    print(f"#                                                           ")
    print(f"#   Excited state fidelity     = {wt: 16.10f}               ")
    print(f"#   Postselect fidelity        = {w: 16.10f}                ")
    print(f"#                                                           ")
    print(f"#   Final photon number        = {pn: 16.10f}               ")
    print(f"#                                                           ")
    print(f"#   Thermal state energy       = {0.5*system.eI: 16.10f}     ")
    print(f"#   Thermal state photon num   = {0.5: 16.10f}              ")
    print(f"#-----------------------------------------------------------")
    sys.stdout.flush()
