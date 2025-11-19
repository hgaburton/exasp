# Excited Adiabatic State Preparation (EXASP)

This repository holds the code required to perform EXASP simulations, as
described in [INSERT PREPRINT HERE].

**Contributors:**
* Hugh Burton 
* Maria-Andreea Filip

All queries should be directed to Dr Hugh Burton (hgaburton[AT]gmail.com)

## Pre-requisites
EXASP uses the Quantel library to perform HF calculations. This can be downloaded and installed from
```
https://github.com/hgaburton/quantel
```
You may need to add the quantel source to the PYTHONPATH variable.
It is suggested that you use the conda environment defined in the Quantel repository.

Also requires Qiskit software to perform quantum simulations. This can be downloaded and installed from
```
pip install qiskit qiskit-nature qiskit-aer qiskit-ibm-runtime
```

## Functionality

Relevant scripts and modules can be found in `./exasp/` and
example inputs and outputs can be found in `./examples/`.

The functionality of each script is as follows:

1. `exact_coupling.py`:  
Compute exact eigenstates for molecular system along EXASP adiabatic pathway.
3. `exact_hubbard_coupling.py`:  
Compute exact eigenstates for Hubbard lattice along EXASP adiabatic pathway.
4. `exact_time_evolution.py`:  
Perform exact EXASP time propagation for a molecular system analytically.
5. `quantum_time_evolution.py`:  
Perform exact or Trotterised EXASP time propagation for a molecular system.
7. `quantum_hubbard_time_evolution.py`:  
Perform exact or Trotterised EXASP time propagation for a Hubbard lattice.
9. `twolevel_ibm.py`:  
Perform quantum circuit simulation for EXASP time propagation for a two-level model
Hamiltonian using noiseless statevector simulations, noisy simulations, or on real quantum hardware
11. `hubbard_dimer_ibm.py`:  
Perform quantum circuit simulation for EXASP time propagation for the Hubbard dimer
using noiseless statevector simulations, noisy simulations, or on real quantum hardware

Running the script with the `--help` command line argument will show the options available.
Example command line arguments and reference output files are availabe in `./examples/'.

