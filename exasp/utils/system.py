""" Module for defining systems and integrals in EXASP using Quantel"""
import numpy as np
from scipy.sparse import csc_matrix
import quantel
from quantel.wfn.rhf import RHF
from quantel.opt.diis import DIIS
from quantel.utils.linalg import orthogonalise

class System:
    """ Class to hold information about an electron-photon system """
    def __init__(self, nphoton, eps, nalfa, nbeta, nfrozen, nactive, mo_coeff=None, dse=True):
        """ Initialise an electron-photon system
            
            Args: 
                nphoton:  number of photon levels to include
                eps:      polarization vector (3 elements)
                nalfa:    number of active alpha electrons
                nbeta:    number of active beta electrons
                nfrozen:  number of frozen orbitals
                nactive:  number of active orbitals
                mo_coeff: orbital coefficients (if None, run RHF to get them)
                dse:      include dipole self-interaction (True) or not (False). [Default=True]
        """
        # Save parameters
        self.nphoton = nphoton
        assert(len(eps) == 3)
        self.eps = np.array(eps)

        # Save information about number of electrons and basis functions
        self.nalfa = nalfa
        self.nbeta = nbeta
        self.nfrozen = nfrozen
        self.nmo = nactive

        # Include dipole self-interaction (True) or not (False)
        self.dse = dse

        # Sanity check
        if(self.nfrozen > self.ints.nmo() or self.nfrozen < 0):
            raise ValueError(f"Number of frozen orbitals ({self.nfrozen}) is invalid")
        if(self.nmo > self.ints.nmo()-self.nfrozen or self.nmo < 0):
            raise ValueError(f"Number of active orbitals ({self.nmo}) is invalid")
        if(self.nalfa < 0 or self.nbeta < 0):
            raise ValueError(f"Number of active electrons ({self.nalfa},{self.nbeta}) is invalid")

        # Initialise with coefficients (default is to run RHF)
        self.set_coefficients(coeff=mo_coeff)


    def set_coefficients(self,coeff=None):
        """Set the orbital coefficients for the electronic system
        
            Args:
                coeff: orbital coefficients (if None, run RHF to get them)
        """
        if(coeff is None):
            # If no coefficients provided, run a HF calculation
            self.wfn = RHF(self.ints)
            self.wfn.get_orbital_guess("gwh")
            DIIS().run(self.wfn, plev=0)
            self.coeff = self.wfn.mo_coeff.copy()
        else:
            # Check the dimensions of input coefficients
            (nr,nc) = coeff.shape
            nbsf, nmo = self.ints.nbsf(), self.ints.nmo()
            if (nr != nbsf) or (nc != nmo):
                raise ValueError(f"Shape coeff ({nr},{nc}) doesn't match expected ({nbsf,nmo})")

            # Orthogonalise input coefficients
            self.coeff = orthogonalise(coeff, self.ints.overlap_matrix())

        # Update integrals
        self.setup_integrals()
        # Setup photon matrices
        self.setup_photon_matrices(self.nphoton)
        # Setup coupling matrices
        self.setup_coupling()

    def setup_integrals(self):
        """Setup molecule and integrals"""
        # Construct MO integral object
        self.mo_ints = quantel.MOintegrals(self.ints)
        self.mo_ints.update_orbitals(self.coeff,self.nfrozen,self.nmo)
        # Construct dipole integrals in MO basis
        self.dip_mo = self.mo_ints.dipole_matrix(True)

        # Build the FCI space
        self.cispace = quantel.CIspace(self.mo_ints,self.nmo,self.nalfa,self.nbeta)
        self.cispace.initialize('FCI')

        # Identity matrix
        self.Im = np.eye(self.cispace.ndet())
        # Compute dipole matrix and Hamiltonian in FCI basis
        self.Hm = self.cispace.build_Hmat()
        # Get the requested dipole component
        self.Dm = np.einsum('x,xpq->pq',self.eps,self.cispace.build_Dmat())
        
    def setup_photon_matrices(self, nphoton):
        """ Setup photon matrices
         
            Args:
                nphoton: number of photon levels to include
        """
        # Identity matrix
        self.Ip = np.eye(nphoton+1)
        # Raising operator
        self.Ap = np.zeros((nphoton+1,nphoton+1))
        for i in range(nphoton):
            self.Ap[i,i+1] = np.sqrt(i+1)
        # Photon x operator
        self.Xp = self.Ap + self.Ap.T
        # Photon Hamiltonian
        self.Hp = self.Ap.T @ self.Ap

    def setup_coupling(self):
        """Compute the coupling matrices in direct product basis as sparse matrices"""
        # System Hamiltonian
        self.Hm_c = csc_matrix(np.kron(self.Ip, self.Hm))
        # Photon Hamiltonian
        self.Hp_c = csc_matrix(np.kron(self.Hp, self.Im))
        # Coupling
        self.Hi_c = csc_matrix(np.kron(self.Xp, self.Dm))
        # Dipole self-interaction
        self.Hd_c = csc_matrix(np.kron(self.Ip, self.Dm @ self.Dm))

    def solve_molecular_hamiltonian(self):
        """Solve the molecular Hamiltonian"""
        return np.linalg.eigh(self.Hm)
    
    def H(self,w,l):
        """Return Hamiltonian for given w and l"""
        if(self.dse):
            return self.Hm_c + w*self.Hp_c - l*np.sqrt(0.5*w)*self.Hi_c + 0.5 * l**2 * self.Hd_c
        else:
            return self.Hm_c + w*self.Hp_c - l*np.sqrt(0.5*w)*self.Hi_c
    
    def dH(self,w,l,dw,dl):
        """Return derivative of Hamiltonian with respect to w and l"""
        dHw = self.Hm_c + l * 0.5 * np.sqrt(0.5 / w) * self.Hi_c
        dHl = np.sqrt(0.5 * w) * self.Hi_c + l * self.Hd_c
        if(self.dse):
            return dw * dHw + dl * dHl
        else:
            return dw * dHw


class MolecularSystem(System):
    """Class to hold information about a molecular system"""
    def __init__(self, xyz, basis, nphoton, eps, nfrozen=None, nactive=None, mo_coeff=None, dse=True):
        """ Initialise the system
        
            Args:
                xyz:       Name of .xyz file containing molecular geometry
                basis:     basis set name
                nphoton:   number of photon levels to include
                eps:       polarization vector (3 elements)
                nfrozen:   number of frozen orbitals (if None, no frozen orbitals)
                nactive:   number of active orbitals (if None, all orbitals active)
                mo_coeff:  orbital coefficients (if None, run RHF to get them)
                dse:       include dipole self-interaction (True) or not (False). [Default=True]
        """
        # Save information about molecule and basis
        self.basis = basis
        self.xyz   = xyz

        # Setup quantel objects
        self.mol  = quantel.Molecule(self.xyz, "angstrom")
        self.ints = quantel.LibintInterface(self.basis, self.mol)

        # Setup active space and electrons
        nfrozen = 0 if (nfrozen is None) else nfrozen
        nactive = self.ints.nmo()-nfrozen if (nactive is None) else nactive
        nalfa = self.mol.nalfa() - nfrozen
        nbeta = self.mol.nbeta() - nfrozen
        
        # Call parent constructor
        super().__init__(nphoton, eps, nalfa, nbeta, nfrozen, nactive, mo_coeff=mo_coeff, dse=dse)

class HubbardSystem(System):
    """Class to hold information about a Hubbard model"""
    def __init__(self,U,t,nphoton, eps, ne:list, dim:list, periodic:list=[False,False,False], mo_coeff=None, dse=True):
        """Initialise the Hubbard system
        
            Args:
                U:         on-site interaction
                t:         hopping parameter
                nphoton:   number of photon levels to include
                eps:       polarization vector (3 elements)
                ne:        list with number of electrons [nalfa, nbeta]
                dim:       list with dimensions of the lattice [nx, ny, nz]
                periodic:  list with periodicity flags for each dimension [px, py, pz]
                mo_coeff:  orbital coefficients (if None, run RHF to get them)
                dse:       include dipole self-interaction (True) or not (False). [Default=True]
        """
        # Save parameters
        self.t = t
        self.U = U
        # Check the input
        if(len(dim) != 3): 
            raise ValueError("HubbardSystem requires at least 3 dimensions")
        if(len(periodic) != 3): 
            raise ValueError("HubbardSystem requires at least 3 periodicity flags")
        
        # Get the numebr of electrons
        if(len(ne) != 2):
            raise ValueError("HubbardSystem requires ne to take form [na,nb]")
        nalfa, nbeta = ne[0], ne[1]
        
        # Initialise the Hubbard system
        self.ints = quantel.HubbardInterface(U,t,nalfa,nbeta,dim,periodic)

        # Call parent constructor
        super().__init__(nphoton,eps,nalfa,nbeta,0,self.ints.nmo(), mo_coeff=mo_coeff, dse=dse)
