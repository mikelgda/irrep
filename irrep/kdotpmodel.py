import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KdotPModel:
    """
    Given a space group and a maximal kpoint of the BZ, creates a k.p model representation
    whose eigenstates belong to a given representation and for the Hamiltonian supplied.
    Includes common operations with this kind of model:
        - Plotting the energies along one k-line
        - Computing the eigenvalues and eigenstates at a given kpoint
        - Computing the traces of the representation
        - Finding the irrep decomposition of the eigenstates after a symmetry-breaking
          perturbation is introduced, given the target space group and kpoint. (WIP)
    """
    def __init__(self, symgroup, kpoint, kpname, rep, h, p, user_defined = None, hermitian = True):
        """
        Define the space symmetry group and the specific kpoint of the expansion.
        The small irreps may be loaded from the tables or user defined. In that case
        'user_defined' should contain the path to the text file containing the operation
        names and their matrix representation (see '_load_custom').
        Otherwise, rep should be a list (even if only one present) of the standard irrep names. When
        more than one name is given, this indicates the direct sum of the representations.
        'hamiltonian' is a matrix function of the same dimension as 'rep' which is expected
        to have two arguments: a 3-tuple of the kpoint and a tuple 'p' of the parameters of the model.
        'p' is supplied at the moment of creation of the KdotPModel instance.

        The model is implicitly assumed to be Hermitian. If not, set 'hermitian=False' by keyword argument.
        """
        self.symgroup = symgroup
        self.kpoint = kpoint
        self.kpname = kpname
        self.rep = rep
        self.hamiltonian = h
        self.parameters = p
        self.hermitian = hermitian
        
        if user_defined:
            self.opnames, self.matrices = self._load_custom(user_defined)
        else:
            self.opnames, self.matrices = self._load_irreps()
    
    def _load_irreps(self):
        """
        Loads small irrep matrices from the standard tables of the BCS.
        Returns an array of Seitz symbols and their corresponding matrices, in the same
        order.
        """
        
        def build_matrix(string):
            string = string.split()
            dim = int(np.sqrt(len(string)//2))
            l = len(string)//2
            mod = np.array(string[:l], dtype = 'float32').reshape((dim, dim))
            arg = np.exp(1j * np.pi * np.array(string[l:], dtype = 'float32')).reshape(dim,dim)
            return mod * arg
        
        irreptable=pd.read_csv('./tables/group_{0}/point_{1}.txt'.format(self.symgroup, self.kpname),
                               index_col = 0, dtype = str)
        irreptable['single'] = ~irreptable["symbol"].str.contains('d')
        irreptable['unit'] = ~irreptable['symbol'].str.contains('\'')
        table = irreptable[irreptable['single']]
        table = table[table['unit']]
        irrepmats = []
        for irname in self.rep:
            irseries = irreptable[irname]
            irmats = np.array([build_matrix(s) for s in irseries])
            irrepmats.append(irmats)
        return table['symbol'].values, irrepmats
    
    def _load_custom(self,path):
        """
        Loads a user defined representation. The file should be formatted as follows
            1. Names for the operations, space-separated
            2. Operations in matrix form, e.g.

                    0 1j 0
                    -1j 0 0
                    0 0 1
                Matrices should be separated by a line break.
        """
        def tokenize(lines):
            chunk = []
            nl = len(lines)
            for i in range(nl):
                if lines[i] == '\n':
                    yield chunk
                    chunk = []
                    continue
                chunk.append(lines[i].split())
                if i == nl-1:
                    yield chunk
        
        with open(path,'r') as infile:
            lines = infile.readlines()
        
        opnames = np.array(lines[0].split(), dtype = str )
        mats = []
        
        for chunk in tokenize(lines[2:]):
            mats.append(chunk)
        
        return opnames, np.array(mats, dtype = complex)
    
    def plot_line(self, kstart = (0,0,-1), kend = (0,0,1), npoints = 100):
        """
        Plots the bands along a line given by evenly spaced values between the 3-tuples
        kstart and kend.
        """
        
        kline = np.linspace(kstart,kend,npoints)
        if self.hermitian:
            energies = np.array([np.sort(np.linalg.eigvalsh(self.hamiltonian(kpoint,self.parameters))) for kpoint in kline])
        else:
            energies = np.array([np.sort(np.linalg.eigvalsh(self.hamiltonian(kpoint,self.parameters))) for kpoint in kline])
        
        plt.plot(energies, color = 'red')
    
    def energies(self, kpoint=(0,0,0)):
        if self.hermitian:
            return np.linalg.eigvalsh(self.hamiltonian(kpoint,self.parameters))
        else:
            return np.linalg.eigvals(self.hamiltonian(kpoint,self.parameters))

    def eigenstates(self,kpoint=(0,0,0)):
        if self.hermitian:
            return np.linalg.eigh(self.hamiltonian(kpoint,self.parameters))
        else:
            return np.linalg.eig(self.hamiltonian(kpoint,self.parameters))
    
    def traces(self,kpoint=(0,0,0)):
        """
        Computes the eigenvectors at a given kpoint and returns the traces of
        the operations, in the same order as in 'self.matrices' or 'self.opnames'.
        """
        states=self.eigenstates(kpoint)[1]
        traces = []
        for matrix in self.matrices:
            traces.append(self._compute_trace(states,matrix))
        return np.array(traces)
    
    def _compute_trace(self,vecs,operation):
        return np.trace(np.transpose(vecs) @ operation @ vecs)
            

            
        
        
        