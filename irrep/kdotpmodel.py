import pandas as pd
import numpy as np


class KdotPModel:
    def __init__(self, symgroup, kpoint, kpname, rep, h, user_defined = None):
        self.symgroup = symgroup
        self.kpoint = kpoint
        self.kpname = kpname
        self.rep = rep
        self.h = h
        
        if user_defined:
            self.opnames, self.matrices = self._load_custom(user_defined)
        else:
            self.opnames, self.matrices = self._load_irreps()
    
    def _load_irreps(self):
        
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
        
        
        