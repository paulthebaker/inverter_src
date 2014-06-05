
# types.py
# classes for mat_inv.py
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>

import numpy as np

# Root epsilon for numerical derivatives
reps = np.sqrt(np.finfo(np.float64).eps)

class Fisher:
    """fisher matrix with eigen values and vectors"""
    mat = []
    val = []
    vec = []

    def update(self, M, G):
        """compute Fisher Info matrix: Fij = (h,i|h,j)"""
        # TODO Fisher is symmetric ... only need lower triangle...
        n = G.shape[0]
        dh = np.zeros([n*n,n,n])
        self.mat = np.zeros([n*n,n*n])
        # compute derivatives of H
        for i in range(n):
            for j in range(n):
                # establish good choice for dg
                if( G[i,j]!=0 ):
                    tmp = reps*G[i,j]
                else:
                    tmp = reps
                dg = (tmp+G[i,j]) - G[i,j]
                dG = np.zeros_like(G)
                dG[i,j]=dg
                # H = np.dot(M,G)
                dh[i*n+j,:,:] = 0.5*( np.dot(M,G+dG) - np.dot(M,G-dG) ) / dg
        # components of F
        for j in range(n*n):
            for i in range(j,n*n):
                self.mat[i,j] = sum(sum( dh[i,:,:]*dh[j,:,:] )) # inner product
        # compute eigen-stuff for Fij use lower triangle
        self.val, tmp = np.linalg.eigh(self.mat, UPLO='L')
        self.vec = tmp.reshape([n*n,n,n])
        
    def __init__(self,M,G):
        """initialize Fisher matrix"""
        self.update(M,G)

