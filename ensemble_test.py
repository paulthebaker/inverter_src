
import sys
import numpy as np
import emcee
from emcee.utils import MPIPool

log2P = np.log(2.0*np.pi)

def logPDF(x, M, sigL, sigP):
    n = M.shape[0]
    G = x.reshape(n,n).copy()
    Id = np.identity(n,float)

    R = np.dot(M,G) - Id
    logL = -0.5*( R.size*(log2P+np.log(sigL)) + np.sum(R*R)/(sigL*sigL) )
    logP = -0.5*( G.size*(log2P+np.log(sigP)) + np.sum(G*G)/(sigP*sigP) )

    logPDF = logL + logP
    return logPDF


mat = np.array([[1, 2, 3, 4],
                [2, 0, 1, 2],
                [3, 1, 3, 0],
                [4, 2, 0, 1]])

sigL = 0.1
sigP = 10.0

nwalkers = 250
ndim = mat.size

x0 = np.random.normal(size=nwalkers*ndim).reshape((nwalkers, ndim))

print(x0)

burn = 100
N = 1000

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

# initialize MCMC
MCMC = emcee.EnsembleSampler(nwalkers, ndim, logPDF, args=[mat, sigL, sigP])

# burn in
pos, prob, state = MCMC.run_mcmc(x0, burn)
MCMC.reset()

# actual run
MCMC.run_mcmc(pos, N)

pool.close()

import matplotlib.pyplot as pl

print("Mean acceptance fraction: {0:.3f}"
                        .format(np.mean(MCMC.acceptance_fraction)))

for i in range(ndim):
        pl.figure()
        pl.hist(MCMC.flatchain[:,i], 100, color="k", histtype="step")
        pl.title("Dimension {0:d}".format(i))

        pl.show()

