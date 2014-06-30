#!/usr/bin/env python3.3

#
#    mcmc_mat_inv.py
#
# a Markov chain Monte Carlo for inverting square matrices
#   an ongoing exercise in Python 
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>
#

"""
the likelihood for the MCMC goes like exp(-R.R/2) where R is the residual S-H
our data, S, is the identity matrix
our model, H, is the matrix product of the input matrix M and the current guess G
each matrix is treated like a vector of the N*N entries
"""

import sys
import mmi_lib as mmi

import time
import numpy as np

import emcee
from emcee.utils import MPIPool


def logPDF(x, M, sigL, sigP):
    """compute log posterior from 1xn^2 param vector"""
    n = M.shape[0]
    G = x.reshape(n,n).copy()
    Id = np.identity(n,float)

    R = np.dot(M,G) - Id
    logL = -0.5 * np.sum(R*R)/(sigL*sigL)
    logP = -0.5 * np.sum(G*G)/(sigP*sigP)

    logPDF = logL + logP
    return logPDF


##### BEGIN MAIN #####
infile_name, outfile_name, number, burn, walk, SEED, MPI = mmi.io.parse_options(sys.argv[1:])

np.random.seed(SEED)
ranS = np.random.get_state()

rawM = mmi.io.get_Mat(infile_name)
detM = np.linalg.det(rawM)
if(detM == 0):
    print("ERROR: det[M]=0, cannot invert")
    exit(2) 
n = rawM.shape[0] # matrix is nxn
scale = np.power(detM,-1.0/n)
M = scale*rawM.copy()  # scale so det(M)=1

N = number   # number o' MCMC samples
Nwalkers = walk
Ndim =M.size
acc = 0

sigL = 0.1
sigP = 3.0

# draw starting location from prior
x0 = np.random.normal(scale=sigP, size=Nwalkers*Ndim).reshape((Nwalkers, Ndim))

# initialize MPI pool for parallelization
if MPI:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

# initialize ensemble MCMC
eMCMC = emcee.EnsembleSampler(Nwalkers, Ndim, logPDF, args=[M, sigL, sigP])#, rstate=ranS)

t_start = time.clock()

# burn in
pos, prob, state = eMCMC.run_mcmc(x0, burn)
eMCMC.reset()

# actual run
eMCMC.run_mcmc(pos, N)
>>>>>>> use ensemble sampler

t_end = time.clock()

# close MPI pool
if MPI:
    pool.close()

# get median for each param
samp = eMCMC.chain.reshape((-1, Ndim))
result = np.array( list( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
              zip(*np.percentile(samp, [16, 50, 84], axis=0))) ))
med = result.reshape(3,Ndim)[0,:]
plus = result.reshape(3,Ndim)[1,:]
minus = result.reshape(3,Ndim)[2,:]

# write chain to file
mmi.io.print_chain(samp, 'chain.dat')

# write rescaled Minv to file
Minv = med.reshape(n,n)
MinvP = plus.reshape(n,n)
MinvM = minus.reshape(n,n)

rawMinv = scale*Minv.copy()
mmi.io.print_Mat(rawMinv, outfile_name)

# print stuff to command line
print("")
print("MCMC runtime: %.4f sec"%(t_end-t_start))
print("")
print("Mean acceptance: {0:.4f}"
                        .format(np.mean(eMCMC.acceptance_fraction)))
print("")

np.set_printoptions(precision=4)
print("M =") 
print(rawM)
print("")

I = np.dot(rawM,rawMinv)
print("M*Minv =")
print(I)
print("")

print("Minv =") 
print(rawMinv)
print("")

MinvTRUE = np.linalg.inv(rawM)
print("Minv TRUE =") 
print(MinvTRUE)
print("")

# TODO: fast fitting factor computation assumes no noise in data
HtHt = rawM.shape[0]
HmHm = np.sum( I*I )
HtHm = np.trace(I)
FF = HtHm/np.sqrt(HtHt*HmHm)
print("fitting factor = %.4f"%(FF))
print("")


