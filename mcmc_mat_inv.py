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


# global values... add to CmdLineOpts?
sigL = 0.1
sigP = 3.0

##### BEGIN MAIN #####
opts = mmi.io.CmdLineOpts(sys.argv[1:])

np.random.seed(opts.seed)
ranS = np.random.get_state()

rawM = mmi.io.get_Mat(opts.inputfile)
detM = np.linalg.det(rawM)
n = rawM.shape[0] # matrix is nxn

if detM == 0:
    print("ERROR: det[M]=0, cannot invert")
    exit(2)

scale = np.power(np.abs(detM),-1.0/n)
M = scale*rawM.copy()  # scale so det(M)=1

N = opts.number   # number o' MCMC steps (1 sample per walker)
B = opts.burn     # number o' burned steps
Nwalkers = opts.walk
Ndim = M.size

Nsamp = N * Nwalkers # total number of samples in chain

# get starting location
if opts.guessfile:
    x01 = mmi.io.get_Mat(opts.guessfile)
    if x01.size != rawM.size:
        print("ERROR: inverse guess must be same size as mat_in")
        exit(2)
    else:
        x01 = x01.reshape(1, Ndim)
        sig = np.amax(x01)*0.01 # random deviation is 1% of maximum element
        dev = np.random.normal(scale=sig,
                               size=Nwalkers*Ndim).reshape((Nwalkers,Ndim))
        x0 = np.tile(x01, [Nwalkers, 1]) + dev
        x0 /= scale
else:
    # draw starting location from prior
    x0 = np.random.normal(scale=sigP,
                          size=Nwalkers*Ndim).reshape((Nwalkers, Ndim))

# initialize MPI pool for parallelization
if opts.mpi:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

# initialize ensemble MCMC
eMCMC = emcee.EnsembleSampler(Nwalkers, Ndim, mmi.prop.log_PDF, args=[M, sigL, sigP])#, rstate=ranS)

tstart = time.clock()

# burn in
pos, prob, state = eMCMC.run_mcmc(x0, B)
eMCMC.reset()

# actual run
eMCMC.run_mcmc(pos, N)
>>>>>>> use ensemble sampler

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()

# get median for each param
samp = eMCMC.flatchain
post = (eMCMC.flatlnprobability).reshape(Nsamp, 1)
result = np.array( list( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
              zip(*np.percentile(samp, [16, 50, 84], axis=0))) ))

ind_MAP = np.argmax(post)
MAP = samp[ind_MAP,:]    # get maximum a posteriori... unused...
med = np.array(*np.percentile(samp, [50], axis=0))
plus = np.array(*np.percentile(samp, [84], axis=0)) - med
minus = med - np.array(*np.percentile(samp, [84], axis=0))

chain = np.hstack( (post, samp) )
# write chain to file
mmi.io.print_chain(chain, opts.chainfile)

# write rescaled Minv to file
rawMinv = scale*med.reshape(n,n)
rawMinvP = scale*plus.reshape(n,n)
rawMinvM = scale*minus.reshape(n,n)

mmi.io.print_Mat(rawMinv, opts.outputfile)

# print stuff to command line
acc = np.mean(eMCMC.acceptance_fraction)
runtime = tend-tstart

mmi.io.print_endrun(rawM, rawMinv, runtime, acc)
