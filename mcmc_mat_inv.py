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
import matplotlib.pyplot as plt

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
        dev = np.random.normal(scale=sig, size=Nwalkers*Ndim).reshape((Nwalkers,Ndim))
        x0 = np.tile(x01, [Nwalkers, 1]) + dev
        x0 /= scale
else:
    # draw starting location from prior
	x0 = np.random.normal(scale=sigP, size=Nwalkers*Ndim).reshape((Nwalkers, Ndim))
	x01 = x0[0].reshape((n,n))

# take scaled identity matrix as covariance
cov = (sigL/Ndim**2)*np.identity(Ndim)


### Ensemble Method ###

# initialize MPI pool for parallelization
if opts.mpi:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

# initialize ensemble MCMC
eMCMC = emcee.EnsembleSampler(Nwalkers, Ndim, mmi.prop.log_PDF, args=[M, sigL, sigP])#, rstate=ranS)

tstart = time.clock()

# for this version, it will incrementally save the progress of the sampler to an external chain file instead of devouring RAM

# burn in
pos, prob, state = eMCMC.run_mcmc(x0, B, storechain=False)
eMCMC.reset()

# actual run
logchain = open('ensemble_log_chain.dat','w')
matchain = open('ensemble_mat_chain.dat','w')

dum_dum = []

for i in range(N):
	for step in eMCMC.sample(pos, iterations=1, storechain=False):
		position = step[0]
		for k in range(position.shape[0]):
			for j in range(position.shape[1]):
				val = position[k,j] * scale
				matchain.write("%f  " % val)
			matchain.write("\n")
		
		log = step[1]
		logchain.write("%f \n" % log[0])
		
		dum_dum = position
	pos = dum_dum

logchain.close()
matchain.close()

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()

#read in matrix chain file
chainmat = np.loadtxt('ensemble_mat_chain.dat').reshape((Nsamp,Ndim))
quarters = np.array_split(chainmat,4)

# get median and uncert
med = np.array(np.percentile(chainmat, [50], axis=0))
plus = np.array(np.percentile(chainmat, [84], axis=0)) - med

# write rescaled Minv
rawMinv = med.reshape(n,n)
rawMinvP = plus.reshape(n,n)

# obtain autocorrelation time from chainmat
autocor = emcee.autocorr.integrated_time(chainmat, axis=0)

# print stuff to command line
acc = np.mean(eMCMC.acceptance_fraction)
runtime = tend-tstart
print("The ensemble sampler yielded the following result:")
mmi.io.print_endrun(rawM, rawMinv, rawMinvP, runtime, acc, autocor)



### Metropolis-Hastings Method ###

# initialize MPI pool for parallelization
if opts.mpi:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

#for MH sampler, redefine run parameters to make the number of iterations equal
N *= Nwalkers
B *= Nwalkers

# initialize Metropolis-Hastings MCMC
mhMCMC = emcee.MHSampler(cov, Ndim, mmi.prop.log_PDF, args=[M, sigL, sigP])

tstart = time.clock()

# burn-in
x0 = x01.reshape(1,Ndim)
pos, prob, state = mhMCMC.run_mcmc(x0[0], B)
mhMCMC.reset()

# actual run
logchain = open('mh_log_chain.dat','w')
matchain = open('mh_mat_chain.dat','w')

dum_dum = []

for i in range(N):
	for step in mhMCMC.sample(pos, iterations=1, storechain=False):
		position = step[0]
		for k in range(position.shape[0]):
			val = position[k] * scale
			matchain.write("%f  " % val)
		matchain.write("\n")
		
		if i % Nwalkers == 0:
			log = step[1]
			logchain.write("%f \n" % log)
		
		dum_dum = position
	pos = dum_dum

logchain.close()
matchain.close()

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()

#read in matrix chain file
chainmat = np.loadtxt('mh_mat_chain.dat').reshape((Nsamp,Ndim))
quarters = np.array_split(chainmat,4)

med = np.array(*np.percentile(chainmat, [50], axis=0))
plus = np.array(*np.percentile(chainmat, [84], axis=0)) - med

# write rescaled Minv
rawMinv = med.reshape(n,n)
rawMinvP = plus.reshape(n,n)

# obtain autocorrelation time from chainmat
autocor = emcee.autocorr.integrated_time(chainmat, axis=0)

# print stuff to command line
acc = np.mean(mhMCMC.acceptance_fraction)
runtime = tend-tstart
print("The Metropolis-Hastings sampler yielded the following result:")
mmi.io.print_endrun(rawM, rawMinv, rawMinvP, runtime, acc, autocor)