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

#obtain covariance matrix for this starting guess
F = mmi.types.Fisher(M, x01)
cov = np.linalg.inv(F.mat)

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
logchain = open(opts.logchainfile,'w')
matchain = open(opts.matchainfile,'w')

dum_dum = []

for i in range(N):
	for step in eMCMC.sample(pos, iterations=1, storechain=False):
		position = step[0]
		for k in range(position.shape[0]):
			for j in range(position.shape[1]):
				matchain.write("%f  " % position[k,j])
			matchain.write("\n")
		
		m = position[0]
		log = mmi.prop.log_PDF(m, M, sigL, sigP)
		logchain.write("%f \n" % log)
		
		dum_dum = position
	pos = dum_dum

logchain.close()
matchain.close()

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()


# make convergence plot for ensemble
print("Done with run, making plot")
post = np.loadtxt(opts.logchainfile)

plt.figure(1)
plt.title('Chain plot for logPost')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.xlim([0,N])
plt.savefig('ensemble_logPost_chain_plot.pdf')

#read in matrix chain file
chainmat = np.loadtxt(opts.matchainfile).reshape((Nsamp,Ndim))

# get median and uncert
med = np.array(*np.percentile(chainmat, [50], axis=0))
plus = np.array(*np.percentile(chainmat, [84], axis=0)) - med
minus = med - np.array(*np.percentile(chainmat, [84], axis=0))

# write rescaled Minv
rawMinv = scale*med.reshape(n,n)
#rawMinvP = scale*plus.reshape(n,n)
#rawMinvM = scale*minus.reshape(n,n)

# to file, if needed
#mmi.io.print_Mat(rawMinv, opts.outputfile)

# print stuff to command line
acc = np.mean(eMCMC.acceptance_fraction)
runtime = tend-tstart
print("The ensemble sampler yielded the following result:")
mmi.io.print_endrun(rawM, rawMinv, runtime)

