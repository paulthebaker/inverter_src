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

# burn in
pos, prob, state = eMCMC.run_mcmc(x0, B)
eMCMC.reset()

# actual run
eMCMC.run_mcmc(pos, N)

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()

# extract sample chain
samp = (eMCMC.flatchain).reshape((Nsamp,Ndim))
# to file, if needed
#mmi.io.print_chain(samp, opts.matchainfile)

# make convergence plot for ensemble
print("Done with run, making plot")
p = eMCMC.lnprobability
post = []
for i in range(N):
	post.append(p[Nwalkers-1,i])
post = np.array(post)

plt.figure(1)
plt.title('Chain plot for logPost')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(0,post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.xlim([0, N/2])
plt.savefig('ensemble_logPost_chain_plot.pdf')

# get median, uncert, and MAP
ind_MAP = np.argmax(post)
MAP = samp[ind_MAP]
med = np.array(*np.percentile(samp, [50], axis=0))
plus = np.array(*np.percentile(samp, [84], axis=0)) - med
minus = med - np.array(*np.percentile(samp, [84], axis=0))

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
mmi.io.print_endrun(rawM, rawMinv, runtime, acc)

# initialize MPI pool for parallelization
if opts.mpi:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

#for MH sampler, redefine run parameters to make the number of iterations equal
N *= Nwalkers
B *= Nwalkers

#initialize Metropolis-Hastings MCMC
mhMCMC = emcee.MHSampler(cov, Ndim, mmi.prop.log_PDF, args=[M, sigL, sigP])

tstart = time.clock()

#burn-in
x0 = x01.reshape(1,Ndim)
pos, prob, state = mhMCMC.run_mcmc(x0[0], B)
mhMCMC.reset()

#recompute covariance
G = pos.reshape((n,n))
F = mmi.types.Fisher(M,G)
cov = np.linalg.inv(F.mat)

#initialize sampler with new cov
mhMCMC = emcee.MHSampler(cov, Ndim, mmi.prop.log_PDF, args=[M, sigL, sigP])

# actual run
mhMCMC.run_mcmc(pos, N)

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()

# extract sample chain
samp = mhMCMC.flatchain
# to file, if needed
#mmi.io.print_chain(samp, opts.matchainfile)

# make convergence plot for mh
print("Done with run, making plot")
p = mhMCMC.lnprobability
post = []
for i in range(N):
	if i%Nwalkers == 0:
		post.append(p[i])
post = np.array(post)

plt.figure(2)
plt.title('Chain plot for logPost')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(0,post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.xlim([0, N/Nwalkers])
plt.savefig('mh_logPost_chain_plot.pdf')

# get median, uncert, and MAP
ind_MAP = np.argmax(post)
MAP = samp[ind_MAP]
med = np.array(*np.percentile(samp, [50], axis=0))
plus = np.array(*np.percentile(samp, [84], axis=0)) - med
minus = med - np.array(*np.percentile(samp, [84], axis=0))

# write rescaled Minv
rawMinv = scale*med.reshape(n,n)
#rawMinvP = scale*plus.reshape(n,n)
#rawMinvM = scale*minus.reshape(n,n)

# to file, if needed
#mmi.io.print_Mat(rawMinv, opts.outputfile)

# print stuff to command line
acc = np.mean(mhMCMC.acceptance_fraction)
runtime = tend-tstart
print("The Metropolis-Hastings sampler yielded the following result:")
mmi.io.print_endrun(rawM, rawMinv, runtime, acc)