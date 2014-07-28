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


# PLOT TIME!!!!
print("Done with run, making plot \n")
post = np.loadtxt(opts.logchainfile)

# plot of log_PDF as a function of iteration
plt.figure(1)
plt.title('Chain plot for logPost')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.xlim([0, N/2])
plt.savefig('ensemble_logPost_chain_plot.pdf')

#read in matrix chain file
chainmat = np.loadtxt(opts.matchainfile).reshape((Nsamp,Ndim))

# get median and uncert
med = np.array(*np.percentile(chainmat, [50], axis=0))
plus = np.array(*np.percentile(chainmat, [84], axis=0)) - med

# write rescaled Minv
rawMinv = scale*med.reshape(n,n)
rawMinvP = scale*plus.reshape(n,n)

rawMinvTrue = np.linalg.inv(M)

# histogram the diagonal elements of the inverse
plt.figure(2)
plt.suptitle('Histograms of inverse matrix elements', fontsize=20)

for i in range(n):
	plt.subplot(1, n, i+1)
	plt.hist(chainmat[:,i+n+1], bins=20, normed=True, log=False, alpha=0.5, facecolor='blue')
	plt.axvline(rawMinv[i,i], linewidth=2, color='blue')
	plt.axvline(rawMinvTrue[i,i], linewidth=2, color='red')
	plt.axvline(rawMinv[i,i] + rawMinvP[i,i], linewidth=1.5, color='purple')
	plt.axvline(rawMinv[i,i] - rawMinvP[i,i], linewidth=1.5, color='purple')
	plt.grid(True, 'major')
	plt.ylim([0, 12])
	mx = rawMinv[i,i] + 2*rawMinvP[i,i]
	mn = rawMinv[i,i] - 2*rawMinvP[i,i]
	plt.xlim([mn, mx])
	plt.xticks(np.linspace(mn+.05, mx-.05, num=5), ['-0.2', '-0.1', 'True', '+0.1', '+0.2'], rotation=60)
	plt.yticks(np.linspace(0,12,num=4))
	plt.subplots_adjust(hspace=0.25, wspace=0.20, left=0.08, right=0.95, top=None, bottom=None)

plt.savefig('ensemble_plots_matinv_hist.pdf')

# to file, if needed
#mmi.io.print_Mat(rawMinv, opts.outputfile)

# obtain autocorrelation time from chainmat
autocor = emcee.autocorr.integrated_time(chainmat, axis=0)

# print stuff to command line
acc = np.mean(eMCMC.acceptance_fraction)
runtime = tend-tstart
print("The ensemble sampler yielded the following result:")
mmi.io.print_endrun(rawM, rawMinv, rawMinvP, rawMinvM, runtime, acc, autocor)

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

# recompute covariance
#G = pos.reshape((n,n))
#F = mmi.types.Fisher(M,G)
#cov = np.linalg.inv(F.mat)

#initialize sampler with new cov
mhMCMC = emcee.MHSampler(cov, Ndim, mmi.prop.log_PDF, args=[M, sigL, sigP])

# actual run
# FOR NOW: reuse chain files
logchain = open(opts.logchainfile,'w')
matchain = open(opts.matchainfile,'w')

dum_dum = []

for i in range(N):
	for step in mhMCMC.sample(pos, iterations=1, storechain=False):
		position = step[0]
		for k in range(position.shape[0]):
			matchain.write("%f  " % position[k])
		matchain.write("\n")
		
		log = mmi.prop.log_PDF(position, M, sigL, sigP)
		logchain.write("%f \n" % log)
		
		dum_dum = position
	pos = dum_dum

logchain.close()
matchain.close()

tend = time.clock()

# close MPI pool
if opts.mpi:
    pool.close()

# PLOT TIME!!!
print("Done with run, making plot \n")
post = np.loadtxt(opts.logchainfile)

# plot of log_PDF as a function of iteration
plt.figure(3)
plt.title('Chain plot for logPost')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(0,post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.xlim([0, N/Nwalkers])
plt.savefig('mh_logPost_chain_plot.pdf')

#read in matrix chain file
chainmat = np.loadtxt(opts.matchainfile).reshape((Nsamp,Ndim))

med = np.array(*np.percentile(chainmat, [50], axis=0))
plus = np.array(*np.percentile(chainmat, [84], axis=0)) - med

# write rescaled Minv
rawMinv = scale*med.reshape(n,n)
rawMinvP = scale*plus.reshape(n,n)

# histogram the diagonal elements of the inverse
plt.figure(4)
plt.suptitle('Histograms of inverse matrix elements', fontsize=20)
	
for i in range(n):
	plt.subplot(1, n, i+1)
	plt.hist(chainmat[:,i+n+1], bins=20, normed=True, log=False, alpha=0.5, facecolor='blue')
	plt.axvline(rawMinv[i,i], linewidth=2, color='blue')
	plt.axvline(rawMinvTrue[i,i], linewidth=2, color='red')
	plt.axvline(rawMinv[i,i] + rawMinvP[i,i], linewidth=1.5, color='purple')
	plt.axvline(rawMinv[i,i] - rawMinvP[i,i], linewidth=1.5, color='purple')
	plt.grid(True, 'major')
	plt.ylim([0, 12])
	mx = rawMinv[i,i] + 2*rawMinvP[i,i]
	mn = rawMinv[i,i] - 2*rawMinvP[i,i]
	plt.xlim([mn, mx])
	plt.xticks(np.linspace(mn+.05, mx-.05, num=5), ['-0.2', '-0.1', 'True', '+0.1', '+0.2'], rotation=60)
	plt.yticks(np.linspace(0,12,num=4))
	plt.subplots_adjust(hspace=0.25, wspace=0.20, left=0.08, right=0.95, top=None, bottom=None)

plt.savefig('mh_plots_matinv_hist.pdf')

# to file, if needed
#mmi.io.print_Mat(rawMinv, opts.outputfile)

# obtain autocorrelation time from chainmat
autocor = emcee.autocorr.integrated_time(chainmat, axis=0)

# print stuff to command line
acc = np.mean(mhMCMC.acceptance_fraction)
runtime = tend-tstart
print("The Metropolis-Hastings sampler yielded the following result:")
mmi.io.print_endrun(rawM, rawMinv, rawMinvP, runtime, acc, autocor)