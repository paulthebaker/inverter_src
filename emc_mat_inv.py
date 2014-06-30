#!/usr/bin/env python3.3

#
#    emc_mat_inv.py
#
# an affine invariant Markov chain Monte Carlo ensemble sampler for inverting square matrices
#   an ongoing exercise in Python 
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>
#

"""
the likelihood for the MCMC goes like exp(R.R/2) where R is the residual S-H
our data, S, is the identity matrix
our model, H, is the matrix product of the input matrix M and the current guess G
each matrix is treated like a vector of the N*N entries
"""

import sys
import mmi_lib as mmi
import time
import numpy as np
import emcee as em
import matplotlib.pyplot as pl

##### BEGIN MAIN #####
infile_name, outfile_name, number, burn, SEED = mmi.io.parse_options(sys.argv[1:])

rawM = mmi.io.get_Mat(infile_name)
detM = np.linalg.det(rawM)
if(detM == 0):
    print("ERROR: det[M]=0, cannot invert")
    exit(2) 
scale = np.power(detM,-1.0/rawM.shape[0])
M = scale*rawM.copy()  # scale so det(M)=1

#define the parameters for the ensemble sampler
numDim = rawM.shape[0]*rawM.shape[1] #the number of elements in the matrix, and the number of dimensions for the walker
numWalk = 256 #why not?

#pick starting positions for each walker
Ga = np.random.normal(0,1,(numWalk,numDim)) #draw from a Gaussian with stdev 1 mean 0, as matrix is normalized, third arg gives shape of output array

#define the target function, the log of the posterior (logPrior + logLikelihood)
def log_Post(G, Mat): #note argument syntax, takes guess first as required by em. Also note that G is a vector, Mat is a matrix
	return mmi.prop.log_L(Mat,G) + mmi.prop.log_P(G)

#initialize the sampler
sampler = em.EnsembleSampler(numWalk, numDim, log_Post, args = [M])

#burn-in
burn_in = sampler.run_mcmc(Ga, burn)
sampler.reset()

#run sampler
sampler.run_mcmc(burn_in[0], number)

#takes the median in each element as the best guess matrix
samples = sampler.flatchain #convert samples to flattened array
values = np.array(*np.percentile(samples, [50], axis=0)) #takes median in each direction of space from unzipped percentile array
Minv = values.reshape((M.shape[0],M.shape[1])) #returns the inverse matrix

# write rescaled Minv to file
rawMinv = scale*Minv.copy()
mmi.io.print_Mat(rawMinv, outfile_name)
 
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
HmHm = sum(sum( I*I ))
HtHm = np.trace(I)
FF = HtHm/np.sqrt(HtHt*HmHm)
print("fitting factor = %.4f"%(FF))
print("")

#create chain file for logLikelihood
print "Printing chain file..."
chain_file = open('chain.dat','w')
raw_samples = sampler.chain
n = 0
for s in range(0, number):
	for w in range(0, numWalk):
		temp = np.hstack( ([[n*s + w]], [[log_Post(scale*raw_samples[w,s],M)]], [scale*raw_samples[w,s]]) )
		np.savetxt(chain_file, temp)
		n += 1