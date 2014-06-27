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
numWalk = numDim*3 #start out with three for each element

Ga = np.array(np.identity(M.shape[0],float))  # first guess is II
Ga = Ga.reshape((1,numDim)) # convert guess into coordinate vector

#define the target function, the log of the posterior (logPrior + logLikelihood)
def log_Post(G, Mat): #note argument syntax, takes guess first as required by em. Also note that G is a vector, Mat is a matrix
	return mmi.prop.log_L(Mat,G) + mmi.prop.log_P(G)

#initialize the sampler
sampler = em.EnsembleSampler(numWalk, numDim, log_Post, args = [M])

#burn-in
sampler.run_mcmc(Ga, 1)
sampler.reset()

#run sampler
sampler.run_mcmc(Ga, number)

#takes output and converts to best guess, for now just last step
Minv = sampler.flatchain[numWalk*number]
#reshape as a matrix
Minv = Minv.reshape((M.shape[0],M.shape[1]))

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


