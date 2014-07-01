#!/usr/bin/env python3.3

#
#    emc_mat_inv.py
#
# affine invariant method compared to metropolis-hastings method
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

infile_name, outfile_name, number, burn, SEED = mmi.io.parse_options(sys.argv[1:])

rawM = mmi.io.get_Mat(infile_name)
detM = np.linalg.det(rawM)
if(detM == 0):
    print("ERROR: det[M]=0, cannot invert")
    exit(2) 
scale = np.power(detM,-1.0/rawM.shape[0])
M = scale*rawM.copy()  # scale so det(M)=1

#display raw matrix 
np.set_printoptions(precision=4)
print("M =") 
print(rawM)
print("")

#display true inverse
MinvTRUE = np.linalg.inv(rawM)
print("Minv TRUE =") 
print(MinvTRUE)
print("")

#define the target function, the log of the posterior (logPrior + logLikelihood)
def log_Post(G, Mat): #note argument syntax, takes guess first as required by em. Also note that G is a vector, Mat is a matrix
    return mmi.prop.log_L(Mat,G) + mmi.prop.log_P(G)

##### ENSEMBLE METHOD BEGIN AND DISPLAY PRINTOUT #####
print("Beginning Ensemble Method")
print("")
#define the parameters for the ensemble sampler
numDim = rawM.shape[0]*rawM.shape[1] #the number of elements in the matrix, and the number of dimensions for the walker
numWalk = 256 #why not?

#pick starting positions for each walker
Ga = np.random.normal(0,1,(numWalk,numDim)) #draw from a Gaussian with stdev 1 mean 0, as matrix is normalized, third arg gives shape of output array

#initialize the sampler
ensemble = em.EnsembleSampler(numWalk, numDim, log_Post, args = [M])

#burn-in
burn_in = ensemble.run_mcmc(Ga, burn)
ensemble.reset()

#run sampler
ensemble.run_mcmc(burn_in[0], number)

#takes the median in each element as the best guess matrix
samples = ensemble.flatchain #convert samples to flattened array
values = np.array(*np.percentile(samples, [50], axis=0)) #takes median in each direction of space from unzipped percentile array
MinvEns = values.reshape((M.shape[0],M.shape[1])) #returns the inverse matrix

# write rescaled inverse to file
rawMinvEns = scale*MinvEns.copy()
mmi.io.print_Mat(rawMinvEns, outfile_name)

IEns = np.dot(rawM,rawMinvEns)
print("For the ensemble sampler, M*Minv =")
print(IEns)
print("")

print("For the ensemble sampler, Minv =") 
print(rawMinvEns)
print("")

# TODO: fast fitting factor computation assumes no noise in data
HtHt = rawM.shape[0]
HmHm = np.sum( IEns*IEns )
HtHm = np.trace(IEns)
FF = HtHm/np.sqrt(HtHt*HmHm)
print("For the ensemble sampler, fitting factor = %.4f"%(FF))
print("")

#create chain file for logLikelihood
print("Printing chain file for the ensemble sampler...")
chain_file = open('ensemble_chain.dat','wb')

n = number*numWalk
number_col = np.array(np.arange(n)).reshape(n,1)
logPost_col = np.array( [log_Post(scale*samples[i,:], M)
                          for i in range(n)] ).reshape(n,1)

chain_output = np.hstack( (number_col, logPost_col, scale*samples) )
np.savetxt(chain_file, chain_output)
chain_file.close()


##### METROPOLIS-HASTINGS BEGIN #####
print("Beginning M-H Method")
print("")

#NB: this sampler uses the covariance matrix to update the positions, so we will just start with the identity matrix as a first guess
cov = np.identity(M.size,float)

#Guess initial starting position
G = np.identity(M.shape[0],float).reshape(numDim)

#initialize sampler
m_h = em.MHSampler(cov, numDim, log_Post, args=[M])

#NB: to ensure an equal number of sample sizes for comparison, number and burn will be rescaled by numWalk
burnMH = burn*numWalk
numberMH = number*numWalk

#burn-in: halfway through, recomputes the Fisher matrix
burn_in1 = m_h.run_mcmc(G, burnMH/2)

#take current best guess matrix...
samples = m_h.chain
Ga = np.array(np.percentile(samples, [50], axis=0))

#and use to compute Fisher matrix, and from that the covariance
G = Ga.reshape(M.shape[0],M.shape[1])
F = mmi.types.Fisher(M,G)
cov = np.linalg.inv(F.mat)
m_h.clear_chain()

#restart sampler from current positions with new Fisher matrix
m_h = em.MHSampler(cov, numDim, log_Post, args=[M])
burn_in2 = m_h.run_mcmc(burn_in1[0],burnMH/2)

#recompute Fisher matrix one more time from current best guess values
Ga = np.array(np.percentile(samples, [50], axis=0))
G = Ga.reshape(M.shape[0],M.shape[1])
F.update(M,G)
cov = np.linalg.inv(F.mat)
m_h.clear_chain()

#run sampler for realz this time
m_h = em.MHSampler(cov,numDim,log_Post,args=[M])
m_h.run_mcmc(burn_in2[0], numberMH)

#take median in each direction as best-guess inverse
samples = m_h.flatchain
values = np.array(*np.percentile(samples, [50], axis=0))
MinvMH = values.reshape((M.shape[0],M.shape[1]))

#write rescaled inverse to file
rawMinvMH = scale*MinvMH.copy()
mmi.io.print_Mat(rawMinvMH, outfile_name)

IMH = np.dot(rawM,rawMinvMH)
print("For the MH sampler, M*Minv =")
print(IMH)
print("")

print("For the MH sampler, Minv =") 
print(rawMinvMH)
print("")

# TODO: fast fitting factor computation assumes no noise in data
HtHt = rawM.shape[0]
HmHm = np.sum( IMH*IMH )
HtHm = np.trace(IMH)
FF = HtHm/np.sqrt(HtHt*HmHm)
print("For the MH sampler, fitting factor = %.4f"%(FF))
print("")

#create chain file for logLikelihood
print("Printing chain file for the MH sampler...")
chain_file = open('mh_chain.dat','wb')

n = numberMH
number_col = np.array(np.arange(n)).reshape(n,1)
logPost_col = np.array( [log_Post(scale*samples[i,:], M)
                          for i in range(n)] ).reshape(n,1)

chain_output = np.hstack( (number_col, logPost_col, scale*samples) )
np.savetxt(chain_file, chain_output)


