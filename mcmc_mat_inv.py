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
the likelihood for the MCMC goes like exp(R.R/2) where R is the residual S-H
our data, S, is the identity matrix
our model, H, is the matrix product of the input matrix M and the current guess G
each matrix is treated like a vector of the N*N entries
"""

import sys
import mmi_lib as mmi

import time
import numpy as np
import random as rand

#TODO: for speedup use cython and gpl C libraries... or maybe Pypy

log2P = np.log(2.0*np.pi)


def log_L(M, G):
    """compute logL for a Guess"""
    # TODO HARDCODED: sigma=0.1
    sigsq = (0.01)
    Id = np.identity(M.shape[0],float)
    R = np.dot(M,G) - Id
    return -0.5*( R.size*log2P + sum(sum(R*R/2.0))/sigsq ) # R*R is element by element product



##### BEGIN MAIN #####
infile_name, outfile_name, number, burn, SEED = mmi.io.parse_options(sys.argv[1:])

rand.seed(SEED)

rawM = mmi.io.get_Mat(infile_name)
detM = np.linalg.det(rawM)
if(detM == 0):
    print("ERROR: det[M]=0, cannot invert")
    exit(2) 
scale = np.power(detM,-1.0/rawM.shape[0])
M = scale*rawM.copy()  # scale so det(M)=1

# initialize MCMC
# let (a) be current state, (b) be proposed new state
Ga = np.identity(M.shape[0],float)  # first guess is I
logLa = log_L(M,Ga)
Minv = Ga.copy()         # init best fit Minv to first guess
logLmax = log_L(M,Minv)

N = number   # number o' MCMC samples
acc = 0

F = mmi.types.Fisher(M,Ga)

chain_file = open('chain.dat','wb')

t_start = time.clock()

for n in range(N):
    # MCMC LOOP
    if (n%1000)==0:
        # update fisher matrix eigen directions
        F.update(M,Ga)

    Gb, dlogQ = mmi.prop.proposal(Ga, F)

    # compute logL and Hastings Ratio
    logLb = log_L(M,Gb)
    logH = logLb - logLa + dlogQ

    b = np.log(rand.random())

    if ( n%1000 == 0 ):  # print progress to stdout
        print(
              "n:%d :  logL = %.4f,  logH = %.4f,  b = %.4f,  acc = %.4f"
              %(n, logLa, logH, b, float(acc)/float(n+1) )
             )

    if ( logH >= b ):  # accept proposal ... (b) -> (a)
        Ga = Gb.copy()
        logLa = logLb
        acc = acc + 1
        if ( logLa > logLmax ):  # new best fit (maximum likelihood)
            Minv = Ga.copy()
            logLmax = logLa

    tmp = np.hstack( ([[n]], [[logLa]], (Ga.copy()).reshape(1,Ga.size)) )
    np.savetxt(chain_file, tmp, fmt='%+.8e')

# end for
chain_file.close()

t_end = time.clock()

# write rescaled Minv to file
rawMinv = scale*Minv.copy()
mmi.io.print_Mat(rawMinv, outfile_name)

# print stuff to command line
print("")
print("MCMC runtime: %.4f sec"%(t_end-t_start))
print("")
print("acceptance = %.4f"%( float(acc)/float(N) ))
print("")
print("max logL =", logLmax)
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
HmHm = sum(sum( I*I ))
HtHm = np.trace(I)
FF = HtHm/np.sqrt(HtHt*HmHm)
print("fitting factor = %.4f"%(FF))
print("")


