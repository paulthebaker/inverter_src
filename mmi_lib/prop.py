
# prop.py
# mcmc move proposal functions for mat_inv.py
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>

import numpy as np

log2P = np.log(2.0*np.pi)
log5 = np.log(5.0)

def log_L(M, G):
    """compute logL for a Guess"""
    # TODO HARDCODED: sigma=0.1
    sigsq = (0.01)
    Id = np.identity(M.shape[0],float)
    R = np.dot(M,G) - Id
    return -0.5*( R.size*log2P + sum(sum(R*R))/sigsq ) # R*R is element by element product


def log_P(G):
    """compute gaussian prior mean=0, stdev=5"""
    #TODO: hard coded stdev=5
    sig = 5.0
    logP = -0.5*( G.size*(log2P+log5) + sum(sum(G*G))/(sig*sig) )
    return logP


def proposal(Ga, F):
    """ propose a new state for MCMC
    
    return new state (Gb) and proposal density (dlogQ) for Hastings Ratio
    dlogQ = logQb - logQa, same as dP and dL: subtracted in HR
    """
    test = np.random.random()
    if test<0.05:
        Gb, dlogQ = prior(Ga)
    elif test<0.525:
        Gb, dlogQ = step(Ga)
    else:
        Gb, dlogQ = fisher(Ga, F)
    
    return (Gb, dlogQ)

def step(Ga):
    """random direction, 1-sigma gaussian step"""
    dG = np.random.normal(size=Ga.shape)/float(Ga.size)
    Gb = Ga + dG
    dlogQ = 0.  # jump is symmetric: Qa == Qb => dQ=0.
    return (Gb, dlogQ)

def fisher(Ga, F):
    """gaussian step along random eigenvector of F""" 
    i = np.random.randint(Ga.size) # choose eigenvector
    dG = 1.0/np.sqrt(F.val[i]) * np.random.normal() * F.vec[i,:,:]
    Gb = Ga + dG
    dlogQ = 0.  # fisher jump is symmetric
    return (Gb, dlogQ)

def prior(Ga):
    """prior draw, gaussian prior"""
    #TODO: sigma=5 hardcoded
    Gb = np.random.normal(scale=5.0,size=Ga.shape)
    dlogQ = log_P(Gb) - log_P(Ga)
    return (Gb, dlogQ)

