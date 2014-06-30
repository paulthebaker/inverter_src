
# hastings.py
# mcmc hastings ratio functions for mat_inv.py
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

