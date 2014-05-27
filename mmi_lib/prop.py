
# prop.py
# mcmc move proposal functions for mat_inv.py
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>

import numpy as np
import random as rand


def proposal(Ga, F):
    """ propose a new state for MCMC
    
    return new state (Gb) and proposal density (dlogQ) for Hastings Ratio
    """
#   TODO: implement suite of proposals... 
#         prior draw, ???
    if rand.random()<0.5:
        Gb, dlogQ = step(Ga)
    else:
        Gb, dlogQ = fisher(Ga, F)
    
    return (Gb, dlogQ)

def step(Ga):
    Gb = Ga.copy()
    # random direction, gaussian 1-sigma jump
    for i in range(Ga.shape[0]):
        for j in range(Ga.shape[1]):
            dG = rand.gauss(0.0,0.1)/float(Ga.size)
            Gb[i,j] = Ga[i,j] + dG
    dlogQ = 0.  # jump is symmetric: Qa == Qb => dQ=0.
    return (Gb, dlogQ)

def fisher(Ga, F):
    i = rand.randrange(Ga. size)
    dG = 1.0/np.sqrt(F.val[i]) * rand.gauss(0.0,1.0) * F.vec[i,:,:]
    Gb = Ga + dG
    dlogQ = 0.  # fisher jump is symmetric
    return (Gb, dlogQ)

