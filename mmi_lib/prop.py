
# prop.py
# mcmc move proposal functions for mat_inv.py
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>

import numpy as np
import random as rand


def proposal(Ga,F):
    """ propose a new state for MCMC
    
    currently only uses a random direction jump
    return new state (Gb) and proposal density (dlogQ) for Hastings Ratio
    """
#   TODO: implement suite of proposals... 
#         prior draw, ???
    Gb = Ga.copy()
#    if rand.random()<0.5:
    if 1:
        # random direction, gaussian 1-sigma jump
        for i in range(Ga.shape[0]):
            for j in range(Ga.shape[1]):
                dG = rand.gauss(0.0,0.1)/float(Ga.size)
                Gb[i,j] = Ga[i,j] + dG
        dlogQ = 0.  # jump is symmetric: Qa == Qb => dQ=0.
        return (Gb, dlogQ)
#    else:
#        # fisher jump


