#!/opt/local/bin/python

#
#    mcmc_mat_inv.py
#
# a Markov chain Monte Carlo for inverting square matrices
#   an ongoing exercise in Python 
#
# by Paul T. Baker, bakerp@geneseo.edu
#

"""
the likelihood for the MCMC goes like exp(R.R/2) where R is the residual S-H
our data, S, is the identity matrix
our model, H, is the matrix product of the input matrix M and the current guess G
each matrix is treated like a vector of the N*N entries
"""

import sys
import time
import getopt
import numpy as np
from scipy import linalg
import random as rand

log2P = np.log(2.0*np.pi)
reps = np.sqrt(np.finfo(np.float64).eps)


def parse_options(argv):
    """command line parser"""
    inputfile = 'mat_in.dat'
    outputfile = 'mat_out.dat'
    seed = None
    number = 10000
    burn = int(number/10)
    try:
        opts, args = getopt.getopt(
                         argv,
                         "hi:o:s:n:b:",
                         ["help","ifile=","ofile=","seed=","number=","burn-in"]
                     )
    except getopt.GetoptError:
        print('  ERROR: invalid options, try --help for more info')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h',"--help"):
            print('mcmc_mat_inv.py [OPTIONS]')
            print('')
            print('  options:')
            print('')
            print('   --help, -h                   display this message')
            print('')
            print('   --ifile, -i <input_file>     contains square matrix to invert')
            print('   --ofile, -o <output_file>    stores inverse of matrix')
            print('')
            print('   --seed, -s <seed>            seed to initialize random()')
            print('   --number, -n <number>        number of samples in MCMC')
            print('   --burn-in, -b <N_burn>       length of burn in')
            print('')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-n", "--number"):
            number = int(arg)
        elif opt in ("-b", "--burn-in"):
            burn = int(arg)
    return (inputfile, outputfile, number, burn, seed)


def get_Mat(filename):
    """read a square matrix from file"""
    mat = np.loadtxt(filename)
    if mat.shape[0] != mat.shape[1]:
        print( '  ERROR: input matrix is not square %dx%d'%(mat.shape[0], mat.shape[1]) )
        sys.exit(2)
    return mat


def print_Mat(mat,filename):
    """print a matrix to file"""
    np.savetxt(filename, mat, fmt='%+.9e')


def fisher(M, G):
    """compute Fisher Info matrix: Fij = (h,i|h,j)"""
    # TODO Fisher is symmetric ... only need lower triangle...
    n = G.shape[0]
    dh = np.zeros([n*n,n,n])
    F = np.zeros([n*n,n*n])
    # compute derivatives of H
    for i in range(n):
        for j in range(n):
            # establish good choice for dg
            if( G[i,j]!=0 ):
                tmp = reps*G[i,j]
            else:
                tmp = reps
            dg = (tmp+G[i,j]) - G[i,j]
            dG = np.zeros_like(G)
            dG[i,j]=dg
            # H = np.dot(M,G)
            dh[i*n+j,:,:] = 0.5*( np.dot(M,G+dG) - np.dot(M,G-dg) ) / dg
    # components of F
    for j in range(n*n):
        for i in range(j,n*n):
            F[i,j] = sum(sum( dh[i,:,:]*dh[j,:,:] )) # inner product
    # compute eigen-stuff for Fij use lower triangle
    val, vec = linalg.eigh(F, lower=True)
    return (val, vec.reshape([n*n,n,n]))


def log_L(M, G):
    """compute logL for a Guess"""
    # TODO HARDCODED: sigma=0.1
    sigsq = (0.01)
    Id = np.identity(M.shape[0],float)
    R = np.dot(M,G) - Id
    return -0.5*( R.size*log2P + sum(sum(R*R/2.0))/sigsq ) # R*R is element by element product


def proposal(Ga):
    """ propose a new state for MCMC
    
    currently only uses a random direction jump
    return new state (Gb) and proposal density (dlogQ) for Hastings Ratio
    """
#   TODO: implement suite of proposals... 
#         fisher jump, prior draw, ???
    Gb = Ga.copy()
    # random direction, gaussian 1-sigma jump
    for i in range(Ga.shape[0]):
        for j in range(Ga.shape[1]):
            dG = rand.gauss(0.0,0.1)/float(Ga.size)
            Gb[i,j] = Ga[i,j] + dG
    dlogQ = 0.  # jump is symmetric: Qa == Qb => dQ=0.
    return (Gb, dlogQ)

##### BEGIN MAIN #####
infile_name, outfile_name, number, burn, SEED = parse_options(sys.argv[1:])

rand.seed(SEED)

rawM = get_Mat(infile_name)
detM = linalg.det(rawM)
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

val,vec = fisher(M,Ga)
print( val[0]*vec[0,:,:] )
print( val[15]*vec[15,:,:] )
sys.exit()

chain_file = open('chain.dat','wb')

t_start = time.clock()

for n in range(N):
    # MCMC LOOP
    Gb, dlogQ = proposal(Ga)

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
print_Mat(rawMinv, outfile_name)

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
print("Minv =") 
print(rawMinv)
print("")
print("M*Minv =")
print(np.dot(rawM,rawMinv))
print("")

