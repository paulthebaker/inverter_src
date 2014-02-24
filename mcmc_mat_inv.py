#!/opt/local/bin/python

#
#    mcmc_mat_inv.py
#
# a Markov chain Monte Carlo for inverting square matrices
#   an ongoing exercise in Python 
#
# by Paul T. Baker, bakerp@geneseo.edu
#

import sys
import getopt
import numpy as np
from scipy import linalg
import random as rand

log2P = np.log(2.0*np.pi)

def parse_options(argv):
    # command line parser
    inputfile = 'mat_in.dat'
    outputfile = 'mat_out.dat'
    seed = None
    number = 1000
    try:
        opts, args = getopt.getopt(
                         argv,
                         "hi:o:s:n:",
                         ["help","ifile=","ofile=","seed=","number="]
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
    return (inputfile, outputfile, number, seed)


def get_Mat(filename):
    mat = np.loadtxt(filename)
    if mat.shape[0] != mat.shape[1]:
        print( '  ERROR: input matrix is not square %dx%d'%(mat.shape[0], mat.shape[1]) )
        sys.exit(2)
    return mat


def print_Mat(mat,filename):
    np.savetxt(filename, mat, fmt='%+.9e')


def log_L(M, G):
    # compute logL for Guess
    # uses sigma=0.1
    sigsq = (0.01)
    Id = np.identity(M.shape[0],float)
    R = np.dot(M,G) - Id
    return -0.5*( R.size*log2P + sum(sum(R*R/2.0))/sigsq ) # R*R is element by element product


##### BEGIN MAIN #####
infile_name, outfile_name, number, SEED = parse_options(sys.argv[1:])

rand.seed(SEED)

rawM = get_Mat(infile_name)
detM = linalg.det(rawM)
scale = np.power(detM,-1.0/rawM.shape[0])
M = scale*rawM.copy()  # scale so det(M)=1

# initialize MCMC
# let (a) be current state, (b) be proposed new state
Ga = np.identity(M.shape[0],float)  # first guess is I
logLa = log_L(M,Ga)
Minv = Ga.copy()       # init best fit Minv
logLmax = log_L(M,Minv)


N = number   # number o' MCMC samples
acc = 0

for n in range(N):
    # TODO
    # pick number of elements to change: k = rand.randrange(M.size)
    # random.sample( ,k)

    # propose a new state Gb (Fischer jump, Gb = Ga+dG)
    Gb = Ga.copy()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dG = rand.gauss(0.0,0.1)/float(M.size)
            Gb[i,j] = Ga[i,j] + dG

    # propose a new state by changing one element of Ga
#    i = rand.randrange(0,Ga.shape[0],1)
#    j = rand.randrange(0,Ga.shape[1],1)
#    jump_type = rand.random()
#    if (jump_type > 0.5):
#        dG = rand.gauss(0.0,0.1)   # 1 sigma jump
#    elif (jump_type > 0.1):
#        dG = rand.gauss(0.0,0.01)  # tiny jump
#    else:
#        dG = rand.gauss(0.0,1.0)   # big jump
#    Gb[i,j] = Ga[i,j] + dG

    # is proposal accepted?
    logLb = log_L(M,Gb)
    logH = logLb - logLa

    b = np.log(rand.random())

    if ( n%1000 == 0 ):
        print(
              "n:%d :  logL = %.4f,  logH = %.4f,  b = %.4f,  acc = %.4f"
              %(n, logLa, logH, b, float(acc)/float(n+1) )
             )

    if ( logH >= b ): # accept ... (b) -> (a)
        Ga = Gb.copy()
        logLa = logLb
        acc = acc + 1
        if ( logLa > logLmax ):  # new best fit (maximum likelihood)
            Minv = Ga.copy()
            logLmax = logLa
# end for

# write rescaled Minv to file
rawMinv = scale*Minv.copy()
print_Mat(rawMinv, outfile_name)

# print stuff to command line
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


