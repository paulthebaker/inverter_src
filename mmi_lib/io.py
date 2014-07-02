
# io.py
# input/output functions for mat_inv.py
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>

import sys
import getopt
import numpy as np

class CmdLineOpts:
    """object for storing command line options"""
    inputfile = 'mat_in.dat'
    outputfile = 'mat_out.dat'
    guessfile = None
    chainfile = 'chain.dat'
    seed = None
    number = 100
    burn = int(number/5)
    walk = 100
    mpi = False

    def __init__(self, argv):
        """command line parser"""
        try:
            opts, args = getopt.getopt(
                            argv,
                            "hi:o:g:c:s:n:b:w:p",
                            ["help",
                            "ifile=", "ofile=", "gfile=", "cfile="
                            "seed=",
                            "number=", "burn", "walk",
                            "MPI"]
                        )
        except getopt.GetoptError:
            print('  ERROR: invalid options, try --help for more info')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ('-h',"--help"):
                print('mcmc_mat_inv.py [OPTIONS]')
                print()
                print('  options:')
                print()
                print('   --help, -h                   display this message')
                print()
                print('   --ifile, -i <input_file>     contains square matrix to invert')
                print('   --ofile, -o <output_file>    stores inverse of matrix')
                print('   --gfile, -g <guess_file>     optional initial guess for inverse')
                print('   --cfile, -c <chain_file>     store Markov chain samples')
                print()
                print('   --seed, -s <seed>            seed to initialize random()')
                print()
                print('   --number, -n <number>        number of samples in MCMC')
                print('   --burn, -b <N_burn>          length of burn in')
                print('   --walk, -w <N_walk>          number of walkers in ensemble')
                print()
                print('   --MPI, -p                    parallelize with MPI')
                print()
                sys.exit()
            elif opt in ("-i", "--ifile"):
                self.inputfile = arg
            elif opt in ("-o", "--ofile"):
                self.outputfile = arg
            elif opt in ("-g", "--gfile"):
                self.guessfile = arg
            elif opt in ("-c", "--cfile"):
                self.chainfile = arg
            elif opt in ("-s", "--seed"):
                self.seed = int(arg)
            elif opt in ("-n", "--number"):
                self.number = int(arg)
            elif opt in ("-b", "--burn"):
                self.burn = int(arg)
            elif opt in ("-w", "--walk"):
                self.walk = int(arg)
            elif opt in ("-p", "--MPI"):
                self.mpi = True


def get_Mat(filename):
    """read a square matrix from file"""
    mat = np.loadtxt(filename)
    if mat.shape[0] != mat.shape[1]:
        print( '  ERROR: input matrix is not square %dx%d'%(mat.shape[0], mat.shape[1]) )
        sys.exit(2)
    return mat


def print_Mat(mat, filename):
    """print a matrix to file"""
    np.savetxt(filename, mat, fmt='%+.9e')


def print_chain(chain, filename):
    """print MCMC samples to chain_file"""
    np.savetxt(filename, chain, fmt='%+.9e')


def print_endrun(M, Minv, dt, acc):
    """print end of run summary to stdout"""
    print()
    print("MCMC runtime: %.4f sec"%dt)
    print()
    print("Mean acceptance: %.4f}"%acc)
    print()

    np.set_printoptions(precision=4)
    print("M =")
    print(M)
    print()

    I = np.dot(M,Minv)
    print("M*Minv =")
    print(I)
    print()

    print("Minv =")
    print(Minv)
    print()

    MinvTRUE = np.linalg.inv(M)
    print("Minv TRUE =")
    print(MinvTRUE)
    print()

    # TODO: fast fitting factor computation assumes no noise in data
    HtHt = M.shape[0]
    HmHm = np.sum( I*I )
    HtHm = np.trace(I)
    FF = HtHm/np.sqrt(HtHt*HmHm)
    print("fitting factor = %.4f"%(FF))
    print()
