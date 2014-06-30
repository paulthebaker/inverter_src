#!/usr/bin/env python3.3
#
#    hist_mat_inv.py
#
# an histogramming tool for verification of the MCMC matrix inverter
#   part of an ongoing exercise in Python 
#
# (c) 2014 Paul T. Baker, bakerp@geneseo.edu
# licence: GNU GPLv3 <http://www.gnu.org/licenses/gpl.txt>
#

"""
reads in two chainfiles and histograms everything!
"""

import sys
import getopt
import mmi_lib as mmi

import numpy as np
import matplotlib.pyplot as plt


def parse_options(argv):
    """command line parser"""
    mat_infile = 'mat_in.dat'
    mat_outfile = 'mat_out.dat'
    rawchainfile = 'raw_chain.dat'
    try:
        opts, args = getopt.getopt(
                         argv,
                         "hi:o:c:",
                         ["help","ifile=","ofile=","cfile="]
                     )
    except getopt.GetoptError:
        print('  ERROR: invalid options, try --help for more info')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h',"--help"):
            print('hist_mat_inv.py [OPTIONS]')
            print('')
            print('  options:')
            print('')
            print('   --help, -h                       display this message and exit')
            print('')
            print('   --ifile, -i <input_file>         input square matrix to invert')
            print('   --ofile, -o <output_file>        output best fit inverse')
            print('   --cfile, -c <chain_file>         chain file output by MCMC')
            print('')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            mat_infile = arg
        elif opt in ("-o", "--ofile"):
            mat_outfile = arg
        elif opt in ("-c", "--cfile"):
            chainfile = arg
    return (mat_infile, mat_outfile, chainfile )


##### BEGIN MAIN #####

infile_name, outfile_name, chainfile_name = parse_options(sys.argv[1:])
chain = np.loadtxt(chainfile_name)
mat_in = mmi.io.get_Mat(infile_name)
mat_out = mmi.io.get_Mat(outfile_name)

inv_best = mat_out.flatten()
inv_true = np.linalg.inv(mat_in).flatten()

index, logL, mat = np.array_split(chain, [1,2], 1)

#logL_hist, tmp_bin = np.histogram(logL, bins=50, density=True)
#logL_bin = np.delete(tmp_bin, tmp_bin.size-1)

open('plots_logPost_hist.pdf','w')

#logPost histogram
plt.figure(1)
plt.grid(True, 'major')
plt.title('Histogram of log(Post)')
plt.xlabel('log(Post)')
plt.ylabel('Probability')
n, bins, patch = plt.hist(logL, bins=50, normed=True, log=False,
                          alpha=0.5, facecolor='red')

plt.savefig('plotslogPost_hist.pdf')

#histogram for each matrix element
open('plots_mat_inv_hist.pdf','w')
plt.figure(2)
plt.suptitle('Histograms of inverse matrix elements', fontsize=20)
N = mat.shape[1]
n = np.sqrt(N)
for i in range(N):
    plt.subplot(n, n, i+1)
    plt.hist(mat[:,i], bins=20, normed=True, log=False,
             alpha=0.5, facecolor='blue')
    plt.axvline(inv_best[i], linewidth=2, color='blue')
    plt.axvline(inv_true[i], linewidth=2, color='red')
    plt.grid(True, 'major')
    plt.ylim([0, 12])
    w = 0.25
    mn = inv_true[i] - w
    mx = inv_true[i] + w
    plt.xlim([mn, mx])
    
    if (i%n == 0):
        plt.yticks(np.linspace(0,12,num=4))
    else:
        plt.yticks(np.linspace(0,12,num=4), ())
    
    if (i >= N-n):
        plt.xticks(np.linspace(mn+.05, mx-.05, num=5), 
               ['-0.2', '-0.1', 'True', '+0.1', '+0.2'],
               rotation=60)
    else:
        plt.xticks(np.linspace(mn+.05, mx-.05, num=5), ())
    
    plt.subplots_adjust(hspace=0.25, wspace=0.20,
                        left=0.08, right=0.95, top=None, bottom=None)

plt.savefig('plotsmatinv_hist.pdf')

#logPost chain plot
open('logPost_chain_plot.pdf','w')
plt.figure(3)
plt.title('Chain plot for logPost')
plt.xlabel('Iteration')
plt.ylabel('logPost')
plt.plot(index, logL)
plt.savefig('logPost_chain_plot.pdf')