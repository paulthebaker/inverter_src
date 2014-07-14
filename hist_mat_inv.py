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
reads in a chainfile and histograms everything!
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
	matchainfile = 'mat_chain.dat'
	logchainfile = 'log_chain.dat'
	try:
		opts, args = getopt.getopt(
                         argv,
                         "hi:o:m:l:",
                         ["help","ifile=","ofile=","mfile=","lfile="]
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
			print('   --help, -h                   display this message and exit')
			print('')
			print('   --ifile, -i <input_file>     input square matrix to invert')
			print('   --ofile, -o <output_file>    output best fit inverse')
			print('   --mfile, -m <matrix_chain_file>     chain file output by MCMC with each matrix guess')
			print('   --lfile, -l <log_chain_file>     chain file output by MCMC with logPost values every numWalkers-th iteration')
			print('')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			mat_infile = arg
		elif opt in ("-o", "--ofile"):
			mat_outfile = arg
		elif opt in ("-m", "--mfile"):
			matchainfile = arg
		elif opt in ("-l", "--lfile"):
			logchainfile = arg
	return (mat_infile, mat_outfile, matchainfile, logchainfile)


##### BEGIN MAIN #####

infile_name, outfile_name, mat_chainfile_name, log_chainfile_name = parse_options(sys.argv[1:])
mat = np.loadtxt(mat_chainfile_name)
logL = np.loadtxt(log_chainfile_name)
mat_in = mmi.io.get_Mat(infile_name)
mat_out = mmi.io.get_Mat(outfile_name)

inv_best = mat_out.flatten()
inv_true = np.linalg.inv(mat_in).flatten()

#TODO: maybe use matplotlib until I figure out gnuplot.py or similar...
plt.figure(1)
plt.grid(True, 'major')
plt.title('Histogram of log(L)')
plt.xlabel('log(L)')
plt.ylabel('Probability')
n, bins, patch = plt.hist(logL, bins=50, normed=True, log=False,
                          alpha=0.5, facecolor='red')

plt.savefig('plots_logL_hist.pdf')


plt.figure(2)
plt.suptitle('Histograms of inverse matrix elements', fontsize=20)
N = mat.shape[1]
n = int(np.sqrt(N))
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

plt.savefig('plots_matinv_hist.pdf')


