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
import mmi_lib as mmi

import numpy as np
import matplotlib.pyplot as plt

##### BEGIN MAIN #####

#TODO: filenames hardcoded...
chain = np.loadtxt("chain.dat")
mat_in = mmi.io.get_Mat("mat_in.dat")
mat_out = mmi.io.get_Mat("mat_out.dat").flatten()

mat_inv = np.linalg.inv(mat_in).flatten()

n, logL, mat = np.array_split(chain, [1,2], 1)

#logL_hist, tmp_bin = np.histogram(logL, bins=50, density=True)
#logL_bin = np.delete(tmp_bin, tmp_bin.size-1)

#TODO: maybe use matplotlib until I figure out gnuplot.py or similar...
plt.figure(1)
plt.grid(True, 'major')
plt.title('Histogram of log(L)')
plt.xlabel('log(L)')
plt.ylabel('Probability')
n, bins, patch = plt.hist(logL, bins=50, normed=True, log=False,
                          alpha=0.5, facecolor='red')

plt.savefig('plots/logL_hist.pdf')


plt.figure(2)
plt.suptitle('Histograms of inverse matrix elements', fontsize=20)
N = mat.shape[1]
n = np.sqrt(N)
for i in range(N):
    plt.subplot(n, n, i+1)
    plt.hist(mat[:,i], bins=20, normed=True, log=False,
             alpha=0.5, facecolor='blue')
    plt.axvline(mat_out[i], linewidth=2, color='blue')
    plt.axvline(mat_inv[i], linewidth=2, color='red')
    plt.grid(True, 'major')
    plt.ylim([0, 12])
    w = 0.25
    mn = mat_inv[i] - w
    mx = mat_inv[i] + w
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

plt.savefig('plots/matinv_hist.pdf')

