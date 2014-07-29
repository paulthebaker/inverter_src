import sys
import mmi_lib as mmi

import time
import numpy as np
import matplotlib.pyplot as plt

mat_in = mmi.io.get_Mat('mat_in.dat')
mat_inv = np.linalg.inv(mat_in)

Ndim = mat_in.shape[0]*mat_in.shape[1]
n = mat_in.shape[0]

row = 2
col = 2

#### MAKE PLOTS FOR THE ENSEMBLE RUN ####

## log plots
post = np.loadtxt('ensemble_log_chain.dat')
post_quar = np.array_split(post,4)

# plot of log_PDF as a function of iteration
plt.figure(1)
plt.title('Chain plot for log_PDF')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.savefig('ensemble_logPost_chain_plot.pdf')

# histogram the log_PDF chain for whole run
plt.figure(2)
plt.suptitle('Histogram of log_PDF')
plt.xlabel('PDF')
plt.ylabel('Occurances')
plt.hist(post, bins=100)
plt.grid(True, 'major')
plt.xlim([-200,0])
plt.savefig('ensemble_plots_log_PDF_hist.pdf')

# histogram the first and last quarters of the PDF chain
quarters = np.array_split(post, 4)

plt.figure(3)
plt.suptitle('Histogram of log_PDF, First Quarter')
plt.xlabel('PDF')
plt.ylabel('Occurances')
plt.hist(quarters[0], bins=100)
plt.grid(True, 'major')
plt.xlim([-200,0])
plt.savefig('ensemble_plots_log_PDF_first.pdf')

plt.figure(4)
plt.suptitle('Histogram of log_PDF, Last Quarter')
plt.xlabel('PDF')
plt.ylabel('Occurances')
plt.hist(quarters[3], bins=100)
plt.grid(True, 'major')
plt.xlim([-200,0])
plt.savefig('ensemble_plots_log_PDF_last.pdf')


## histogram diagonal elements
chainmat = np.loadtxt('ensemble_mat_chain.dat')
mat_quar = np.array_split(chainmat,4)
second = mat_quar[1]
last = mat_quar[3]

# histogram diagonal elements for 2nd quarter
med = np.array(np.percentile(second, [50], axis=0)).reshape((n,n))
plus = np.array(np.percentile(second, [84], axis=0)).reshape((n,n))
minus = np.array(np.percentile(second, [16], axis=0)).reshape((n,n))

plt.figure(5)
plt.suptitle("Diagonal Elements of Matrix for 2nd Quarter of Simulation")

plt.subplot(row,col,1)
plt.title("Element 1,1")
plt.hist(second[:,0], bins=100)
plt.axvline(mat_inv[0,0], linewidth=2, color='red')
plt.axvline(med[0,0], linewidth=2, color='blue')
plt.axvline(plus[0,0], linewidth=1.5, color='purple')
plt.axvline(minus[0,0], linewidth=1.5, color='purple')
plt.ylabel('PDF')

plt.subplot(row,col,2)
plt.title("Element 2,2")
plt.hist(second[:,5], bins=100)
#plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[1,1], linewidth=2, color='red')
plt.axvline(med[1,1], linewidth=2, color='blue')
plt.axvline(plus[1,1], linewidth=1.5, color='purple')
plt.axvline(minus[1,1], linewidth=1.5, color='purple')

plt.subplot(row,col,3)
plt.title("Element 3,3")
plt.hist(second[:,10], bins=100)
#plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[2,2], linewidth=2, color='red')
plt.axvline(med[2,2], linewidth=2, color='blue')
plt.axvline(plus[2,2], linewidth=1.5, color='purple')
plt.axvline(minus[2,2], linewidth=1.5, color='purple')
plt.xlabel('Element Value')
plt.ylabel('PDF')

plt.subplot(row,col,4)
plt.title("Element 4,4")
plt.hist(second[:,15], bins=100)
#plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[3,3], linewidth=2, color='red')
plt.axvline(med[3,3], linewidth=2, color='blue')
plt.axvline(plus[3,3], linewidth=1.5, color='purple')
plt.axvline(minus[3,3], linewidth=1.5, color='purple')
plt.xlabel('Element Value')

plt.savefig('ensemble_diag_second_quarter.pdf')

# histogram diagonal elements for last quarter
med = np.array(np.percentile(last, [50], axis=0)).reshape((n,n))
plus = np.array(np.percentile(last, [84], axis=0)).reshape((n,n))
minus = np.array(np.percentile(last, [16], axis=0)).reshape((n,n))

plt.figure(6)
plt.suptitle("Diagonal Elements of Matrix for 4th Quarter of Simulation")

plt.subplot(row,col,1)
plt.title("Element 1,1")
plt.hist(last[:,0], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[0,0], linewidth=2, color='red')
plt.axvline(med[0,0], linewidth=2, color='blue')
plt.axvline(plus[0,0], linewidth=1.5, color='purple')
plt.axvline(minus[0,0], linewidth=1.5, color='purple')
plt.ylabel('PDF')

plt.subplot(row,col,2)
plt.title("Element 2,2")
plt.hist(last[:,5], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[1,1], linewidth=2, color='red')
plt.axvline(med[1,1], linewidth=2, color='blue')
plt.axvline(plus[1,1], linewidth=1.5, color='purple')
plt.axvline(minus[1,1], linewidth=1.5, color='purple')

plt.subplot(row,col,3)
plt.title("Element 3,3")
plt.hist(last[:,10], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[2,2], linewidth=2, color='red')
plt.axvline(med[2,2], linewidth=2, color='blue')
plt.axvline(plus[2,2], linewidth=1.5, color='purple')
plt.axvline(minus[2,2], linewidth=1.5, color='purple')
plt.xlabel('Element Value')
plt.ylabel('PDF')

plt.subplot(row,col,4)
plt.title("Element 4,4")
plt.hist(last[:,15], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[3,3], linewidth=2, color='red')
plt.axvline(med[3,3], linewidth=2, color='blue')
plt.axvline(plus[3,3], linewidth=1.5, color='purple')
plt.axvline(minus[3,3], linewidth=1.5, color='purple')
plt.xlabel('Element Value')

plt.savefig('ensemble_diag_last_quarter.pdf')



#### MAKE PLOTS FOR THE METROPOLIS-HASTINGS RUN ####

## log plots
post = np.loadtxt('mh_log_chain.dat')
post_quar = np.array_split(post,4)

# plot of log_PDF as a function of iteration
plt.figure(7)
plt.title('Chain plot for log_PDF')
plt.xlabel('Iteration')
plt.ylabel('logPost')
index = range(post.shape[0])
plt.plot(index, post, 'x')
plt.ylim([-200,0])
plt.savefig('mh_logPost_chain_plot.pdf')

# histogram the log_PDF chain for whole run
plt.figure(8)
plt.suptitle('Histogram of log_PDF')
plt.xlabel('PDF')
plt.ylabel('Occurances')
plt.hist(post, bins=100)
plt.grid(True, 'major')
plt.xlim([-200,0])
plt.savefig('mh_plots_log_PDF_hist.pdf')

# histogram the first and last quarters of the PDF chain
quarters = np.array_split(post, 4)

plt.figure(9)
plt.suptitle('Histogram of log_PDF, First Quarter')
plt.xlabel('PDF')
plt.ylabel('Occurances')
plt.hist(quarters[0], bins=100)
plt.grid(True, 'major')
plt.xlim([-200,0])
plt.savefig('mh_plots_log_PDF_first.pdf')

plt.figure(10)
plt.suptitle('Histogram of log_PDF, Last Quarter')
plt.xlabel('PDF')
plt.ylabel('Occurances')
plt.hist(quarters[3], bins=100)
plt.grid(True, 'major')
plt.xlim([-200,0])
plt.savefig('mh_plots_log_PDF_last.pdf')


## histogram diagonal elements
chainmat = np.loadtxt('mh_mat_chain.dat')
mat_quar = np.array_split(chainmat,4)
second = mat_quar[1]
last = mat_quar[3]

# histogram diagonal elements for 2nd quarter
med = np.array(np.percentile(second, [50], axis=0)).reshape((n,n))
plus = np.array(np.percentile(second, [84], axis=0)).reshape((n,n))
minus = np.array(np.percentile(second, [16], axis=0)).reshape((n,n))

plt.figure(11)
plt.suptitle("Diagonal Elements of Matrix for 2nd Quarter of Simulation")

plt.subplot(row,col,1)
plt.title("Element 1,1")
plt.hist(second[:,0], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[0,0], linewidth=2, color='red')
plt.axvline(med[0,0], linewidth=2, color='blue')
plt.axvline(plus[0,0], linewidth=1.5, color='purple')
plt.axvline(minus[0,0], linewidth=1.5, color='purple')
plt.ylabel('PDF')

plt.subplot(row,col,2)
plt.title("Element 2,2")
plt.hist(second[:,5], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[1,1], linewidth=2, color='red')
plt.axvline(med[1,1], linewidth=2, color='blue')
plt.axvline(plus[1,1], linewidth=1.5, color='purple')
plt.axvline(minus[1,1], linewidth=1.5, color='purple')

plt.subplot(row,col,3)
plt.title("Element 3,3")
plt.hist(second[:,10], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[2,2], linewidth=2, color='red')
plt.axvline(med[2,2], linewidth=2, color='blue')
plt.axvline(plus[2,2], linewidth=1.5, color='purple')
plt.axvline(minus[2,2], linewidth=1.5, color='purple')
plt.xlabel('Element Value')
plt.ylabel('PDF')

plt.subplot(row,col,4)
plt.title("Element 4,4")
plt.hist(second[:,15], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[3,3], linewidth=2, color='red')
plt.axvline(med[3,3], linewidth=2, color='blue')
plt.axvline(plus[3,3], linewidth=1.5, color='purple')
plt.axvline(minus[3,3], linewidth=1.5, color='purple')
plt.xlabel('Element Value')

plt.savefig('mh_diag_second_quarter.pdf')

# histogram diagonal elements for last quarter
med = np.array(np.percentile(last, [50], axis=0)).reshape((n,n))
plus = np.array(np.percentile(last, [84], axis=0)).reshape((n,n))
minus = np.array(np.percentile(last, [16], axis=0)).reshape((n,n))

plt.figure(12)

plt.subplot(row,col,1)
plt.title("Element 1,1")
plt.hist(last[:,0], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[0,0], linewidth=2, color='red')
plt.axvline(med[0,0], linewidth=2, color='blue')
plt.axvline(plus[0,0], linewidth=1.5, color='purple')
plt.axvline(minus[0,0], linewidth=1.5, color='purple')
plt.ylabel('PDF')

plt.subplot(row,col,2)
plt.title("Element 2,2")
plt.hist(last[:,5], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[1,1], linewidth=2, color='red')
plt.axvline(med[1,1], linewidth=2, color='blue')
plt.axvline(plus[1,1], linewidth=1.5, color='purple')
plt.axvline(minus[1,1], linewidth=1.5, color='purple')

plt.subplot(row,col,3)
plt.title("Element 3,3")
plt.hist(last[:,10], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[2,2], linewidth=2, color='red')
plt.axvline(med[2,2], linewidth=2, color='blue')
plt.axvline(plus[2,2], linewidth=1.5, color='purple')
plt.axvline(minus[2,2], linewidth=1.5, color='purple')
plt.xlabel('Element Value')
plt.ylabel('PDF')

plt.subplot(row,col,4)
plt.title("Element 4,4")
plt.hist(last[:,15], bins=100)
plt.autoscale(enable=True, axis='x', tight=True)
plt.axvline(mat_inv[3,3], linewidth=2, color='red')
plt.axvline(med[3,3], linewidth=2, color='blue')
plt.axvline(plus[3,3], linewidth=1.5, color='purple')
plt.axvline(minus[3,3], linewidth=1.5, color='purple')
plt.xlabel('Element Value')

plt.savefig('mh_diag_last_quarter.pdf')