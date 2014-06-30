
import sys
import numpy as np
import emcee
from emcee.utils import MPIPool

log2P = np.log(2.0*np.pi)

def logPDF(x, M, sigL, sigP):
    n = M.shape[0]
    G = x.reshape(n,n).copy()
    Id = np.identity(n,float)

    R = np.dot(M,G) - Id
    logL = -0.5*( R.size*(log2P+np.log(sigL)) + np.sum(R*R)/(sigL*sigL) )
    logP = -0.5*( G.size*(log2P+np.log(sigP)) + np.sum(G*G)/(sigP*sigP) )

    logPDF = logL + logP
    return logPDF


mat = np.array([[1, 2, 3, 4],
                [2, 0, 1, 2],
                [3, 1, 3, 0],
                [4, 2, 0, 1]])

sigL = 0.1
sigP = 10.0

nwalkers = 250
ndim = mat.size

x0 = np.random.normal(size=nwalkers*ndim).reshape((nwalkers, ndim))

print(x0)

burn = 100
N = 100

#pool = MPIPool()
#if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

# initialize MCMC
MCMC = emcee.EnsembleSampler(nwalkers, ndim, logPDF, args=[mat, sigL, sigP])

print(" begin burn-in")
# burn in
pos, prob, state = MCMC.run_mcmc(x0, burn)
MCMC.reset()

print(" begin MCMC")
# actual run
MCMC.run_mcmc(pos, N)

#pool.close()

print(" end of run bookkeeping...")
result = np.array( list( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
              zip(*np.percentile(samp, [16, 50, 84], axis=0))) ))
med = result.reshape(3,Ndim)[0,:]
plus = result.reshape(3,Ndim)[1,:]
minus = result.reshape(3,Ndim)[2,:]

# write chain to file
mmi.io.print_chain(samp, 'chain.dat')

# write rescaled Minv to file
Minv = med.reshape(n,n)
MinvP = plus.reshape(n,n)
MinvM = minus.reshape(n,n)

rawMinv = Minv.copy()
#mmi.io.print_Mat(rawMinv, outfile_name)

# print stuff to command line
print("")
print("MCMC runtime: %.4f sec"%(t_end-t_start))
print("")
print("Mean acceptance: {0:.4f}"
                        .format(np.mean(eMCMC.acceptance_fraction)))
print("")

np.set_printoptions(precision=4)
print("M =") 
print(rawM)
print("")

I = np.dot(rawM,rawMinv)
print("M*Minv =")
print(I)
print("")

print("Minv =") 
print(rawMinv)
print("")

MinvTRUE = np.linalg.inv(rawM)
print("Minv TRUE =") 
print(MinvTRUE)
print("")

# TODO: fast fitting factor computation assumes no noise in data
HtHt = rawM.shape[0]
HmHm = np.sum( I*I )
HtHm = np.trace(I)
FF = HtHm/np.sqrt(HtHt*HmHm)
print("fitting factor = %.4f"%(FF))
print("")

