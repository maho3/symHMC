import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import copy
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

class HMC():
    """
    Hamiltonian Monte Carlo sampler
    https://mc-stan.org/docs/2_21/reference-manual/hamiltonian-monte-carlo.html
    """
    
    def __init__(self, logp, dlogp, dt, L, M, n_args, prop='leapfrog'):
        """
        Arguments:
            logp: 
                Function which accepts two inputs (args, data) and returns log
                probability of the objective model
                
            dlogp:
                Function which which accepts two inputs (args, data) and returns the
                gradient of the log probability w.r.t. the parameters as a np array
                
            dt: Time step
            
            L: Number of leapfrog iterations to calculate
            
            M: covariance of momentum sampling multivariate normal
            
            n_args: number of parameters to sample
        """
        self.logp = logp
        self.dlogp = dlogp
        self.dt = dt
        self.L = L
        self.M = M
        self.n_args = n_args
        
        if prop=='leapfrog':
            self.proposal = self.prop_lf
        elif prop=='yoshida':
            self.proposal = self.prop_yo
        else:
            raise
    
    def U(self, x, data):
        return -self.logp(x,data)
    
    def dU(self, x, data):
        return self.dlogp(x,data)
    
    def K(self, v):
        return -mvnorm.logpdf(v, mean=np.zeros(self.n_args), cov=self.M)
    
    def prop_lf(self, x, v, data):
        # Leapfrog
        # https://en.wikiself.dlogp(x, data)pedia.org/wiki/Leapfrog_integration
        x, v = copy.deepcopy((x,v))

        for _ in range(self.L):
            v += (self.dt/2)*self.dU(x, data)
            x += self.dt*np.dot(np.linalg.inv(self.M), v)
            v += (self.dt/2)*self.dU(x, data)
        
        return x, v
    
    def prop_yo(self, x, v, data):
        # 4th order Yoshida integrator
        # https://en.wikiself.dlogp(x, data)pedia.org/wiki/Leapfrog_integration
        x, v = copy.deepcopy((x,v))
        
        crt2 = (2.**(1./3))
        w0 = -crt2/(2-crt2)
        w1 = 1./(2-crt2)
        c1=c4=w1/2
        c2=c3=(w0+w1)/2
        d1=d3=w1
        d2=w0
        for _ in range(self.L):
            x += c1*np.dot(np.linalg.inv(self.M), v)*self.dt
            v += d1*self.dU(x, data)*self.dt
            x += c2*np.dot(np.linalg.inv(self.M), v)*self.dt
            v += d2*self.dU(x, data)*self.dt
            x += c3*np.dot(np.linalg.inv(self.M), v)*self.dt
            v += d3*self.dU(x, data)*self.dt
            x += c4*np.dot(np.linalg.inv(self.M), v)*self.dt

        return x, v
    
    
    def sample(self, N_samp, data, init_x=None, verbose=False):
        """Samples one HMC chain for N_samp samples."""
        
        if init_x is None:
            x = 2*np.random.rand(self.n_args)-0.5
        else:
            x = init_x
            
        xs = np.zeros(shape=(N_samp, self.n_args))
        xs[0] = x
        
        i_s = range(1,N_samp)
        if verbose: # Print progress bar
            i_s = tqdm(i_s)
        
        for i in i_s:
            v = mvnorm.rvs(mean=np.zeros(self.n_args), cov=self.M)
            xnew, vnew = self.proposal(x, v, data)

            alpha = np.exp(self.U(x, data) +  self.K(v)
                           - self.U(xnew, data) - self.K(vnew))

            if np.random.rand() <= min(1, alpha):
                xs[i] = xnew
                x = xnew
            else:
                xs[i] = xs[i-1]
                
        return xs
    
    def _pool_helper(self, N_samp, data, init_x, verbose, seed):
        np.random.seed(seed)
        return self.sample(N_samp, data, init_x, verbose)
    
    def sample_pool(self, N_samp, data, N_chains, init_x=None, verbose=False):
        """Samples N_chains HMC chains in parallel, each for N_samp samples"""
        
        f = partial(self._pool_helper, N_samp, data, init_x, verbose)
        
        with mp.Pool(N_chains, initargs=(N_samp, data, init_x)) as pool:
            out = np.array(pool.map(f, 
                                    np.random.randint(0, 1e5, N_chains),
                                    ))
            
        return out
