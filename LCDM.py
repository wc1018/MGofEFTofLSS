import os
import numpy as np
import h5py as h5
from montepython.likelihood_class import Likelihood

class wc_LCDM(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        
        # load data
        with h5.File(os.path.join(self.data_directory, self.data_file ), 'r') as f:
            self.k = f['k'][()]
            self.pk = f['pk_mean'][()]
            cov = f['pk_cov'][()]
            self.z = f['z'][()]
            sample = f['pk_sample'][()]
       
        N_s = sample.shape[0]
        N_d = sample.shape[1]
        # N_t = len(data.mcmc_parameters)
        N_t = self.Nt
        
        def rescale_P(N_s, N_d, N_t):
            B = (N_s-N_d-2)/((N_s-N_d-1)*(N_s-N_d-4))
            P = (N_s-1)*(1+B*(N_d-N_t))/(N_s-N_d+N_t-1)
            return P
            
        
        self.cov = rescale_P(N_s, N_d, N_t)*cov


        self.need_cosmo_arguments(data, {'output':'mPk', 'P_k_max_1/Mpc':3.0})
        # end of initialization
        
    # compute likelihood

    def loglkl(self, cosmo, data):

        theory_pk = []
        h = cosmo.h()
        for k in self.k:
             theory_pk.append(cosmo.pk(k*h, self.z)*h**3)
            
        theory_pk = np.array(theory_pk)
        data_pk = self.pk
        cov = self.cov
        diff = theory_pk - data_pk
    
        
        
        # final chi square
        cov_inv = np.linalg.inv(cov)
        chi_squared = diff.T @ cov_inv @ diff

        # return ln(L)
        lglkl = - 0.5 * chi_squared

        return lglkl
