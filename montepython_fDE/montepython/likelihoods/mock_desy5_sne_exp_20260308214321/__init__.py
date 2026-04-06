import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
from astropy.table import Table


class mock_desy5_sne_exp_20260308214321(Likelihood):

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        # Read mock SN data
        filename = os.path.join(self.data_directory, self.data_file)
        sn_data = Table.read(filename, format='ascii.csv', comment='#', delimiter='\s')

        self.zCMB = np.array(sn_data['zHD'], dtype=np.float64)
        self.zHEL = np.array(sn_data['zHEL'], dtype=np.float64)
        self.mu_obs = np.array(sn_data['MU'], dtype=np.float64)
        self.num_sn = len(self.zCMB)

        # Read inverse covariance (upper triangular npz)
        cov_file = os.path.join(self.data_directory, self.cov_file)
        d = np.load(cov_file)
        n = d[d.files[0]][0]
        self.inv_cov = np.zeros((n, n))
        self.inv_cov[np.triu_indices(n)] = d[d.files[1]]
        i_lower = np.tril_indices(n, -1)
        self.inv_cov[i_lower] = self.inv_cov.T[i_lower]

    def loglkl(self, cosmo, data):
        theory_mu = np.zeros(self.num_sn)
        for i in range(self.num_sn):
            z_cmb = self.zCMB[i]
            z_hel = self.zHEL[i]
            theory_mu[i] = 5.0 * np.log10(
                (1.0 + z_cmb) * (1.0 + z_hel) * cosmo.angular_distance(z_cmb)
            ) + 25.0

        # Analytic marginalization over M (Betoule+ 2014, eq. 15)
        delta = theory_mu - self.mu_obs
        chit2 = delta @ self.inv_cov @ delta
        B = np.sum(self.inv_cov @ delta)
        C = np.sum(self.inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2.0 * np.pi))
        return -0.5 * chi2
