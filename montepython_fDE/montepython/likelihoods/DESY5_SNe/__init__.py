"""
.. module:: DESY5_SNe
    :synopsis: DES Y5 Supernova likelihood 

.. moduleauthor:: Tanvi Karwal 

Based heavily on the likelihood given with DESY5 Dovekie 
Major changes to read in DES Y5 SNe data, and to allow for option
to use only DES SNe or all SNe in the sample.

Preserving directions from og liklihood: 
The likelihood module for the DES-SN5YR sample
This likelihood analytically marginalize over M (SN absolute brightness).
M is fully degenerate with H0. Do not try to measure H0 with SN data

.. code::

    C00 = mag_covmat_file

.. note::

    Since there are a lot of file manipulation involved, the "pandas" library
    has to be installed -- it is an 8-fold improvement in speed over numpy, and
    a 2-fold improvement over a fast Python implementation. The "numexpr"
    library is also needed for doing the fast array manipulations, done with
    blas daxpy function in the original c++ code. Both can be installed with
    pip (Python package manager) easily.

"""
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood_sn
from astropy.table import Table
import os


class DESY5_SNe(Likelihood_sn):

    def __init__(self, path, data, command_line):

	    #Read the data and covariance matrix
        try:
            Likelihood_sn.__init__(self, path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The DESY5_SNe data files were not found. Please check if "
                "the following files are in the data/DESY5_SNe directory: "
                "\n-> DESY5_SNe.dataset"
                "\n-> DES-Dovekie_HD.csv"
                "\n-> STAT+SYS.npz")

        # are there conflicting experiments?
        conflicting_experiments = [
            'Pantheon', 'Pantheon_Plus_SH0ES', 'Pantheon_Plus']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'DESY5_SNe reports conflicting SN or H0 measurments from: %s' %(experiment))
            
        # Read in and construct the data vector 
        filename = os.path.join(self.data_directory, self.data_file)
        print("Loading Dovekie SN data from {} with the column names: ".format(filename))
        data = Table.read(  filename, 
                            format='ascii.csv', 
                            comment='#', 
                            delimiter='\s',
                            )
        self.origlen = len(data)
        print(data.colnames)

        # The only columns that we actually need here are the redshift,
        # distance modulus and distance modulus error for the likelihood 
        self.ww = (data['zHD']>0.00) 
        #use the vpec corrected redshift for zCMB 
        self.zCMB = data['zHD'][self.ww] 
        self.zHEL = data['zHEL'][self.ww]
        # distance modulus and relative stat uncertainties. Note MUERR instead or MUERR_FINAL
        self.mu_obs = data['MU'][self.ww]
        self.mu_obs_err = data['MUERR'][self.ww]

        # We also want the survey IDs in case we want to cut to only DES SNe later
        self.survey_id = data['IDSURVEY'][self.ww]


        # Next build the covariance matrix
        filename = os.path.join(self.data_directory, self.covmat_file)
        print("Loading Dovekie SN covariance from {}".format(filename))

        # The file format for the covariance has been changed in the new SNANA version in use after DES
        # Covtot_inv is the inverse of stat+sys because it makes no sense to invert covsys
        # This data file is stored with .npz (which will work with np.load)
        # in upper triangular format
        # changes are made to accomodate anticipated much larger files from LSST
        d = np.load(filename)
        n = d[d.files[0]][0]
        inv_cov = np.zeros((n, n))
        inv_cov[np.triu_indices(n)] = d[d.files[1]]

        # Reflect to lower triangular part to make it symmetric
        i_lower = np.tril_indices(n, -1)
        inv_cov[i_lower] = inv_cov.T[i_lower]
	    # Unfortunately, to make this work in cosmosis we need to invert the inverse covariance and return that ... sigh ...
        C = np.linalg.inv(inv_cov)

        # Return the covariance; the parent class knows to invert this
        # later to get the precision matrix that we need for the likelihood.
        C = C[self.ww][:, self.ww]


        # Next we check if we want to retain just DES SNe or all SNe
        # The data are ordered roughly by survey which simplifies removing everything that is not DES 
        # the DES survey ID is 10. We keep that, remove everything else. 
        # We also save the mask for later use in loglkl
        if self.DES_only: 
            self.mask = self.survey_id == 10
        else:
            self.mask = self.survey_id > -np.inf
        # Apply the mask to the covariance matrix
        newcov = C[np.ix_(self.mask, self.mask)]
        # And apply to the data vectors 
        self.zCMB = self.zCMB[self.mask]
        self.zHEL = self.zHEL[self.mask]
        self.mu_obs = self.mu_obs[self.mask]
        self.mu_obs_err = self.mu_obs_err[self.mask]
        print("Number of SNe used in likelihood after any masking: ", len(self.mu_obs))

        # Finally, save the inverse covariance for use in loglkl
        self.inv_cov = np.linalg.inv(newcov)


    def loglkl(self, cosmo, data):
        """
        Compute negative log-likelihood (eq.15 Betoule et al. 2014)

        """
        # Recover the distance moduli from CLASS (a size N vector of double
        # containing the predicted distance modulus for each SN in the JLA
        # sample, given the redshift of the supernova.)

	    # Masking these according to whether we are keeping only DES SNe or all SNe
        theory_mu = np.empty((np.sum(self.mask), ))
        obs_mu = np.empty((np.sum(self.mask), ))
        good_z = 0
        # Loop over all SNe to compute the moduli at their redshifts with the mask 
        for index in self.zCMB:
            z_cmb = self.zCMB[good_z]
            z_hel = self.zHEL[good_z]
            Mb_corr = self.mu_obs[good_z]

            theory_mu[good_z] = 5.0 * np.log10((1.0 + z_cmb) * (1.0 + z_hel)*cosmo.angular_distance(z_cmb)) + 25.
            obs_mu[good_z] = Mb_corr
            good_z+=1

        # Compute the residuals (estimate of distance moduli - exact moduli)
        delta = np.array([theory_mu - obs_mu])
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ self.inv_cov @ deltaT)
        B = np.sum(delta @ self.inv_cov)
        C = np.sum(self.inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))
        return -0.5*chi2