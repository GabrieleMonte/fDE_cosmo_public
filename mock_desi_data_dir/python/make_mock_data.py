import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.ndimage import gaussian_filter
import datetime

class DESY5_like_data:
    def __init__(self, data_file, covmat_file, DES_only=False):
        self.data_file = data_file
        self.covmat_file = covmat_file

        # Read the CSV using astropy Table (same as the real likelihood)
        from astropy.table import Table
        raw = Table.read(data_file, format='ascii.csv', comment='#', delimiter='\s')

        # Mask out z <= 0 (same as real likelihood: data['zHD'] > 0)
        ww = (raw['zHD'] > 0.0)
        self.zCMB = np.array(raw['zHD'][ww], dtype=np.float64)
        self.zHEL = np.array(raw['zHEL'][ww], dtype=np.float64)
        self.mu_obs = np.array(raw['MU'][ww], dtype=np.float64)
        self.mu_obs_err = np.array(raw['MUERR'][ww], dtype=np.float64)
        self.survey_id = np.array(raw['IDSURVEY'][ww], dtype=int)

        # Keep the full raw table for writing mock CSVs later
        self._raw_table = raw
        self._ww = ww
        # Column names for writing
        self._colnames = raw.colnames

        # Load inverse covariance (upper triangular in npz), invert to get cov
        d = np.load(covmat_file, allow_pickle=True)
        n = d[d.files[0]][0]
        inv_cov_full = np.zeros((n, n))
        inv_cov_full[np.triu_indices(n)] = d[d.files[1]]
        i_lower = np.tril_indices(n, -1)
        inv_cov_full[i_lower] = inv_cov_full.T[i_lower]
        cov_full = np.linalg.inv(inv_cov_full)

        # Apply z > 0 mask to covariance
        cov_full = cov_full[ww][:, ww]

        # Optionally keep only DES SNe (survey_id == 10)
        if DES_only:
            self.mask = self.survey_id == 10
        else:
            self.mask = np.ones(len(self.zCMB), dtype=bool)

        self.cov = cov_full[np.ix_(self.mask, self.mask)]
        self.zCMB = self.zCMB[self.mask]
        self.zHEL = self.zHEL[self.mask]
        self.mu_obs = self.mu_obs[self.mask]
        self.mu_obs_err = self.mu_obs_err[self.mask]
        self.survey_id = self.survey_id[self.mask]
        self.num_sn = len(self.zCMB)

    def make_fake_DESY5_data(self, cosmo, mean_noise=False, seed=None, return_theory=False):
        """Generate mock DESY5 distance moduli from a cosmological model."""
        theory_mu = np.zeros(self.num_sn)
        for i in range(self.num_sn):
            z_cmb = self.zCMB[i]
            z_hel = self.zHEL[i]
            theory_mu[i] = 5.0 * np.log10(
                (1.0 + z_cmb) * (1.0 + z_hel) * cosmo.angular_distance(z_cmb)
            ) + 25.0

        if mean_noise:
            rng = np.random.default_rng(seed)
            noise = rng.multivariate_normal(
                mean=np.zeros(self.num_sn), cov=self.cov
            )
            mock = theory_mu + noise
        else:
            mock = theory_mu.copy()

        out = {'dat': mock, 'cov': self.cov}
        if return_theory:
            out['th'] = theory_mu
        return out

    def make_fake_likelihood(self, cosmo, model, mean_noise=False, seed=None, extra_lk=None):
        """Create a MontePython DESY5-like SNe likelihood from mock data.

        Parameters
        ----------
        extra_lk : str, optional
            Name of an additional likelihood (e.g. a BAO likelihood) to include
            in the generated param files.
        """
        result = self.make_fake_DESY5_data(cosmo, mean_noise=mean_noise, seed=seed)
        mock_mu = result['dat']
        mock_cov = result['cov']

        now = datetime.datetime.now()
        tag = model + "_" + now.strftime("%Y%m%d%H%M%S")
        lk_name = "mock_desy5_sne_" + tag

        lk_dir = "../../montepython_fDE/montepython/likelihoods/" + lk_name
        dat_dir = "../../montepython_fDE/data/" + lk_name

        if not os.path.exists(lk_dir):
            os.makedirs(lk_dir)
        if not os.path.exists(dat_dir):
            os.makedirs(dat_dir)

        # --- write mock CSV (same format as DES-Dovekie_HD.csv) ---
        csv_filename = lk_name + "_HD.csv"
        csv_path = os.path.join(dat_dir, csv_filename)
        with open(csv_path, 'w') as f:
            f.write("# Mock DESY5-like SNe data\n")
            f.write("# model: {}\n".format(model))
            f.write("# Cosmological parameters: ")
            for key, val in cosmo.pars.items():
                f.write("{}={}, ".format(key, val))
            f.write("\n")
            f.write("VARNAMES: CID IDSURVEY zHD zHEL MU MUERR MUERR_VPEC MUERR_SYS PROBIA_BEAMS\n")
            # Write rows using original table structure but with mock MU
            raw = self._raw_table
            ww = self._ww
            # Build index mapping: filtered+masked rows
            ww_indices = np.where(ww)[0]
            if hasattr(self, 'mask'):
                masked_indices = ww_indices[self.mask]
            else:
                masked_indices = ww_indices
            for i, row_idx in enumerate(masked_indices):
                row = raw[int(row_idx)]
                f.write("SN: {:<12s} {:>3d} {:.5f} {:.5f} {:.5f}   {:.5f}  {:.5f}  {:.5f}  {:.5f}\n".format(
                    str(row['CID']), int(row['IDSURVEY']),
                    float(row['zHD']), float(row['zHEL']),
                    mock_mu[i],
                    float(row['MUERR']), float(row['MUERR_VPEC']),
                    float(row['MUERR_SYS']), float(row['PROBIA_BEAMS']),
                ))

        # --- write inverse covariance in npz (upper triangular, same format) ---
        cov_filename = lk_name + "_cov.npz"
        cov_path = os.path.join(dat_dir, cov_filename)
        inv_cov = np.linalg.inv(mock_cov)
        n = inv_cov.shape[0]
        np.savez(cov_path,
                 np.array([n]),
                 inv_cov[np.triu_indices(n)])

        # --- write __init__.py ---
        init_path = os.path.join(lk_dir, "__init__.py")
        init_content = '''import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
from astropy.table import Table


class {class_name}(Likelihood):

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        # Read mock SN data
        filename = os.path.join(self.data_directory, self.data_file)
        sn_data = Table.read(filename, format='ascii.csv', comment='#', delimiter='\\s')

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
'''.format(class_name=lk_name)

        with open(init_path, 'w') as f:
            f.write(init_content)

        # --- write .data file ---
        data_file_path = os.path.join(lk_dir, lk_name + ".data")
        with open(data_file_path, 'w') as f:
            f.write("# mock DESY5-like SNe data\n")
            f.write("# model: {}\n".format(model))
            f.write("{}.data_directory      = data.path['data']\n".format(lk_name))
            f.write("{}.data_file           = '{}/{}'\n".format(
                lk_name, lk_name, csv_filename))
            f.write("{}.cov_file            = '{}/{}'\n".format(
                lk_name, lk_name, cov_filename))

        print("Created likelihood: {}".format(lk_name))
        print("  Likelihood dir: {}".format(lk_dir))
        print("  Data dir:       {}".format(dat_dir))

        # Generate param files
        self.make_param_files(lk_name, extra_lk=extra_lk)

        return lk_name

    def make_param_files(self, lk_name, extra_lk=None):
        """Generate MontePython .param files for 5 analysis models,
        using this SNe likelihood, Qcmb, and optionally an extra likelihood.

        Parameters
        ----------
        lk_name : str
            Name of this SNe likelihood.
        extra_lk : str, optional
            Name of an additional likelihood (e.g. a BAO likelihood) to include.
        """
        output_dir = "../../montepython_fDE/mock_desi_like_input/"

        if extra_lk:
            exp_list = "'{lk}','{extra}','Qcmb'".format(lk=lk_name, extra=extra_lk)
        else:
            exp_list = "'{lk}','Qcmb'".format(lk=lk_name)

        common_header = (
            "#------Experiments to test (separated with commas)-----\n"
            "data.experiments=[" + exp_list + "]\n"
            "\n"
            "data.over_sampling=[1]\n"
            "\n"
            "# Cosmological parameters list\n"
            "\n"
            "data.parameters['omega_b']      = [  2.2377,   None, None,      0.015, 0.01, 'cosmo']\n"
            "data.parameters['omega_cdm']    = [ 0.131,   None, None,     0.0013,    1, 'cosmo']\n"
            "data.parameters['h']            = [ 0.68,     .2, 1,       0.01,    1, 'cosmo']\n"
        )

        omega_lambda_zero = "data.cosmo_arguments['Omega_Lambda'] = 0\n"

        derived_block = (
            "\n"
            "# Derived parameters\n"
            "data.parameters['H0']              = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_Lambda']    = [1, None, None, 0,     1,   'derived']\n"
            "data.parameters['theta_s_100']     = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_m']         = [1, None, None, 0,     1,   'derived']\n"
        )

        fixed_block = (
            "\n"
            "# Other cosmo parameters (fixed parameters, precision parameters, etc.)\n"
            "data.cosmo_arguments['sBBN file'] = data.path['cosmo']+'/external/bbn/sBBN.dat'\n"
            "{eos_line}"
            "{extra_fixed}"
            "data.cosmo_arguments['k_pivot'] = 0.05\n"
            "\n"
            "data.cosmo_arguments['m_ncdm'] = 0.02\n"
            "data.cosmo_arguments['N_ur'] = 0.00441\n"
            "data.cosmo_arguments['N_ncdm'] = 1\n"
            "data.cosmo_arguments['deg_ncdm'] = 3\n"
            "data.cosmo_arguments['T_ncdm'] = 0.71611\n"
            "\n"
            "data.cosmo_arguments['ln10^{{10}}A_s'] =   3.036\n"
            "data.cosmo_arguments['n_s']          =   0.9649\n"
            "data.cosmo_arguments['tau_reio']     =   0.0544\n"
            "data.cosmo_arguments['output']\t=\t''\n"
        )

        mcmc_block = (
            "#------ Mcmc parameters ----\n"
            "\n"
            "data.N=10\n"
            "data.write_step=5\n"
        )

        models = {
            'lcdm': {
                'extra_params': '',
                'eos_line': '',
                'extra_fixed': '',
            },
            'fp': {
                'extra_params': omega_lambda_zero + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld'] = 2/3\n",
            },
            'fa': {
                'extra_params': omega_lambda_zero + "data.parameters['fa_fld']\t= [\t0,\t-2,\t2, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'faDE'\n",
                'extra_fixed': '',
            },
            'fp_wp': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n"
                    "data.parameters['wp_fld']\t= [\t0,\t-3,\t1, \t0.2,\t1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE_2'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld']= 2/3\n",
            },
            'w0wa': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['w0_fld']       = [    -1.0,   -3.0,     1.0,        0.1,    1, 'cosmo']\n"
                    "data.parameters['w0wa_fld']         = [    -1.0,   -5.0,     0.0,        0.1,    1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'CLP'\n",
                'extra_fixed': '',
            },
        }

        for model_tag, cfg in models.items():
            content = common_header
            content += cfg['extra_params']
            content += derived_block
            content += fixed_block.format(
                eos_line=cfg['eos_line'],
                extra_fixed=cfg['extra_fixed'],
            )
            content += mcmc_block

            if extra_lk:
                fname = "{lk}_{extra}_Qcmb_{model}.param".format(
                    lk=lk_name, extra=extra_lk, model=model_tag)
            else:
                fname = "{lk}_Qcmb_{model}.param".format(lk=lk_name, model=model_tag)
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'w') as f:
                f.write(content)
            print("  Wrote param file: {}".format(fpath))  # DESY5


class LSSTY3_like_data:
    def __init__(self, data_file, covmat_file):
        self.data_file = data_file
        self.covmat_file = covmat_file

        # Read the CSV using astropy Table (same as the real likelihood)
        from astropy.table import Table
        raw = Table.read(data_file, format='ascii.csv', comment='#', delimiter='\s')

        # Mask out z <= 0 (same as real likelihood: data['zHD'] > 0)
        ww = (raw['zHD'] > 0.0)
        self.zCMB = np.array(raw['zHD'][ww], dtype=np.float64)
        self.zHEL = np.array(raw['zHEL'][ww], dtype=np.float64)
        self.mu_obs = np.array(raw['MU'][ww], dtype=np.float64)
        self.mu_obs_err = np.array(raw['MUERR'][ww], dtype=np.float64)

        # Keep the full raw table for writing mock CSVs later
        self._raw_table = raw
        self._ww = ww
        self._colnames = raw.colnames

        # Load inverse covariance (upper triangular in npz), invert to get cov
        d = np.load(covmat_file, allow_pickle=True)
        n = d[d.files[0]][0]
        inv_cov_full = np.zeros((n, n))
        inv_cov_full[np.triu_indices(n)] = d[d.files[1]]
        i_lower = np.tril_indices(n, -1)
        inv_cov_full[i_lower] = inv_cov_full.T[i_lower]
        cov_full = np.linalg.inv(inv_cov_full)

        # Apply z > 0 mask to covariance
        self.cov = cov_full[ww][:, ww]
        self.num_sn = len(self.zCMB)

    def make_fake_LSSTY3_data(self, cosmo, mean_noise=False, seed=None, return_theory=False):
        """Generate mock LSST Y3 distance moduli from a cosmological model."""
        theory_mu = np.zeros(self.num_sn)
        for i in range(self.num_sn):
            z_cmb = self.zCMB[i]
            z_hel = self.zHEL[i]
            theory_mu[i] = 5.0 * np.log10(
                (1.0 + z_cmb) * (1.0 + z_hel) * cosmo.angular_distance(z_cmb)
            ) + 25.0

        if mean_noise:
            rng = np.random.default_rng(seed)
            noise = rng.multivariate_normal(
                mean=np.zeros(self.num_sn), cov=self.cov
            )
            mock = theory_mu + noise
        else:
            mock = theory_mu.copy()

        out = {'dat': mock, 'cov': self.cov}
        if return_theory:
            out['th'] = theory_mu
        return out

    def make_fake_likelihood(self, cosmo, model, mean_noise=False, seed=None, extra_lk=None):
        """Create a MontePython LSSTY3-like SNe likelihood from mock data.

        Parameters
        ----------
        extra_lk : str, optional
            Name of an additional likelihood (e.g. a BAO likelihood) to include
            in the generated param files.
        """
        result = self.make_fake_LSSTY3_data(cosmo, mean_noise=mean_noise, seed=seed)
        mock_mu = result['dat']
        mock_cov = result['cov']

        now = datetime.datetime.now()
        tag = model + "_" + now.strftime("%Y%m%d%H%M%S")
        lk_name = "mock_lssty3_sne_" + tag

        lk_dir = "../../montepython_fDE/montepython/likelihoods/" + lk_name
        dat_dir = "../../montepython_fDE/data/" + lk_name

        if not os.path.exists(lk_dir):
            os.makedirs(lk_dir)
        if not os.path.exists(dat_dir):
            os.makedirs(dat_dir)

        # --- write mock CSV (same format as LSSTY3_HD.csv) ---
        csv_filename = lk_name + "_HD.csv"
        csv_path = os.path.join(dat_dir, csv_filename)
        with open(csv_path, 'w') as f:
            f.write("# Mock LSSTY3-like SNe data\n")
            f.write("# model: {}\n".format(model))
            f.write("# Cosmological parameters: ")
            for key, val in cosmo.pars.items():
                f.write("{}={}, ".format(key, val))
            f.write("\n")
            f.write("VARNAMES: CID IDSURVEY zHD zHEL MU MUERR MUERR_VPEC MUERR_SYS PROBIA_BEAMS\n")
            raw = self._raw_table
            ww = self._ww
            ww_indices = np.where(ww)[0]
            for i, row_idx in enumerate(ww_indices):
                row = raw[int(row_idx)]
                f.write("SN: {:<12s} {:>3d} {:.5f} {:.5f} {:.5f}   {:.5f}  {:.5f}  {:.5f}  {:.5f}\n".format(
                    str(row['CID']), int(row['IDSURVEY']),
                    float(row['zHD']), float(row['zHEL']),
                    mock_mu[i],
                    float(row['MUERR']), float(row['MUERR_VPEC']),
                    float(row['MUERR_SYS']), float(row['PROBIA_BEAMS']),
                ))

        # --- write inverse covariance in npz (upper triangular, same format) ---
        cov_filename = lk_name + "_cov.npz"
        cov_path = os.path.join(dat_dir, cov_filename)
        inv_cov = np.linalg.inv(mock_cov)
        n = inv_cov.shape[0]
        np.savez(cov_path,
                 np.array([n]),
                 inv_cov[np.triu_indices(n)])

        # --- write __init__.py ---
        init_path = os.path.join(lk_dir, "__init__.py")
        init_content = '''import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
from astropy.table import Table


class {class_name}(Likelihood):

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        # Read mock SN data
        filename = os.path.join(self.data_directory, self.data_file)
        sn_data = Table.read(filename, format='ascii.csv', comment='#', delimiter='\\s')

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
'''.format(class_name=lk_name)

        with open(init_path, 'w') as f:
            f.write(init_content)

        # --- write .data file ---
        data_file_path = os.path.join(lk_dir, lk_name + ".data")
        with open(data_file_path, 'w') as f:
            f.write("# mock LSSTY3-like SNe data\n")
            f.write("# model: {}\n".format(model))
            f.write("{}.data_directory      = data.path['data']\n".format(lk_name))
            f.write("{}.data_file           = '{}/{}'\n".format(
                lk_name, lk_name, csv_filename))
            f.write("{}.cov_file            = '{}/{}'\n".format(
                lk_name, lk_name, cov_filename))

        print("Created likelihood: {}".format(lk_name))
        print("  Likelihood dir: {}".format(lk_dir))
        print("  Data dir:       {}".format(dat_dir))

        # Generate param files
        self.make_param_files(lk_name, extra_lk=extra_lk)

        return lk_name

    def make_param_files(self, lk_name, extra_lk=None):
        """Generate MontePython .param files for 5 analysis models.

        extra_lk can be a string or a list of strings for multiple extra likelihoods.
        """
        output_dir = "../../montepython_fDE/mock_desi_like_input/"

        if extra_lk:
            if isinstance(extra_lk, str):
                extras = [extra_lk]
            else:
                extras = list(extra_lk)
            all_exps = [lk_name] + extras + ['Qcmb']
            exp_list = ",".join(["'{}'".format(e) for e in all_exps])
            extra_tag = "_".join(extras)
        else:
            exp_list = "'{lk}','Qcmb'".format(lk=lk_name)
            extra_tag = None

        common_header = (
            "#------Experiments to test (separated with commas)-----\n"
            "data.experiments=[" + exp_list + "]\n"
            "\n"
            "data.over_sampling=[1]\n"
            "\n"
            "# Cosmological parameters list\n"
            "\n"
            "data.parameters['omega_b']      = [  2.2377,   None, None,      0.015, 0.01, 'cosmo']\n"
            "data.parameters['omega_cdm']    = [ 0.131,   None, None,     0.0013,    1, 'cosmo']\n"
            "data.parameters['h']            = [ 0.68,     .2, 1,       0.01,    1, 'cosmo']\n"
        )

        omega_lambda_zero = "data.cosmo_arguments['Omega_Lambda'] = 0\n"

        derived_block = (
            "\n"
            "# Derived parameters\n"
            "data.parameters['H0']              = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_Lambda']    = [1, None, None, 0,     1,   'derived']\n"
            "data.parameters['theta_s_100']     = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_m']         = [1, None, None, 0,     1,   'derived']\n"
        )

        fixed_block = (
            "\n"
            "# Other cosmo parameters (fixed parameters, precision parameters, etc.)\n"
            "data.cosmo_arguments['sBBN file'] = data.path['cosmo']+'/external/bbn/sBBN.dat'\n"
            "{eos_line}"
            "{extra_fixed}"
            "data.cosmo_arguments['k_pivot'] = 0.05\n"
            "\n"
            "data.cosmo_arguments['m_ncdm'] = 0.02\n"
            "data.cosmo_arguments['N_ur'] = 0.00441\n"
            "data.cosmo_arguments['N_ncdm'] = 1\n"
            "data.cosmo_arguments['deg_ncdm'] = 3\n"
            "data.cosmo_arguments['T_ncdm'] = 0.71611\n"
            "\n"
            "data.cosmo_arguments['ln10^{{10}}A_s'] =   3.036\n"
            "data.cosmo_arguments['n_s']          =   0.9649\n"
            "data.cosmo_arguments['tau_reio']     =   0.0544\n"
            "data.cosmo_arguments['output']\t=\t''\n"
        )

        mcmc_block = (
            "#------ Mcmc parameters ----\n"
            "\n"
            "data.N=10\n"
            "data.write_step=5\n"
        )

        models = {
            'lcdm': {
                'extra_params': '',
                'eos_line': '',
                'extra_fixed': '',
            },
            'fp': {
                'extra_params': omega_lambda_zero + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld'] = 2/3\n",
            },
            'fa': {
                'extra_params': omega_lambda_zero + "data.parameters['fa_fld']\t= [\t0,\t-2,\t2, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'faDE'\n",
                'extra_fixed': '',
            },
            'fp_wp': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n"
                    "data.parameters['wp_fld']\t= [\t0,\t-3,\t1, \t0.2,\t1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE_2'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld']= 2/3\n",
            },
            'w0wa': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['w0_fld']       = [    -1.0,   -3.0,     1.0,        0.1,    1, 'cosmo']\n"
                    "data.parameters['w0wa_fld']         = [    -1.0,   -5.0,     0.0,        0.1,    1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'CLP'\n",
                'extra_fixed': '',
            },
        }

        for model_tag, cfg in models.items():
            content = common_header
            content += cfg['extra_params']
            content += derived_block
            content += fixed_block.format(
                eos_line=cfg['eos_line'],
                extra_fixed=cfg['extra_fixed'],
            )
            content += mcmc_block

            if extra_tag:
                fname = "{lk}_{extra}_Qcmb_{model}.param".format(
                    lk=lk_name, extra=extra_tag, model=model_tag)
            else:
                fname = "{lk}_Qcmb_{model}.param".format(lk=lk_name, model=model_tag)
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'w') as f:
                f.write(content)
            print("  Wrote param file: {}".format(fpath))


class DESI_like_data:
    def __init__(self, mean_file, cov_file):
        self.mean_file = mean_file
        self.cov_file = cov_file
        self.z = np.array([], 'float64')
        self.data_array = np.array([], 'float64')
        self.quantity = []
        # read redshifts and data points
        with open(self.mean_file, 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.data_array = np.append(self.data_array, float(this_line[1]))
                    self.quantity.append(str(this_line[2]))
        # read covariance matrix
        self.cov_data = np.loadtxt(self.cov_file)
        # number of bins
        self.num_bins = np.shape(self.z)[0]
        # number of data points
        self.num_points = np.shape(self.cov_data)[0]
    def make_fake_DESI_data(self, cosmo, mean_noise=False, seed=None, return_theory=False):
        # make fake data by adding Gaussian noise to the theory array
        # the noise is drawn from a multivariate normal distribution with mean 0 and covariance given by the covariance matrix
        theory = np.zeros(self.num_bins)
        rs = cosmo.rs_drag()
        for i in range(self.num_bins):
            DM_at_z = cosmo.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo.Hubble(self.z[i])
            theo_DM_over_rs = DM_at_z / rs
            theo_DH_over_rs = 1. / H_at_z / rs
            theo_DV_over_rs = (self.z[i] * DM_at_z**2 / H_at_z)**(1./3.) / rs
            # calculate theory predictions
            if self.quantity[i] == 'DV_over_rs':
                theory[i] = theo_DV_over_rs
            elif self.quantity[i] == 'DM_over_rs':
                theory[i] = theo_DM_over_rs
            elif self.quantity[i] == 'DH_over_rs':
               theory[i] = theo_DH_over_rs
        rng = np.random.default_rng(seed)
        noise = rng.multivariate_normal(
            mean=np.zeros(self.num_bins, dtype=np.float64),
            cov=self.cov_data
        )
        if mean_noise:
            mock = theory + noise
        else:
            mock = theory.copy()
        fake_cov = self.cov_data
        if return_theory:
            return {'dat':mock, 'cov':fake_cov, 'th':theory}
        else:
            return {'dat':mock, 'cov':fake_cov}
    def make_fake_likelihood(self, cosmo, model,mean_noise=False,seed=False):
        # generate the mock data
        result = self.make_fake_DESI_data(cosmo, mean_noise=mean_noise,seed=seed)
        mock_data = result['dat']
        mock_cov = result['cov']

        # create unique name based on model and timestamp
        now = datetime.datetime.now()
        tag = model + "_" + now.strftime("%Y%m%d%H%M%S")
        lk_name = "mock_bao_desi_dr2_" + tag

        lk_dir = "../../montepython_fDE/montepython/likelihoods/" + lk_name
        dat_dir = "../../montepython_fDE/data/" + lk_name

        if not os.path.exists(lk_dir):
            os.makedirs(lk_dir)
        if not os.path.exists(dat_dir):
            os.makedirs(dat_dir)

        # --- write mean file ---
        mean_filename = lk_name + "_mean.txt"
        mean_path = os.path.join(dat_dir, mean_filename)
        with open(mean_path, 'w') as f:
            f.write("# mock DESI-like BAO data\n")
            f.write("# model: {}\n".format(model))
            f.write("# Cosmological parameters: ")
            for key, val in cosmo.pars.items():
                f.write("#   {} = {}, ".format(key, val))
            f.write("\n# [z] [value at z] [quantity]\n")
            for i in range(self.num_bins):
                f.write("{:.8f} {:.12f} {}\n".format(
                    self.z[i], mock_data[i], self.quantity[i]))

        # --- write covariance file ---
        cov_filename = lk_name + "_cov.txt"
        cov_path = os.path.join(dat_dir, cov_filename)
        np.savetxt(cov_path, mock_cov, fmt='%.8e')

        # --- write __init__.py ---
        init_path = os.path.join(lk_dir, "__init__.py")
        init_content = '''import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

#  adapted from bao_boss_dr12 likelihood
class {class_name}(Likelihood):

    # initialization routine
    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        # Note: need to check for conflicting experiments manually

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.data_array = np.array([], 'float64')
        self.quantity = []

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.data_array = np.append(self.data_array, float(this_line[1]))
                    self.quantity.append(str(this_line[2]))

        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))

        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.cov_data)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # Hubble distance D_H = 1 / H(z),
        # sound horizon at baryon drag rs and
        # angle-averaged distance D_V = (z * D_M^2 * D_H)^(1/3)
        
        diff = np.zeros(self.num_bins)
        for i in range(self.num_bins):

            DM_at_z = cosmo.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo.Hubble(self.z[i])
            rs = cosmo.rs_drag()

            theo_DM_over_rs = DM_at_z / rs
            theo_DH_over_rs = 1. / H_at_z / rs
            theo_DV_over_rs = (self.z[i] * DM_at_z**2 / H_at_z)**(1./3.) / rs

            # calculate difference between the sampled point and observations
            if self.quantity[i] == 'DV_over_rs':
                diff[i] = theo_DV_over_rs - self.data_array[i]
            elif self.quantity[i] == 'DM_over_rs':
                diff[i] = theo_DM_over_rs - self.data_array[i]
            elif self.quantity[i] == 'DH_over_rs':
                diff[i] = theo_DH_over_rs - self.data_array[i]
        
        # compute chi squared
        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = np.dot(np.dot(diff,inv_cov_data),diff)

        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl
'''.format(class_name=lk_name)

        with open(init_path, 'w') as f:
            f.write(init_content)

        # --- write .data file ---
        data_file_path = os.path.join(lk_dir, lk_name + ".data")
        with open(data_file_path, 'w') as f:
            f.write("# mock DESI-like BAO data\n")
            f.write("# model: {}\n".format(model))
            f.write("{}.data_directory      = data.path['data']\n".format(lk_name))
            f.write("{}.data_file           = '{}/{}'\n".format(
                lk_name, lk_name, mean_filename))
            f.write("{}.cov_file            = '{}/{}'\n".format(
                lk_name, lk_name, cov_filename))

        print("Created likelihood: {}".format(lk_name))
        print("  Likelihood dir: {}".format(lk_dir))
        print("  Data dir:       {}".format(dat_dir))

        # automatically generate param files for all models
        self.make_param_files(lk_name)

        return lk_name

    def make_param_files(self, lk_name):
        """
        Generate MontePython .param files for 5 analysis models
        (lcdm, fp, fa, fp_wp, w0wa), using lk_name as the BAO likelihood.
        
        Parameters
        ----------
        lk_name : str
            Name of the likelihood created by make_fake_likelihood().
        output_dir : str, optional
            Directory where param files are saved. Defaults to current directory.
        """
        output_dir = "../../montepython_fDE/mock_desi_like_input/"

        # ---------- shared blocks ----------
        common_header = (
            "#------Experiments to test (separated with commas)-----\n"
            "data.experiments=['{lk}','Qcmb']\n"
            "\n"
            "data.over_sampling=[1]\n"
            "\n"
            "# Cosmological parameters list\n"
            "\n"
            "data.parameters['omega_b']      = [  2.2377,   None, None,      0.015, 0.01, 'cosmo']\n"
            "data.parameters['omega_cdm']    = [ 0.131,   None, None,     0.0013,    1, 'cosmo']\n"
            "data.parameters['h']            = [ 0.68,     .2, 1,       0.01,    1, 'cosmo']\n"
        ).format(lk=lk_name)

        omega_lambda_zero = "data.cosmo_arguments['Omega_Lambda'] = 0\n"

        derived_block = (
            "\n"
            "# Derived parameters\n"
            "data.parameters['H0']              = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_Lambda']    = [1, None, None, 0,     1,   'derived']\n"
            "data.parameters['theta_s_100']     = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_m']         = [1, None, None, 0,     1,   'derived']\n"
        )

        fixed_block = (
            "\n"
            "# Other cosmo parameters (fixed parameters, precision parameters, etc.)\n"
            "data.cosmo_arguments['sBBN file'] = data.path['cosmo']+'/external/bbn/sBBN.dat'\n"
            "{eos_line}"
            "{extra_fixed}"
            "data.cosmo_arguments['k_pivot'] = 0.05\n"
            "\n"
            "data.cosmo_arguments['m_ncdm'] = 0.02\n"
            "data.cosmo_arguments['N_ur'] = 0.00441\n"
            "data.cosmo_arguments['N_ncdm'] = 1\n"
            "data.cosmo_arguments['deg_ncdm'] = 3\n"
            "data.cosmo_arguments['T_ncdm'] = 0.71611\n"
            "\n"
            "data.cosmo_arguments['ln10^{{10}}A_s'] =   3.036\n"
            "data.cosmo_arguments['n_s']          =   0.9649\n"
            "data.cosmo_arguments['tau_reio']     =   0.0544\n"
            "data.cosmo_arguments['output']\t=\t''\n"
        )

        mcmc_block = (
            "#------ Mcmc parameters ----\n"
            "\n"
            "data.N=10\n"
            "data.write_step=5\n"
        )

        # ---------- model-specific definitions ----------
        models = {
            'lcdm': {
                'extra_params': '',
                'eos_line': '',
                'extra_fixed': '',
            },
            'fp': {
                'extra_params': omega_lambda_zero + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld'] = 2/3\n",
            },
            'fa': {
                'extra_params': omega_lambda_zero + "data.parameters['fa_fld']\t= [\t0,\t-2,\t2, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'faDE'\n",
                'extra_fixed': '',
            },
            'fp_wp': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n"
                    "data.parameters['wp_fld']\t= [\t0,\t-3,\t1, \t0.2,\t1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE_2'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld']= 2/3\n",
            },
            'w0wa': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['w0_fld']       = [    -1.0,   -3.0,     1.0,        0.1,    1, 'cosmo']\n"
                    "data.parameters['w0wa_fld']         = [    -1.0,   -5.0,     0.0,        0.1,    1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'CLP'\n",
                'extra_fixed': '',
            },
        }

        for model_tag, cfg in models.items():
            content = common_header
            content += cfg['extra_params']
            content += derived_block
            content += fixed_block.format(
                eos_line=cfg['eos_line'],
                extra_fixed=cfg['extra_fixed'],
            )
            content += mcmc_block

            fname = "{lk}_Qcmb_{model}.param".format(lk=lk_name, model=model_tag)
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'w') as f:
                f.write(content)
            print("  Wrote param file: {}".format(fpath))

class DESI_DR3_like_data:
    """Mock data generator for DESI DR3 BAO (background-only: DA_over_rs and Hz_rs).

    Use with the lowz or highz data files. For a combined analysis, create
    two instances and call make_fake_likelihood on each.

    Parameters
    ----------
    mean_file : str
        Path to data vector file (e.g. desi_dr3_lowz_bao_mean.txt).
    cov_file : str
        Path to covariance matrix file (e.g. desi_dr3_lowz_bao_cov.txt).
    label : str
        'lowz' or 'highz', used for naming the mock likelihood.
    """
    # speed of light in km/s (CLASS returns H/c in 1/Mpc)
    c_km_s = 299792.458

    def __init__(self, mean_file, cov_file, label='lowz'):
        self.mean_file = mean_file
        self.cov_file = cov_file
        self.label = label
        self.z = np.array([], 'float64')
        self.data_array = np.array([], 'float64')
        self.quantity = []
        with open(self.mean_file, 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.data_array = np.append(self.data_array, float(this_line[1]))
                    self.quantity.append(str(this_line[2]))
        self.cov_data = np.loadtxt(self.cov_file)
        self.num_bins = np.shape(self.z)[0]
        self.num_points = np.shape(self.cov_data)[0]

    def make_fake_DESI_DR3_data(self, cosmo, mean_noise=False, seed=None, return_theory=False):
        """Generate mock DESI DR3 BAO data from a cosmological model.

        Computes DA_over_rs and Hz_rs from CLASS background quantities.
        Note: cosmo.Hubble(z) returns H(z)/c in 1/Mpc units.
        """
        theory = np.zeros(self.num_bins)
        rs = cosmo.rs_drag()
        for i in range(self.num_bins):
            DA_at_z = cosmo.angular_distance(self.z[i])
            H_at_z = cosmo.Hubble(self.z[i]) * self.c_km_s  # km/s/Mpc

            theo_DA_over_rs = DA_at_z / rs
            theo_Hz_rs = H_at_z * rs

            if self.quantity[i] == 'DA_over_rs':
                theory[i] = theo_DA_over_rs
            elif self.quantity[i] == 'Hz_rs':
                theory[i] = theo_Hz_rs

        rng = np.random.default_rng(seed)
        noise = rng.multivariate_normal(
            mean=np.zeros(self.num_bins, dtype=np.float64),
            cov=self.cov_data
        )
        if mean_noise:
            mock = theory + noise
        else:
            mock = theory.copy()
        fake_cov = self.cov_data
        if return_theory:
            return {'dat': mock, 'cov': fake_cov, 'th': theory}
        else:
            return {'dat': mock, 'cov': fake_cov}

    def make_fake_likelihood(self, cosmo, model, mean_noise=False, seed=False):
        """Create a MontePython DESI-DR3-like BAO likelihood from mock data."""
        result = self.make_fake_DESI_DR3_data(cosmo, mean_noise=mean_noise, seed=seed)
        mock_data = result['dat']
        mock_cov = result['cov']

        now = datetime.datetime.now()
        tag = model + "_" + now.strftime("%Y%m%d%H%M%S")
        lk_name = "mock_bao_desi_dr3_{}_".format(self.label) + tag

        lk_dir = "../../montepython_fDE/montepython/likelihoods/" + lk_name
        dat_dir = "../../montepython_fDE/data/" + lk_name

        if not os.path.exists(lk_dir):
            os.makedirs(lk_dir)
        if not os.path.exists(dat_dir):
            os.makedirs(dat_dir)

        # --- write mean file ---
        mean_filename = lk_name + "_mean.txt"
        mean_path = os.path.join(dat_dir, mean_filename)
        with open(mean_path, 'w') as f:
            f.write("# mock DESI-DR3-like BAO data ({})\n".format(self.label))
            f.write("# model: {}\n".format(model))
            f.write("# Cosmological parameters: ")
            for key, val in cosmo.pars.items():
                f.write("#   {} = {}, ".format(key, val))
            f.write("\n# [z] [value at z] [quantity]\n")
            for i in range(self.num_bins):
                f.write("{:.8f} {:.12f} {}\n".format(
                    self.z[i], mock_data[i], self.quantity[i]))

        # --- write covariance file ---
        cov_filename = lk_name + "_cov.txt"
        cov_path = os.path.join(dat_dir, cov_filename)
        np.savetxt(cov_path, mock_cov, fmt='%.8e')

        # --- write __init__.py (with DA_over_rs and Hz_rs support) ---
        init_path = os.path.join(lk_dir, "__init__.py")
        init_content = '''import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class {class_name}(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.z = np.array([], 'float64')
        self.data_array = np.array([], 'float64')
        self.quantity = []

        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.data_array = np.append(self.data_array, float(this_line[1]))
                    self.quantity.append(str(this_line[2]))

        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))
        self.num_bins = np.shape(self.z)[0]
        self.num_points = np.shape(self.cov_data)[0]
        self.c_km_s = conts.c / 1000.0

    def loglkl(self, cosmo, data):

        diff = np.zeros(self.num_bins)
        for i in range(self.num_bins):

            DA_at_z = cosmo.angular_distance(self.z[i])
            H_at_z = cosmo.Hubble(self.z[i]) * self.c_km_s
            rs = cosmo.rs_drag()

            theo_DA_over_rs = DA_at_z / rs
            theo_Hz_rs = H_at_z * rs

            if self.quantity[i] == 'DA_over_rs':
                diff[i] = theo_DA_over_rs - self.data_array[i]
            elif self.quantity[i] == 'Hz_rs':
                diff[i] = theo_Hz_rs - self.data_array[i]

        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = np.dot(np.dot(diff, inv_cov_data), diff)

        loglkl = - 0.5 * chi2

        return loglkl
'''.format(class_name=lk_name)

        with open(init_path, 'w') as f:
            f.write(init_content)

        # --- write .data file ---
        data_file_path = os.path.join(lk_dir, lk_name + ".data")
        with open(data_file_path, 'w') as f:
            f.write("# mock DESI-DR3-like BAO data ({})\n".format(self.label))
            f.write("# model: {}\n".format(model))
            f.write("{}.data_directory      = data.path['data']\n".format(lk_name))
            f.write("{}.data_file           = '{}/{}'\n".format(
                lk_name, lk_name, mean_filename))
            f.write("{}.cov_file            = '{}/{}'\n".format(
                lk_name, lk_name, cov_filename))

        print("Created likelihood: {}".format(lk_name))
        print("  Likelihood dir: {}".format(lk_dir))
        print("  Data dir:       {}".format(dat_dir))

        self.make_param_files(lk_name)

        return lk_name

    def make_param_files(self, lk_name):
        """Generate MontePython .param files for 5 analysis models."""
        output_dir = "../../montepython_fDE/mock_desi_like_input/"

        common_header = (
            "#------Experiments to test (separated with commas)-----\n"
            "data.experiments=['{lk}','Qcmb']\n"
            "\n"
            "data.over_sampling=[1]\n"
            "\n"
            "# Cosmological parameters list\n"
            "\n"
            "data.parameters['omega_b']      = [  2.2377,   None, None,      0.015, 0.01, 'cosmo']\n"
            "data.parameters['omega_cdm']    = [ 0.131,   None, None,     0.0013,    1, 'cosmo']\n"
            "data.parameters['h']            = [ 0.68,     .2, 1,       0.01,    1, 'cosmo']\n"
        ).format(lk=lk_name)

        omega_lambda_zero = "data.cosmo_arguments['Omega_Lambda'] = 0\n"

        derived_block = (
            "\n"
            "# Derived parameters\n"
            "data.parameters['H0']              = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_Lambda']    = [1, None, None, 0,     1,   'derived']\n"
            "data.parameters['theta_s_100']     = [0, None, None, 0,     1,   'derived']\n"
            "data.parameters['Omega_m']         = [1, None, None, 0,     1,   'derived']\n"
        )

        fixed_block = (
            "\n"
            "# Other cosmo parameters (fixed parameters, precision parameters, etc.)\n"
            "data.cosmo_arguments['sBBN file'] = data.path['cosmo']+'/external/bbn/sBBN.dat'\n"
            "{eos_line}"
            "{extra_fixed}"
            "data.cosmo_arguments['k_pivot'] = 0.05\n"
            "\n"
            "data.cosmo_arguments['m_ncdm'] = 0.02\n"
            "data.cosmo_arguments['N_ur'] = 0.00441\n"
            "data.cosmo_arguments['N_ncdm'] = 1\n"
            "data.cosmo_arguments['deg_ncdm'] = 3\n"
            "data.cosmo_arguments['T_ncdm'] = 0.71611\n"
            "\n"
            "data.cosmo_arguments['ln10^{{10}}A_s'] =   3.036\n"
            "data.cosmo_arguments['n_s']          =   0.9649\n"
            "data.cosmo_arguments['tau_reio']     =   0.0544\n"
            "data.cosmo_arguments['output']\t=\t''\n"
        )

        mcmc_block = (
            "#------ Mcmc parameters ----\n"
            "\n"
            "data.N=10\n"
            "data.write_step=5\n"
        )

        models = {
            'lcdm': {
                'extra_params': '',
                'eos_line': '',
                'extra_fixed': '',
            },
            'fp': {
                'extra_params': omega_lambda_zero + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld'] = 2/3\n",
            },
            'fa': {
                'extra_params': omega_lambda_zero + "data.parameters['fa_fld']\t= [\t0,\t-2,\t2, \t0.2,\t1, 'cosmo']\n",
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'faDE'\n",
                'extra_fixed': '',
            },
            'fp_wp': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['fp_fld']\t= [\t0,\t-1,\t3, \t0.2,\t1, 'cosmo']\n"
                    "data.parameters['wp_fld']\t= [\t0,\t-3,\t1, \t0.2,\t1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'fpDE_2'\n",
                'extra_fixed': "data.cosmo_arguments['ap_fld']= 2/3\n",
            },
            'w0wa': {
                'extra_params': (
                    omega_lambda_zero
                    + "data.parameters['w0_fld']       = [    -1.0,   -3.0,     1.0,        0.1,    1, 'cosmo']\n"
                    "data.parameters['w0wa_fld']         = [    -1.0,   -5.0,     0.0,        0.1,    1, 'cosmo']\n"
                ),
                'eos_line': "data.cosmo_arguments['fluid_equation_of_state'] = 'CLP'\n",
                'extra_fixed': '',
            },
        }

        for model_tag, cfg in models.items():
            content = common_header
            content += cfg['extra_params']
            content += derived_block
            content += fixed_block.format(
                eos_line=cfg['eos_line'],
                extra_fixed=cfg['extra_fixed'],
            )
            content += mcmc_block

            fname = "{lk}_Qcmb_{model}.param".format(lk=lk_name, model=model_tag)
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'w') as f:
                f.write(content)
            print("  Wrote param file: {}".format(fpath))
