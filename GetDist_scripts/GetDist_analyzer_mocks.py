from getdist import loadMCSamples, MCSamples
from getdist import plots
import os
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

burnin = 0.05
smooth1D = 0.4
smooth2D = 0.6
base_path = '/scratch/09218/gab97/chains/'
load_settings = {'ignore_rows': burnin, 'smooth_scale_1D': smooth1D, 'smooth_scale_2D': smooth2D}

file_stamps = {
    'v0': '2026-02-19_50030_',
    'v1': '2026-02-20_50030_',
    'v2': '2026-03-01_50030_',
    'v3': '2026-03-08_50030_',
    'v4': '2026-03-09_50030_',
    'v5': '2026-03-14_50030_',
}

# === Analysis definitions ===
# Each entry: (short_name, chain_subdir, file_stamp_key, model_type)
analyses = [
    ('desi_cmb_lcdm_mock_w0wa',   'mock_desi_dr2_lcdm_Qcmb_w0wa_v1',            'v0', 'w0wa'),
    ('desi_cmb_lcdm_mock_fa',   'mock_desi_dr2_lcdm_Qcmb_fa_v1',            'v0', 'fa'),
    ('desi_cmb_lcdm_mock_fp',   'mock_desi_dr2_lcdm_Qcmb_fp_v1',            'v0', 'fp'),
    ('desi_cmb_lcdm_mock_fp_wp',   'mock_desi_dr2_lcdm_Qcmb_fp_wp_v1',            'v0', 'fp_wp'),
    ('desi_cmb_w0wa_mock_w0wa',   'mock_desi_dr2_w0wa_Qcmb_w0wa_v1',            'v0', 'w0wa'),
    ('desi_cmb_w0wa_mock_fa',   'mock_desi_dr2_w0wa_Qcmb_fa_v2',            'v1', 'fa'),
    ('desi_cmb_w0wa_mock_fp',   'mock_desi_dr2_w0wa_Qcmb_fp_v2',            'v1', 'fp'),
    ('desi_cmb_w0wa_mock_fp_wp',   'mock_desi_dr2_w0wa_Qcmb_fp_wp_v1',            'v0', 'fp_wp'),
    ('desi_cmb_exp_mock_w0wa',   'mock_desi_dr2_exp_Qcmb_w0wa_v1',            'v0', 'w0wa'),
    ('desi_cmb_exp_mock_fa',   'mock_desi_dr2_exp_Qcmb_fa_v1',            'v0', 'fa'),
    ('desi_cmb_exp_mock_fp',   'mock_desi_dr2_exp_Qcmb_fp_v1',            'v0', 'fp'),
    ('desi_cmb_exp_mock_fp_wp',   'mock_desi_dr2_exp_Qcmb_fp_wp_v1',            'v0', 'fp_wp'),
    ('desi_cmb_exp2_mock_w0wa',   'mock_desi_dr2_exp2_Qcmb_w0wa_v1',            'v0', 'w0wa'),
    ('desi_cmb_exp2_mock_fa',   'mock_desi_dr2_exp2_Qcmb_fa_v1',            'v0', 'fa'),
    ('desi_cmb_exp2_mock_fp',   'mock_desi_dr2_exp2_Qcmb_fp_v1',            'v0', 'fp'),
    ('desi_cmb_exp2_mock_fp_wp',   'mock_desi_dr2_exp2_Qcmb_fp_wp_v1',            'v0', 'fp_wp'),
    ('desi_cmb_fpwp_mock_w0wa',   'mock_desi_dr2_fpwp_Qcmb_w0wa_v1',            'v0', 'w0wa'),
    ('desi_cmb_fpwp_mock_fa',   'mock_desi_dr2_fpwp_Qcmb_fa_v1',            'v0', 'fa'),
    ('desi_cmb_fpwp_mock_fp',   'mock_desi_dr2_fpwp_Qcmb_fp_v1',            'v0', 'fp'),
    ('desi_cmb_fpwp_mock_fp_wp',   'mock_desi_dr2_fpwp_Qcmb_fp_wp_v1',            'v0', 'fp_wp'),
    ('desi_cmb_lcdm_mock_lcdm',   'mock_desi_dr2_lcdm_Qcmb_lcdm_v1',            'v2', 'lcdm'),
    ('desi_cmb_w0wa_mock_lcdm',   'mock_desi_dr2_w0wa_Qcmb_lcdm_v1',            'v2', 'lcdm'),
    ('desi_cmb_fpwp_mock_lcdm',   'mock_desi_dr2_fpwp_Qcmb_lcdm_v1',            'v2', 'lcdm'),
    ('desi_cmb_exp_mock_lcdm',   'mock_desi_dr2_exp_Qcmb_lcdm_v1',            'v2', 'lcdm'),
    ('desi_cmb_exp2_mock_lcdm',   'mock_desi_dr2_exp2_Qcmb_lcdm_v1',            'v2', 'lcdm'),
    ('desi_des_cmb_lcdm_mock_w0wa',   'mock_desi_dr2_DESY5_lcdm_Qcmb_w0wa_v1',            'v3', 'w0wa'),
    ('desi_des_cmb_lcdm_mock_lcdm',   'mock_desi_dr2_DESY5_lcdm_Qcmb_lcdm_v1',            'v3', 'lcdm'),
    ('desi_des_cmb_lcdm_mock_fp',   'mock_desi_dr2_DESY5_lcdm_Qcmb_fp_v1',            'v4', 'fp'),
    ('desi_des_cmb_lcdm_mock_fp_wp',   'mock_desi_dr2_DESY5_lcdm_Qcmb_fp_wp_v1',            'v4', 'fp_wp'),
    ('desi_des_cmb_exp_mock_w0wa',   'mock_desi_dr2_DESY5_exp_Qcmb_w0wa_v2',            'v4', 'w0wa'),
    ('desi_des_cmb_exp_mock_lcdm',   'mock_desi_dr2_DESY5_exp_Qcmb_lcdm_v2',            'v4', 'lcdm'),
    ('desi_des_cmb_exp_mock_fp',   'mock_desi_dr2_DESY5_exp_Qcmb_fp_v2',            'v4', 'fp'),
    ('desi_des_cmb_exp_mock_fp_wp',   'mock_desi_dr2_DESY5_exp_Qcmb_fp_wp_v2',            'v4', 'fp_wp'),
    ('desi_des_cmb_exp2_mock_w0wa',   'mock_desi_dr2_DESY5_exp2_Qcmb_w0wa_v2',            'v4', 'w0wa'),
    ('desi_des_cmb_exp2_mock_lcdm',   'mock_desi_dr2_DESY5_exp2_Qcmb_lcdm_v2',            'v4', 'lcdm'),
    ('desi_des_cmb_exp2_mock_fp',   'mock_desi_dr2_DESY5_exp2_Qcmb_fp_v2',            'v4', 'fp'),
    ('desi_des_cmb_exp2_mock_fp_wp',   'mock_desi_dr2_DESY5_exp2_Qcmb_fp_wp_v2',            'v4', 'fp_wp'),
    ('desi3_lsst_cmb_exp_mock_w0wa',   'mock_desi_dr3_lssty3_exp_Qcmb_w0wa_v1',            'v5', 'w0wa'),
    ('desi3_lsst_cmb_exp_mock_lcdm',   'mock_desi_dr3_lssty3_exp_Qcmb_lcdm_v1',            'v5', 'lcdm'),
    ('desi3_lsst_cmb_exp_mock_fp',     'mock_desi_dr3_lssty3_exp_Qcmb_fp_v1',            'v5', 'fp'),
    ('desi3_lsst_cmb_exp_mock_fp_wp',   'mock_desi_dr3_lssty3_exp_Qcmb_fpwp_v1',            'v5', 'fp_wp'),
    ('desi3_lsst_cmb_exp2_mock_w0wa',   'mock_desi_dr3_lssty3_exp2_Qcmb_w0wa_v1',            'v5', 'w0wa'),
    ('desi3_lsst_cmb_exp2_mock_lcdm',   'mock_desi_dr3_lssty3_exp2_Qcmb_lcdm_v1',            'v5', 'lcdm'),
    ('desi3_lsst_cmb_exp2_mock_fp',   'mock_desi_dr3_lssty3_exp2_Qcmb_fp_v1',            'v5', 'fp'),
    ('desi3_lsst_cmb_exp2_mock_fp_wp',   'mock_desi_dr3_lssty3_exp2_Qcmb_fpwp_v1',            'v5', 'fp_wp'),
    ('desi3_lsst_cmb_lcdm_mock_w0wa',   'mock_desi_dr3_lssty3_lcdm_Qcmb_w0wa_v1',            'v5', 'w0wa'),
    ('desi3_lsst_cmb_lcdm_mock_lcdm',   'mock_desi_dr3_lssty3_lcdm_Qcmb_lcdm_v1',            'v5', 'lcdm'),
    ('desi3_lsst_cmb_lcdm_mock_fp',   'mock_desi_dr3_lssty3_lcdm_Qcmb_fp_v1',            'v5', 'fp'),
    ('desi3_lsst_cmb_lcdm_mock_fp_wp',   'mock_desi_dr3_lssty3_lcdm_Qcmb_fpwp_v1',            'v5', 'fp_wp'),
    ]

# === Parameter lists & triangle pairs per model type ===
def make_pairs(pars):
    """Generate lower-triangle pairs from an ordered parameter list."""
    return [(pars[j], pars[i]) for j in range(1, len(pars)) for i in range(j)]

MODEL_PARS = {
    'lcdm':  ["omega_b", "omega_m", "theta_s_100", "H0"],
    'w0wa':  ["omega_b", "omega_m", "theta_s_100", "w0_fld", "wa_fld", "H0"],
    'fa':    ["omega_b", "omega_m", "theta_s_100", "fa_fld", "H0"],
    'fp':    ["omega_b", "omega_m", "theta_s_100", "fp_fld", "H0"],
    'fp_wp': ["omega_b", "omega_m", "theta_s_100", "fp_fld", "wp_fld", "H0"],
}



# === Derived-parameter recipes ===
def add_derived(samples, model_type):
    """Add omega_m (always) and wa_fld (for w0wa models), with optional filtering."""
    p = samples.getParams()
    omega_m = p.omega_cdm + p.omega_b / 100
    samples.addDerived(omega_m, name='omega_m', label=r'\omega_m')

    if model_type == 'w0wa':
        wa_fld = p.w0wa_fld - p.w0_fld
        samples.addDerived(wa_fld, name='wa_fld', label=r'w_a')
        # Filter: w0+wa < 0  and  wa > -3
        p = samples.getParams()
        samples.filter(p.w0wa_fld < 0)
        p = samples.getParams()
        samples.filter(p.wa_fld > -3)

def save_posteriors(chain_sample,pars,pairs,name):
    save_dict= {}
    for param in trange(len(pars), desc="Computing 1D densities"):
        p=pars[param]
        density1D=chain_sample.get1DDensity(p)
        x_1D=density1D.x
        P_1D=density1D.P
        save_dict[f"{name}_{p}_1D_x"]=x_1D
        save_dict[f"{name}_{p}_1D_P"]=P_1D
    for i in trange(len(pairs), desc="Computing 2D densities"):
        p1, p2 = pairs[i]
        density2D = chain_sample.get2DDensity(p2, p1)
        # Extract data
        xvs = density2D.x            # 1D array of x values
        yvs = density2D.y            # 1D array of y values
        p_grid = density2D.P         # 2D array of density
        #density2D.contours = [0.6827, 0.9545, 0.9973]
        contour_levels = density2D.getContourLevels(contours=(0.68,0.95,0.99))  # e.g. [level_68, level_95]
        # Build a key prefix
        key_prefix = f"{name}_{p1}__{p2}"
        # Store in the dictionary
        save_dict[key_prefix + "_x"] = xvs
        save_dict[key_prefix + "_y"] = yvs
        save_dict[key_prefix + "_p_grid"] = p_grid
        save_dict[key_prefix + "_contour_levels"] = contour_levels
    # -------------------------------------------------
    # Save everything to an .npz file
    # -------------------------------------------------
    output_file = f"output_mocks/{name}.npz"
    np.savez(output_file, **save_dict)
    return None

for idx in range(36,len(analyses)):
    name, subdir, stamp_key, model_type = analyses[idx]
    chain_dir = f"{base_path}{subdir}/{file_stamps[stamp_key]}"
    samples = loadMCSamples(chain_dir, settings=load_settings)
    add_derived(samples, model_type)
    pars = MODEL_PARS[model_type]
    pairs = make_pairs(pars)
    save_posteriors(samples, pars, pairs, name)
