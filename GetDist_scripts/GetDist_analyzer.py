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
    'v0': '2026-02-17_100030_',
    'v1': '2026-02-15_300030_',
    'v2': '2026-02-18_100030_',
    'v3': '2026-02-18_50030_',
    'v4': '2026-02-19_100030_',
    'v5': '2026-02-28_100030_',
    'v6': '2026-02-28_50030_',

}

# === Analysis definitions ===
# Each entry: (short_name, chain_subdir, file_stamp_key, model_type)
analyses = [
    ('desi_cmb_lcdm',   'desi_dr2_Qcmb_lcdm_v1',            'v2', 'lcdm'),
    ('desi_cmb_w0wa',   'desi_dr2_Qcmb_w0wa_v1',            'v0', 'w0wa'),
    #('desi_cmb_fa',     'desi_dr2_Qcmb_fa_v1',              'v0', 'fa'),
    ('desi_cmb_fp',     'desi_dr2_Qcmb_fp_v1',              'v2', 'fp'),
    ('desi_cmb_fp_wp',  'desi_dr2_Qcmb_fp_wp_v1',           'v2', 'fp_wp'),
    #('desi_p18_w0wa',   'desi_dr2_p18TTTEEE_lensing_w0wa_v1','v1', 'w0wa'),
    ('desi_cmb_des_fp_wp',  'desi_dr2_Qcmb_DESY5_fp_wp_v1', 'v3', 'fp_wp'),
    ('desi_cmb_pp_fp_wp',  'desi_dr2_Qcmb_PP_fp_wp_v1', 'v3', 'fp_wp'),
    ('desi_cmb_des_w0wa',  'desi_dr2_Qcmb_DESY5_w0wa_v1', 'v3', 'w0wa'),
    ('desi_cmb_pp_w0wa',  'desi_dr2_Qcmb_PP_w0wa_v1', 'v3', 'w0wa'),
    ('desi_cmb_des_fa',  'desi_dr2_Qcmb_DESY5_fa_v1', 'v3', 'fa'),
    ('desi_cmb_pp_fa',  'desi_dr2_Qcmb_PP_fa_v1', 'v3', 'fa'),
    ('desi_cmb_des_fp',  'desi_dr2_Qcmb_DESY5_fp_v1', 'v3', 'fp'),
    ('desi_cmb_pp_fp',  'desi_dr2_Qcmb_PP_fp_v1', 'v3', 'fp'),
    ('desi_cmb_des_lcdm',  'desi_dr2_Qcmb_DESY5_lcdm_v1', 'v4', 'lcdm'),
    ('desi_cmb_pp_lcdm',  'desi_dr2_Qcmb_PP_lcdm_v1', 'v4', 'lcdm'),
    ('desi_cmb_lcdm_v2',   'desi_dr2_Qcmb_lcdm_v2',            'v6', 'lcdm_v2'),
    ('desi_cmb_w0wa_v3',   'desi_dr2_Qcmb_w0wa_v2',            'v6', 'w0wa_v3'),
    ('desi_cmb_w0wa_v2',   'desi_dr2_Qcmb_w0wa_v3',            'v6', 'w0wa_v2'),
    ('desi_cmb_des_lcdm_v2',  'desi_dr2_Qcmb_DESY5_lcdm_v2', 'v5', 'lcdm_v2'),
    ('desi_cmb_des_w0wa_v2',  'desi_dr2_Qcmb_DESY5_w0wa_v2', 'v5', 'w0wa_v2'),
    ('desi_cmb_des_old_lcdm_v2',  'desi_dr2_Qcmb_DESY5old_lcdm_v2', 'v5', 'lcdm_v2'),
    ('desi_cmb_des_old_w0wa_v2',  'desi_dr2_Qcmb_DESY5old_w0wa_v2', 'v5', 'w0wa_v2'),
    ]
# === Parameter lists & triangle pairs per model type ===
def make_pairs(pars):
    """Generate lower-triangle pairs from an ordered parameter list."""
    return [(pars[j], pars[i]) for j in range(1, len(pars)) for i in range(j)]

MODEL_PARS = {
    'lcdm':  ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "H0"],
    'w0wa':  ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "w0_fld", "wa_fld", "H0"],
    'lcdm_v2':  ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "H0"],
    'w0wa_v2':  ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "w0_fld", "wa_fld", "H0"],
    'w0wa_v3':  ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "w0_fld", "wa_fld", "H0"],
    'fa':    ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "fa_fld", "H0"],
    'fp':    ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "fp_fld", "H0"],
    'fp_wp': ["omega_b", "omega_cb", "omega_m", "Omega_m", "theta_s_100", "fp_fld", "wp_fld", "H0"],
}

# === Derived-parameter recipes ===
omega_ncdm = 3 * 0.02 / 93.14  # 3 degenerate species, m_ncdm = 0.02 eV each

def add_derived(samples, model_type):
    """Add omega_cb, omega_m, Omega_m, and wa_fld (for w0wa models), with optional filtering."""
    # Remove existing derived params if present, so we can re-derive them
    p = samples.getParams()
    if model_type not in ('lcdm_v2', 'w0wa_v2'):
        indices_to_delete = []
        for par_name in ['omega_m', 'Omega_m']:
            idx = samples.paramNames.numberOfName(par_name)
            if idx is not None:
                indices_to_delete.append(idx)
        if indices_to_delete:
            samples.changeSamples(np.delete(samples.samples, indices_to_delete, axis=1))
            samples.paramNames.deleteIndices(indices_to_delete)
        omega_cb = p.omega_cdm + p.omega_b / 100
        omega_m = omega_cb + omega_ncdm
        Omega_m = omega_m / (p.H0 / 100.)**2
        samples.addDerived(omega_cb, name='omega_cb', label=r'\omega_{cb}')
        samples.addDerived(omega_m, name='omega_m', label=r'\omega_m')
        samples.addDerived(Omega_m, name='Omega_m', label=r'\Omega_m')
    else:
        omega_m = p.Omega_m * p.h**2
        omega_cb = omega_m - omega_ncdm
        samples.addDerived(omega_cb, name='omega_cb', label=r'\omega_{cb}')
        samples.addDerived(omega_m, name='omega_m', label=r'\omega_m')
    if model_type == 'w0wa':
        wa_fld = p.w0wa_fld - p.w0_fld
        samples.addDerived(wa_fld, name='wa_fld', label=r'w_a')
        # Filter: w0+wa < 0  and  wa > -3
        p = samples.getParams()
        samples.filter(p.w0wa_fld < 0)
        p = samples.getParams()
        samples.filter(p.wa_fld > -3)
    if model_type == 'w0wa_v2' or model_type=='w0wa_v3':
        w0wa_fld = p.w0_fld + p.wa_fld

def save_posteriors(chain_sample,pars,pairs,name):
    save_dict= {}
    for param in trange(len(pars), desc="Computing 1D densities"):
        p=pars[param]
        density1D=chain_sample.get1DDensity(p)
        x_1D=density1D.x
        P_1D=density1D.P
        save_dict[f"{name}_{p}_1D_x"]=x_1D
        save_dict[f"{name}_{p}_1D_P"]=P_1D
        save_dict[f"{name}_{p}_1D_mean"]=chain_sample.mean(p)
        save_dict[f"{name}_{p}_1D_68limits"]=density1D.getLimits([0.68])
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
    output_file = f"output/{name}.npz"
    np.savez(output_file, **save_dict)
    return None

for idx in range(4,len(analyses)):
    name, subdir, stamp_key, model_type = analyses[idx]
    chain_dir = f"{base_path}{subdir}/{file_stamps[stamp_key]}"
    print(chain_dir)
    samples = loadMCSamples(chain_dir, settings=load_settings)
    samples = loadMCSamples(chain_dir, settings=load_settings)
    like_stats = samples.getLikeStats()
    # Best-fit sample's -log(posterior) value
    #print("Min -logL (best sample):", like_stats.logLike_sample)
    #print("Min chi2 (approx):", 2 * like_stats.logLike_sample)
    # Mean -log(posterior) across samples
    #print("Mean -logL:", like_stats.meanLogLike)
    # Bayesian complexity: 2 * (<-logL> - min(-logL))
    #print("Complexity:", like_stats.complexity)
    add_derived(samples, model_type)
    pars = MODEL_PARS[model_type]
    pairs = make_pairs(pars)
    save_posteriors(samples, pars, pairs, name)
