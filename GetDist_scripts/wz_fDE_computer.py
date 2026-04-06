import numpy as np


# ---------------------------------------------------------------------------
# Physics: w(z) and fDE(z) for the w0wa (CPL) parametrization
# ---------------------------------------------------------------------------
def w_of_z_w0wa(z, w0, wa):
    """CPL parametrization: w(z) = w0 + wa * z/(1+z)."""
    return w0 + wa * z / (1.0 + z)


def fDE_of_z_w0wa(z, w0, wa):
    """Dark energy density ratio fDE(z) = rho_DE(z)/rho_DE(0).
    Eq.(4) of 2503.14743v2: (1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z))."""
    return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))


# ---------------------------------------------------------------------------
# Physics: w(z) and fDE(z) for the fa parametrization
# fDE = 1 + fa*(1 - a),  wDE = -1 + a*fa / (3*fDE)
# ---------------------------------------------------------------------------
def fDE_of_z_fa(z, fa):
    """fDE(z) = 1 + fa*(1 - a) where a = 1/(1+z)."""
    a = 1.0 / (1.0 + z)
    return 1.0 + fa * (1.0 - a)


def w_of_z_fa(z, fa):
    """w(z) = -1 + a*fa / (3*fDE)."""
    a = 1.0 / (1.0 + z)
    fDE = fDE_of_z_fa(z, fa)
    return -1.0 + a * fa / (3.0 * fDE)


# ---------------------------------------------------------------------------
# Physics: w(z) and fDE(z) for the fp parametrization
# fDE = 1 + (fp-1)/(1-ap) * (1-a),  same as fa with fa = (fp-1)/(1-ap)
# ---------------------------------------------------------------------------
def fDE_of_z_fp(z, fp, ap=2/3):
    """fDE(z) = 1 + (fp-1)/(1-ap) * (1-a)."""
    fa = (fp - 1.0) / (1.0 - ap)
    return fDE_of_z_fa(z, fa)


def w_of_z_fp(z, fp, ap=2/3):
    """w(z) = -1 + a*fa / (3*fDE) with fa = (fp-1)/(1-ap)."""
    fa = (fp - 1.0) / (1.0 - ap)
    return w_of_z_fa(z, fa)


# ---------------------------------------------------------------------------
# Physics: w(z) and fDE(z) for the fp-wp parametrization
# ---------------------------------------------------------------------------
def fDE_of_z_fpwp(z, fp, wp, ap=2/3):
    """Dark energy density fraction fDE(a) in the (fp, wp, ap) parametrization.
    fDE(a) = fp - 3(1+wp)*fp/ap * (a-ap) + fb*(a-ap)^2
    with fb fixed by fDE(1)=1."""
    a = 1.0 / (1.0 + z)
    fb = (1.0 - fp + 3.0 * (1.0 + wp) * fp / ap * (1.0 - ap)) / (1.0 - ap) ** 2
    return fp - 3.0 * (1.0 + wp) * fp / ap * (a - ap) + fb * (a - ap) ** 2


def w_of_z_fpwp(z, fp, wp, ap=2/3):
    """Dark energy equation of state w(a) from d ln fDE / d ln a = -3(1+w)."""
    a = 1.0 / (1.0 + z)
    fDE = fDE_of_z_fpwp(z, fp, wp, ap)
    fb = (1.0 - fp + 3.0 * (1.0 + wp) * fp / ap * (1.0 - ap)) / (1.0 - ap) ** 2
    dfDE_da = -3.0 * (1.0 + wp) * fp / ap + 2.0 * fb * (a - ap)
    return -1.0 - a * dfDE_da / (3.0 * fDE)


# ---------------------------------------------------------------------------
# Sampling from a 2D posterior grid
# ---------------------------------------------------------------------------
def _sample_2d(data, key, n_samples, rng_seed=42):
    """Draw samples from a 2D marginalized posterior stored in an npz."""
    x_1d = data[f"{key}_x"]
    y_1d = data[f"{key}_y"]
    P = data[f"{key}_p_grid"]

    X, Y = np.meshgrid(x_1d, y_1d)

    prob = P.ravel().copy()
    prob[prob < 0] = 0.0
    prob /= prob.sum()

    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(prob), size=n_samples, p=prob)

    return X.ravel()[idx], Y.ravel()[idx]


# ---------------------------------------------------------------------------
# Sampling from a 1D posterior
# ---------------------------------------------------------------------------
def _sample_1d(data, key, n_samples, rng_seed=42):
    """Draw samples from a 1D marginalized posterior stored in an npz."""
    x = data[f"{key}_1D_x"]
    P = data[f"{key}_1D_P"]

    prob = P.copy()
    prob[prob < 0] = 0.0
    prob /= prob.sum()

    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(prob), size=n_samples, p=prob)

    return x[idx]


# ---------------------------------------------------------------------------
# Compute percentile bands
# ---------------------------------------------------------------------------
def _compute_bands(vals):
    """Return mean, lo2, lo1, hi1, hi2 from a (n_samples, n_points) array."""
    pcts = [2.275, 15.865, 84.135, 97.725]
    lo2, lo1, hi1, hi2 = np.percentile(vals, pcts, axis=0)
    return np.mean(vals, axis=0), lo2, lo1, hi1, hi2


# ---------------------------------------------------------------------------
# w0wa bands
# ---------------------------------------------------------------------------
def get_wz_fDE_bands_w0wa(
    data,
    prefix,
    z_max=2.5,
    n_points=500,
    n_samples=10000,
    rng_seed=42,
):
    """
    Compute mean and 1/2-sigma bands for w(z) and fDE(z) from a w0-wa posterior.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
    prefix : str  (e.g. "desi_cmb_w0wa")
    z_max, n_points, n_samples, rng_seed : see defaults

    Returns
    -------
    dict with keys: z, w_mean, w_lo1, w_hi1, w_lo2, w_hi2,
                    fDE_mean, fDE_lo1, fDE_hi1, fDE_lo2, fDE_hi2
    """
    z_arr = np.linspace(0, z_max, n_points)
    key = f"{prefix}_wa_fld__w0_fld"
    w0_s, wa_s = _sample_2d(data, key, n_samples, rng_seed)

    w_vals = w_of_z_w0wa(z_arr[None, :], w0_s[:, None], wa_s[:, None])
    f_vals = fDE_of_z_w0wa(z_arr[None, :], w0_s[:, None], wa_s[:, None])

    w_mean, w_lo2, w_lo1, w_hi1, w_hi2 = _compute_bands(w_vals)
    f_mean, f_lo2, f_lo1, f_hi1, f_hi2 = _compute_bands(f_vals)

    return {
        "z": z_arr,
        "w_mean": w_mean, "w_lo1": w_lo1, "w_hi1": w_hi1, "w_lo2": w_lo2, "w_hi2": w_hi2,
        "fDE_mean": f_mean, "fDE_lo1": f_lo1, "fDE_hi1": f_hi1, "fDE_lo2": f_lo2, "fDE_hi2": f_hi2,
    }


# ---------------------------------------------------------------------------
# fp-wp bands
# ---------------------------------------------------------------------------
def get_wz_fDE_bands_fpwp(
    data,
    prefix,
    ap=2/3,
    z_max=2.5,
    n_points=500,
    n_samples=10000,
    rng_seed=42,
):
    """
    Compute mean and 1/2-sigma bands for w(z) and fDE(z) from a fp-wp posterior.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
    prefix : str  (e.g. "desi_cmb_fp_wp")
    ap : float
        Pivot scale factor (default 0.5).
    z_max, n_points, n_samples, rng_seed : see defaults

    Returns
    -------
    dict with keys: z, w_mean, w_lo1, w_hi1, w_lo2, w_hi2,
                    fDE_mean, fDE_lo1, fDE_hi1, fDE_lo2, fDE_hi2
    """
    z_arr = np.linspace(0, z_max, n_points)
    key = f"{prefix}_wp_fld__fp_fld"
    fp_s, wp_s = _sample_2d(data, key, n_samples, rng_seed)

    w_vals = w_of_z_fpwp(z_arr[None, :], fp_s[:, None], wp_s[:, None], ap)
    f_vals = fDE_of_z_fpwp(z_arr[None, :], fp_s[:, None], wp_s[:, None], ap)

    w_mean, w_lo2, w_lo1, w_hi1, w_hi2 = _compute_bands(w_vals)
    f_mean, f_lo2, f_lo1, f_hi1, f_hi2 = _compute_bands(f_vals)

    return {
        "z": z_arr,
        "w_mean": w_mean, "w_lo1": w_lo1, "w_hi1": w_hi1, "w_lo2": w_lo2, "w_hi2": w_hi2,
        "fDE_mean": f_mean, "fDE_lo1": f_lo1, "fDE_hi1": f_hi1, "fDE_lo2": f_lo2, "fDE_hi2": f_hi2,
    }


# ---------------------------------------------------------------------------
# fa bands
# ---------------------------------------------------------------------------
def get_wz_fDE_bands_fa(
    data,
    prefix,
    z_max=2.5,
    n_points=500,
    n_samples=10000,
    rng_seed=42,
):
    """
    Compute mean and 1/2-sigma bands for w(z) and fDE(z) from an fa posterior.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
    prefix : str  (e.g. "desi_cmb_fa")
    """
    z_arr = np.linspace(0, z_max, n_points)
    key = f"{prefix}_fa_fld"
    fa_s = _sample_1d(data, key, n_samples, rng_seed)

    w_vals = w_of_z_fa(z_arr[None, :], fa_s[:, None])
    f_vals = fDE_of_z_fa(z_arr[None, :], fa_s[:, None])

    w_mean, w_lo2, w_lo1, w_hi1, w_hi2 = _compute_bands(w_vals)
    f_mean, f_lo2, f_lo1, f_hi1, f_hi2 = _compute_bands(f_vals)

    return {
        "z": z_arr,
        "w_mean": w_mean, "w_lo1": w_lo1, "w_hi1": w_hi1, "w_lo2": w_lo2, "w_hi2": w_hi2,
        "fDE_mean": f_mean, "fDE_lo1": f_lo1, "fDE_hi1": f_hi1, "fDE_lo2": f_lo2, "fDE_hi2": f_hi2,
    }


# ---------------------------------------------------------------------------
# fp bands
# ---------------------------------------------------------------------------
def get_wz_fDE_bands_fp(
    data,
    prefix,
    ap=2/3,
    z_max=2.5,
    n_points=500,
    n_samples=10000,
    rng_seed=42,
):
    """
    Compute mean and 1/2-sigma bands for w(z) and fDE(z) from an fp posterior.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
    prefix : str  (e.g. "desi_cmb_fp")
    ap : float
        Pivot scale factor (default 2/3).
    """
    z_arr = np.linspace(0, z_max, n_points)
    key = f"{prefix}_fp_fld"
    fp_s = _sample_1d(data, key, n_samples, rng_seed)

    w_vals = w_of_z_fp(z_arr[None, :], fp_s[:, None], ap)
    f_vals = fDE_of_z_fp(z_arr[None, :], fp_s[:, None], ap)

    w_mean, w_lo2, w_lo1, w_hi1, w_hi2 = _compute_bands(w_vals)
    f_mean, f_lo2, f_lo1, f_hi1, f_hi2 = _compute_bands(f_vals)

    return {
        "z": z_arr,
        "w_mean": w_mean, "w_lo1": w_lo1, "w_hi1": w_hi1, "w_lo2": w_lo2, "w_hi2": w_hi2,
        "fDE_mean": f_mean, "fDE_lo1": f_lo1, "fDE_hi1": f_hi1, "fDE_lo2": f_lo2, "fDE_hi2": f_hi2,
    }


