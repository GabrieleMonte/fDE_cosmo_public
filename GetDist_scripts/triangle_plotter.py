import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use('../class_fDE/fDE_notebooks/mine.mplstyle')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amssymb}",
    "axes.axisbelow": False,
})

# ---------------------------------------------------------------------------
# Parameter -> TeX label mapping
# ---------------------------------------------------------------------------
PAR_TEX = {
    "omega_b":      r"$100\,\omega_b$",
    "omega_cb":     r"$\omega_{cb}$",
    "omega_cdm":    r"$\omega_{\rm cdm}$",
    "omega_m":      r"$\omega_m$",
    "theta_s_100":  r"$100\,\theta_s$",
    "100theta_s":   r"$100\,\theta_s$",
    "H0":           r"$H_0$",
    "h":            r"$h$",
    "n_s":          r"$n_s$",
    "ln10^{10}A_s": r"$\ln(10^{10}A_s)$",
    "A_s":          r"$A_s$",
    "sigma8":       r"$\sigma_8$",
    "S8":           r"$S_8$",
    "z_reio":       r"$z_{\rm reio}$",
    "tau_reio":     r"$\tau_{\rm reio}$",
    "w0_fld":       r"$w_0$",
    "wa_fld":       r"$w_a$",
    "wp_fld":       r"$w_p$",
    "fa_fld":       r"$f_a$",
    "fp_fld":       r"$f_p$",
    "Omega_Lambda":  r"$\Omega_\Lambda$",
    "Omega_m":       r"$\Omega_m$",
    "rdrag":         r"$r_{\rm drag}$",
}


def _get_label(par):
    """Return TeX label for a parameter, falling back to raw string."""
    return PAR_TEX.get(par, rf"${par}$")


def color_shades(color, n, lightest_alpha=0.25):
    """Generate n shades from `color` to lighter, blended toward white.

    Parameters
    ----------
    color : str or RGB tuple
        Base color (e.g. "tab:blue").
    n : int
        Number of shades.
    lightest_alpha : float
        Fraction of base color in the lightest shade (0=white, 1=original).

    Returns
    -------
    list of hex color strings, from darkest to lightest.
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    fracs = np.linspace(1.0, lightest_alpha, n)
    return [mcolors.to_hex(rgb * f + white * (1 - f)) for f in fracs]


def _auto_lim(arr, pad_frac=0.02):
    """Compute (lo, hi) with a small padding from a data array."""
    lo, hi = arr.min(), arr.max()
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _load_2d(data, prefix, par_y, par_x):
    """Load 2D contour data, swapping axes if needed.

    Tries key ``{prefix}_{par_y}__{par_x}`` first.  If that doesn't exist,
    falls back to ``{prefix}_{par_x}__{par_y}`` with X<->Y swap and P^T.

    Returns (X, Y, P, contour_levels) ready for ``contour(X, Y, P, ...)``.
    """
    key = f"{prefix}_{par_y}__{par_x}"
    if f"{key}_x" in data:
        return (data[f"{key}_x"], data[f"{key}_y"],
                data[f"{key}_p_grid"], data[f"{key}_contour_levels"])
    # Try swapped order
    key_swap = f"{prefix}_{par_x}__{par_y}"
    X_raw = data[f"{key_swap}_x"]
    Y_raw = data[f"{key_swap}_y"]
    P_raw = data[f"{key_swap}_p_grid"]
    cl = data[f"{key_swap}_contour_levels"]
    return Y_raw, X_raw, P_raw.T, cl


# ---------------------------------------------------------------------------
# Main triangle plotter
# ---------------------------------------------------------------------------
def triangle_plot(
    datasets,
    pars_1d,
    style="default",
    figsize=None,
    xlims=None,
    xticks=None,
    ref_values=None,
    tick_labelsize=7,
    contour_lw=0.8,
    fill_alpha=0.4,
    show_legend=True,
    legend_loc=None,
    legend_fontsize=9,
    legend_title=None,
    hspace=0,
    wspace=0,
    save_path=None,
    save_dpi=150,
    show=True,
):
    """
    Generalised triangle (corner) plot for 1D posteriors + 2D contours.

    Parameters
    ----------
    datasets : list of dict
        Each dict must contain:
            "data"   : loaded np.load object (.npz)
            "prefix" : analysis_type string  (e.g. "desi_cmb_w0wa")
            "colors" : [main_color, light_color]
            "label"  : str for legend
    pars_1d : list of str
        Parameter names (minimum 3). Diagonal = 1D, lower triangle = 2D.
    style : str
        "default" — all datasets filled with ``fill_alpha`` (original).
        "paper"  — dataset 0 as dashed contours only; datasets 1-2 as
        opaque filled contours with zorder layering (paper style).
    figsize : tuple, optional
        Figure size. Default scales with npar.
    xlims : dict, optional
        Manual axis limits {par_name: (lo, hi)}. Overrides auto-limits.
    xticks : dict, optional
        Manual tick positions {par_name: [tick1, tick2, ...]}. Applied to
        all panels sharing that parameter axis (diagonal x, 2D x and y).
    ref_values : dict, optional
        Reference/default values {par_name: value} to draw as dashed gray
        lines. E.g. {"w0_fld": -1, "wa_fld": 0}. None by default (no lines).
    tick_labelsize : float
    contour_lw : float
    fill_alpha : float
    legend_loc : tuple, optional
        Legend location (e.g. (x, y) in axes coords). Default: top-right corner.
    legend_fontsize : float
    hspace : float
        Vertical spacing between subplots (default 0).
    wspace : float
        Horizontal spacing between subplots (default 0).
    save_path : str, optional
        Base path (without extension) to save .png and .pdf.
    save_dpi : int
    show : bool

    Returns
    -------
    fig, axes
    """
    npar = len(pars_1d)
    assert npar >= 3, "Need at least 3 parameters for a triangle plot."

    if xlims is None:
        xlims = {}
    if xticks is None:
        xticks = {}
    if ref_values is None:
        ref_values = {}
    if figsize is None:
        figsize = (1.6 * npar, 1.6 * npar)

    fig, axes = plt.subplots(npar, npar, figsize=figsize)

    # --- Hide upper triangle ---
    for i in range(npar):
        for j in range(i + 1, npar):
            axes[i, j].set_visible(False)

    # --- Pre-compute auto limits per parameter ---
    auto_xlims = {}
    for par in pars_1d:
        lo_all, hi_all = np.inf, -np.inf
        for ds in datasets:
            key_x = f"{ds['prefix']}_{par}_1D_x"
            if key_x in ds["data"]:
                x = ds["data"][key_x]
                lo_all = min(lo_all, x.min())
                hi_all = max(hi_all, x.max())
        if lo_all < hi_all:
            pad = (hi_all - lo_all) * 0.02
            auto_xlims[par] = (lo_all - pad, hi_all + pad)

    def get_lim(par):
        return xlims.get(par, auto_xlims.get(par, None))

    # --- Diagonal: 1D posteriors ---
    for i, par in enumerate(pars_1d):
        ax = axes[i, i]
        for ds in datasets:
            key_x = f"{ds['prefix']}_{par}_1D_x"
            key_P = f"{ds['prefix']}_{par}_1D_P"
            x = ds["data"][key_x]
            P = ds["data"][key_P]
            ax.plot(x, P, color=ds["colors"][0], lw=1.2)
        # y-axis: ticks on both sides, range [0, 1]
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([0., 0.5, 1.])
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(labelright=True, labelleft=False)
        lim = get_lim(par)
        if lim is not None:
            ax.set_xlim(lim)
        if par in xticks:
            ax.set_xticks(xticks[par])
        if par in ref_values:
            ax.axvline(ref_values[par], lw=0.6, ls='--', c='tab:gray',
                       zorder=8 if style == "paper" else 5)

    # --- Lower triangle: 2D contours ---
    paper_zorders = [2, 3, 4]
    for i in range(1, npar):
        for j in range(i):
            ax = axes[i, j]
            par_y = pars_1d[i]
            par_x = pars_1d[j]

            for di, ds in enumerate(datasets):
                X, Y, P, cl_raw = _load_2d(
                    ds["data"], ds["prefix"], par_y, par_x)

                if style == "paper":
                    cl = sorted(cl_raw)[1:]  # keep 68% & 95%
                    levels = sorted(np.append([P.max() + 1], cl))
                    ax.contourf(
                        X, Y, P, levels=levels,
                        colors=[ds["colors"][1], ds["colors"][0]],
                        zorder=paper_zorders[di],
                    )
                    ax.contour(
                        X, Y, P, levels=levels,
                        colors=[ds["colors"][0], ds["colors"][0]],
                        linewidths=contour_lw,
                        zorder=paper_zorders[di],
                    )
                    ax.contour(
                        X, Y, P,
                        levels=[cl[1], P.max() + 1],
                        colors=[ds["colors"][0]],
                        linewidths=contour_lw, zorder=6,
                    )
                else:  # default
                    cl = sorted(cl_raw)[1:]  # keep 68% CL level
                    levels = sorted(np.append([P.max() + 1], cl))
                    ax.contourf(
                        X, Y, P, levels=levels,
                        colors=[ds["colors"][1], ds["colors"][0]],
                        alpha=fill_alpha,
                    )
                    ax.contour(
                        X, Y, P, levels=levels,
                        colors=[ds["colors"][0], ds["colors"][0]],
                        linewidths=contour_lw,
                    )

            ref_zorder = 8 if style == "paper" else 5
            xlim = get_lim(par_x)
            ylim = get_lim(par_y)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if par_x in xticks:
                ax.set_xticks(xticks[par_x])
            if par_y in xticks:
                ax.set_yticks(xticks[par_y])
            if par_x in ref_values:
                ax.axvline(ref_values[par_x], lw=0.6, ls='--',
                           c='tab:gray', zorder=ref_zorder)
            if par_y in ref_values:
                ax.axhline(ref_values[par_y], lw=0.6, ls='--',
                           c='tab:gray', zorder=ref_zorder)

    # --- Axis labels ---
    for i in range(npar):
        axes[npar - 1, i].set_xlabel(_get_label(pars_1d[i]))
        if i > 0:
            axes[i, 0].set_ylabel(_get_label(pars_1d[i]))

    # --- Tick label cleanup ---
    for i in range(npar):
        for j in range(i + 1):
            ax = axes[i, j]
            ax.tick_params(labelsize=tick_labelsize)
            # suppress x-labels except bottom row
            if i < npar - 1:
                ax.set_xticklabels([])
            # suppress y-labels except leftmost column (skip diagonal)
            if j > 0 and i != j:
                ax.set_yticklabels([])

    # --- Legend (68/95 CL patch pairs, top-right corner by default) ---
    if show_legend and (len(datasets) > 1 or datasets[0].get("label")):
        handles, labels = [], []
        for ds in datasets:
            if ds.get("label"):
                patch68 = Patch(facecolor=ds["colors"][0])
                patch95 = Patch(facecolor=ds["colors"][1])
                handles.append((patch68, patch95))
                labels.append(ds["label"])
        if handles:
            leg_ax = axes[0, npar - 1]
            leg_ax.set_visible(True)
            leg_ax.axis("off")
            if legend_loc is not None:
                leg = leg_ax.legend(
                    handles=handles, labels=labels,
                    loc="upper right", bbox_to_anchor=legend_loc,
                    fontsize=legend_fontsize, frameon=False,
                    title=legend_title, title_fontsize=legend_fontsize,
                    handler_map={tuple: mpl.legend_handler.HandlerTuple(
                        ndivide=None, pad=0.0)},
                )
            else:
                leg = leg_ax.legend(
                    handles=handles, labels=labels, loc="center",
                    fontsize=legend_fontsize, frameon=False,
                    title=legend_title, title_fontsize=legend_fontsize,
                    handler_map={tuple: mpl.legend_handler.HandlerTuple(
                        ndivide=None, pad=0.0)},
                )
            # Right-align the whole legend column (title + entries)
            leg._legend_box.align = "right"

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # --- Save ---
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path + ".png", dpi=save_dpi, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
        plt.close()
    return fig, axes
