# fDE_cosmo

This repository accompanies the paper

> **Parameterizing Dark Energy at the density level: A two-parameter alternative to CPL**
> Gabriele Montefalcone and Richard Stiskalek, [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

It contains the modified Boltzmann solver, likelihood and input files, analysis scripts, and posterior data needed to reproduce all results and figures in the paper.

---

## Dark energy parameterization

We parameterize the normalized dark energy (DE) density as a second-order polynomial in `(1 - a)`:

$$f_{\rm DE}(a) \equiv \frac{\rho_{\rm DE}(a)}{\rho_{\rm DE,0}} = 1 + f_a\,(1 - a) + f_b\,(1 - a)^2\,,$$

with $\Lambda$CDM recovered for $f_a = f_b = 0$. The corresponding equation of state follows from the continuity equation.

### Pivot parameterization $(w_p,\, f_p)$

We reparameterize the density expansion in terms of two physically transparent quantities evaluated at a pivot redshift $z_p$:

$$f_p \equiv f_{\rm DE}(z_p)\,, \qquad w_p \equiv w(z_p)\,,$$

yielding

$$f_{\rm DE}(z) = f_p - 3\,(1 + w_p)\,f_p \left(\frac{z_p - z}{1 + z}\right) + \frac{1 - f_p + 3\,(1 + w_p)\,f_p\,z_p}{z_p^2}\left(\frac{z_p - z}{1 + z}\right)^2\,,$$

with $\Lambda$CDM recovered for $(w_p,\, f_p) = (-1,\, 1)$. The pivot is set to $z_p = 0.5$ throughout.

This formulation directly parameterizes the quantity that enters the Friedmann equation, avoiding the degeneracies inherent to the CPL $(w_0,\, w_a)$ basis. Both parameters are individually well constrained: $w_p$ by BAO + CMB, and $f_p$ by the independent $\Omega_m$ measurement from SNe.

---

## Repository structure

### `class_fDE/`
Modified [CLASS](https://github.com/lesgourg/class_public) Boltzmann solver (based on [class_ede](https://github.com/mwt5345/class_ede)) implementing the $f_{\rm DE}$ parameterization at the background level. The DE density evolution is specified through the fluid sector via `fluid_equation_of_state`:
- `'fpDE'`: single-parameter $f_p$ formulation (Eq. 11 of the paper)
- `'fpDE_2'`: two-parameter $(w_p, f_p)$ formulation (Eq. 8 of the paper)

The `fDE_notebooks/` subdirectory contains the quintessence benchmark validation notebooks that produce Figures 1 and A1.

### `montepython_fDE/`
Likelihood modules and input parameter files for [MontePython](https://github.com/brinckmann/montepython_public). This directory contains only the likelihoods and data relevant to this work (not the full MontePython code):
- **`montepython/likelihoods/`**: DESI DR2 BAO (`bao_desi_DR2`), compressed CMB (`Qcmb`), Pantheon+ (`Pantheon_Plus`), DESY5 SNe (`DESY5_SNe`), and example mock likelihoods for the validation analyses
- **`data/`**: Corresponding data files for each likelihood
- **`desi_like_input/`**: MontePython `.param` files for all model/dataset combinations analyzed in the paper ($\Lambda$CDM, $w_0 w_a$CDM, $w_p f_p$CDM, $f_p$CDM)
- **`mock_desi_like_input/`**: Parameter files for the mock data analyses of Appendix B

### `GetDist_scripts/`
Analysis and plotting scripts:
- **`paper_plots.ipynb`**: Main text figures (Figures 2, 3) and Table 1 constraints
- **`paper_plots_appendix.ipynb`**: Appendix figures (Figures A2--A9)
- **`GetDist_analyzer.py`** / **`GetDist_analyzer_mocks.py`**: Process MCMC chains into the compressed `.npz` posterior files
- **`triangle_plotter.py`**: Contour plotting utilities
- **`wz_fDE_computer.py`**: Reconstructs $f_{\rm DE}(z)$ and $w(z)$ evolution bands from posterior samples
- **`output/`** and **`output_mocks/`**: Pre-computed posterior data (`.npz`) for reproducing all figures without re-running chains

### `mock_desi_data_dir/`
Scripts and notebooks for generating mock DESI BAO and SNe data vectors from fiducial cosmologies ($\Lambda$CDM and exponential quintessence), as described in Appendix B.

### `evidence_scripts/`
Scripts for computing the Bayesian evidence and $\Delta\chi^2_{\rm MAP}$ values reported in Table 2.

---

## Getting started

### 1. Compile CLASS

```bash
cd class_fDE
./compile_class_fDE.sh
```

This compiles the C library and installs the Python wrapper as `classy_fDE`.

### 2. Install MontePython

Follow the [MontePython installation instructions](https://github.com/brinckmann/montepython_public). Then point MontePython to `class_fDE` as the Boltzmann solver by setting the path in `montepython_fDE/default.conf`.

### 3. Run an MCMC analysis

Example: $w_p f_p$CDM with DESI + $Q_{\rm CMB}$ + Pantheon+

```bash
cd montepython_fDE
python montepython/MontePython.py run \
    -p desi_like_input/desi_dr2_Qcmb_PP_fp_wp.param \
    -o chains/desi_pp_fpwp \
    -N 100000
```

### 4. Reproduce figures

The `GetDist_scripts/output/` directory contains all pre-computed posteriors. Open the paper notebooks directly:

```bash
cd GetDist_scripts
jupyter notebook paper_plots.ipynb
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{Montefalcone:2026xxx,
    author = {Montefalcone, Gabriele and Stiskalek, Richard},
    title = "{Parameterizing Dark Energy at the density level: A two-parameter alternative to CPL}",
    eprint = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    year = "2026"
}
```

We also acknowledge the use of [CLASS](https://github.com/lesgourg/class_public), [MontePython](https://github.com/brinckmann/montepython_public), and [GetDist](https://github.com/cmbant/getdist).
