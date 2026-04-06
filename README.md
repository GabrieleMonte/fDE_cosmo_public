# fDE_cosmo

This repository accompanies the paper

> **Parameterizing Dark Energy at the density level: A two-parameter alternative to CPL**
> Gabriele Montefalcone and Richard Stiskalek, [arXiv:2603.25735](https://arxiv.org/abs/2603.25735)

It contains the modified Boltzmann solver, likelihood and input files, analysis scripts, and posterior data needed to reproduce all results and figures in the paper.

---

## Dark energy parameterization

We parameterize the normalized dark energy (DE) density as a second-order polynomial in $(1 - a)$:

$$f_{\rm DE}(a) \equiv \frac{\rho_{\rm DE}(a)}{\rho_{\rm DE,0}} = 1 + f_a\,(1 - a) + f_b\,(1 - a)^2$$

with the cosmological constant recovered for $f_a = f_b = 0$. The corresponding equation of state follows from the continuity equation.

### Pivot parameterization

We reparameterize the density expansion in terms of two physically transparent quantities evaluated at a pivot redshift $z_p$:

$$f_p \equiv f_{\rm DE}(z_p) \qquad w_p \equiv w(z_p)$$

yielding

$$f_{\rm DE}(z) = f_p - 3\,(1 + w_p)\,f_p \Big(\frac{z_p - z}{1 + z}\Big) + \frac{1 - f_p + 3\,(1 + w_p)\,f_p\,z_p}{z_p^2}\Big(\frac{z_p - z}{1 + z}\Big)^2$$

with the cosmological constant recovered for $(w_p,\, f_p) = (-1,\, 1)$. The pivot is set to $z_p = 0.5$ throughout.

This formulation directly parameterizes the quantity that enters the Friedmann equation, avoiding the degeneracies inherent to the CPL $(w_0,\, w_a)$ basis. Both parameters are individually well constrained: $w_p$ by BAO + CMB, and $f_p$ by the independent $\Omega_m$ measurement from SNe.

---

## Repository structure

### `class_fDE/`
Modified [CLASS](https://github.com/lesgourg/class_public) Boltzmann solver (based on [class_ede](https://github.com/mwt5345/class_ede)) implementing the $f_{\rm DE}$ parameterization at the background level. The DE density evolution is specified through the fluid sector via `fluid_equation_of_state`:
- `'fpDE'`: single-parameter $f_p$ formulation (Eq. 11 of the paper)
- `'fpDE_2'`: two-parameter $(w_p, f_p)$ formulation (Eq. 8 of the paper)

The `fDE_notebooks/` subdirectory contains the quintessence benchmark validation notebooks that produce Figures 1 and A1.

### `montepython_fDE/`
Likelihood modules and input parameter files for [MontePython](https://github.com/brinckmann/montepython_public). This directory contains only the likelihoods and data relevant to this work, not the full MontePython sampler code. To run MCMC analyses, these likelihoods and data directories should be placed within a working MontePython installation.

The likelihoods provided are:
- **DESI DR2 BAO** (`bao_desi_DR2`): MontePython implementation from [Herold & Ferreira (2024)](https://arxiv.org/abs/2407.04777)
- **Compressed CMB** (`Qcmb`): Gaussian prior on $\theta_s$, $\omega_b$, $\omega_{cb}$ as defined in [DESI Collaboration (2025)](https://arxiv.org/abs/2503.14738)
- **Pantheon+** (`Pantheon_Plus`) and **DESY5 SNe** (`DESY5_SNe`): MontePython implementations from [Herold & Karwal (2024)](https://arxiv.org/abs/2412.00965)
- **Mock likelihoods**: Example mock DESI BAO and DESY5 data for the validation analyses of Appendix B

The directory also includes:
- **`desi_like_input/`**: MontePython `.param` files for all model and dataset combinations analyzed in the paper
- **`mock_desi_like_input/`**: Parameter files for the mock data analyses

### `GetDist_scripts/`
Analysis and plotting scripts:
- **`paper_plots.ipynb`**: Main text figures (Figures 2, 3) and Table 1 constraints
- **`paper_plots_appendix.ipynb`**: Appendix figures (Figures A2--A9)
- **`GetDist_analyzer.py`** / **`GetDist_analyzer_mocks.py`**: Process MCMC chains into the compressed `.npz` posterior files
- **`triangle_plotter.py`**: Contour plotting utilities
- **`wz_fDE_computer.py`**: Reconstructs $f_{\rm DE}(z)$ and $w(z)$ evolution bands from posterior samples
- **`output/`** and **`output_mocks/`**: Pre-computed posterior data (`.npz`) for reproducing all figures without re-running chains

### `mock_desi_data_dir/`
Scripts and notebooks for generating mock DESI BAO and SNe data vectors from fiducial cosmologies, as described in Appendix B.

### `evidence_scripts/`
Scripts for computing the Bayesian evidence and $\Delta\chi^2_{\rm MAP}$ values reported in Table 2.

---

## Getting started

### Compile CLASS

To compile the modified CLASS code and install the Python wrapper (`classy_fDE`):

```bash
cd class_fDE
./compile_class_fDE.sh
```

### Reproduce the paper figures

All pre-computed posterior data are included in the `GetDist_scripts/output/` and `GetDist_scripts/output_mocks/` directories, so the paper figures can be reproduced directly without re-running any MCMC chains. To generate the main text figures (Figures 2 and 3) and the parameter constraints of Table 1, run `GetDist_scripts/paper_plots.ipynb`. The appendix figures (Figures A2--A9) are produced by `GetDist_scripts/paper_plots_appendix.ipynb`. The quintessence benchmark validation plots (Figures 1 and A1) are generated by the notebooks in `class_fDE/fDE_notebooks/`, which require `classy_fDE` to be compiled and installed.

### Run MCMC analyses

To rerun the MCMC analyses from scratch, install [MontePython](https://github.com/brinckmann/montepython_public) and copy the likelihood modules from `montepython_fDE/montepython/likelihoods/` and the data files from `montepython_fDE/data/` into your MontePython installation. The `.param` files in `montepython_fDE/desi_like_input/` specify the parameter settings for each model and dataset combination analyzed in the paper.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Montefalcone:2026iga,
    author = "Montefalcone, Gabriele and Stiskalek, Richard",
    title = "{Parameterizing Dark Energy at the density level: A two-parameter alternative to CPL}",
    eprint = "2603.25735",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "UTWI-10-2026",
    month = "3",
    year = "2026"
}
```

## Acknowledgments

We acknowledge the use of [CLASS](https://github.com/lesgourg/class_public), [MontePython](https://github.com/brinckmann/montepython_public), and [GetDist](https://github.com/cmbant/getdist). 
The MontePython implementations of the DESI DR2 BAO, Pantheon+ and DESY5 SNe likelihoods were originally written by Laura Herold and Tanvi Karwal and are available respectively at [MontePython_desilike](https://github.com/LauraHerold/MontePython_desilike/tree/main) and [cosmo_likelihoods](https://github.com/tkarwal/cosmo_likelihoods).

## AI Disclosure

The preparation of this public repository was aided by Claude (Anthropic). Specifically, AI assistance was used to organize the directory structure for public release, add descriptive comments and markdown sections to the notebooks, and produce an initial draft of this README. No AI tools were used in the scientific analysis or in writing the paper. All AI-generated content was reviewed and verified by the authors.
