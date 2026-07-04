CLASS_fDE
==============================================

`class_fDE` is a modified version of the [CLASS](https://github.com/lesgourg/class_public) Boltzmann
solver (based on [class_ede](https://github.com/mwt5345/class_ede)) that implements the $f_{\rm DE}$
dark energy density parameterization at the background level, as described in
[arXiv:2603.25735](https://arxiv.org/abs/2603.25735). See the [top-level README](../README.md) for
the full physics motivation and how this fits into the rest of the repository (likelihoods, MCMC
inputs, plotting scripts).

## Overview

The normalized DE density is evolved as a second-order polynomial in $(1-a)$,
$f_{\rm DE}(a) = 1 + f_a(1-a) + f_b(1-a)^2$, with $w(a)$ derived from the continuity equation. The
parameterization used is selected at runtime through the `fluid_equation_of_state` input:

| `fluid_equation_of_state` | Parameters | Description |
| --- | --- | --- |
| `CLP` | `w0_fld`, `wa_fld` (or `w0wa_fld`), `cs2_fld` | Standard CPL $w_0w_a$CDM |
| `faDE` | `fa_fld`, `cs2_fld` | Single-parameter $f_a$ formulation |
| `fpDE` | `fp_fld`, `ap_fld`, `cs2_fld` | Single-parameter pivot formulation, $f_p$ (paper Eq. 11) |
| `faDE_2` | `fa_fld`, `dfa_fld`, `cs2_fld` | Two-parameter $(f_a, f_b)$ formulation |
| `fpDE_2` | `fp_fld`, `wp_fld`, `ap_fld`, `cs2_fld` | Two-parameter pivot formulation, $(w_p, f_p)$ (paper Eq. 8) |
| `EDE` | (pre-existing, unrelated to this paper) | Early dark energy |

For all `f(a/p)DE` variants, $w(a)$ is computed directly from $f_{\rm DE}(a)$ via
$w(a) = -1 + a f_{\rm DE}'(a) / \bigl(3 f_{\rm DE}(a)\bigr)$. A small regularization sets
$w(a) = 0$ whenever $|f_{\rm DE}(a)| < 10^{-8}$, avoiding the numerical singularity that would
otherwise occur wherever $f_{\rm DE}(a)$ crosses zero. This is implemented in
[`source/background.c`](source/background.c) (`background_w_fld`), with the corresponding input
parsing in [`source/input.c`](source/input.c).

## Example Notebook

[`fDE_notebooks/fDE_examples.ipynb`](fDE_notebooks/fDE_examples.ipynb) benchmarks the `fpDE`/`fpDE_2`
background evolution against a quintessence scalar field solved directly by CLASS, reproducing
Figures 1 and A1 of the paper. It requires `classy_fDE` to be compiled and installed (see below).

## Installation

To compile CLASS and install the `classy_fDE` Python wrapper in one step:

```bash
cd class_fDE
./compile_class_fDE.sh    # = make clean && make && pip install .
python -c 'from classy_fDE import Class; print("OK")'
```

Or, following the standard CLASS installation if you only need the C binary:

```bash
git clone https://github.com/GabrieleMonte/fDE_cosmo_public.git
cd fDE_cosmo_public/class_fDE
make
```

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
