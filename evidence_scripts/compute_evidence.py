"""Compute harmonic evidence for DESI chains."""
import argparse
import json
import re
from pathlib import Path

import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

LOG10E = np.log10(np.e)


###############################################################################
# Reading MontePython chains
###############################################################################


def parse_log_param(log_param_path):
    """Parse a MontePython `log.param` file."""
    varying = []
    derived = []

    with open(log_param_path) as f:
        for line in f:
            if line.strip().startswith('#'):
                continue
            if 'data.parameters' not in line:
                continue

            m = re.search(r"data\.parameters\['(.+?)'\]\s*=\s*\[(.+)\]", line)
            if m is None:
                continue

            name = m.group(1)
            entries = [x.strip().strip("'\"") for x in m.group(2).split(',')]

            def _parse(s):
                s = s.strip()
                if s in ('None', '-1'):
                    return None
                try:
                    return float(s)
                except ValueError:
                    return s

            arr = [_parse(e) for e in entries]
            ptype = arr[5] if len(arr) > 5 else None

            if ptype in ('derived', 'derived_lkl'):
                derived.append(name)
            elif arr[3] is not None and arr[3] != 0:
                varying.append({
                    'name': name,
                    'center': arr[0],
                    'min': arr[1],
                    'max': arr[2],
                    'sigma': arr[3],
                    'scale': arr[4],
                })

    return varying, derived


def read_chains(chain_dir, burnin_fraction=0.3, min_samples=100):
    """Read MontePython MCMC chains from a directory."""
    chain_dir = Path(chain_dir)

    log_param = chain_dir / 'log.param'
    if not log_param.exists():
        raise FileNotFoundError(
            f"No log.param found in {chain_dir}. Is this a MontePython "
            f"output directory?")
    varying, derived = parse_log_param(log_param)
    n_varying = len(varying)
    print(f"Found {n_varying} varying parameters: "
          f"{[p['name'] for p in varying]}")
    print(f"Found {len(derived)} derived parameters (will be ignored).")

    chain_files = sorted(chain_dir.glob("*__*.txt"))
    if not chain_files:
        raise FileNotFoundError(
            f"No chain files (*__*.txt) found in {chain_dir}.")
    print(f"Found {len(chain_files)} chain file(s).")

    chains_params = []
    chains_loglike = []

    for cf in chain_files:
        rows = []
        with open(cf) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    rows.append([float(x) for x in line.split()])
                except ValueError:
                    continue

        if not rows:
            print(f"  Warning: {cf.name} is empty, skipping.")
            continue

        data = np.array(rows)
        n = len(data)
        i0 = int(burnin_fraction * n)
        data = data[i0:]

        if len(data) == 0:
            print(f"  Warning: {cf.name} has no samples after burn-in, "
                  f"skipping.")
            continue

        loglike = -data[:, 1]
        params = data[:, 2:2 + n_varying]

        # Check for non-finite values
        bad_loglike = ~np.isfinite(loglike)
        bad_params = ~np.all(np.isfinite(params), axis=1)
        bad = bad_loglike | bad_params
        n_bad = np.sum(bad)
        if n_bad > 0:
            print(f"  Warning: {cf.name}: dropping {n_bad} rows with "
                  f"non-finite values.")
            loglike = loglike[~bad]
            params = params[~bad]

        if len(loglike) < min_samples:
            print(f"  Warning: {cf.name}: only {len(loglike)} samples "
                  f"(< {min_samples}), skipping.")
            continue

        chains_params.append(params)
        chains_loglike.append(loglike)

        print(f"  {cf.name}: {n} rows total, {len(loglike)} after burn-in "
              f"and filtering.")

    if not chains_params:
        raise RuntimeError("No valid chain data found.")

    return chains_params, chains_loglike, varying, derived


###############################################################################
# Evidence estimators
###############################################################################


def compute_prior_volume_correction(varying_params):
    """Compute ln(prior volume) for bounded parameters.

    Returns None if any parameter has unbounded (improper) prior.
    """
    ln_V = 0.0
    for p in varying_params:
        if p['min'] is None or p['max'] is None:
            return None
        ln_V += np.log(p['max'] - p['min'])
    return ln_V


def prepare_3d_array(chains_list, min_chain_length=None):
    """Stack a list of arrays into a 3D array (nchains, nsamples, ...)."""
    if min_chain_length is None:
        min_chain_length = min(len(c) for c in chains_list)
    return np.array([c[:min_chain_length] for c in chains_list])


def split_chains(params, loglike, nchains):
    """Split samples into `nchains` sub-chains."""
    n = len(params)
    chunk = n // nchains
    params_list = [params[i * chunk:(i + 1) * chunk] for i in range(nchains)]
    loglike_list = [loglike[i * chunk:(i + 1) * chunk] for i in range(nchains)]
    return params_list, loglike_list


def harmonic_evidence(samples_3d, logposterior_3d, temperature=0.8,
                      epochs_num=30, verbose=True):
    """Estimate the evidence using the `harmonic` package."""
    import harmonic as hm

    ndim = samples_3d.shape[-1]

    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples_3d, logposterior_3d)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=0.5)

    model = hm.model.RQSplineModel(
        ndim, standardize=True, temperature=temperature)
    model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose)

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)

    ln_evidence = -ev.ln_evidence_inv

    # Errors are on ln(1/Z); negate and swap for ln(Z) = -ln(1/Z)
    err_neg, err_pos = ev.compute_ln_inv_evidence_errors()
    ln_evidence_err = (-err_pos, -err_neg)

    return ln_evidence, ln_evidence_err


###############################################################################
# Main
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Compute harmonic evidence for a single chain directory.")
    parser.add_argument("chain_dir", help="Path to MontePython chain dir")
    parser.add_argument("--burnin", type=float, default=0.3)
    parser.add_argument("--nchains", type=int, default=10,
                        help="Number of sub-chains for harmonic (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for this chain's results")
    args = parser.parse_args()

    chain_dir = Path(args.chain_dir)
    name = chain_dir.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read chains
    chains_params, chains_loglike, varying, derived = read_chains(
        chain_dir, burnin_fraction=args.burnin)

    # Print parameters and priors (MontePython priors are always flat/uniform)
    n_unbounded = sum(
        1 for p in varying if p['min'] is None or p['max'] is None)
    print(f"\nComputing evidence over {len(varying)} parameters:")
    max_name_len = max(len(p['name']) for p in varying)
    for p in varying:
        lo = f"{p['min']:.6g}" if p['min'] is not None else "-inf"
        hi = f"{p['max']:.6g}" if p['max'] is not None else "+inf"
        bounded = p['min'] is not None and p['max'] is not None
        label = "flat" if bounded else "flat (unbounded)"
        print(f"  {p['name']:<{max_name_len}}  {label:<20s} [{lo}, {hi}]")
    if n_unbounded > 0:
        print(f"\n  WARNING: {n_unbounded} parameter(s) have unbounded flat "
              f"priors. Absolute evidence is undefined;\n"
              f"           Bayes factors are valid only between models "
              f"sharing these priors.")

    print(f"\nDerived parameters ({len(derived)}, not used in evidence):")
    for d in derived:
        print(f"  {d}")
    print()

    # Merge all chains
    all_params = np.concatenate(chains_params)
    all_loglike = np.concatenate(chains_loglike)
    n_total = len(all_params)

    # Print sample ranges and prior volume waste
    print(f"Sample ranges and prior volume penalty:")
    log10_prior_waste_total = 0.0
    has_unbounded = False
    for i, p in enumerate(varying):
        lo, hi = all_params[:, i].min(), all_params[:, i].max()
        sample_width = hi - lo
        if p['min'] is not None and p['max'] is not None:
            prior_width = p['max'] - p['min']
            waste = np.log10(prior_width / sample_width)
            log10_prior_waste_total += waste
            print(f"  {p['name']:<{max_name_len}}  "
                  f"[{lo:.6g}, {hi:.6g}]  "
                  f"prior/sample = {waste:+.2f}")
        else:
            has_unbounded = True
            print(f"  {p['name']:<{max_name_len}}  "
                  f"[{lo:.6g}, {hi:.6g}]  "
                  f"prior/sample = inf (unbounded)")
    if has_unbounded:
        print(f"  Total prior volume penalty >= {log10_prior_waste_total:.2f}"
              f" (unbounded params not included)")
    else:
        print(f"  Total prior volume penalty = "
              f"{log10_prior_waste_total:.2f}")
    print()

    # Rough posterior diagnostics
    log10_mean_like = (logsumexp(all_loglike) - np.log(n_total)) * LOG10E
    log10_max_like = np.max(all_loglike) * LOG10E
    log10_mean_loglike = np.mean(all_loglike) * LOG10E
    chi2_MAP = -2.0 * np.max(all_loglike)
    print(f"Posterior diagnostics:")
    print(f"  max  log10 L     = {log10_max_like:.4f}")
    print(f"  mean log10 L     = {log10_mean_loglike:.4f}")
    print(f"  log10 <L>        = {log10_mean_like:.4f}  (logsumexp)")
    print(f"  chi2_MAP         = {chi2_MAP:.4f}")
    print()

    # Corner plot
    labels = [p['name'] for p in varying]
    fig = corner.corner(all_params, labels=labels, plot_contours=False,
                        no_fill_contours=True, show_titles=True,
                        title_fmt=".4f")
    fig.savefig(output_dir / "corner.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Corner plot saved to {output_dir / 'corner.png'}")

    nchains = args.nchains
    print(f"Concatenated and splitting into {nchains} sub-chains "
          f"(~{n_total // nchains} samples each).")
    chains_params, chains_loglike = split_chains(
        all_params, all_loglike, nchains)
    min_len = min(len(c) for c in chains_params)

    samples_3d = prepare_3d_array(chains_params, min_len)
    logpost_3d = prepare_3d_array(chains_loglike, min_len)

    # Compute harmonic evidence
    ln_Z, ln_Z_err = harmonic_evidence(
        samples_3d, logpost_3d,
        temperature=args.temperature,
        epochs_num=args.epochs)

    # Prior volume correction
    ln_V = compute_prior_volume_correction(varying)

    # Convert to log10
    log10_Z = ln_Z * LOG10E
    log10_Z_err_lo = ln_Z_err[0] * LOG10E
    log10_Z_err_hi = ln_Z_err[1] * LOG10E

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Chain: {name}")
    print(f"log10 Z = {log10_Z:.4f}  ({log10_Z_err_lo:+.4f} / "
          f"{log10_Z_err_hi:+.4f})")
    if ln_V is not None:
        log10_V = ln_V * LOG10E
        log10_Z_corr = log10_Z - log10_V
        print(f"log10 V (prior volume) = {log10_V:.4f}")
        print(f"log10 Z (prior-corrected) = {log10_Z_corr:.4f}")
    print(f"{'=' * 60}")

    # Save npz
    np.savez(output_dir / "evidence.npz", ln_Z=ln_Z, ln_Z_err=ln_Z_err,
             ln_V=ln_V, chain_name=name)

    # Save JSON for the report
    result = {
        "chain": name,
        "n_params": len(varying),
        "params": [
            {"name": p['name'], "min": p['min'], "max": p['max']}
            for p in varying
        ],
        "log10_Z": float(log10_Z),
        "log10_Z_err": [float(log10_Z_err_lo), float(log10_Z_err_hi)],
        "log10_V": float(ln_V * LOG10E) if ln_V is not None else None,
        "chi2_MAP": float(chi2_MAP),
        "log10_max_L": float(log10_max_like),
        "log10_mean_L": float(log10_mean_loglike),
        "log10_avg_L": float(log10_mean_like),
        "log10_prior_waste": float(log10_prior_waste_total),
        "has_unbounded": has_unbounded,
        "n_samples_used": int(nchains * min_len),
    }
    with open(output_dir / "evidence.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
