"""Generate a text report from evidence JSON results."""
import json
import math
import re
import sys
from pathlib import Path

from scipy.stats import norm

# Dataset groups keyed by the substring between "desi_dr2_Qcmb_" and the
# model name. Longer prefixes must come first so "" doesn't match everything.
DATASET_GROUPS = [
    ("DESY5_", "DESI + CMB + DESY5"),
    ("PP_", "DESI + CMB + Pantheon+"),
    ("", "DESI + CMB"),
]

MODEL_ORDER = ["lcdm", "w0wa", "fa", "fp", "fp_wp"]


def parse_chain_name(name):
    """Extract (dataset_key, model) from a chain name."""
    prefix = "desi_dr2_Qcmb_"
    rest = name[len(prefix):]

    for ds_key, _ in DATASET_GROUPS:
        if rest.startswith(ds_key):
            model_version = rest[len(ds_key):]
            model = re.sub(r"_v\d+$", "", model_version)
            return ds_key, model

    return None, None


def fmt_waste(r):
    """Format the prior waste column."""
    w = r['log10_prior_waste']
    if r['has_unbounded']:
        return f">={w:+.2f}"
    return f"{w:+.2f}"


def fmt_delta(log10_Z, log10_Z_lcdm):
    """Format delta log10 Z relative to LCDM."""
    if not math.isfinite(log10_Z) or not math.isfinite(log10_Z_lcdm):
        return "---"
    return f"{log10_Z - log10_Z_lcdm:+.4f}"


def dlog10Z_to_sigma(log10_Z, log10_Z_lcdm):
    """Convert dlog10Z to equivalent Gaussian sigma.

    Treats the Bayes factor B = Z/Z_LCDM as posterior odds (equal model
    priors), giving p(LCDM) = 1/(1+B). Positive sigma means the extended
    model is preferred.
    """
    if not math.isfinite(log10_Z) or not math.isfinite(log10_Z_lcdm):
        return "---"
    dlog10Z = log10_Z - log10_Z_lcdm
    if dlog10Z == 0:
        return "0.0"

    B = 10**dlog10Z
    p_lcdm = 1.0 / (1.0 + B)

    if dlog10Z > 0:
        sigma = norm.isf(p_lcdm)
    else:
        sigma = -norm.isf(1.0 - p_lcdm)
    return f"{sigma:+.1f}"


def fmt_prior(p):
    """Format a single parameter's prior as a string."""
    lo = f"{p['min']:.6g}" if p['min'] is not None else "-inf"
    hi = f"{p['max']:.6g}" if p['max'] is not None else "+inf"
    return f"[{lo}, {hi}]"


def write_model_summary(out, all_results):
    """Write a summary of model parameters and priors."""
    models_seen = {}
    for group in all_results.values():
        for model, r in group.items():
            if model not in models_seen:
                models_seen[model] = r['params']

    out.write("MODEL PARAMETERS AND PRIORS\n")
    out.write("=" * 70 + "\n\n")

    for model in MODEL_ORDER:
        if model not in models_seen:
            continue
        params = models_seen[model]
        out.write(f"  {model}  ({len(params)} params)\n")
        for p in params:
            out.write(f"    {p['name']:<12}  flat  {fmt_prior(p)}\n")
        out.write("\n")


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output"

    json_files = sorted(output_dir.glob("*/evidence.json"))
    if not json_files:
        print(f"No evidence.json files found in subdirectories of "
              f"{output_dir}")
        sys.exit(1)

    results = {}
    for jf in json_files:
        with open(jf) as f:
            r = json.load(f)
        ds_key, model = parse_chain_name(r['chain'])
        if ds_key is None:
            continue
        results.setdefault(ds_key, {})[model] = r

    report_path = output_dir / "evidence_report.txt"
    with open(report_path, "w") as out:
        out.write("=" * 70 + "\n")
        out.write("EVIDENCE REPORT (harmonic estimator)\n")
        out.write("=" * 70 + "\n\n")

        write_model_summary(out, results)

        out.write("EVIDENCE RESULTS\n")
        out.write("=" * 70 + "\n\n")
        out.write("  waste = log10(prior_width / sample_range), "
                  "summed over params\n")
        out.write("  (>= means some priors are unbounded)\n")
        out.write("  dlog10Z = log10(Z / Z_LCDM)\n")
        out.write("  sigma = equivalent Gaussian significance "
                  "(positive = model preferred over LCDM)\n")
        out.write("  dchi2 = chi2_MAP - chi2_MAP(LCDM)  "
                  "(negative = better fit)\n\n")

        for ds_key, ds_label in DATASET_GROUPS:
            group = results.get(ds_key)
            if not group:
                continue

            out.write("-" * 70 + "\n")
            out.write(f"  {ds_label}\n")
            out.write("-" * 70 + "\n")

            lcdm = group.get("lcdm")
            log10_Z_lcdm = lcdm['log10_Z'] if lcdm else float('nan')
            chi2_MAP_lcdm = lcdm['chi2_MAP'] if lcdm else float('nan')

            header = (f"  {'Model':<8}  {'log10 Z':>10}  "
                      f"{'err lo':>8}  {'err hi':>8}  "
                      f"{'dlog10Z':>8}  {'sigma':>6}  "
                      f"{'chi2_MAP':>10}  {'dchi2':>8}  "
                      f"{'max lgL':>8}  {'mean lgL':>9}  {'lg<L>':>8}  "
                      f"{'waste':>8}  "
                      f"{'N_par':>5}  {'N_samp':>8}")
            out.write(header + "\n")

            models = [m for m in MODEL_ORDER if m in group]
            for model in models:
                r = group[model]
                err = r['log10_Z_err']
                delta = fmt_delta(r['log10_Z'], log10_Z_lcdm)
                sigma = dlog10Z_to_sigma(r['log10_Z'], log10_Z_lcdm)
                chi2 = r['chi2_MAP']
                dchi2 = chi2 - chi2_MAP_lcdm if (
                    math.isfinite(chi2) and math.isfinite(chi2_MAP_lcdm)
                ) else float('nan')
                dchi2_str = f"{dchi2:+8.2f}" if math.isfinite(dchi2) else "---"
                out.write(
                    f"  {model:<8}  "
                    f"{r['log10_Z']:>10.4f}  "
                    f"{err[0]:>+8.4f}  "
                    f"{err[1]:>+8.4f}  "
                    f"{delta:>8s}  "
                    f"{sigma:>6s}  "
                    f"{chi2:>10.2f}  "
                    f"{dchi2_str:>8s}  "
                    f"{r['log10_max_L']:>8.4f}  "
                    f"{r['log10_mean_L']:>9.4f}  "
                    f"{r['log10_avg_L']:>8.4f}  "
                    f"{fmt_waste(r):>8s}  "
                    f"{r['n_params']:>5d}  "
                    f"{r['n_samples_used']:>8d}\n")

            out.write("\n")

    print(f"Report written to {report_path}")
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
