"""
quantile_bins_robustness.py — §5.4 Pre-Registered Robustness Check

Repeats primary analyses using quantile-based (equal-count) bins
instead of equal-width bins, to verify results are not an artefact
of bin placement.

Approach:
  - For each model × dataset, compute 20 quantile bin edges from T=1.0
    NLP data (equal trial count per bin), held constant across temperatures
  - Re-run ROC construction + UVSD fitting with these bins
  - Compare d_a, c, AUC against equal-width results from full_results.json
  - Report max |Δd_a|, max |ΔAUC|, and whether conclusions change

Usage:
  python quantile_bins_robustness.py

Output:
  results/analysis/quantile_bins_robustness.json
"""

import json
import math
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
from scipy.stats import linregress, norm
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"C:\sdt_calibration")
N_BINS = 20
ALL_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
NLP_GIBBERISH_THRESHOLD = -10.0
ECE_BINS = 15
N_RESTARTS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paradigm_a_results(model: str, dataset: str) -> list:
    ds_name = "triviaqa" if dataset == "triviaqa" else "nq"
    path = BASE_DIR / "results" / "paradigm_a" / f"{model}_{ds_name}.jsonl"
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


# ---------------------------------------------------------------------------
# ROC construction (mirrors analysis_pipeline.py but with custom bin_edges)
# ---------------------------------------------------------------------------

def construct_roc(nlp_signal, nlp_noise, bin_edges):
    """Construct ROC from binned NLP values with Hautus correction."""
    signal_counts = np.histogram(nlp_signal, bins=bin_edges)[0]
    noise_counts = np.histogram(nlp_noise, bins=bin_edges)[0]

    # Hautus (1995) log-linear correction
    signal_counts_c = signal_counts.astype(float) + 0.5
    noise_counts_c = noise_counts.astype(float) + 0.5

    n_signal = np.sum(signal_counts_c)
    n_noise = np.sum(noise_counts_c)

    cum_signal = np.cumsum(signal_counts_c[::-1])[::-1]
    cum_noise = np.cumsum(noise_counts_c[::-1])[::-1]

    hit_rates = cum_signal[1:] / n_signal
    fa_rates = cum_noise[1:] / n_noise

    return {
        "hit_rates": hit_rates,
        "fa_rates": fa_rates,
        "bin_edges": bin_edges,
        "signal_counts": signal_counts,
        "noise_counts": noise_counts,
        "n_signal": int(np.sum(signal_counts)),
        "n_noise": int(np.sum(noise_counts)),
    }


# ---------------------------------------------------------------------------
# SDT fitting (vectorised with ndtr, mirrors analysis_pipeline.py)
# ---------------------------------------------------------------------------

def evsdt_nll(params, signal_counts, noise_counts):
    d_prime = params[0]
    criteria = np.sort(params[1:])
    cum_p_signal = np.concatenate([[1.0], 1.0 - ndtr(criteria - d_prime), [0.0]])
    cum_p_noise = np.concatenate([[1.0], 1.0 - ndtr(criteria), [0.0]])
    p_signal = np.clip(np.diff(-cum_p_signal), 1e-10, 1.0)
    p_noise = np.clip(np.diff(-cum_p_noise), 1e-10, 1.0)
    return -np.sum(signal_counts * np.log(p_signal)) - np.sum(noise_counts * np.log(p_noise))


def uvsdt_nll(params, signal_counts, noise_counts):
    d_a = params[0]
    s = params[1]
    criteria = np.sort(params[2:])
    sigma_signal = 1.0 / s if s > 0 else 1.0
    mu_signal = d_a * math.sqrt(2.0 / (1.0 + s * s))
    cum_p_signal = np.concatenate([[1.0], 1.0 - ndtr((criteria - mu_signal) / sigma_signal), [0.0]])
    cum_p_noise = np.concatenate([[1.0], 1.0 - ndtr(criteria), [0.0]])
    p_signal = np.clip(np.diff(-cum_p_signal), 1e-10, 1.0)
    p_noise = np.clip(np.diff(-cum_p_noise), 1e-10, 1.0)
    return -np.sum(signal_counts * np.log(p_signal)) - np.sum(noise_counts * np.log(p_noise))


def fit_sdt_models(roc, n_restarts=N_RESTARTS):
    """Fit EV and UV SDT models. Returns dict with parameters."""
    hit_rates = roc["hit_rates"]
    fa_rates = roc["fa_rates"]
    signal_counts = roc["signal_counts"].astype(float) + 0.5
    noise_counts = roc["noise_counts"].astype(float) + 0.5
    n_criteria = len(hit_rates)

    # AUC (non-parametric)
    fa = np.concatenate([[0.0], fa_rates, [1.0]])
    hr = np.concatenate([[0.0], hit_rates, [1.0]])
    sort_idx = np.argsort(fa)
    try:
        auc = float(np.trapezoid(hr[sort_idx], fa[sort_idx]))
    except AttributeError:
        auc = float(np.trapz(hr[sort_idx], fa[sort_idx]))

    # z-ROC initialisation
    z_hr = norm.ppf(np.clip(hit_rates, 0.001, 0.999))
    z_fa = norm.ppf(np.clip(fa_rates, 0.001, 0.999))
    try:
        slope, intercept, _, _, _ = linregress(z_fa, z_hr)
    except Exception:
        slope, intercept = 1.0, 1.0

    s_init = np.clip(slope, 0.3, 3.0)
    d_a_init = np.clip(np.sqrt(2.0 / (1.0 + s_init**2)) * intercept, 0.01, 5.0)

    # Initial criteria from z-ROC
    init_criteria = norm.ppf(np.clip(np.linspace(0.05, 0.95, n_criteria), 0.01, 0.99))

    # Fit EV
    def fit_ev(init_p):
        bounds = [(0, 5)] + [(None, None)] * n_criteria
        try:
            res = minimize(evsdt_nll, init_p, args=(signal_counts, noise_counts),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 5000, "ftol": 1e-8})
            return res
        except Exception:
            return None

    ev_init = np.concatenate([[d_a_init], init_criteria])
    best_ev = fit_ev(ev_init)
    for _ in range(n_restarts):
        perturbed = ev_init + np.random.randn(len(ev_init)) * 0.3
        perturbed[0] = np.clip(perturbed[0], 0.01, 4.99)
        r = fit_ev(perturbed)
        if r and (best_ev is None or r.fun < best_ev.fun):
            best_ev = r

    # Fit UV
    def fit_uv(init_p):
        bounds = [(0, 5), (0.3, 3.0)] + [(None, None)] * n_criteria
        try:
            res = minimize(uvsdt_nll, init_p, args=(signal_counts, noise_counts),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 5000, "ftol": 1e-8})
            return res
        except Exception:
            return None

    uv_init = np.concatenate([[d_a_init, s_init], init_criteria])
    best_uv = fit_uv(uv_init)
    for _ in range(n_restarts):
        perturbed = uv_init + np.random.randn(len(uv_init)) * 0.3
        perturbed[0] = np.clip(perturbed[0], 0.01, 4.99)
        perturbed[1] = np.clip(perturbed[1], 0.31, 2.99)
        r = fit_uv(perturbed)
        if r and (best_uv is None or r.fun < best_uv.fun):
            best_uv = r

    # Extract results
    ev_params = best_ev.x if best_ev else np.zeros(1 + n_criteria)
    uv_params = best_uv.x if best_uv else np.zeros(2 + n_criteria)

    d_prime_ev = float(ev_params[0])
    d_a_uv = float(uv_params[0])
    s_uv = float(uv_params[1])

    # Criterion c (mean of middle criteria, per Macmillan & Creelman)
    ev_criteria = np.sort(ev_params[1:])
    uv_criteria = np.sort(uv_params[2:])
    c_ev = float(np.mean(ev_criteria))
    c_uv = float(np.mean(uv_criteria))

    n_total = np.sum(signal_counts) + np.sum(noise_counts)
    ev_nll = float(best_ev.fun) if best_ev else float("inf")
    uv_nll = float(best_uv.fun) if best_uv else float("inf")
    n_ev_params = 1 + n_criteria
    n_uv_params = 2 + n_criteria

    return {
        "ev": {"d_prime": d_prime_ev, "c": c_ev, "nll": ev_nll,
               "aic": 2 * n_ev_params + 2 * ev_nll,
               "bic": n_ev_params * np.log(n_total) + 2 * ev_nll},
        "uv": {"d_a": d_a_uv, "s": s_uv, "c": c_uv, "nll": uv_nll,
               "aic": 2 * n_uv_params + 2 * uv_nll,
               "bic": n_uv_params * np.log(n_total) + 2 * uv_nll},
        "auc": auc,
        "converged": best_uv is not None and best_uv.success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models = ["llama3_instruct", "mistral_instruct", "llama3_base"]
    datasets = ["triviaqa", "nq"]

    # Load equal-width results for comparison
    ew_path = BASE_DIR / "results" / "analysis" / "full_results.json"
    with open(ew_path, "r", encoding="utf-8") as f:
        ew_results = json.load(f)

    quantile_results = {}
    comparisons = []

    for model in models:
        quantile_results[model] = {}
        for dataset in datasets:
            try:
                trials = load_paradigm_a_results(model, dataset)
            except FileNotFoundError:
                print(f"  Skipping {model} x {dataset} (no data)")
                continue

            print(f"\n{model} x {dataset}: {len(trials)} trials")

            by_temp = defaultdict(list)
            for t in trials:
                by_temp[t["temperature"]].append(t)

            # Compute QUANTILE bin edges from T=1.0 data
            t10_trials = by_temp.get(1.0, [])
            if not t10_trials:
                print("  No T=1.0 data, skipping")
                continue

            all_nlp_t10 = np.array([t["nlp"] for t in t10_trials])
            # Quantile edges: equal count per bin
            quantiles = np.linspace(0, 100, N_BINS + 1)
            q_bin_edges = np.percentile(all_nlp_t10, quantiles)
            # Ensure strictly increasing (add tiny eps for duplicates)
            for i in range(1, len(q_bin_edges)):
                if q_bin_edges[i] <= q_bin_edges[i - 1]:
                    q_bin_edges[i] = q_bin_edges[i - 1] + 1e-10
            # Extend edges slightly to capture all data
            q_bin_edges[0] -= 1e-6
            q_bin_edges[-1] += 1e-6

            print(f"  Quantile bin edges: [{q_bin_edges[0]:.3f} ... {q_bin_edges[-1]:.3f}]")

            results_by_temp = {}
            for temp in ALL_TEMPS:
                temp_trials = by_temp.get(temp, [])
                if not temp_trials:
                    continue

                nlp_vals = np.array([t["nlp"] for t in temp_trials])
                correct = np.array([t["correct"] for t in temp_trials])

                nlp_signal = nlp_vals[correct]
                nlp_noise = nlp_vals[~correct]

                if len(nlp_signal) < 10 or len(nlp_noise) < 10:
                    print(f"    T={temp}: too few signal/noise")
                    continue

                roc = construct_roc(nlp_signal, nlp_noise, q_bin_edges)
                sdt = fit_sdt_models(roc)

                temp_key = str(temp)
                results_by_temp[temp_key] = {
                    "d_a": sdt["uv"]["d_a"],
                    "s": sdt["uv"]["s"],
                    "c": sdt["uv"]["c"],
                    "d_prime_ev": sdt["ev"]["d_prime"],
                    "c_ev": sdt["ev"]["c"],
                    "auc": sdt["auc"],
                    "converged": sdt["converged"],
                    "ev_aic": sdt["ev"]["aic"],
                    "uv_aic": sdt["uv"]["aic"],
                    "n_signal": roc["n_signal"],
                    "n_noise": roc["n_noise"],
                }

                # Compare with equal-width
                # full_results.json nests paradigm A data under "paradigm_a"
                ew_pa = ew_results.get("paradigm_a", {})
                ew_model = ew_pa.get(model, {})
                ew_ds = ew_model.get(dataset, {})
                ew_temp = ew_ds.get(temp_key, {})

                if ew_temp:
                    ew_da = ew_temp.get("uv", {}).get("d_a")
                    ew_auc = ew_temp.get("auc")
                    ew_c = ew_temp.get("c")  # top-level c from fit_sdt_models

                    if ew_da is not None and ew_auc is not None:
                        delta_da = sdt["uv"]["d_a"] - ew_da
                        delta_auc = sdt["auc"] - ew_auc
                        delta_c = sdt["uv"]["c"] - ew_c if ew_c is not None else None

                        comparisons.append({
                            "model": model,
                            "dataset": dataset,
                            "temperature": temp,
                            "ew_d_a": ew_da,
                            "q_d_a": sdt["uv"]["d_a"],
                            "delta_d_a": delta_da,
                            "ew_auc": ew_auc,
                            "q_auc": sdt["auc"],
                            "delta_auc": delta_auc,
                            "ew_c": ew_c,
                            "q_c": sdt["uv"]["c"],
                            "delta_c": delta_c,
                        })

                        status = ""
                        if abs(delta_da) > 0.1:
                            status += " DA>0.1!"
                        if abs(delta_auc) > 0.02:
                            status += " AUC>0.02!"
                        print(f"    T={temp}: Δd_a={delta_da:+.4f}  ΔAUC={delta_auc:+.4f}{status}")
                    else:
                        print(f"    T={temp}: no EW comparison available")

            quantile_results[model][dataset] = results_by_temp

    # Summary
    print("\n" + "=" * 70)
    print("  QUANTILE BINS ROBUSTNESS SUMMARY")
    print("=" * 70)

    if comparisons:
        all_delta_da = [abs(c["delta_d_a"]) for c in comparisons]
        all_delta_auc = [abs(c["delta_auc"]) for c in comparisons]
        all_delta_c = [abs(c["delta_c"]) for c in comparisons if c["delta_c"] is not None]

        print(f"\n  N conditions compared: {len(comparisons)}")
        print(f"  |Δd_a|: max={max(all_delta_da):.4f}, mean={np.mean(all_delta_da):.4f}, median={np.median(all_delta_da):.4f}")
        print(f"  |ΔAUC|: max={max(all_delta_auc):.4f}, mean={np.mean(all_delta_auc):.4f}, median={np.median(all_delta_auc):.4f}")
        if all_delta_c:
            print(f"  |Δc|:   max={max(all_delta_c):.4f}, mean={np.mean(all_delta_c):.4f}, median={np.median(all_delta_c):.4f}")

        n_da_exceed = sum(1 for d in all_delta_da if d > 0.1)
        n_auc_exceed = sum(1 for d in all_delta_auc if d > 0.02)
        print(f"\n  Conditions with |Δd_a| > 0.1: {n_da_exceed}/{len(comparisons)}")
        print(f"  Conditions with |ΔAUC| > 0.02: {n_auc_exceed}/{len(comparisons)}")

        if n_da_exceed == 0 and n_auc_exceed == 0:
            print("\n  CONCLUSION: Results ROBUST to binning scheme.")
        else:
            print(f"\n  CONCLUSION: {n_da_exceed + n_auc_exceed} conditions show material sensitivity to binning.")

    # Save
    output = {
        "quantile_results": quantile_results,
        "comparisons": comparisons,
        "summary": {
            "n_conditions": len(comparisons),
            "max_abs_delta_d_a": max(all_delta_da) if comparisons else None,
            "mean_abs_delta_d_a": float(np.mean(all_delta_da)) if comparisons else None,
            "max_abs_delta_auc": max(all_delta_auc) if comparisons else None,
            "mean_abs_delta_auc": float(np.mean(all_delta_auc)) if comparisons else None,
            "n_da_exceed_0_1": n_da_exceed if comparisons else 0,
            "n_auc_exceed_0_02": n_auc_exceed if comparisons else 0,
        },
        "metadata": {
            "n_bins": N_BINS,
            "bin_type": "quantile (equal-count)",
            "reference": "equal-width bins from full_results.json",
            "threshold_d_a": 0.1,
            "threshold_auc": 0.02,
        }
    }

    out_path = BASE_DIR / "results" / "analysis" / "quantile_bins_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
