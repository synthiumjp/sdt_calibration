"""
secondary_analyses.py — Secondary Hypotheses and Exploratory Analyses

Extracts and tests H4, H5, H6, E1, E5 from existing full_results.json.
No new inference needed — all data was computed in the main pipeline.

Pre-registration references:
  H4 (§4.2): Unequal variance — z-ROC slope < 1.0
  H5 (§4.2): Domain-specific sensitivity — d_a varies across domains
  H6 (§4.2): High-T degradation — d_a(generation) drops at T>1.0 vs Analysis A
  E1 (§4.3): Instruction-tuning as criterion shift — base vs instruct
  E5 (§4.3): z-ROC slope × temperature — slope changes with T

Usage:
    python secondary_analyses.py
    python secondary_analyses.py --base-dir C:\\sdt_calibration
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(base_dir: str) -> dict:
    path = Path(base_dir) / "results" / "analysis" / "full_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_secondary_analyses(base_dir: str = r"C:\sdt_calibration"):
    results = load_results(base_dir)
    pa = results.get("paradigm_a", {})
    aa = results.get("analysis_a", {})
    domain = results.get("domain_level", {})

    output = {}

    print("=" * 70)
    print("Secondary Hypotheses and Exploratory Analyses")
    print("=" * 70)

    # ===================================================================
    # H4: Unequal Variance (§4.2)
    # z-ROC slope significantly < 1.0
    # Prediction: slope ≈ 0.80 (as in recognition memory)
    # ===================================================================
    print("\n--- H4: Unequal Variance (z-ROC slope) ---")

    h4_results = {}
    for model in pa:
        h4_results[model] = {}
        for dataset in pa[model]:
            slopes_by_temp = {}
            s_values_by_temp = {}
            for temp_key, r in pa[model][dataset].items():
                z_roc = r.get("z_roc", {})
                uv = r.get("uv", {})
                slope = z_roc.get("slope")
                s_val = uv.get("s")
                r_sq = z_roc.get("r_squared")
                if slope is not None:
                    slopes_by_temp[temp_key] = {
                        "slope": slope,
                        "s": s_val,
                        "r_squared": r_sq,
                    }

            # Test at T=1.0 (primary)
            t10 = slopes_by_temp.get("1.0", {})
            if t10:
                slope = t10["slope"]
                s_val = t10.get("s")
                # One-sample t-test: slope < 1.0?
                # We only have a point estimate, not a distribution.
                # Report the slope and s value; bootstrap CIs will provide the interval.
                h4_results[model][dataset] = {
                    "t1.0_slope": slope,
                    "t1.0_s": s_val,
                    "t1.0_r_squared": t10.get("r_squared"),
                    "slope_below_1": slope < 1.0 if slope is not None else None,
                    "all_temps": slopes_by_temp,
                }

                # EV vs UV model comparison at T=1.0
                comparison = pa[model][dataset].get("1.0", {}).get("comparison", {})
                if comparison:
                    h4_results[model][dataset]["ev_vs_uv"] = comparison

                print(f"  {model} × {dataset} (T=1.0): "
                      f"z-ROC slope={slope:.3f}, s={s_val:.3f}, "
                      f"R²={t10.get('r_squared', 0):.3f}, "
                      f"preferred={comparison.get('preferred', 'N/A')}")

                # Mean slope across all temps
                all_slopes = [v["slope"] for v in slopes_by_temp.values()
                              if v.get("slope") is not None]
                if all_slopes:
                    mean_slope = np.mean(all_slopes)
                    print(f"    Mean slope across temps: {mean_slope:.3f} "
                          f"(range: {min(all_slopes):.3f}–{max(all_slopes):.3f})")

    output["h4"] = h4_results

    # ===================================================================
    # H5: Domain-Specific Sensitivity (§4.2)
    # d_a varies across knowledge domains
    # ===================================================================
    print("\n--- H5: Domain-Specific Sensitivity ---")

    h5_results = {}
    pa_domain = domain.get("paradigm_a", {})

    for model in pa_domain:
        domains_data = pa_domain[model]
        if not domains_data:
            continue

        domain_names = sorted(domains_data.keys())
        d_a_values = []
        domain_labels = []
        for d in domain_names:
            d_a = domains_data[d].get("d_a")
            if d_a is not None:
                d_a_values.append(d_a)
                domain_labels.append(d)

        if len(d_a_values) < 3:
            print(f"  {model}: too few domains ({len(d_a_values)})")
            continue

        d_a_arr = np.array(d_a_values)
        d_a_range = float(np.max(d_a_arr) - np.min(d_a_arr))
        d_a_cv = float(np.std(d_a_arr) / np.mean(d_a_arr))

        # Kruskal-Wallis would need per-trial data; with aggregate d_a
        # we report the range and CV as descriptive measures
        h5_results[model] = {
            "domains": {d: domains_data[d].get("d_a") for d in domain_labels},
            "d_a_range": d_a_range,
            "d_a_cv": d_a_cv,
            "d_a_mean": float(np.mean(d_a_arr)),
            "d_a_std": float(np.std(d_a_arr)),
            "n_domains": len(d_a_values),
        }

        print(f"  {model}: {len(d_a_values)} domains, "
              f"d_a range={d_a_range:.3f}, CV={d_a_cv:.3f}")
        for d in domain_labels:
            print(f"    {d}: d_a={domains_data[d].get('d_a'):.3f}, "
                  f"acc={domains_data[d].get('accuracy', 0):.3f}")

    output["h5"] = h5_results

    # ===================================================================
    # H6: High-Temperature Sensitivity Degradation (§4.2)
    # d_a from generation (Analysis B = Paradigm A) decreases at T>1.0
    # d_a from force-decode (Analysis A) stays constant
    # ===================================================================
    print("\n--- H6: High-Temperature Sensitivity Degradation ---")

    h6_results = {}
    for model in pa:
        h6_results[model] = {}
        for dataset in pa[model]:
            temps_data = pa[model][dataset]

            # Analysis B: d_a from Paradigm A at each temperature
            d_a_by_temp = {}
            for temp_key, r in temps_data.items():
                d_a = r.get("uv", {}).get("d_a")
                if d_a is not None:
                    d_a_by_temp[float(temp_key)] = d_a

            # Analysis A: force-decode d_a (temperature-invariant)
            aa_data = aa.get(model, {}).get(dataset, {})
            d_a_force_decode = aa_data.get("uv", {}).get("d_a")

            # Compare T=1.0 vs T=1.5 and T=2.0
            d_a_t10 = d_a_by_temp.get(1.0)
            d_a_t15 = d_a_by_temp.get(1.5)
            d_a_t20 = d_a_by_temp.get(2.0)

            h6_result = {
                "d_a_by_temp": {str(k): v for k, v in sorted(d_a_by_temp.items())},
                "d_a_force_decode": d_a_force_decode,
                "d_a_t1.0": d_a_t10,
                "d_a_t1.5": d_a_t15,
                "d_a_t2.0": d_a_t20,
            }

            # H6 prediction: d_a(gen) decreases at high T
            if d_a_t10 is not None and d_a_t15 is not None and d_a_t20 is not None:
                # Does d_a decrease from T=1.0 to T>1.0?
                h6_degrades = d_a_t15 < d_a_t10 and d_a_t20 < d_a_t10
                h6_result["degrades_at_high_t"] = h6_degrades

                # Spearman correlation of d_a with T for high temps only
                high_temps = [t for t in sorted(d_a_by_temp.keys()) if t >= 1.0]
                high_d_a = [d_a_by_temp[t] for t in high_temps]
                if len(high_temps) >= 3:
                    rho, p = stats.spearmanr(high_temps, high_d_a)
                    h6_result["high_t_spearman_rho"] = float(rho)
                    h6_result["high_t_spearman_p"] = float(p)
                    h6_result["high_t_trend"] = "decreasing" if rho < 0 else "increasing"

            h6_results[model][dataset] = h6_result

            fd_str = f"{d_a_force_decode:.3f}" if d_a_force_decode is not None else "N/A"
            print(f"  {model} × {dataset}:")
            print(f"    Analysis A (force-decode): d_a={fd_str}")
            print(f"    Analysis B (generation):  ", end="")
            for t in sorted(d_a_by_temp.keys()):
                marker = " *" if t > 1.0 else ""
                print(f"T={t}:{d_a_by_temp[t]:.3f}{marker}  ", end="")
            print()
            if "degrades_at_high_t" in h6_result:
                print(f"    H6 degrades at high T: {h6_result['degrades_at_high_t']}")
            if "high_t_trend" in h6_result:
                print(f"    High-T trend: {h6_result['high_t_trend']} "
                      f"(ρ={h6_result['high_t_spearman_rho']:.3f}, "
                      f"p={h6_result['high_t_spearman_p']:.3f})")

    output["h6"] = h6_results

    # ===================================================================
    # E1: Instruction-Tuning as Criterion Shift (§4.3)
    # Compare base vs instruct: d_a(base) ≈ d_a(instruct), c differs
    # ===================================================================
    print("\n--- E1: Instruction-Tuning as Criterion Shift ---")

    e1_results = {}
    for dataset in ["triviaqa", "nq"]:
        instruct_t10 = pa.get("llama3_instruct", {}).get(dataset, {}).get("1.0", {})
        base_t10 = pa.get("llama3_base", {}).get(dataset, {}).get("1.0", {})

        if not instruct_t10 or not base_t10:
            continue

        d_a_instruct = instruct_t10.get("uv", {}).get("d_a")
        d_a_base = base_t10.get("uv", {}).get("d_a")
        c_instruct = instruct_t10.get("c")
        c_base = base_t10.get("c")
        auc_instruct = instruct_t10.get("auc")
        auc_base = base_t10.get("auc")
        acc_instruct = instruct_t10.get("accuracy")
        acc_base = base_t10.get("accuracy")

        d_a_diff = abs(d_a_instruct - d_a_base) if d_a_instruct and d_a_base else None
        c_diff = abs(c_instruct - c_base) if c_instruct and c_base else None

        e1_result = {
            "instruct": {"d_a": d_a_instruct, "c": c_instruct,
                         "auc": auc_instruct, "accuracy": acc_instruct},
            "base": {"d_a": d_a_base, "c": c_base,
                     "auc": auc_base, "accuracy": acc_base},
            "d_a_difference": d_a_diff,
            "c_difference": c_diff,
            # E1 supported if d_a similar but c different
            "d_a_similar": d_a_diff < 0.2 if d_a_diff is not None else None,
            "c_different": c_diff > 0.2 if c_diff is not None else None,
            "e1_pattern": (d_a_diff < 0.2 and c_diff > 0.2)
                if d_a_diff is not None and c_diff is not None else None,
        }
        e1_results[dataset] = e1_result

        print(f"  {dataset} (T=1.0):")
        print(f"    Instruct: d_a={d_a_instruct:.3f}, c={c_instruct:.3f}, "
              f"AUC={auc_instruct:.3f}, acc={acc_instruct:.3f}")
        print(f"    Base:     d_a={d_a_base:.3f}, c={c_base:.3f}, "
              f"AUC={auc_base:.3f}, acc={acc_base:.3f}")
        print(f"    Δd_a={d_a_diff:.3f}, Δc={c_diff:.3f}")
        print(f"    E1 pattern (d_a similar, c different): {e1_result['e1_pattern']}")

    output["e1"] = e1_results

    # ===================================================================
    # E5: z-ROC Slope × Temperature (§4.3)
    # Does the z-ROC slope change with temperature?
    # If slope varies: T affects evidence distribution shape, not just criterion
    # ===================================================================
    print("\n--- E5: z-ROC Slope × Temperature ---")

    e5_results = {}
    for model in pa:
        e5_results[model] = {}
        for dataset in pa[model]:
            temps = []
            slopes = []
            s_values = []

            for temp_key, r in sorted(pa[model][dataset].items(),
                                       key=lambda x: float(x[0])):
                z_roc = r.get("z_roc", {})
                uv = r.get("uv", {})
                slope = z_roc.get("slope")
                s_val = uv.get("s")
                if slope is not None:
                    temps.append(float(temp_key))
                    slopes.append(slope)
                    s_values.append(s_val)

            if len(temps) < 4:
                continue

            temps_arr = np.array(temps)
            slopes_arr = np.array(slopes)
            s_arr = np.array(s_values)

            # Spearman correlation: slope × temperature
            rho_slope, p_slope = stats.spearmanr(temps_arr, slopes_arr)
            rho_s, p_s = stats.spearmanr(temps_arr, s_arr)

            # Moderate temps only (T ≤ 1.0) for comparison with H1 range
            mod_mask = temps_arr <= 1.0
            if np.sum(mod_mask) >= 3:
                rho_mod, p_mod = stats.spearmanr(temps_arr[mod_mask], slopes_arr[mod_mask])
            else:
                rho_mod, p_mod = float("nan"), float("nan")

            e5_results[model][dataset] = {
                "temps": temps,
                "slopes": slopes,
                "s_values": s_values,
                "spearman_rho_slope_T": float(rho_slope),
                "spearman_p_slope_T": float(p_slope),
                "spearman_rho_s_T": float(rho_s),
                "spearman_p_s_T": float(p_s),
                "moderate_rho": float(rho_mod),
                "moderate_p": float(p_mod),
                "slope_range": float(max(slopes) - min(slopes)),
                "slope_changes_with_T": bool(p_slope < 0.05),
            }

            print(f"  {model} × {dataset}:")
            print(f"    Slopes: ", end="")
            for t, sl in zip(temps, slopes):
                print(f"T={t}:{sl:.3f}  ", end="")
            print()
            print(f"    ρ(slope,T)={rho_slope:.3f} (p={p_slope:.4f}), "
                  f"ρ(s,T)={rho_s:.3f} (p={p_s:.4f})")
            print(f"    Moderate only: ρ={rho_mod:.3f} (p={p_mod:.4f})")
            print(f"    Slope changes with T: {p_slope < 0.05}")

    output["e5"] = e5_results

    # ===================================================================
    # Save
    # ===================================================================
    output_dir = Path(base_dir) / "results" / "analysis"
    output["metadata"] = {"timestamp": datetime.now().isoformat()}

    out_file = output_dir / "secondary_analyses.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secondary hypotheses and exploratory analyses")
    parser.add_argument("--base-dir", default=r"C:\sdt_calibration")
    args = parser.parse_args()
    run_secondary_analyses(args.base_dir)
