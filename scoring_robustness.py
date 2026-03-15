"""
scoring_robustness.py — Scoring Robustness Check per §A.7

Re-scores Paradigm A trials at string similarity thresholds {0.80, 0.85, 0.90}
using the best_similarity values already stored in the JSONL data.
Then re-runs ROC + SDT fitting at each threshold and compares d_a, c, AUC.

Pre-reg criterion (§A.7):
  "Report whether d_a, c, and AUC change materially
   (>±0.1 d' units or >±0.02 AUC) across thresholds."

Usage:
    python scoring_robustness.py
    python scoring_robustness.py --base-dir C:\\sdt_calibration
"""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Import from the main pipeline (same directory)
from analysis_pipeline import (
    construct_roc,
    fit_sdt_models,
    N_BINS,
    ALL_TEMPS,
)


THRESHOLDS = [0.80, 0.85, 0.90]
MATERIALITY_D_A = 0.1   # >±0.1 d' units
MATERIALITY_AUC = 0.02  # >±0.02 AUC


def load_paradigm_a_trials(model: str, dataset: str, base_dir: str) -> list:
    """Load raw Paradigm A JSONL trials."""
    path = Path(base_dir) / "results" / "paradigm_a" / f"{model}_{dataset}.jsonl"
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def rescore_trial(trial: dict, threshold: float) -> bool:
    """Re-score a trial at a given similarity threshold.

    Logic mirrors §A.7:
      - Refusals stay as refusals (correct=False)
      - Exact matches stay correct (match_type == "exact")
      - Similarity matches: correct if best_similarity >= threshold
      - Otherwise incorrect
    """
    match_type = trial.get("match_type", "incorrect")

    if match_type == "refusal":
        return False
    elif match_type == "exact":
        return True
    elif match_type == "similarity":
        # Was scored correct at 0.85; re-evaluate at new threshold
        return trial.get("best_similarity", 0.0) >= threshold
    elif match_type == "incorrect":
        # Was scored incorrect at 0.85; might be correct at lower threshold
        return trial.get("best_similarity", 0.0) >= threshold
    else:
        return False


def run_robustness_check(base_dir: str = r"C:\sdt_calibration"):
    """Run scoring robustness check across all thresholds."""
    output_dir = Path(base_dir) / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["llama3_instruct", "mistral_instruct", "llama3_base"]
    datasets = ["triviaqa", "nq"]

    results = {}

    print("=" * 70)
    print("Scoring Robustness Check (§A.7)")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Materiality: d_a > ±{MATERIALITY_D_A}, AUC > ±{MATERIALITY_AUC}")
    print("=" * 70)

    for model in models:
        results[model] = {}
        for dataset in datasets:
            try:
                trials = load_paradigm_a_trials(model, dataset, base_dir)
            except FileNotFoundError:
                print(f"\n  Skipping {model} × {dataset} (no data)")
                continue

            print(f"\n  {model} × {dataset}: {len(trials)} trials")

            # Group by temperature
            by_temp = defaultdict(list)
            for t in trials:
                by_temp[t["temperature"]].append(t)

            # Bin edges from T=1.0 at primary threshold (held constant)
            t10_trials = by_temp.get(1.0, [])
            if not t10_trials:
                continue
            all_nlp_t10 = np.array([t["nlp"] for t in t10_trials])
            bin_edges = np.linspace(
                np.min(all_nlp_t10), np.max(all_nlp_t10) + 1e-10, N_BINS + 1
            )

            results[model][dataset] = {}

            for temp in ALL_TEMPS:
                temp_trials = by_temp.get(temp, [])
                if not temp_trials:
                    continue

                threshold_results = {}
                for threshold in THRESHOLDS:
                    # Re-score all trials at this threshold
                    correct = np.array([
                        rescore_trial(t, threshold) for t in temp_trials
                    ])
                    nlp_vals = np.array([t["nlp"] for t in temp_trials])

                    nlp_signal = nlp_vals[correct]
                    nlp_noise = nlp_vals[~correct]

                    acc = float(np.mean(correct))

                    if len(nlp_signal) < 10 or len(nlp_noise) < 10:
                        threshold_results[str(threshold)] = {
                            "accuracy": acc,
                            "n_signal": int(np.sum(correct)),
                            "n_noise": int(np.sum(~correct)),
                            "error": "too few signal/noise",
                        }
                        continue

                    roc = construct_roc(nlp_signal, nlp_noise, bin_edges=bin_edges)
                    sdt = fit_sdt_models(roc)

                    threshold_results[str(threshold)] = {
                        "accuracy": acc,
                        "d_a": sdt["uv"].get("d_a"),
                        "c": sdt["c"],
                        "auc": sdt["auc"],
                        "s": sdt["uv"].get("s"),
                        "n_signal": int(np.sum(correct)),
                        "n_noise": int(np.sum(~correct)),
                        "uv_converged": sdt["uv"].get("converged", False),
                    }

                # Compute materiality (compare 0.80 and 0.90 against primary 0.85)
                primary = threshold_results.get("0.85", {})
                comparisons = {}
                for threshold in [0.80, 0.90]:
                    alt = threshold_results.get(str(threshold), {})
                    if "error" in primary or "error" in alt:
                        continue

                    d_a_diff = abs(
                        (alt.get("d_a") or 0) - (primary.get("d_a") or 0)
                    )
                    auc_diff = abs(
                        (alt.get("auc") or 0) - (primary.get("auc") or 0)
                    )
                    c_diff = abs(
                        (alt.get("c") or 0) - (primary.get("c") or 0)
                    )

                    comparisons[str(threshold)] = {
                        "d_a_diff": float(d_a_diff),
                        "auc_diff": float(auc_diff),
                        "c_diff": float(c_diff),
                        "d_a_material": bool(d_a_diff > MATERIALITY_D_A),
                        "auc_material": bool(auc_diff > MATERIALITY_AUC),
                    }

                results[model][dataset][str(temp)] = {
                    "thresholds": threshold_results,
                    "comparisons": comparisons,
                }

                # Print summary for T=1.0
                if temp == 1.0:
                    print(f"    T=1.0 scoring robustness:")
                    for thr in THRESHOLDS:
                        r = threshold_results.get(str(thr), {})
                        if "error" not in r:
                            d_a_str = f"{r['d_a']:.3f}" if r.get("d_a") is not None else "N/A"
                            print(f"      threshold={thr}: acc={r['accuracy']:.3f} "
                                  f"d_a={d_a_str} c={r.get('c', 0):.3f} "
                                  f"AUC={r.get('auc', 0):.3f} "
                                  f"(sig={r['n_signal']}, noi={r['n_noise']})")

                    for thr_key, comp in comparisons.items():
                        flag = " *** MATERIAL" if comp["d_a_material"] or comp["auc_material"] else ""
                        print(f"      vs 0.85→{thr_key}: "
                              f"Δd_a={comp['d_a_diff']:.4f} "
                              f"ΔAUC={comp['auc_diff']:.4f} "
                              f"Δc={comp['c_diff']:.4f}{flag}")

    # Summary: any material differences?
    print("\n" + "=" * 70)
    print("SUMMARY: Material differences across scoring thresholds")
    print("=" * 70)

    any_material = False
    for model in results:
        for dataset in results[model]:
            for temp_key, temp_data in results[model][dataset].items():
                for thr_key, comp in temp_data.get("comparisons", {}).items():
                    if comp.get("d_a_material") or comp.get("auc_material"):
                        any_material = True
                        print(f"  {model} × {dataset} × T={temp_key} × thr={thr_key}: "
                              f"Δd_a={comp['d_a_diff']:.4f} "
                              f"ΔAUC={comp['auc_diff']:.4f}")

    if not any_material:
        print("  No material differences found. Results robust to scoring threshold.")

    # Save
    output_file = output_dir / "scoring_robustness.json"
    output = {
        "results": results,
        "metadata": {
            "thresholds": THRESHOLDS,
            "materiality_d_a": MATERIALITY_D_A,
            "materiality_auc": MATERIALITY_AUC,
            "any_material_difference": any_material,
            "timestamp": datetime.now().isoformat(),
        },
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring robustness check (§A.7)")
    parser.add_argument("--base-dir", default=r"C:\sdt_calibration")
    args = parser.parse_args()
    run_robustness_check(args.base_dir)
