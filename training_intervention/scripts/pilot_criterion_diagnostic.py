#!/usr/bin/env python3
"""
pilot_criterion_diagnostic.py — Compute Type-1 criterion (c) for pilot models.

Diagnoses whether training shifted the decision criterion vs. improved
metacognitive sensitivity (meta-d'). Uses the same trial data as
pilot_recompute.py.

Type-1 SDT from the Type-2 count vectors:
  - nR_S1 = counts for S1 (incorrect) trials, bins low→high NLP
  - nR_S2 = counts for S2 (correct) trials, bins low→high NLP
  - Hit rate H = P("responded S2" | S2) = sum of right half of nR_S2 / sum(nR_S2)
  - FA rate F = P("responded S2" | S1) = sum of right half of nR_S1 / sum(nR_S1)
  - d' = z(H) - z(F)
  - c  = -0.5 * (z(H) + z(F))   [criterion: positive = conservative, negative = liberal]

Usage:
    python scripts/pilot_criterion_diagnostic.py \
        --trials-dir results/pilot \
        --baseline-csv results/pilot/baseline_llamacpp_science.csv \
        --domain Science
"""

import argparse
import json
import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def load_trials(trials_dir: str) -> dict:
    """Load all trial JSON files."""
    models = {}
    for fpath in sorted(glob.glob(os.path.join(trials_dir, "pilot_trials_*.json"))):
        name = os.path.basename(fpath).replace("pilot_trials_", "").replace(".json", "")
        with open(fpath) as f:
            data = json.load(f)
        # Normalise column names
        rows = []
        for t in data:
            rows.append({
                "question": t.get("question", ""),
                "correct": t.get("correct", t.get("is_correct", False)),
                "nlp": float(t.get("nlp", 0.0)),
                "response": t.get("model_answer", t.get("response", "")),
            })
        models[name] = pd.DataFrame(rows)
    return models


def load_baseline_csv(csv_path: str) -> pd.DataFrame:
    """Load Q5_K_M baseline CSV."""
    df = pd.read_csv(csv_path)
    # Normalise
    if "is_correct" in df.columns and "correct" not in df.columns:
        df["correct"] = df["is_correct"]
    return df


def compute_sdt_with_criterion(df: pd.DataFrame, n_ratings: int = 4) -> dict:
    """
    Compute Type-1 d', c, and Type-2 meta-d'/M-ratio from trial data.
    
    Returns dict with d_prime, criterion_c, meta_d_prime, m_ratio,
    plus NLP distribution stats.
    """
    from metadpy.mle import metad

    nlp = df["nlp"].values.astype(float)
    correct = df["correct"].values.astype(bool)
    n_total = len(nlp)
    n_correct = int(correct.sum())
    n_incorrect = n_total - n_correct

    if n_correct < 5 or n_incorrect < 5:
        return {"error": f"Too few trials (correct={n_correct}, incorrect={n_incorrect})"}

    # NLP distribution stats
    correct_nlps = nlp[correct]
    incorrect_nlps = nlp[~correct]
    nlp_stats = {
        "mean_nlp_correct": float(np.mean(correct_nlps)),
        "mean_nlp_incorrect": float(np.mean(incorrect_nlps)),
        "std_nlp_correct": float(np.std(correct_nlps)),
        "std_nlp_incorrect": float(np.std(incorrect_nlps)),
        "nlp_gap": float(np.mean(correct_nlps) - np.mean(incorrect_nlps)),
        "nlp_overall_mean": float(np.mean(nlp)),
        "nlp_overall_std": float(np.std(nlp)),
    }

    # --- BIN NLP (same as pilot_recompute.py) ---
    total_bins = 2 * n_ratings
    try:
        bin_labels = pd.qcut(nlp, q=total_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        ranks = pd.Series(nlp).rank(method="first")
        bin_labels = pd.cut(ranks, bins=total_bins, labels=False) + 1
        bin_labels = bin_labels.values

    if hasattr(bin_labels, "values"):
        bin_labels = bin_labels.values
    bin_labels = np.array(bin_labels, dtype=int)
    actual_n_bins = int(np.max(bin_labels))

    if actual_n_bins < total_bins:
        effective_n_ratings = max(actual_n_bins // 2, 2)
        total_bins = 2 * effective_n_ratings
        try:
            bin_labels = pd.qcut(nlp, q=total_bins, labels=False, duplicates="drop") + 1
        except ValueError:
            ranks = pd.Series(nlp).rank(method="first")
            bin_labels = pd.cut(ranks, bins=total_bins, labels=False) + 1
            bin_labels = bin_labels.values
        if hasattr(bin_labels, "values"):
            bin_labels = bin_labels.values
        bin_labels = np.array(bin_labels, dtype=int)
    else:
        effective_n_ratings = n_ratings

    # --- BUILD COUNT VECTORS ---
    n_vec = 2 * effective_n_ratings
    nR_S1 = np.zeros(n_vec, dtype=float)
    nR_S2 = np.zeros(n_vec, dtype=float)

    for i in range(n_vec):
        bin_mask = bin_labels == (i + 1)
        nR_S1[i] = float(np.sum(bin_mask & ~correct))
        nR_S2[i] = float(np.sum(bin_mask & correct))

    # Hautus log-linear correction
    nR_S1_adj = nR_S1 + 0.5
    nR_S2_adj = nR_S2 + 0.5

    # --- TYPE-1 HIT/FA and CRITERION ---
    # Right half = "responded S2" (high confidence / high NLP)
    # Hit = P(high NLP | correct), FA = P(high NLP | incorrect)
    half = effective_n_ratings
    H = np.sum(nR_S2_adj[half:]) / np.sum(nR_S2_adj)
    F = np.sum(nR_S1_adj[half:]) / np.sum(nR_S1_adj)

    # Clip to avoid inf
    H = np.clip(H, 0.001, 0.999)
    F = np.clip(F, 0.001, 0.999)

    z_H = stats.norm.ppf(H)
    z_F = stats.norm.ppf(F)

    d_prime_manual = z_H - z_F
    criterion_c = -0.5 * (z_H + z_F)

    # --- META-D' via metadpy ---
    try:
        result = metad(nR_S1=nR_S1_adj, nR_S2=nR_S2_adj,
                       nRatings=effective_n_ratings, padding=False)
        d_prime = float(result["dprime"].iloc[0])
        meta_d_prime = float(result["meta_d"].iloc[0])
        m_ratio = float(result["m_ratio"].iloc[0])
    except Exception as e:
        d_prime = d_prime_manual
        meta_d_prime = float("nan")
        m_ratio = float("nan")

    return {
        "d_prime": d_prime,
        "criterion_c": float(criterion_c),
        "meta_d_prime": meta_d_prime,
        "m_ratio": m_ratio,
        "m_diff": meta_d_prime - d_prime if not np.isnan(meta_d_prime) else float("nan"),
        "accuracy": n_correct / n_total,
        "n_trials": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "hit_rate": float(H),
        "fa_rate": float(F),
        **nlp_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-dir", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--domain", default="Science")
    args = parser.parse_args()

    print("=" * 70)
    print("PILOT CRITERION DIAGNOSTIC")
    print("=" * 70)

    # Load baseline
    baseline_df = load_baseline_csv(args.baseline_csv)
    baseline_sdt = compute_sdt_with_criterion(baseline_df)

    print(f"\n  Q5_K_M BASELINE:")
    print(f"    d'={baseline_sdt['d_prime']:.3f}, c={baseline_sdt['criterion_c']:.3f}, "
          f"meta-d'={baseline_sdt['meta_d_prime']:.3f}, M-ratio={baseline_sdt['m_ratio']:.3f}")
    print(f"    H={baseline_sdt['hit_rate']:.3f}, FA={baseline_sdt['fa_rate']:.3f}, "
          f"acc={baseline_sdt['accuracy']:.3f}")
    print(f"    NLP gap={baseline_sdt['nlp_gap']:.4f}, "
          f"correct={baseline_sdt['mean_nlp_correct']:.4f}±{baseline_sdt['std_nlp_correct']:.4f}, "
          f"incorrect={baseline_sdt['mean_nlp_incorrect']:.4f}±{baseline_sdt['std_nlp_incorrect']:.4f}")

    # Load all models
    models = load_trials(args.trials_dir)

    # Compute SDT for each
    all_results = {}
    for name, df in sorted(models.items()):
        sdt = compute_sdt_with_criterion(df)
        all_results[name] = sdt

    # Print f16 baseline separately
    if "baseline_unadapted_gguf" in all_results:
        f16 = all_results["baseline_unadapted_gguf"]
        print(f"\n  F16 BASELINE (unadapted):")
        print(f"    d'={f16['d_prime']:.3f}, c={f16['criterion_c']:.3f}, "
              f"meta-d'={f16['meta_d_prime']:.3f}, M-ratio={f16['m_ratio']:.3f}")
        print(f"    H={f16['hit_rate']:.3f}, FA={f16['fa_rate']:.3f}, "
              f"acc={f16['accuracy']:.3f}")
        print(f"    NLP gap={f16['nlp_gap']:.4f}, "
              f"correct={f16['mean_nlp_correct']:.4f}±{f16['std_nlp_correct']:.4f}, "
              f"incorrect={f16['mean_nlp_incorrect']:.4f}±{f16['std_nlp_incorrect']:.4f}")

    # Summary table — GGUF models only (apples-to-apples)
    print(f"\n{'=' * 70}")
    print(f"GGUF MODELS vs F16 BASELINE (criterion diagnostic)")
    print(f"{'=' * 70}")
    
    f16_baseline = all_results.get("baseline_unadapted_gguf", baseline_sdt)
    f16_c = f16_baseline["criterion_c"]
    f16_d = f16_baseline["d_prime"]
    f16_md = f16_baseline["meta_d_prime"]
    f16_m = f16_baseline["m_ratio"]

    header = (f"{'Model':<30} {'M-ratio':>7} {'ΔM':>7} {'d`':>6} {'Δd`':>6} "
              f"{'meta-d`':>7} {'c':>6} {'Δc':>7} {'Acc':>5} {'NLP gap':>8}")
    print(header)
    print("-" * len(header))

    # Print f16 baseline row
    print(f"{'F16 BASELINE':<30} {f16_m:>7.3f} {'---':>7} {f16_d:>6.3f} {'---':>6} "
          f"{f16_md:>7.3f} {f16_c:>6.3f} {'---':>7} {f16_baseline['accuracy']:>5.3f} "
          f"{f16_baseline['nlp_gap']:>8.4f}")

    # Print GGUF adapted models
    gguf_models = sorted([k for k in all_results if k.endswith("_gguf") and k != "baseline_unadapted_gguf"])
    for name in gguf_models:
        sdt = all_results[name]
        delta_m = sdt["m_ratio"] - f16_m
        delta_d = sdt["d_prime"] - f16_d
        delta_c = sdt["criterion_c"] - f16_c
        print(f"{name:<30} {sdt['m_ratio']:>7.3f} {delta_m:>+7.3f} {sdt['d_prime']:>6.3f} "
              f"{delta_d:>+6.3f} {sdt['meta_d_prime']:>7.3f} {sdt['criterion_c']:>6.3f} "
              f"{delta_c:>+7.3f} {sdt['accuracy']:>5.3f} {sdt['nlp_gap']:>8.4f}")

    # Print HF models for comparison
    print(f"\n{'=' * 70}")
    print(f"HF MODELS (cross-backend reference only)")
    print(f"{'=' * 70}")
    print(header)
    print("-" * len(header))

    hf_models = sorted([k for k in all_results if not k.endswith("_gguf")])
    for name in hf_models:
        sdt = all_results[name]
        delta_m = sdt["m_ratio"] - baseline_sdt["m_ratio"]
        delta_d = sdt["d_prime"] - baseline_sdt["d_prime"]
        delta_c = sdt["criterion_c"] - baseline_sdt["criterion_c"]
        print(f"{name:<30} {sdt['m_ratio']:>7.3f} {delta_m:>+7.3f} {sdt['d_prime']:>6.3f} "
              f"{delta_d:>+6.3f} {sdt['meta_d_prime']:>7.3f} {sdt['criterion_c']:>6.3f} "
              f"{delta_c:>+7.3f} {sdt['accuracy']:>5.3f} {sdt['nlp_gap']:>8.4f}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print(f"  Positive c = conservative (lower overall confidence)")
    print(f"  Negative c = liberal (higher overall confidence)")
    print(f"  Large |Δc| with small |Δmeta-d'| = criterion shift, not metacognitive gain")
    print(f"  Large |Δmeta-d'| with small |Δc| = genuine metacognitive change")

    # Save results
    output_path = os.path.join(args.trials_dir, "pilot_criterion_diagnostic.json")
    save_data = {
        "q5km_baseline": baseline_sdt,
        "f16_baseline": f16_baseline,
        "models": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
