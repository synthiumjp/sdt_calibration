#!/usr/bin/env python3
"""
pilot_recompute.py (v2) — Recompute M-ratio from saved pilot trial data.

Changes from v1:
  - Baseline M-ratio computed from C₁ llama-cpp inference CSV (same questions,
    same pipeline as M1) rather than M1 Table 6 values. This avoids the NLP
    scale mismatch between Set A and C₁ question sets (see session log 2 Apr).
  - M1 Table 6 values retained for reference but not used in decision gate.

Fixes from pilot_evaluate.py:
  1. arviz 1.0 broke metadpy import — pin arviz<1.0
  2. NLP binning: pd.qcut into 2*nRatings=8 equal-count bins (was np.digitize
     with quartile edges → empty bins 2-3).

No GPU needed. Reads trial JSON + baseline CSV, computes metadpy M-ratio,
applies the pilot decision gate (ΔM > 0.10, Δd′ < 0.05).

Usage:
    python pilot_recompute.py \
        --trials-dir results/pilot \
        --baseline-csv results/pilot/baseline_llamacpp_science.csv \
        --domain Science

Requires: metadpy, arviz<1.0, numpy, pandas
"""

import argparse
import json
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from metadpy.mle import metad  # MLE fit, returns DataFrame


# M1 Table 6 values (arXiv 2603.25112) — for reference only, not used in gate
M1_TABLE6 = {
    "Science & Technology": 0.788,
    "Science":              0.788,
    "Arts & Literature":    0.925,
    "Arts":                 0.925,
    "Geography":            1.459,
    "History":              1.096,
}

N_RATINGS = 4  # confidence levels per side → 2*4 = 8 total bins


def load_trial_data(trials_dir: str) -> dict[str, pd.DataFrame]:
    """Load all pilot_trials_*.json files → dict of model_name → DataFrame."""
    pattern = os.path.join(trials_dir, "pilot_trials_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No trial files found matching {pattern}")
        sys.exit(1)

    models = {}
    for fpath in files:
        model_name = Path(fpath).stem.replace("pilot_trials_", "")
        with open(fpath, "r", encoding="utf-8") as f:
            trials = json.load(f)
        df = pd.DataFrame(trials)

        # Normalise column names
        if "correct" in df.columns and "is_correct" not in df.columns:
            df["is_correct"] = df["correct"]

        required = {"nlp", "is_correct"}
        missing = required - set(df.columns)
        if missing:
            print(f"  WARNING: {fpath} missing columns {missing}, skipping")
            continue

        models[model_name] = df
        print(f"  Loaded {model_name}: {len(df)} trials")

    return models


def load_baseline_csv(csv_path: str) -> pd.DataFrame:
    """Load C₁ llama-cpp baseline CSV and normalise columns."""
    df = pd.read_csv(csv_path)

    # Normalise column names
    if "correct" in df.columns and "is_correct" not in df.columns:
        df["is_correct"] = df["correct"]
    if "is_correct" not in df.columns:
        raise ValueError(f"Baseline CSV missing 'correct' or 'is_correct' column. "
                         f"Found: {list(df.columns)}")

    print(f"  Loaded baseline CSV: {csv_path}")
    print(f"    N={len(df)}, acc={df.is_correct.mean():.3f}, "
          f"mean NLP={df.nlp.mean():.4f}, std={df.nlp.std():.4f}")

    c = df[df.is_correct == True]["nlp"]
    ic = df[df.is_correct == False]["nlp"]
    print(f"    NLP gap (correct - incorrect): {c.mean() - ic.mean():.4f}")

    return df


def compute_type2_sdt(df: pd.DataFrame, n_ratings: int = 4,
                      domain: str = None) -> dict:
    """
    Compute Type-2 SDT metrics from trial data.

    Binning fix: uses pd.qcut to create 2*n_ratings equal-count NLP bins,
    producing fully-populated nR_S1/nR_S2 count vectors.

    metadpy convention for nR_S1/nR_S2 (2*nRatings entries each):
      Left half (indices 0..nRatings-1):  "responded S1" with confidence high→low
      Right half (indices nRatings..2n-1): "responded S2" with confidence low→high

    For LLMs: S1=incorrect, S2=correct; low NLP → "responded S1",
    high NLP → "responded S2". Bins ordered low→high NLP map
    directly to indices 0→(2*nRatings-1).
    """
    if domain and "domain" in df.columns:
        df = df[df["domain"] == domain].copy()

    if len(df) == 0:
        return {"error": "No trials", "n_trials": 0}

    nlp = df["nlp"].values.astype(float)
    correct = df["is_correct"].values.astype(bool)
    n_total = len(nlp)
    n_correct = int(correct.sum())
    n_incorrect = n_total - n_correct

    if n_correct < 5 or n_incorrect < 5:
        return {"error": f"Too few trials (correct={n_correct}, incorrect={n_incorrect})",
                "n_trials": n_total, "n_correct": n_correct, "n_incorrect": n_incorrect}

    # --- BIN NLP ---
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

    # Adjust nRatings if ties collapsed some bins
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
        print(f"  Adjusted nRatings: {n_ratings} → {effective_n_ratings} (ties in NLP)")
    else:
        effective_n_ratings = n_ratings

    # --- BUILD COUNT VECTORS ---
    n_vec = 2 * effective_n_ratings
    nR_S1 = np.zeros(n_vec, dtype=float)  # incorrect trials
    nR_S2 = np.zeros(n_vec, dtype=float)  # correct trials

    for i in range(n_vec):
        bin_mask = bin_labels == (i + 1)
        nR_S1[i] = float(np.sum(bin_mask & ~correct))
        nR_S2[i] = float(np.sum(bin_mask & correct))

    # Hautus (2005) log-linear correction
    nR_S1_adj = nR_S1 + 0.5
    nR_S2_adj = nR_S2 + 0.5

    # --- FIT META-D' via MLE ---
    try:
        result = metad(nR_S1=nR_S1_adj, nR_S2=nR_S2_adj,
                       nRatings=effective_n_ratings, padding=False)
        d_prime = float(result["dprime"].iloc[0])
        meta_d_prime = float(result["meta_d"].iloc[0])
        m_ratio = float(result["m_ratio"].iloc[0])
    except Exception as e:
        return {"error": str(e), "n_trials": n_total,
                "nR_S1_raw": nR_S1.tolist(), "nR_S2_raw": nR_S2.tolist()}

    return {
        "d_prime": d_prime,
        "meta_d_prime": meta_d_prime,
        "m_ratio": m_ratio,
        "n_trials": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": n_correct / n_total,
        "effective_n_ratings": effective_n_ratings,
        "nR_S1_raw": nR_S1.tolist(),
        "nR_S2_raw": nR_S2.tolist(),
        "nR_S1_adj": nR_S1_adj.tolist(),
        "nR_S2_adj": nR_S2_adj.tolist(),
    }


def apply_decision_gate(results: dict, baseline_m: float,
                        baseline_d: float = None) -> dict:
    """
    Pilot decision gate (v1.2 §11):
      PASS:      ΔM > 0.10 AND Δd′ < 0.05
      GREY_ZONE: ΔM ∈ [0.05, 0.10)
      FAIL:      ΔM < 0.05 or wrong direction
    """
    if "error" in results:
        return {"decision": "ERROR", "reason": results["error"]}

    m = results["m_ratio"]
    d = results["d_prime"]
    delta_m = m - baseline_m
    delta_d = abs(d - baseline_d) if baseline_d is not None else None

    gate = {"baseline_m": baseline_m, "model_m": m, "delta_m": delta_m,
            "baseline_d_prime": baseline_d, "model_d_prime": d,
            "delta_d_prime": delta_d}

    if delta_m > 0.10:
        if delta_d is not None and delta_d > 0.05:
            gate["decision"] = "PASS_BUT_D_SHIFT"
            gate["reason"] = (f"ΔM={delta_m:+.3f} > 0.10 but "
                              f"Δd'={delta_d:.3f} > 0.05 — check d' preservation")
        else:
            gate["decision"] = "PASS"
            d_str = f", Δd'={delta_d:.3f}" if delta_d is not None else ""
            gate["reason"] = f"ΔM={delta_m:+.3f} > 0.10{d_str}"
    elif delta_m > 0.05:
        gate["decision"] = "GREY_ZONE"
        gate["reason"] = f"ΔM={delta_m:+.3f} in [0.05, 0.10)"
    elif delta_m > 0:
        gate["decision"] = "FAIL_WEAK"
        gate["reason"] = f"ΔM={delta_m:+.3f} right direction but < 0.05"
    else:
        gate["decision"] = "FAIL"
        gate["reason"] = f"ΔM={delta_m:+.3f} no improvement"

    return gate


def main():
    parser = argparse.ArgumentParser(
        description="Recompute pilot M-ratio from saved trial data (v2: C₁ baseline)")
    parser.add_argument("--trials-dir", default="results/pilot",
                        help="Directory with pilot_trials_*.json")
    parser.add_argument("--baseline-csv", default=None,
                        help="C₁ llama-cpp baseline CSV (default: auto-detect in trials-dir)")
    parser.add_argument("--domain", default="Science",
                        help="Domain to evaluate (default: Science)")
    parser.add_argument("--n-ratings", type=int, default=4,
                        help="nRatings per side (default: 4 → 8 bins)")
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: <trials-dir>/pilot_recompute.json)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.trials_dir, "pilot_recompute.json")

    # Auto-detect baseline CSV
    if args.baseline_csv is None:
        candidates = [
            os.path.join(args.trials_dir, "baseline_llamacpp_science.csv"),
            os.path.join(args.trials_dir, "baseline_llamacpp_science_64tok.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.baseline_csv = c
                break
        if args.baseline_csv is None:
            print("ERROR: No baseline CSV found. Provide --baseline-csv path.")
            print(f"  Searched: {candidates}")
            sys.exit(1)

    print("=" * 70)
    print("PILOT RECOMPUTE v2 — C₁ llama-cpp baseline")
    print("=" * 70)
    print(f"  Trials dir:    {args.trials_dir}")
    print(f"  Baseline CSV:  {args.baseline_csv}")
    print(f"  Domain:        {args.domain}")
    print(f"  nRatings:      {args.n_ratings} (→ {2 * args.n_ratings} NLP bins)")
    print()

    # --- COMPUTE BASELINE M-RATIO FROM C₁ LLAMA-CPP DATA ---
    print("Computing baseline M-ratio from C₁ llama-cpp inference...")
    baseline_df = load_baseline_csv(args.baseline_csv)
    baseline_sdt = compute_type2_sdt(
        baseline_df, n_ratings=args.n_ratings,
        domain=args.domain if "domain" in baseline_df.columns else None
    )

    if "error" in baseline_sdt:
        print(f"  ERROR computing baseline: {baseline_sdt['error']}")
        sys.exit(1)

    baseline_m = baseline_sdt["m_ratio"]
    baseline_d = baseline_sdt["d_prime"]
    m1_ref = M1_TABLE6.get(args.domain, None)

    print(f"  Baseline d':      {baseline_d:.3f}")
    print(f"  Baseline meta-d': {baseline_sdt['meta_d_prime']:.3f}")
    print(f"  Baseline M-ratio: {baseline_m:.3f}")
    print(f"  nR_S1 raw: {[int(x) for x in baseline_sdt['nR_S1_raw']]}")
    print(f"  nR_S2 raw: {[int(x) for x in baseline_sdt['nR_S2_raw']]}")
    if m1_ref:
        print(f"  (M1 Table 6 reference: {m1_ref:.3f} — not used in gate)")
    print()

    # --- LOAD AND COMPUTE ADAPTED MODELS ---
    print("Loading adapted model trial data...")
    models = load_trial_data(args.trials_dir)
    print(f"  {len(models)} models loaded\n")

    all_results = {
        "_baseline": {
            "sdt": baseline_sdt,
            "source": args.baseline_csv,
            "gate": {"decision": "BASELINE", "reason": "Reference model"},
        }
    }

    for model_name, df in sorted(models.items()):
        print(f"\n>>> {model_name}")

        if "domain" in df.columns:
            n_domain = len(df[df["domain"] == args.domain])
            print(f"    {args.domain} trials: {n_domain}/{len(df)}")

        sdt = compute_type2_sdt(df, n_ratings=args.n_ratings,
                                domain=args.domain if "domain" in df.columns else None)

        if "error" not in sdt:
            print(f"    Accuracy:  {sdt['accuracy']:.3f} ({sdt['n_correct']}/{sdt['n_trials']})")
            print(f"    d':        {sdt['d_prime']:.3f}")
            print(f"    meta-d':   {sdt['meta_d_prime']:.3f}")
            print(f"    M-ratio:   {sdt['m_ratio']:.3f}")
            print(f"    nR_S1 raw: {[int(x) for x in sdt['nR_S1_raw']]}")
            print(f"    nR_S2 raw: {[int(x) for x in sdt['nR_S2_raw']]}")
        else:
            print(f"    ERROR: {sdt['error']}")

        gate = apply_decision_gate(sdt, baseline_m, baseline_d)
        print(f"    Gate: {gate['decision']} — {gate['reason']}")

        all_results[model_name] = {"sdt": sdt, "gate": gate}

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print(f"SUMMARY  (C₁ baseline M-ratio = {baseline_m:.3f}, "
          f"d' = {baseline_d:.3f}, domain = {args.domain})")
    if m1_ref:
        print(f"          M1 Table 6 reference = {m1_ref:.3f} (not used in gate)")
    print("=" * 70)
    header_d = "d'"
    print(f"{'Model':<28} {'M-ratio':>8} {'ΔM':>8} {header_d:>7} {'Acc':>6} {'Gate':>15}")
    print("-" * 75)

    # Print baseline first
    bs = all_results["_baseline"]["sdt"]
    print(f"{'BASELINE (llama-cpp)':<28} {bs['m_ratio']:>8.3f} {'---':>8} "
          f"{bs['d_prime']:>7.3f} {bs['accuracy']:>6.3f} {'REFERENCE':>15}")

    for name in sorted(k for k in all_results if k != "_baseline"):
        r = all_results[name]
        s, g = r["sdt"], r["gate"]
        if "error" in s:
            print(f"{name:<28} {'ERR':>8} {'':>8} {'':>7} {'':>6} {g['decision']:>15}")
        else:
            print(f"{name:<28} {s['m_ratio']:>8.3f} {g['delta_m']:>+8.3f} "
                  f"{s['d_prime']:>7.3f} {s['accuracy']:>6.3f} {g['decision']:>15}")

    # --- SAVE ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
