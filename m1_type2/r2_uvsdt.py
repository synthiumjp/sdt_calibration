"""
R2: Unequal-Variance Meta-d' Robustness Check
==============================================
Tests whether M-ratio estimates change under UVSDT (s != 1).

Uses UVSDT slopes from Cacioli (2026) Type-1 paper (Table 2).
Gemma-2 slope estimated from the same pipeline.

Usage:
    python r2_uvsdt.py --data C:\sdt_calibration\results\m1_trial_data.csv

Author: JP Cacioli
"""

import argparse
import numpy as np
import pandas as pd
from metadpy.mle import fit_metad

# ── Configuration ──
SEED = 42
N_RATINGS = 4
MODELS = ["llama3_instruct", "mistral_instruct", "llama3_base", "gemma2_instruct"]
MODEL_LABELS = {
    "llama3_instruct": "Llama-3-Instruct",
    "mistral_instruct": "Mistral-Instruct",
    "llama3_base": "Llama-3-Base",
    "gemma2_instruct": "Gemma-2-Instruct",
}

# UVSDT slopes from Type-1 paper (z-ROC slopes at T=1.0 on TriviaQA)
# From secondary_analyses.json h4 → model → triviaqa → t1.0_slope
# Gemma-2 computed from trial data (z-ROC regression, 20 bins)
UVSDT_SLOPES = {
    "llama3_instruct": 0.6316,
    "mistral_instruct": 0.5668,
    "llama3_base": 0.7793,
    "gemma2_instruct": 0.9861,
}


def compute_metad_with_s(sub, s=1):
    """Compute meta-d' with specified variance ratio."""
    correct = sub.correct.values
    rating = sub.rating.values

    nR_S1 = np.array(
        [((correct == 0) & (rating == r)).sum() for r in range(1, 2 * N_RATINGS + 1)],
        dtype=float,
    ) + 0.5
    nR_S2 = np.array(
        [((correct == 1) & (rating == r)).sum() for r in range(1, 2 * N_RATINGS + 1)],
        dtype=float,
    ) + 0.5

    try:
        result = fit_metad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=N_RATINGS, s=s, verbose=0)
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="R2: Unequal-variance meta-d' robustness check")
    parser.add_argument("--data", required=True, help="Path to m1_trial_data.csv")
    args = parser.parse_args()

    # ── Load data ──
    df = pd.read_csv(args.data)
    df = df.drop_duplicates(
        subset=["model", "dataset", "temperature", "question_index"], keep="first"
    )

    # Domain collapse
    domain_map = {
        "History & Politics": "History & Politics",
        "Arts & Literature": "Arts & Literature",
        "Geography": "Geography",
        "Science & Technology": "Science & Technology",
        "Unclassified": "Unclassified",
        "Pop Culture & Entertainment": "Unclassified",
        "Sports": "Unclassified",
    }
    df["domain_collapsed"] = df["domain"].map(domain_map)
    df["correct"] = df["correct"].astype(int)

    # ── Compute bin edges at T=1.0 ──
    bin_edges = {}
    for model in df.model.unique():
        for dataset in df.dataset.unique():
            sub = df[(df.model == model) & (df.dataset == dataset) & (df.temperature == 1.0)]
            quantiles = np.linspace(0, 1, 2 * N_RATINGS + 1)[1:-1]
            bin_edges[(model, dataset)] = np.quantile(sub.nlp.values, quantiles)

    # ── Assign ratings ──
    ratings = np.zeros(len(df), dtype=int)
    for i, row in enumerate(df.itertuples()):
        e = bin_edges[(row.model, row.dataset)]
        ratings[i] = int(np.digitize(row.nlp, e)) + 1
    df["rating"] = ratings

    # ── R2: Compare EVSDT (s=1) vs UVSDT ──
    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]

    print("=" * 70)
    print("R2: UNEQUAL-VARIANCE META-D' ROBUSTNESS CHECK")
    print("=" * 70)
    print(f"\n{'Model':<22s}  {'s':>5s}  {'d_EVSDT':>8s}  {'M_EVSDT':>8s}  {'d_UVSDT':>8s}  {'M_UVSDT':>8s}  {'ΔM':>8s}")
    print("-" * 70)

    max_delta = 0
    ordering_evsdt = []
    ordering_uvsdt = []

    for model in MODELS:
        sub = tqa10[tqa10.model == model].reset_index(drop=True)
        s = UVSDT_SLOPES[model]

        # EVSDT (s=1)
        r_ev = compute_metad_with_s(sub, s=1)
        # UVSDT
        r_uv = compute_metad_with_s(sub, s=s)

        if r_ev is not None and r_uv is not None:
            delta = r_uv["m_ratio"] - r_ev["m_ratio"]
            max_delta = max(max_delta, abs(delta))
            ordering_evsdt.append((model, r_ev["m_ratio"]))
            ordering_uvsdt.append((model, r_uv["m_ratio"]))

            print(f"{MODEL_LABELS[model]:<22s}  {s:>5.2f}  "
                  f"{r_ev['dprime']:>8.3f}  {r_ev['m_ratio']:>8.3f}  "
                  f"{r_uv['dprime']:>8.3f}  {r_uv['m_ratio']:>8.3f}  "
                  f"{delta:>+8.3f}")
        else:
            print(f"{MODEL_LABELS[model]:<22s}  FAILED")

    # ── Check ordering preservation ──
    rank_ev = [m for m, _ in sorted(ordering_evsdt, key=lambda x: x[1], reverse=True)]
    rank_uv = [m for m, _ in sorted(ordering_uvsdt, key=lambda x: x[1], reverse=True)]
    ordering_preserved = rank_ev == rank_uv

    print(f"\nMax |ΔM-ratio|: {max_delta:.3f}")
    print(f"EVSDT ranking:  {' > '.join(MODEL_LABELS[m] for m in rank_ev)}")
    print(f"UVSDT ranking:  {' > '.join(MODEL_LABELS[m] for m in rank_uv)}")
    print(f"Ordering preserved: {'YES' if ordering_preserved else 'NO'}")

    # ── One-line summary for paper ──
    print("\n" + "=" * 70)
    print("SENTENCE FOR §4.8:")
    print("=" * 70)
    if ordering_preserved:
        print(f"Under the unequal-variance model (using z-ROC slopes from the Type-1 analysis),")
        print(f"M-ratio estimates shifted by at most {max_delta:.3f}; model ordering was preserved.")
    else:
        print(f"Under the unequal-variance model, M-ratio estimates shifted by at most {max_delta:.3f}")
        print(f"and model ordering changed — see details above.")


if __name__ == "__main__":
    main()
