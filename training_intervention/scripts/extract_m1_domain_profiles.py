"""
Extract per-domain M1 profiles for Llama-3-8B-Instruct from trial data.

Usage:
  cd C:\sdt_calibration
  .venv\Scripts\activate
  python training_intervention\scripts\extract_m1_domain_profiles.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import sys

# ── Configuration ──────────────────────────────────────────────────────────
TRIAL_DATA_PATH = Path(r"C:\sdt_calibration\results\m1_trial_data.csv")
OUTPUT_PATH = Path(r"C:\sdt_calibration\training_intervention\results\m1_domain_profiles.json")

MODEL_FILTER = "llama3_instruct"   # exact or substring match
DATASET_FILTER = "triviaqa"        # M1 diagnostic was TriviaQA
TEMPERATURE_FILTER = 1.0           # M1 used T=1.0

# v1.2 classification thresholds
M_RATIO_LOWER = 0.85
M_RATIO_UPPER = 1.15
AUROC2_THRESHOLD = 0.55

# ── SDT Functions ──────────────────────────────────────────────────────────

def compute_type1_sdt(is_correct, nlp):
    correct_nlp = nlp[is_correct == 1]
    incorrect_nlp = nlp[is_correct == 0]

    if len(correct_nlp) < 5 or len(incorrect_nlp) < 5:
        return {"d_prime": np.nan, "criterion": np.nan}

    mean_c = np.mean(correct_nlp)
    mean_i = np.mean(incorrect_nlp)
    sd_c = np.std(correct_nlp, ddof=1)
    sd_i = np.std(incorrect_nlp, ddof=1)

    n_c, n_i = len(correct_nlp), len(incorrect_nlp)
    pooled_sd = np.sqrt(((n_c - 1) * sd_c**2 + (n_i - 1) * sd_i**2) / (n_c + n_i - 2))

    if pooled_sd < 1e-10:
        return {"d_prime": 0.0, "criterion": 0.0}

    d_prime = (mean_c - mean_i) / pooled_sd
    criterion = -0.5 * (mean_c + mean_i) / pooled_sd

    return {"d_prime": float(d_prime), "criterion": float(criterion)}


def compute_type2_sdt(is_correct, nlp, n_bins=4):
    correct_nlp = nlp[is_correct == 1]
    incorrect_nlp = nlp[is_correct == 0]

    if len(correct_nlp) < 10 or len(incorrect_nlp) < 10:
        return {"meta_d_prime": np.nan, "m_ratio": np.nan, "auroc2": np.nan}

    # Bin NLP into quantiles for Type-2 ROC
    all_nlp = nlp
    try:
        bin_edges = np.unique(np.quantile(all_nlp, np.linspace(0, 1, n_bins + 1)))
        if len(bin_edges) < 3:
            return {"meta_d_prime": np.nan, "m_ratio": np.nan, "auroc2": np.nan}
    except Exception:
        return {"meta_d_prime": np.nan, "m_ratio": np.nan, "auroc2": np.nan}

    n_correct_bins = np.histogram(correct_nlp, bins=bin_edges)[0]
    n_incorrect_bins = np.histogram(incorrect_nlp, bins=bin_edges)[0]

    # Cumulative from right (higher NLP = higher confidence)
    n_correct_cumR = np.cumsum(n_correct_bins[::-1])[::-1]
    n_incorrect_cumR = np.cumsum(n_incorrect_bins[::-1])[::-1]

    total_correct = len(correct_nlp)
    total_incorrect = len(incorrect_nlp)

    hr2 = np.clip(n_correct_cumR[1:] / max(total_correct, 1), 0.01, 0.99)
    far2 = np.clip(n_incorrect_cumR[1:] / max(total_incorrect, 1), 0.01, 0.99)

    # AUROC2
    far2_roc = np.concatenate([[0], far2, [1]])
    hr2_roc = np.concatenate([[0], hr2, [1]])
    sort_idx = np.argsort(far2_roc)
    auroc2 = float(np.trapz(hr2_roc[sort_idx], far2_roc[sort_idx]))

    # Type-1 d'
    t1 = compute_type1_sdt(is_correct, nlp)
    d_prime = t1["d_prime"]
    if np.isnan(d_prime):
        return {"meta_d_prime": np.nan, "m_ratio": np.nan, "auroc2": auroc2,
                "d_prime": np.nan, "criterion": np.nan}

    # meta-d' from median-split Type-2 HR/FAR
    median_nlp = np.median(all_nlp)
    high_conf_correct = np.sum((nlp >= median_nlp) & (is_correct == 1))
    low_conf_correct = np.sum((nlp < median_nlp) & (is_correct == 1))
    high_conf_incorrect = np.sum((nlp >= median_nlp) & (is_correct == 0))
    low_conf_incorrect = np.sum((nlp < median_nlp) & (is_correct == 0))

    hr2_med = np.clip(high_conf_correct / max(high_conf_correct + low_conf_correct, 1), 0.01, 0.99)
    far2_med = np.clip(high_conf_incorrect / max(high_conf_incorrect + low_conf_incorrect, 1), 0.01, 0.99)

    meta_d_prime = float(stats.norm.ppf(hr2_med) - stats.norm.ppf(far2_med))
    m_ratio = meta_d_prime / d_prime if abs(d_prime) > 0.01 else np.nan

    return {
        "meta_d_prime": meta_d_prime,
        "m_ratio": float(m_ratio) if not np.isnan(m_ratio) else None,
        "auroc2": auroc2,
        "d_prime": float(d_prime),
        "criterion": float(t1["criterion"]),
    }


def classify_domain(profile, median_d_prime, nlp_top_quartile_threshold):
    m = profile["m_ratio"]
    d = profile["d_prime"]
    auroc2 = profile["auroc2"]
    mean_nlp_incorrect = profile["mean_nlp_incorrect"]

    if m is None or np.isnan(m):
        return "unclassifiable"
    if M_RATIO_LOWER <= m <= M_RATIO_UPPER:
        return "well-calibrated"
    if d > median_d_prime and m < M_RATIO_LOWER and auroc2 > AUROC2_THRESHOLD:
        return "under-monitoring"
    if d < median_d_prime and mean_nlp_incorrect >= nlp_top_quartile_threshold:
        return "over-confident"
    return "unclassified"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not TRIAL_DATA_PATH.exists():
        print(f"ERROR: Not found: {TRIAL_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(TRIAL_DATA_PATH)
    print(f"Loaded {len(df)} trials")
    print(f"Columns: {list(df.columns)}")

    # ── Rename 'correct' -> 'is_correct' ───────────────────────────────────
    if "correct" in df.columns and "is_correct" not in df.columns:
        df = df.rename(columns={"correct": "is_correct"})
        print("Renamed 'correct' -> 'is_correct'")

    # ── Filter: model ──────────────────────────────────────────────────────
    mask = df["model"].astype(str).str.contains(MODEL_FILTER, case=False)
    df = df[mask]
    print(f"\nAfter model filter ('{MODEL_FILTER}'): {len(df)}")

    # ── Filter: dataset ────────────────────────────────────────────────────
    if DATASET_FILTER:
        mask = df["dataset"].astype(str).str.contains(DATASET_FILTER, case=False)
        df = df[mask]
        print(f"After dataset filter ('{DATASET_FILTER}'): {len(df)}")

    # ── Filter: temperature ────────────────────────────────────────────────
    if TEMPERATURE_FILTER is not None:
        df = df[df["temperature"] == TEMPERATURE_FILTER]
        print(f"After temperature filter ({TEMPERATURE_FILTER}): {len(df)}")

    if len(df) == 0:
        print("ERROR: No trials remaining after filters.")
        sys.exit(1)

    # ── Drop NaN domains ───────────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["domain"])
    df["domain"] = df["domain"].astype(str)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN domain")

    df["is_correct"] = df["is_correct"].astype(int)
    df["nlp"] = df["nlp"].astype(float)

    domains = sorted(df["domain"].unique())
    print(f"\nDomains: {domains}")
    print(f"Total trials: {len(df)}")

    # ── Per-domain profiles ────────────────────────────────────────────────
    profiles = {}
    for domain in domains:
        ddf = df[df["domain"] == domain]
        is_correct = ddf["is_correct"].values
        nlp = ddf["nlp"].values

        sdt = compute_type2_sdt(is_correct, nlp)

        correct_nlp = nlp[is_correct == 1]
        incorrect_nlp = nlp[is_correct == 0]

        profiles[domain] = {
            "n": int(len(ddf)),
            "n_correct": int(np.sum(is_correct)),
            "n_incorrect": int(np.sum(is_correct == 0)),
            "accuracy": float(np.mean(is_correct)),
            "d_prime": sdt.get("d_prime", np.nan),
            "criterion": sdt.get("criterion", np.nan),
            "meta_d_prime": sdt.get("meta_d_prime", np.nan),
            "m_ratio": sdt.get("m_ratio"),
            "auroc2": sdt.get("auroc2", np.nan),
            "mean_nlp_correct": float(np.mean(correct_nlp)) if len(correct_nlp) > 0 else np.nan,
            "mean_nlp_incorrect": float(np.mean(incorrect_nlp)) if len(incorrect_nlp) > 0 else np.nan,
            "std_nlp_correct": float(np.std(correct_nlp, ddof=1)) if len(correct_nlp) > 1 else np.nan,
            "std_nlp_incorrect": float(np.std(incorrect_nlp, ddof=1)) if len(incorrect_nlp) > 1 else np.nan,
            "nlp_gap": float(np.mean(correct_nlp) - np.mean(incorrect_nlp))
                       if len(correct_nlp) > 0 and len(incorrect_nlp) > 0 else np.nan,
        }

    # ── Classification ─────────────────────────────────────────────────────
    d_primes = [p["d_prime"] for p in profiles.values() if not np.isnan(p["d_prime"])]
    median_d_prime = float(np.median(d_primes)) if d_primes else 0.0

    nlp_incorrects = [p["mean_nlp_incorrect"] for p in profiles.values()
                      if not np.isnan(p["mean_nlp_incorrect"])]
    nlp_q75 = float(np.percentile(nlp_incorrects, 75)) if nlp_incorrects else 0.0

    print(f"\n── Classification Parameters ──")
    print(f"Median d': {median_d_prime:.3f}")
    print(f"NLP (incorrect) 75th pctl: {nlp_q75:.4f}")

    for domain, p in profiles.items():
        p["classification"] = classify_domain(p, median_d_prime, nlp_q75)
        cls = p["classification"]
        if cls == "under-monitoring":
            p["prescription"] = "confidence_amplification"
        elif cls == "over-confident":
            p["prescription"] = "abstention_training"
        else:
            p["prescription"] = "none"

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"M1 DOMAIN PROFILES — {MODEL_FILTER} | {DATASET_FILTER} | T={TEMPERATURE_FILTER}")
    print(f"{'='*100}")

    hdr = (f"{'Domain':<16} {'N':>5} {'Acc':>6} {'d_prime':>7} {'meta_d':>8} "
           f"{'M-ratio':>8} {'AUROC2':>7} {'NLP_gap':>8}  {'Class':<18} {'Rx'}")
    print(hdr)
    print("-" * len(hdr))

    for domain in sorted(profiles.keys()):
        p = profiles[domain]
        m_str = f"{p['m_ratio']:.3f}" if p['m_ratio'] is not None else "N/A"
        print(f"{domain:<16} {p['n']:>5} {p['accuracy']:>6.3f} {p['d_prime']:>7.3f} "
              f"{p['meta_d_prime']:>8.3f} {m_str:>8} {p['auroc2']:>7.3f} "
              f"{p['nlp_gap']:>8.4f}  {p['classification']:<18} {p['prescription']}")

    # Between-domain meta-d' SD
    md_vals = [p["meta_d_prime"] for p in profiles.values() if not np.isnan(p["meta_d_prime"])]
    if len(md_vals) > 1:
        sd = float(np.std(md_vals, ddof=1))
        print(f"\nBetween-domain meta-d' SD: {sd:.4f}")
        print(f"Suggested TOST delta (0.5 x SD): {0.5 * sd:.4f}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "model": MODEL_FILTER,
        "dataset": DATASET_FILTER,
        "temperature": TEMPERATURE_FILTER,
        "source": str(TRIAL_DATA_PATH),
        "classification_params": {
            "median_d_prime": median_d_prime,
            "nlp_top_quartile_threshold": nlp_q75,
            "m_ratio_band": [M_RATIO_LOWER, M_RATIO_UPPER],
            "auroc2_threshold": AUROC2_THRESHOLD,
        },
        "domains": profiles,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {OUTPUT_PATH}")

    # ── Prescription summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PRESCRIPTION SUMMARY")
    print(f"{'='*60}")
    for d in sorted(profiles.keys()):
        p = profiles[d]
        print(f"  {d}: {p['classification']} -> {p['prescription']}")

    targeted = [d for d, p in profiles.items() if p["prescription"] != "none"]
    excluded = [d for d, p in profiles.items() if p["prescription"] == "none"]
    print(f"\nTargeted: {targeted}")
    print(f"Excluded: {excluded}")

    # Wrong-prescription target
    strong = [(d, p["m_ratio"]) for d, p in profiles.items()
              if p["m_ratio"] is not None and p["classification"] in ("well-calibrated",)]
    if strong:
        strong.sort(key=lambda x: x[1], reverse=True)
        print(f"Wrong-prescription target (Cond 4): {strong[0][0]} (M-ratio={strong[0][1]:.3f})")

    return profiles


if __name__ == "__main__":
    main()
