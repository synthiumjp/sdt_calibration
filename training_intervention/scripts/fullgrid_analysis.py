"""
Full-Grid Confirmatory Analysis — Pre-registration 2

Computes meta-d′ per domain per condition from trial data, then runs all
pre-registered confirmatory tests:

  H1: Treatment effect (Cond 2 > Cond 1 in Science)
  H2: Non-degradation (|Cond 2 − Cond 1| < δ in History, Arts, Geography)
  H3: Conditional advantage (Cond 2 > Cond 3 in Science)
  H4: Causal test (Cond 2 > Cond 4 in Science)
  H5: Accuracy descriptives (all conditions × domains)

All bootstrap: 10,000 resamples, seed=42, question-level within domain.
TOST δ = 0.17 (half the between-domain meta-d′ SD from M1).

Usage:
  cd C:\\sdt_calibration
  .venv\\Scripts\\activate
  cd training_intervention
  python scripts/fullgrid_analysis.py

  # Quick check (100 bootstraps for speed):
  python scripts/fullgrid_analysis.py --n-boot 100

Output:
  results/fullgrid/confirmatory_results.json — all metrics and test results
  results/fullgrid/confirmatory_report.txt  — human-readable report

Author: JP Cacioli / Synthium
Date: April 2026
"""

import json
import argparse
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# metadpy MLE fit for meta-d′
from metadpy.mle import fit_metad

# ============================================================
# CONSTANTS (from Pre-reg 2)
# ============================================================

SEED = 42
N_BOOTSTRAP = 10_000
TOST_DELTA = 0.17  # half the between-domain meta-d′ SD from M1
N_RATINGS = 4      # quartile bins → 8 total bins for Type-2 table
HAUTUS_CORRECTION = 0.5  # log-linear correction

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results" / "fullgrid"

DOMAINS = [
    "Science",
    "History",
    "Arts",
    "Geography",
]

TARGET_DOMAIN = "Science"
NON_TARGET_DOMAINS = [d for d in DOMAINS if d != TARGET_DOMAIN]

# Conditions
ALL_CONDITIONS = [1, 2, 3, 4, 5, 6, 7]
CONDITION_LABELS = {
    1: "Baseline (no intervention)",
    2: "Conditional SFT (Science)",
    3: "Agnostic SFT (matched)",
    4: "Wrong-prescription SFT (Geography)",
    5: "Prompt cue",
    6: "Per-domain temp scaling",
    7: "Conditional SFT (Science, low LR)",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_trials(cond_id):
    """Load trial data for a condition."""
    path = RESULTS_DIR / f"fullgrid_trials_cond{cond_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Trial data not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    return trials


def trials_to_df(trials):
    """Convert trial list to DataFrame."""
    df = pd.DataFrame(trials)
    # Ensure correct types
    df["is_correct"] = df["is_correct"].astype(bool)
    df["nlp"] = pd.to_numeric(df["nlp"], errors="coerce")
    # Drop trials with invalid NLP
    n_before = len(df)
    df = df[np.isfinite(df["nlp"])]
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} trials with invalid NLP")
    return df


# ============================================================
# SDT COMPUTATION
# ============================================================

def compute_bin_edges(df):
    """
    Compute 2*N_RATINGS quantile bin edges from ALL trials in df.
    Matches M1: np.linspace(0, 1, 2*N_RATINGS+1)[1:-1] → 7 quantile edges → 8 bins.
    Bin edges are computed GLOBALLY (across all domains within a condition).
    """
    quantiles = np.linspace(0, 1, 2 * N_RATINGS + 1)[1:-1]
    edges = np.quantile(df["nlp"].values, quantiles)
    return edges


def assign_ratings(df, bin_edges):
    """
    Assign 1..2*N_RATINGS ratings based on NLP and precomputed bin edges.
    Matches M1: np.digitize(nlp, edges) + 1.
    """
    df = df.copy()
    df["rating"] = [int(np.digitize(nlp, bin_edges)) + 1 for nlp in df["nlp"].values]
    return df


def compute_metad(df_domain):
    """
    Compute meta-d′, d′, M-ratio for a single domain subset.
    Assumes df_domain already has a 'rating' column (1..2*N_RATINGS).

    Matches M1 exactly:
      - nR_S1[r] = count of incorrect trials with rating == r, + 0.5
      - nR_S2[r] = count of correct trials with rating == r, + 0.5
      - fit_metad(nR_S1, nR_S2, nRatings=N_RATINGS, s=1)

    Falls back to manual optimization if metadpy hits ZeroDivisionError.

    Returns dict with dprime, meta_d, m_ratio, n, accuracy, or None if failed.
    """
    if len(df_domain) < 20:
        return None

    if "rating" not in df_domain.columns:
        print("    ERROR: 'rating' column missing — call assign_ratings first")
        return None

    correct = df_domain["is_correct"].astype(int).values
    rating = df_domain["rating"].values

    nR_S1 = (
        np.array(
            [((correct == 0) & (rating == r)).sum()
             for r in range(1, 2 * N_RATINGS + 1)],
            dtype=float,
        )
        + HAUTUS_CORRECTION
    )
    nR_S2 = (
        np.array(
            [((correct == 1) & (rating == r)).sum()
             for r in range(1, 2 * N_RATINGS + 1)],
            dtype=float,
        )
        + HAUTUS_CORRECTION
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_metad(
                nR_S1=nR_S1, nR_S2=nR_S2,
                nRatings=N_RATINGS, s=1, verbose=0
            )

        dprime = float(result["dprime"])
        meta_d = float(result["meta_d"])
        m_ratio = float(result["m_ratio"])

    except ZeroDivisionError:
        # Fallback: compute d' and meta-d' manually using the
        # Maniscalco & Lau approach with clamped probabilities
        result = _fit_metad_safe(nR_S1, nR_S2, N_RATINGS)
        if result is None:
            return None
        dprime = result["dprime"]
        meta_d = result["meta_d"]
        m_ratio = result["m_ratio"]

    except Exception as e:
        print(f"    fit_metad failed: {e}")
        return None

    accuracy = float(np.mean(correct))

    return {
        "dprime": dprime,
        "meta_d": meta_d,
        "m_ratio": m_ratio,
        "n": len(df_domain),
        "accuracy": accuracy,
    }


def _fit_metad_safe(nR_S1, nR_S2, nRatings, s=1):
    """
    Safe fallback for fit_metad when scipy's trust-constr hits ZeroDivisionError.
    Monkey-patches the minimize call to use L-BFGS-B instead.
    """
    from scipy.stats import norm as norm_dist
    from scipy.optimize import minimize as sp_minimize

    # Replicate metadpy's setup (lines 696-746 of mle.py)
    nCriteria = 2 * nRatings - 1

    ratingHR = [sum(nR_S2[c:]) / sum(nR_S2) for c in range(1, 2 * nRatings)]
    ratingFAR = [sum(nR_S1[c:]) / sum(nR_S1) for c in range(1, 2 * nRatings)]

    t1_index = nRatings - 1

    d1 = (1 / s) * norm_dist.ppf(ratingHR[t1_index]) - norm_dist.ppf(ratingFAR[t1_index])

    if abs(d1) < 1e-10:
        return None

    c1 = (-1 / (1 + s)) * (norm_dist.ppf(ratingHR) + norm_dist.ppf(ratingFAR))
    t1c1 = c1[t1_index]
    t2_index = [i for i in range(2 * nRatings - 1) if i != t1_index]
    t2c1 = c1[t2_index]

    meta_d1 = d1
    guess = [meta_d1] + list(t2c1 - (meta_d1 * (t1c1 / d1)))

    # Import metadpy's log-likelihood function
    from metadpy.mle import fit_meta_d_logL

    # Wrap to catch ZeroDivisionError inside the objective
    def safe_logL(params, *args):
        try:
            return fit_meta_d_logL(params, *args)
        except (ZeroDivisionError, FloatingPointError):
            return 1e10  # large penalty

    # Bounds matching metadpy
    LB = [-10.0] + list(-20 * np.ones((nCriteria - 1) // 2)) + list(np.zeros((nCriteria - 1) // 2))
    UB = [10.0] + list(np.zeros((nCriteria - 1) // 2)) + list(20 * np.ones((nCriteria - 1) // 2))
    bounds = list(zip(LB, UB))

    try:
        result = sp_minimize(
            safe_logL,
            guess,
            args=(nR_S1, nR_S2, nRatings, d1, t1c1, s),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        meta_d_fit = result.x[0]
        m_ratio = meta_d_fit / d1 if abs(d1) > 1e-10 else float('nan')

        return {
            "dprime": float(d1),
            "meta_d": float(meta_d_fit),
            "m_ratio": float(m_ratio),
        }

    except Exception as e:
        print(f"    _fit_metad_safe failed: {e}")
        return None


def compute_all_metrics(df, cond_id):
    """Compute SDT metrics for all domains within one condition.
    Bin edges computed globally (across all domains), then metrics per domain."""
    # Compute bin edges globally for this condition (matching M1)
    bin_edges = compute_bin_edges(df)
    df = assign_ratings(df, bin_edges)

    results = {}
    for domain in DOMAINS:
        df_d = df[df["domain"] == domain]
        if len(df_d) == 0:
            continue
        metrics = compute_metad(df_d)
        if metrics:
            metrics["domain"] = domain
            metrics["condition"] = cond_id
            results[domain] = metrics
        else:
            print(f"    WARNING: Could not compute metrics for "
                  f"cond{cond_id} × {domain} (N={len(df_d)})")
    return results


# ============================================================
# BOOTSTRAP INFRASTRUCTURE
# ============================================================

def bootstrap_delta_metad(df_a, df_b, domain, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap the contrast: meta-d′(A) − meta-d′(B) within a domain.
    Question-level resampling within domain (per pre-reg §19).

    Bin edges are computed globally per condition (across all domains),
    then ratings assigned, then domain subset extracted. On each bootstrap
    resample, we re-bin within the domain subset using the resampled data's
    own quantile edges (matching M1's bootstrap approach).

    Returns dict with:
      point_estimate, ci_lower, ci_upper (95% percentile),
      ci90_lower, ci90_upper (90% for TOST),
    """
    rng = np.random.RandomState(seed)

    # Assign ratings globally for point estimate
    edges_a = compute_bin_edges(df_a)
    edges_b = compute_bin_edges(df_b)
    df_a = assign_ratings(df_a, edges_a)
    df_b = assign_ratings(df_b, edges_b)

    df_a_d = df_a[df_a["domain"] == domain].reset_index(drop=True)
    df_b_d = df_b[df_b["domain"] == domain].reset_index(drop=True)

    n_a = len(df_a_d)
    n_b = len(df_b_d)

    if n_a < 20 or n_b < 20:
        print(f"    Insufficient data for bootstrap: A={n_a}, B={n_b}")
        return None

    # Point estimate
    metrics_a = compute_metad(df_a_d)
    metrics_b = compute_metad(df_b_d)
    if metrics_a is None or metrics_b is None:
        return None
    point_est = metrics_a["meta_d"] - metrics_b["meta_d"]

    # Bootstrap
    deltas = []
    failures = 0
    for i in range(n_boot):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)

        boot_a_d = df_a_d.iloc[idx_a].copy()
        boot_b_d = df_b_d.iloc[idx_b].copy()

        # Re-bin within the bootstrap resample
        try:
            boot_edges_a = np.quantile(boot_a_d["nlp"].values,
                                        np.linspace(0, 1, 2 * N_RATINGS + 1)[1:-1])
            boot_a_d["rating"] = [int(np.digitize(nlp, boot_edges_a)) + 1
                                   for nlp in boot_a_d["nlp"].values]

            boot_edges_b = np.quantile(boot_b_d["nlp"].values,
                                        np.linspace(0, 1, 2 * N_RATINGS + 1)[1:-1])
            boot_b_d["rating"] = [int(np.digitize(nlp, boot_edges_b)) + 1
                                   for nlp in boot_b_d["nlp"].values]
        except Exception:
            failures += 1
            continue

        boot_met_a = compute_metad(boot_a_d)
        boot_met_b = compute_metad(boot_b_d)

        if boot_met_a is None or boot_met_b is None:
            failures += 1
            continue

        deltas.append(boot_met_a["meta_d"] - boot_met_b["meta_d"])

    if len(deltas) < n_boot * 0.5:
        print(f"    Too many bootstrap failures: {failures}/{n_boot}")
        return None

    deltas = np.array(deltas)

    if failures > 0:
        print(f"    Bootstrap: {failures}/{n_boot} failures "
              f"({len(deltas)} valid)")

    return {
        "point_estimate": float(point_est),
        "ci95_lower": float(np.percentile(deltas, 2.5)),
        "ci95_upper": float(np.percentile(deltas, 97.5)),
        "ci90_lower": float(np.percentile(deltas, 5.0)),
        "ci90_upper": float(np.percentile(deltas, 95.0)),
        "n_valid_boots": len(deltas),
        "n_failures": failures,
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
    }


# ============================================================
# CONFIRMATORY TESTS
# ============================================================

def test_h1(df_cond2, df_cond1, n_boot):
    """H1: Δmeta-d′(Cond2 − Cond1) > 0 in Science. 95% CI lower > 0."""
    print("\n--- H1: Treatment effect (Cond 2 vs Cond 1, Science) ---")
    result = bootstrap_delta_metad(df_cond2, df_cond1, TARGET_DOMAIN, n_boot)
    if result is None:
        return {"supported": None, "reason": "bootstrap failed"}

    supported = result["ci95_lower"] > 0
    result["supported"] = supported
    result["test"] = "H1"
    result["domain"] = TARGET_DOMAIN
    result["contrast"] = "Cond2 - Cond1"

    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"  Δmeta-d′ = {result['point_estimate']:.3f} "
          f"[{result['ci95_lower']:.3f}, {result['ci95_upper']:.3f}]")
    print(f"  → {verdict}")
    return result


def test_h2(df_cond2, df_cond1, n_boot):
    """
    H2: Non-degradation. |Δmeta-d′(Cond2 − Cond1)| < δ in non-target domains.
    TOST: 90% CI within [−0.17, +0.17]. Must pass all three domains.
    """
    print("\n--- H2: Non-degradation (Cond 2 vs Cond 1, non-target domains) ---")
    results = {}
    all_pass = True

    for domain in NON_TARGET_DOMAINS:
        print(f"\n  Domain: {domain}")
        r = bootstrap_delta_metad(df_cond2, df_cond1, domain, n_boot)
        if r is None:
            results[domain] = {"supported": None, "reason": "bootstrap failed"}
            all_pass = False
            continue

        # TOST: 90% CI must lie entirely within [−δ, +δ]
        within_bounds = (r["ci90_lower"] > -TOST_DELTA and
                         r["ci90_upper"] < TOST_DELTA)
        r["supported"] = within_bounds
        r["test"] = "H2"
        r["domain"] = domain
        r["contrast"] = "Cond2 - Cond1"
        r["tost_delta"] = TOST_DELTA

        verdict = "EQUIVALENT" if within_bounds else "NOT EQUIVALENT"
        print(f"  Δmeta-d′ = {r['point_estimate']:.3f} "
              f"90% CI [{r['ci90_lower']:.3f}, {r['ci90_upper']:.3f}]")
        print(f"  δ = ±{TOST_DELTA:.2f}")
        print(f"  → {verdict}")

        if not within_bounds:
            all_pass = False
        results[domain] = r

    results["overall_supported"] = all_pass
    print(f"\n  H2 overall: {'SUPPORTED' if all_pass else 'NOT SUPPORTED'}")
    return results


def test_h3(df_cond2, df_cond3, n_boot):
    """H3: Conditional advantage. meta-d′(Cond2) > meta-d′(Cond3) in Science."""
    print("\n--- H3: Conditional advantage (Cond 2 vs Cond 3, Science) ---")
    result = bootstrap_delta_metad(df_cond2, df_cond3, TARGET_DOMAIN, n_boot)
    if result is None:
        return {"supported": None, "reason": "bootstrap failed"}

    supported = result["ci95_lower"] > 0
    result["supported"] = supported
    result["test"] = "H3"
    result["domain"] = TARGET_DOMAIN
    result["contrast"] = "Cond2 - Cond3"

    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"  Δmeta-d′ = {result['point_estimate']:.3f} "
          f"[{result['ci95_lower']:.3f}, {result['ci95_upper']:.3f}]")
    print(f"  → {verdict}")
    return result


def test_h4(df_cond2, df_cond4, n_boot):
    """H4: Causal test. meta-d′(Cond2) > meta-d′(Cond4) in Science."""
    print("\n--- H4: Causal test (Cond 2 vs Cond 4, Science) ---")
    result = bootstrap_delta_metad(df_cond2, df_cond4, TARGET_DOMAIN, n_boot)
    if result is None:
        return {"supported": None, "reason": "bootstrap failed"}

    supported = result["ci95_lower"] > 0
    result["supported"] = supported
    result["test"] = "H4"
    result["domain"] = TARGET_DOMAIN
    result["contrast"] = "Cond2 - Cond4"

    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"  Δmeta-d′ = {result['point_estimate']:.3f} "
          f"[{result['ci95_lower']:.3f}, {result['ci95_upper']:.3f}]")
    print(f"  → {verdict}")
    return result


def descriptive_h5(all_dfs):
    """H5: Accuracy and d_a per domain per condition (descriptive)."""
    print("\n--- H5: Accuracy and discriminability (descriptive) ---")
    rows = []

    for cond_id in ALL_CONDITIONS:
        if cond_id not in all_dfs:
            continue
        df = all_dfs[cond_id]

        # Assign ratings globally for this condition
        bin_edges = compute_bin_edges(df)
        df = assign_ratings(df, bin_edges)

        for domain in DOMAINS:
            df_d = df[df["domain"] == domain]
            if len(df_d) == 0:
                continue
            metrics = compute_metad(df_d)
            row = {
                "condition": cond_id,
                "condition_label": CONDITION_LABELS[cond_id],
                "domain": domain,
                "n": len(df_d),
                "accuracy": float(df_d["is_correct"].mean()),
            }
            if metrics:
                row.update({
                    "dprime": metrics["dprime"],
                    "meta_d": metrics["meta_d"],
                    "m_ratio": metrics["m_ratio"],
                })
            else:
                row.update({"dprime": None, "meta_d": None, "m_ratio": None})

            # NLP statistics
            nlps = df_d["nlp"].values
            correct_nlps = df_d[df_d["is_correct"]]["nlp"].values
            incorrect_nlps = df_d[~df_d["is_correct"]]["nlp"].values
            row["nlp_mean"] = float(np.mean(nlps))
            row["nlp_std"] = float(np.std(nlps))
            if len(correct_nlps) > 0 and len(incorrect_nlps) > 0:
                row["nlp_gap"] = float(np.mean(correct_nlps) - np.mean(incorrect_nlps))
            else:
                row["nlp_gap"] = None

            rows.append(row)

    return rows


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(metrics_table, h1, h2, h3, h4, output_path):
    """Write human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CONFIRMATORY ANALYSIS REPORT — Pre-registration 2")
    lines.append("Prescribe, Don't Average: Domain-Conditional Metacognitive Training")
    lines.append("=" * 70)

    # Metrics table
    lines.append("\n\nMETRICS TABLE (per condition × domain)")
    lines.append("-" * 70)
    header = f"{'Cond':>4} {'Domain':<25} {'N':>5} {'Acc':>6} {'d′':>7} {'meta-d′':>8} {'M-ratio':>8} {'NLP gap':>8}"
    lines.append(header)
    lines.append("-" * 70)

    for row in sorted(metrics_table, key=lambda r: (r["condition"], r["domain"])):
        dp = f"{row['dprime']:.3f}" if row.get("dprime") is not None else "  —"
        md = f"{row['meta_d']:.3f}" if row.get("meta_d") is not None else "  —"
        mr = f"{row['m_ratio']:.3f}" if row.get("m_ratio") is not None else "  —"
        ng = f"{row['nlp_gap']:.4f}" if row.get("nlp_gap") is not None else "  —"
        lines.append(
            f"{row['condition']:>4} {row['domain']:<25} "
            f"{row['n']:>5} {row['accuracy']:>6.3f} {dp:>7} {md:>8} {mr:>8} {ng:>8}"
        )

    # H1
    lines.append("\n\n" + "=" * 70)
    lines.append("H1: TREATMENT EFFECT")
    lines.append(f"  Cond 2 vs Cond 1 in {TARGET_DOMAIN}")
    if h1.get("supported") is not None:
        lines.append(f"  Δmeta-d′ = {h1['point_estimate']:.3f} "
                      f"95% CI [{h1['ci95_lower']:.3f}, {h1['ci95_upper']:.3f}]")
        lines.append(f"  → {'SUPPORTED' if h1['supported'] else 'NOT SUPPORTED'}")
    else:
        lines.append(f"  → COULD NOT TEST: {h1.get('reason', 'unknown')}")

    # H2
    lines.append("\n" + "=" * 70)
    lines.append("H2: NON-DEGRADATION (TOST)")
    for domain in NON_TARGET_DOMAINS:
        r = h2.get(domain, {})
        lines.append(f"\n  {domain}:")
        if r.get("supported") is not None:
            lines.append(f"    Δmeta-d′ = {r['point_estimate']:.3f} "
                          f"90% CI [{r['ci90_lower']:.3f}, {r['ci90_upper']:.3f}]")
            lines.append(f"    δ = ±{TOST_DELTA:.2f}")
            lines.append(f"    → {'EQUIVALENT' if r['supported'] else 'NOT EQUIVALENT'}")
        else:
            lines.append(f"    → COULD NOT TEST: {r.get('reason', 'unknown')}")
    overall = h2.get("overall_supported")
    lines.append(f"\n  H2 overall: {'SUPPORTED' if overall else 'NOT SUPPORTED'}")

    # H3
    lines.append("\n" + "=" * 70)
    lines.append("H3: CONDITIONAL ADVANTAGE")
    lines.append(f"  Cond 2 vs Cond 3 in {TARGET_DOMAIN}")
    if h3.get("supported") is not None:
        lines.append(f"  Δmeta-d′ = {h3['point_estimate']:.3f} "
                      f"95% CI [{h3['ci95_lower']:.3f}, {h3['ci95_upper']:.3f}]")
        lines.append(f"  → {'SUPPORTED' if h3['supported'] else 'NOT SUPPORTED'}")
    else:
        lines.append(f"  → COULD NOT TEST: {h3.get('reason', 'unknown')}")

    # H4
    lines.append("\n" + "=" * 70)
    lines.append("H4: CAUSAL TEST")
    lines.append(f"  Cond 2 vs Cond 4 in {TARGET_DOMAIN}")
    if h4.get("supported") is not None:
        lines.append(f"  Δmeta-d′ = {h4['point_estimate']:.3f} "
                      f"95% CI [{h4['ci95_lower']:.3f}, {h4['ci95_upper']:.3f}]")
        lines.append(f"  → {'SUPPORTED' if h4['supported'] else 'NOT SUPPORTED'}")
    else:
        lines.append(f"  → COULD NOT TEST: {h4.get('reason', 'unknown')}")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {output_path}")
    print(report)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full-grid confirmatory analysis")
    parser.add_argument(
        "--n-boot", type=int, default=N_BOOTSTRAP,
        help=f"Number of bootstrap resamples (default: {N_BOOTSTRAP})"
    )
    args = parser.parse_args()
    n_boot = args.n_boot

    print(f"Confirmatory analysis — {n_boot} bootstrap resamples, seed={SEED}")
    print(f"TOST δ = {TOST_DELTA}")

    # Load all trial data
    all_dfs = {}
    for cond_id in ALL_CONDITIONS:
        try:
            trials = load_trials(cond_id)
            df = trials_to_df(trials)
            all_dfs[cond_id] = df
            print(f"  Cond {cond_id}: {len(df)} trials loaded")
        except FileNotFoundError as e:
            print(f"  Cond {cond_id}: {e}")

    # Check required conditions
    required = {1, 2, 3, 4}
    missing = required - set(all_dfs.keys())
    if missing:
        print(f"\nERROR: Missing required conditions: {missing}")
        print("Cannot run confirmatory tests without conditions 1, 2, 3, 4.")
        print("Run fullgrid_evaluate.py first.")
        return

    t0 = time.time()

    # H5: Descriptive metrics (all conditions)
    metrics_table = descriptive_h5(all_dfs)

    # H1: Treatment effect
    h1 = test_h1(all_dfs[2], all_dfs[1], n_boot)

    # H2: Non-degradation
    h2 = test_h2(all_dfs[2], all_dfs[1], n_boot)

    # H3: Conditional advantage
    h3 = test_h3(all_dfs[2], all_dfs[3], n_boot)

    # H4: Causal test
    h4 = test_h4(all_dfs[2], all_dfs[4], n_boot)

    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed/60:.1f} minutes")

    # Save structured results
    results = {
        "n_bootstrap": n_boot,
        "seed": SEED,
        "tost_delta": TOST_DELTA,
        "elapsed_minutes": elapsed / 60,
        "metrics_table": metrics_table,
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
    }
    results_path = RESULTS_DIR / "confirmatory_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # Generate report
    report_path = RESULTS_DIR / "confirmatory_report.txt"
    generate_report(metrics_table, h1, h2, h3, h4, report_path)


if __name__ == "__main__":
    main()
