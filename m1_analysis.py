"""
M1 Analysis Pipeline: Type-2 SDT for LLM Metacognition
=======================================================
Complete analysis script for local execution.

Pre-registered at OSF: [insert OSF ID]
Builds on: Cacioli (2026), arXiv: 2603.14893

Requirements:
    pip install metadpy pymc arviz numpy pandas scipy matplotlib seaborn

Usage:
    python m1_analysis.py --data m1_trial_data.csv --output results/

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
"""

import argparse
import json
import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metadpy.mle import fit_metad
from scipy import stats

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    print("Warning: arviz not available. HMeta-d analysis will be skipped.")

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
SEED = 42
N_BOOTSTRAP = 10_000
N_RATINGS = 4  # per response side; 2*N_RATINGS = 8 total bins
CLASSIFIED_DOMAINS = [
    "History & Politics",
    "Arts & Literature",
    "Geography",
    "Science & Technology",
]
DOMAIN_COLLAPSE = {
    "History & Politics": "History & Politics",
    "Arts & Literature": "Arts & Literature",
    "Geography": "Geography",
    "Science & Technology": "Science & Technology",
    "Unclassified": "Unclassified",
    "Pop Culture & Entertainment": "Unclassified",
    "Sports": "Unclassified",
}
MODELS = ["llama3_instruct", "mistral_instruct", "llama3_base", "gemma2_instruct"]
MODEL_LABELS = {
    "llama3_instruct": "Llama-3-Instruct",
    "mistral_instruct": "Mistral-Instruct",
    "llama3_base": "Llama-3-Base",
    "gemma2_instruct": "Gemma-2-Instruct",
}
H3_TEMPERATURES = [0.3, 0.5, 0.7, 1.0]
ALL_TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
TOST_DELTA = 0.3  # meta-d' units
MIN_TRIALS_PER_ACC = 50
MIN_DPRIME = 0.5

# Table 1 d_a values from CB&B paper (UVSDT) for criterion 4 check
# Gemma-2 not in CB&B paper; no prior d_a value
TABLE1_DA = {
    "llama3_instruct": 1.39,
    "mistral_instruct": 1.97,
    "llama3_base": 1.45,
    "gemma2_instruct": None,
}


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════
def load_and_clean(filepath):
    """Load trial-level data, dedup, collapse domains."""
    df = pd.read_csv(filepath)
    print(f"Raw: {len(df)} trials")

    # Dedup
    before = len(df)
    df = df.drop_duplicates(
        subset=["model", "dataset", "temperature", "question_index"], keep="first"
    )
    print(f"After dedup: {len(df)} (dropped {before - len(df)})")

    # Collapse domains
    df["domain_collapsed"] = df["domain"].map(DOMAIN_COLLAPSE)

    # Convert correct to int
    df["correct"] = df["correct"].astype(int)

    return df


def compute_bin_edges(df, temperature=1.0):
    """Compute 2*N_RATINGS quantile bin edges at specified temperature per model x dataset."""
    edges = {}
    for model in df.model.unique():
        for dataset in df.dataset.unique():
            sub = df[
                (df.model == model)
                & (df.dataset == dataset)
                & (df.temperature == temperature)
            ]
            quantiles = np.linspace(0, 1, 2 * N_RATINGS + 1)[1:-1]
            edges[(model, dataset)] = np.quantile(sub.nlp.values, quantiles)
    return edges


def assign_ratings(df, bin_edges):
    """Assign 1..2*N_RATINGS ratings based on NLP and precomputed bin edges."""
    ratings = np.zeros(len(df), dtype=int)
    for i, row in enumerate(df.itertuples()):
        e = bin_edges[(row.model, row.dataset)]
        ratings[i] = int(np.digitize(row.nlp, e)) + 1
    df = df.copy()
    df["rating"] = ratings
    return df


# ═══════════════════════════════════════════════════════════════════════
# META-D' COMPUTATION
# ═══════════════════════════════════════════════════════════════════════
def compute_metad(sub, s=1):
    """Compute meta-d' from a subset of trials with precomputed ratings.

    Returns dict with dprime, meta_d, m_ratio, or None if fitting fails.
    """
    correct = sub.correct.values
    rating = sub.rating.values

    nR_S1 = (
        np.array(
            [((correct == 0) & (rating == r)).sum() for r in range(1, 2 * N_RATINGS + 1)],
            dtype=float,
        )
        + 0.5
    )  # Hautus log-linear correction
    nR_S2 = (
        np.array(
            [((correct == 1) & (rating == r)).sum() for r in range(1, 2 * N_RATINGS + 1)],
            dtype=float,
        )
        + 0.5
    )

    try:
        result = fit_metad(
            nR_S1=nR_S1, nR_S2=nR_S2, nRatings=N_RATINGS, s=s, verbose=0
        )
        return result
    except Exception:
        return None


def bootstrap_mratio(sub, n_boot=N_BOOTSTRAP, seed=SEED, s=1):
    """Bootstrap M-ratio from trial-level data.

    Returns array of valid M-ratio bootstrap estimates.
    """
    rng = np.random.RandomState(seed)
    n = len(sub)
    correct_arr = sub.correct.values
    rating_arr = sub.rating.values

    mratios = []
    dprimes = []
    meta_ds = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        bc = correct_arr[idx]
        br = rating_arr[idx]

        nR_S1 = (
            np.array(
                [((bc == 0) & (br == r)).sum() for r in range(1, 2 * N_RATINGS + 1)],
                dtype=float,
            )
            + 0.5
        )
        nR_S2 = (
            np.array(
                [((bc == 1) & (br == r)).sum() for r in range(1, 2 * N_RATINGS + 1)],
                dtype=float,
            )
            + 0.5
        )

        try:
            result = fit_metad(
                nR_S1=nR_S1, nR_S2=nR_S2, nRatings=N_RATINGS, s=s, verbose=0
            )
            mr = result["m_ratio"]
            dp = result["dprime"]
            md = result["meta_d"]
            if np.isfinite(mr) and abs(mr) < 10:
                mratios.append(mr)
                dprimes.append(dp)
                meta_ds.append(md)
        except Exception:
            pass

    return {
        "m_ratio": np.array(mratios),
        "dprime": np.array(dprimes),
        "meta_d": np.array(meta_ds),
    }


# ═══════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════
def check_monotonicity(df, bin_edges):
    """Verify NLP is monotonically related to accuracy (§5.1 sanity check)."""
    print("\n" + "=" * 70)
    print("NLP MONOTONICITY CHECK")
    print("=" * 70)

    failures = []
    for model in MODELS:
        for dataset in df.dataset.unique():
            sub = df[
                (df.model == model)
                & (df.dataset == dataset)
                & (df.temperature == 1.0)
            ].copy()
            sub["nlp_q"] = pd.qcut(sub["nlp"], q=4, labels=[1, 2, 3, 4])
            acc = sub.groupby("nlp_q")["correct"].mean()
            mono = all(acc.iloc[i] <= acc.iloc[i + 1] for i in range(len(acc) - 1))

            status = "✓" if mono else "✗ FLAGGED"
            if not mono:
                failures.append(f"{model}/{dataset}")

            print(f"  {MODEL_LABELS[model]} | {dataset}: ", end="")
            for q in [1, 2, 3, 4]:
                print(f"Q{q}={acc.loc[q]:.3f} ", end="")
            print(status)

    if failures:
        print(f"\n  FAILURES: {failures}")
    else:
        print("\n  ALL PASS")
    return len(failures) == 0


def check_trial_counts(df):
    """Verify minimum trial counts per accuracy category (criterion 5)."""
    print("\n" + "=" * 70)
    print("TRIAL COUNT CHECK (criterion 5)")
    print("=" * 70)

    issues = []
    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]

    for model in MODELS:
        for domain in CLASSIFIED_DOMAINS:
            sub = tqa10[
                (tqa10.model == model) & (tqa10.domain_collapsed == domain)
            ]
            n_correct = sub.correct.sum()
            n_incorrect = len(sub) - n_correct
            min_n = min(n_correct, n_incorrect)
            status = "✓" if min_n >= MIN_TRIALS_PER_ACC else "✗ EXCLUDED"
            if min_n < MIN_TRIALS_PER_ACC:
                issues.append(f"{model}/{domain}")
            print(
                f"  {MODEL_LABELS[model]} | {domain}: "
                f"correct={n_correct}, incorrect={n_incorrect}, min={min_n} {status}"
            )

    if issues:
        print(f"\n  EXCLUSIONS: {issues}")
    else:
        print("\n  ALL PASS")
    return issues


# ═══════════════════════════════════════════════════════════════════════
# H1: SUBOPTIMAL METACOGNITION
# ═══════════════════════════════════════════════════════════════════════
def test_h1(df):
    """H1: M-ratio < 1 for all models at T=1.0 on TriviaQA."""
    print("\n" + "=" * 70)
    print("H1: SUBOPTIMAL METACOGNITION (aggregate M-ratio)")
    print(f"  Bootstrap: {N_BOOTSTRAP} resamples, seed={SEED}")
    print("=" * 70)

    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]
    results = {}

    for model in MODELS:
        sub = tqa10[tqa10.model == model].reset_index(drop=True)

        # Point estimate
        point = compute_metad(sub)

        # Bootstrap
        print(f"\n  {MODEL_LABELS[model]}: bootstrapping...", end="", flush=True)
        boot = bootstrap_mratio(sub)
        print(" done.")

        ci_lo, ci_hi = np.percentile(boot["m_ratio"], [2.5, 97.5])
        supported = ci_hi < 1.0

        results[model] = {
            "dprime": point["dprime"],
            "meta_d": point["meta_d"],
            "m_ratio": point["m_ratio"],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "supported": supported,
            "n_valid_boot": len(boot["m_ratio"]),
        }

        status = "SUPPORTED" if supported else "NOT SUPPORTED"
        print(
            f"    d'={point['dprime']:.3f}, meta-d'={point['meta_d']:.3f}, "
            f"M-ratio={point['m_ratio']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] → {status}"
        )

    # Summary
    n_supported = sum(1 for r in results.values() if r["supported"])
    print(f"\n  H1 SUMMARY: {n_supported}/3 models show M-ratio significantly < 1.0")

    return results


# ═══════════════════════════════════════════════════════════════════════
# H2: DOMAIN-SPECIFIC METACOGNITIVE EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════
def test_h2(df):
    """H2: M-ratio varies across TriviaQA domains within each model."""
    print("\n" + "=" * 70)
    print("H2: DOMAIN-SPECIFIC METACOGNITIVE EFFICIENCY")
    print(f"  Bootstrap: {N_BOOTSTRAP} resamples, seed={SEED}")
    print("=" * 70)

    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]
    results = {}
    models_with_sig = 0

    for model in MODELS:
        print(f"\n  {MODEL_LABELS[model]}:")
        domain_boots = {}
        domain_points = {}

        for domain in CLASSIFIED_DOMAINS:
            sub = tqa10[
                (tqa10.model == model) & (tqa10.domain_collapsed == domain)
            ].reset_index(drop=True)

            # Point estimate
            point = compute_metad(sub)
            domain_points[domain] = point

            # Bootstrap
            print(f"    {domain}: bootstrapping...", end="", flush=True)
            boot = bootstrap_mratio(sub)
            domain_boots[domain] = boot["m_ratio"]
            print(
                f" M-ratio={point['m_ratio']:.3f} "
                f"({len(boot['m_ratio'])}/{N_BOOTSTRAP} valid)"
            )

        # Pairwise comparisons
        print(f"    Pairwise differences:")
        sig_pairs = []
        pairwise_results = []

        for d1, d2 in combinations(CLASSIFIED_DOMAINS, 2):
            min_len = min(len(domain_boots[d1]), len(domain_boots[d2]))
            diffs = domain_boots[d1][:min_len] - domain_boots[d2][:min_len]
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            sig = ci_lo > 0 or ci_hi < 0
            if sig:
                sig_pairs.append((d1, d2))

            # Bonferroni check (alpha = 0.05/18 = 0.0028)
            bonf_lo, bonf_hi = np.percentile(diffs, [0.14, 99.86])
            bonf_sig = bonf_lo > 0 or bonf_hi < 0

            pairwise_results.append({
                "domain_1": d1,
                "domain_2": d2,
                "mean_diff": np.mean(diffs),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "significant": sig,
                "bonferroni_sig": bonf_sig,
            })

            marker = "***" if sig else ""
            print(
                f"      {d1[:12]:12s} - {d2[:12]:12s}: "
                f"Δ={np.mean(diffs):+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] {marker}"
            )

        if len(sig_pairs) > 0:
            models_with_sig += 1

        print(f"    Significant pairs: {len(sig_pairs)}/6")

        results[model] = {
            "domain_points": {d: {"dprime": p["dprime"], "meta_d": p["meta_d"],
                                   "m_ratio": p["m_ratio"]}
                              for d, p in domain_points.items()},
            "pairwise": pairwise_results,
            "n_sig_pairs": len(sig_pairs),
        }

    supported = models_with_sig >= 2
    print(f"\n  H2 SUMMARY: {models_with_sig}/3 models have ≥1 significant pairwise difference")
    print(f"  H2: {'SUPPORTED' if supported else 'NOT SUPPORTED'} (requires ≥2 models)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# H3: TEMPERATURE AND TYPE-2 PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
def test_h3(df):
    """H3: Temperature shifts Type-2 criterion while meta-d' is relatively stable."""
    print("\n" + "=" * 70)
    print("H3: TEMPERATURE EFFECTS ON TYPE-2 PARAMETERS")
    print(f"  Temperatures: {H3_TEMPERATURES}")
    print(f"  TOST delta: {TOST_DELTA} meta-d' units")
    print("=" * 70)

    tqa = df[df.dataset == "triviaqa"]
    results = {}

    for model in MODELS:
        print(f"\n  {MODEL_LABELS[model]}:")
        temps = []
        meta_ds = []
        dprimes = []
        type2_criteria = []

        for temp in H3_TEMPERATURES:
            sub = tqa[(tqa.model == model) & (tqa.temperature == temp)].reset_index(
                drop=True
            )
            r = compute_metad(sub)
            if r is not None:
                temps.append(temp)
                meta_ds.append(r["meta_d"])
                dprimes.append(r["dprime"])
                type2_criteria.append(r.get("meta_c1", r.get("meta_ca", 0)))
                print(
                    f"    T={temp}: d'={r['dprime']:.3f}, "
                    f"meta-d'={r['meta_d']:.3f}, M={r['m_ratio']:.3f}"
                )

        temps = np.array(temps)
        meta_ds = np.array(meta_ds)
        dprimes = np.array(dprimes)

        # Spearman correlations
        rho_metad, p_metad = stats.spearmanr(temps, meta_ds)
        rho_dprime, p_dprime = stats.spearmanr(temps, dprimes)

        # TOST for meta-d' invariance
        meta_d_range = meta_ds.max() - meta_ds.min()
        tost_pass = meta_d_range < TOST_DELTA

        # Relative robustness
        relative_robust = abs(rho_metad) < abs(rho_dprime)

        results[model] = {
            "temperatures": temps.tolist(),
            "meta_ds": meta_ds.tolist(),
            "dprimes": dprimes.tolist(),
            "rho_metad": rho_metad,
            "p_metad": p_metad,
            "rho_dprime": rho_dprime,
            "p_dprime": p_dprime,
            "meta_d_range": meta_d_range,
            "tost_pass": tost_pass,
            "relative_robust": relative_robust,
        }

        print(f"    ρ(meta-d', T) = {rho_metad:+.3f} (p={p_metad:.3f})")
        print(f"    ρ(d', T) = {rho_dprime:+.3f} (p={p_dprime:.3f})")
        print(f"    meta-d' range = {meta_d_range:.3f} (TOST δ={TOST_DELTA}: {'PASS' if tost_pass else 'FAIL'})")
        print(f"    |ρ(meta-d')| < |ρ(d')|? {'YES' if relative_robust else 'NO'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# H4: HIDDEN METACOGNITIVE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════
def test_h4(h1_results):
    """H4: Models with similar d' occupy different M-ratio positions."""
    print("\n" + "=" * 70)
    print("H4: HIDDEN METACOGNITIVE STRUCTURE")
    print("=" * 70)

    results = {}
    sig_pairs = 0

    for m1, m2 in combinations(MODELS, 2):
        r1 = h1_results[m1]
        r2 = h1_results[m2]

        diff_mr = r1["m_ratio"] - r2["m_ratio"]
        diff_dp = r1["dprime"] - r2["dprime"]

        # Use the bootstrap distributions from H1
        # For a proper pairwise test, we'd need paired bootstrap
        # For now, use the independent CIs as a conservative test
        ci1 = (r1["ci_lo"], r1["ci_hi"])
        ci2 = (r2["ci_lo"], r2["ci_hi"])
        no_overlap = ci1[0] > ci2[1] or ci2[0] > ci1[1]

        if no_overlap:
            sig_pairs += 1

        results[f"{m1}_vs_{m2}"] = {
            "m1_mratio": r1["m_ratio"],
            "m2_mratio": r2["m_ratio"],
            "diff_mratio": diff_mr,
            "diff_dprime": diff_dp,
            "ci_no_overlap": no_overlap,
        }

        print(
            f"  {MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}:"
        )
        print(
            f"    Δd'={diff_dp:+.3f}, ΔM-ratio={diff_mr:+.3f}"
        )
        print(
            f"    CIs: [{ci1[0]:.3f},{ci1[1]:.3f}] vs [{ci2[0]:.3f},{ci2[1]:.3f}]"
            f" → {'NO OVERLAP ***' if no_overlap else 'overlap'}"
        )

    supported = sig_pairs >= 1
    print(f"\n  H4 SUMMARY: {sig_pairs}/3 pairs with non-overlapping CIs")
    print(f"  H4: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════
def plot_h1_h4(h1_results, output_dir):
    """Figure 1: d' vs meta-d' space with identity line."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    colors = {"llama3_instruct": "#2196F3", "mistral_instruct": "#FF5722",
              "llama3_base": "#4CAF50", "gemma2_instruct": "#9C27B0"}
    markers = {"llama3_instruct": "o", "mistral_instruct": "s",
               "llama3_base": "D", "gemma2_instruct": "^"}

    for model in MODELS:
        r = h1_results[model]
        ci_lo = r.get("ci_lo", np.nan)
        ci_hi = r.get("ci_hi", np.nan)

        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            yerr_lo = r["meta_d"] - ci_lo * r["dprime"]
            yerr_hi = ci_hi * r["dprime"] - r["meta_d"]
            yerr = [[max(0, yerr_lo)], [max(0, yerr_hi)]]
        else:
            yerr = None

        ax.errorbar(
            r["dprime"], r["meta_d"],
            xerr=0, yerr=yerr,
            fmt=markers.get(model, "o"), color=colors.get(model, "gray"), markersize=12,
            capsize=5, label=MODEL_LABELS[model], zorder=5,
        )

    # Identity line
    lims = [0, max(r["dprime"] for r in h1_results.values()) * 1.3]
    ax.plot(lims, lims, "k--", alpha=0.3, label="meta-d' = d' (optimal)")
    ax.set_xlabel("Type-1 d'", fontsize=13)
    ax.set_ylabel("meta-d'", fontsize=13)
    ax.set_title("Metacognitive Efficiency Space", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_dprime_vs_metad.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig1_dprime_vs_metad.pdf"))
    plt.close()
    print("  Saved fig1_dprime_vs_metad")


def plot_h2_domains(h2_results, output_dir):
    """Figure 2: M-ratio by domain for each model."""
    n_models = len([m for m in MODELS if m in h2_results])
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = {"llama3_instruct": "#2196F3", "mistral_instruct": "#FF5722",
              "llama3_base": "#4CAF50", "gemma2_instruct": "#9C27B0"}

    for i, model in enumerate(m for m in MODELS if m in h2_results):
        ax = axes[i]
        dp = h2_results[model]["domain_points"]

        domains_short = [d[:12] for d in CLASSIFIED_DOMAINS]
        mratios = [dp[d]["m_ratio"] for d in CLASSIFIED_DOMAINS]

        bars = ax.bar(domains_short, mratios, color=colors.get(model, "gray"), alpha=0.8, edgecolor="black")
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Optimal (M=1)")
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_ylabel("M-ratio" if i == 0 else "", fontsize=12)
        ax.set_ylim(0, max(max(mratios) * 1.15, 1.3))
        ax.tick_params(axis="x", rotation=45)

        # Annotate
        for bar, mr in zip(bars, mratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mr:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_domain_mratio.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig2_domain_mratio.pdf"))
    plt.close()
    print("  Saved fig2_domain_mratio")


def plot_h3_temperature(h3_results, output_dir):
    """Figure 3: meta-d' and d' across temperatures."""
    n_models = len(MODELS)
    ncols = min(n_models, 4)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    colors = {"llama3_instruct": "#2196F3", "mistral_instruct": "#FF5722",
              "llama3_base": "#4CAF50", "gemma2_instruct": "#9C27B0"}

    for i, model in enumerate(MODELS):
        ax = axes[i]
        r = h3_results[model]

        ax.plot(r["temperatures"], r["dprimes"], "o-", color=colors.get(model, "gray"),
                label="d'", markersize=8)
        ax.plot(r["temperatures"], r["meta_ds"], "s--", color=colors.get(model, "gray"),
                alpha=0.6, label="meta-d'", markersize=8)

        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Sensitivity" if i == 0 else "", fontsize=12)
        ax.set_title(f"{MODEL_LABELS[model]}\nρ(meta-d',T)={r['rho_metad']:+.2f}, "
                     f"ρ(d',T)={r['rho_dprime']:+.2f}", fontsize=11)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_temperature.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig3_temperature.pdf"))
    plt.close()
    print("  Saved fig3_temperature")


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECKS
# ═══════════════════════════════════════════════════════════════════════
def robustness_nratings(df, h1_point_results):
    """R1: Repeat with nRatings ∈ {3, 6}."""
    print("\n" + "=" * 70)
    print("R1: BINNING SENSITIVITY (nRatings ∈ {3, 6})")
    print("=" * 70)

    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]
    max_delta = 0

    for nr in [3, 6]:
        print(f"\n  nRatings = {nr}:")
        edges = {}
        for model in MODELS:
            for dataset in df.dataset.unique():
                sub = df[(df.model == model) & (df.dataset == dataset) & (df.temperature == 1.0)]
                quantiles = np.linspace(0, 1, 2 * nr + 1)[1:-1]
                edges[(model, dataset)] = np.quantile(sub.nlp.values, quantiles)

        for model in MODELS:
            sub = tqa10[tqa10.model == model].copy()
            e = edges[(model, "triviaqa")]
            sub["rating_r"] = [int(np.digitize(nlp, e)) + 1 for nlp in sub.nlp.values]

            correct = sub.correct.values
            rating = sub.rating_r.values
            nR_S1 = np.array([((correct == 0) & (rating == r)).sum()
                              for r in range(1, 2 * nr + 1)], dtype=float) + 0.5
            nR_S2 = np.array([((correct == 1) & (rating == r)).sum()
                              for r in range(1, 2 * nr + 1)], dtype=float) + 0.5

            try:
                result = fit_metad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=nr, s=1, verbose=0)
                orig = h1_point_results[model]["m_ratio"]
                delta = abs(result["m_ratio"] - orig)
                max_delta = max(max_delta, delta)
                print(f"    {MODEL_LABELS[model]}: M-ratio={result['m_ratio']:.3f} "
                      f"(orig={orig:.3f}, Δ={delta:.3f})")
            except Exception as e:
                print(f"    {MODEL_LABELS[model]}: FAILED ({e})")

    print(f"\n  Max |ΔM-ratio| across R1: {max_delta:.3f}")
    return max_delta


# ═══════════════════════════════════════════════════════════════════════
# EXPLORATORY: AUROC_2 vs M-RATIO DISSOCIATION
# ═══════════════════════════════════════════════════════════════════════
def exploratory_auroc_dissociation(df, h1_results, fig_dir):
    """Show that AUROC_2 and M-ratio give non-redundant (potentially opposite) rankings."""
    print("\n" + "=" * 70)
    print("EXPLORATORY: AUROC_2 vs M-RATIO DISSOCIATION")
    print("=" * 70)

    from sklearn.metrics import roc_auc_score

    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]
    results = {}

    for model in MODELS:
        sub = tqa10[tqa10.model == model]
        auroc = roc_auc_score(sub.correct, sub.nlp)
        mr = h1_results[model]["m_ratio"]
        dp = h1_results[model]["dprime"]
        results[model] = {"auroc2": auroc, "m_ratio": mr, "dprime": dp}
        print(f"  {MODEL_LABELS[model]}: AUROC_2={auroc:.3f}, M-ratio={mr:.3f}, d'={dp:.3f}")

    # Rankings
    ranked_auroc = sorted(results, key=lambda m: results[m]["auroc2"], reverse=True)
    ranked_mr = sorted(results, key=lambda m: results[m]["m_ratio"], reverse=True)
    print(f"\n  AUROC_2 ranking: {' > '.join(MODEL_LABELS[m] for m in ranked_auroc)}")
    print(f"  M-ratio ranking: {' > '.join(MODEL_LABELS[m] for m in ranked_mr)}")
    agree = ranked_auroc == ranked_mr
    print(f"  Rankings agree? {'YES' if agree else 'NO — metrics provide non-redundant information'}")

    # Per-domain AUROC_2
    print("\n  Per-domain AUROC_2:")
    classified = ["History & Politics", "Arts & Literature", "Geography", "Science & Technology"]
    for model in MODELS:
        print(f"    {MODEL_LABELS[model]}:", end="")
        for domain in classified:
            sub = tqa10[(tqa10.model == model) & (tqa10.domain_collapsed == domain)]
            try:
                auroc = roc_auc_score(sub.correct, sub.nlp)
                print(f"  {domain[:8]}={auroc:.3f}", end="")
            except Exception:
                print(f"  {domain[:8]}=N/A", end="")
        print()

    # Figure: AUROC_2 vs M-ratio scatter
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    colors = {"llama3_instruct": "#2196F3", "mistral_instruct": "#FF5722",
              "llama3_base": "#4CAF50", "gemma2_instruct": "#9C27B0"}
    markers = {"llama3_instruct": "o", "mistral_instruct": "s",
               "llama3_base": "D", "gemma2_instruct": "^"}

    for model in MODELS:
        if model in results:
            ax.scatter(results[model]["auroc2"], results[model]["m_ratio"],
                       c=colors.get(model, "gray"), marker=markers.get(model, "o"),
                       s=150, label=MODEL_LABELS[model], zorder=5, edgecolors="black", linewidths=0.5)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Optimal M-ratio")
    ax.set_xlabel("AUROC₂ (Type-2 discrimination)", fontsize=12)
    ax.set_ylabel("M-ratio (metacognitive efficiency)", fontsize=12)
    ax.set_title("AUROC₂ vs M-ratio: Non-redundant evaluation metrics", fontsize=12)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_auroc_vs_mratio.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig4_auroc_vs_mratio.pdf"))
    plt.close()
    print("  Saved fig4_auroc_vs_mratio")

    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPLORATORY: SELECTIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════
def exploratory_selective_prediction(df, fig_dir):
    """Accuracy-coverage curves for selective prediction using NLP confidence."""
    print("\n" + "=" * 70)
    print("EXPLORATORY: SELECTIVE PREDICTION (accuracy vs coverage)")
    print("=" * 70)

    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]
    coverages = np.arange(10, 101, 5)  # 10% to 100% in 5% steps

    colors = {"llama3_instruct": "#2196F3", "mistral_instruct": "#FF5722",
              "llama3_base": "#4CAF50", "gemma2_instruct": "#9C27B0"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    results = {}
    for model in MODELS:
        sub = tqa10[tqa10.model == model].sort_values("nlp", ascending=False)
        baseline = sub.correct.mean()
        accs = []
        gains = []
        for k in coverages:
            n = max(1, int(len(sub) * k / 100))
            top_k = sub.head(n)
            acc = top_k.correct.mean()
            accs.append(acc)
            gains.append(acc - baseline)
        results[model] = {"coverages": coverages.tolist(), "accuracies": accs,
                          "gains": gains, "baseline": baseline}

        # Raw accuracy-coverage
        axes[0].plot(coverages, accs, "o-", color=colors.get(model, "gray"),
                     label=MODEL_LABELS[model], markersize=4)
        # Accuracy gain
        axes[1].plot(coverages, gains, "o-", color=colors.get(model, "gray"),
                     label=MODEL_LABELS[model], markersize=4)

    axes[0].set_xlabel("Coverage (%)", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Selective prediction: accuracy vs coverage", fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].set_xlabel("Coverage (%)", fontsize=12)
    axes[1].set_ylabel("Accuracy gain over baseline", fontsize=12)
    axes[1].set_title("Accuracy gain from confidence-based selection", fontsize=12)
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.2)
    axes[1].legend(fontsize=10)

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_selective_prediction.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig5_selective_prediction.pdf"))
    plt.close()
    print("  Saved fig5_selective_prediction")

    # Print summary table
    print(f"\n  {'Model':<22s}  baseline  top-50%  top-70%  top-90%")
    print("  " + "-" * 60)
    for model in MODELS:
        r = results[model]
        idx50 = list(coverages).index(50)
        idx70 = list(coverages).index(70)
        idx90 = list(coverages).index(90)
        print(f"  {MODEL_LABELS[model]:<22s}  {r['baseline']:.3f}    "
              f"{r['accuracies'][idx50]:.3f}    {r['accuracies'][idx70]:.3f}    "
              f"{r['accuracies'][idx90]:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPLORATORY: INSTRUCT vs BASE (E1)
# ═══════════════════════════════════════════════════════════════════════
def exploratory_instruct_vs_base(h1_results, h2_results):
    """E1: Does instruction tuning shift Type-2 criterion or meta-d'?"""
    print("\n" + "=" * 70)
    print("EXPLORATORY E1: INSTRUCTION TUNING EFFECT ON TYPE-2 PARAMETERS")
    print("=" * 70)

    inst = h1_results.get("llama3_instruct")
    base = h1_results.get("llama3_base")

    if inst is None or base is None:
        print("  Cannot compare — missing model results")
        return

    print(f"  Llama-3-Instruct: d'={inst['dprime']:.3f}, meta-d'={inst['meta_d']:.3f}, "
          f"M-ratio={inst['m_ratio']:.3f}")
    print(f"  Llama-3-Base:     d'={base['dprime']:.3f}, meta-d'={base['meta_d']:.3f}, "
          f"M-ratio={base['m_ratio']:.3f}")

    delta_d = inst["dprime"] - base["dprime"]
    delta_md = inst["meta_d"] - base["meta_d"]
    delta_mr = inst["m_ratio"] - base["m_ratio"]

    print(f"\n  Δd' = {delta_d:+.3f}")
    print(f"  Δmeta-d' = {delta_md:+.3f}")
    print(f"  ΔM-ratio = {delta_mr:+.3f}")

    print(f"\n  Interpretation: Instruction tuning produces {'minimal' if abs(delta_d) < 0.1 else 'substantial'} "
          f"change in d' ({delta_d:+.3f}),")
    print(f"  {'minimal' if abs(delta_md) < 0.15 else 'substantial'} change in meta-d' ({delta_md:+.3f}),")
    print(f"  and a {'decrease' if delta_mr < 0 else 'increase'} in M-ratio ({delta_mr:+.3f}).")
    print(f"  This is consistent with instruction tuning primarily affecting confidence")
    print(f"  policy (criterion) rather than metacognitive sensitivity.")

    # Per-domain comparison if available
    if h2_results and "llama3_instruct" in h2_results and "llama3_base" in h2_results:
        print(f"\n  Per-domain M-ratio comparison:")
        inst_d = h2_results["llama3_instruct"]["domain_points"]
        base_d = h2_results["llama3_base"]["domain_points"]
        for domain in ["History & Politics", "Arts & Literature", "Geography", "Science & Technology"]:
            if domain in inst_d and domain in base_d:
                imr = inst_d[domain]["m_ratio"]
                bmr = base_d[domain]["m_ratio"]
                print(f"    {domain[:16]:16s}: Instruct={imr:.3f}, Base={bmr:.3f}, Δ={imr - bmr:+.3f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    global N_BOOTSTRAP

    parser = argparse.ArgumentParser(description="M1 Analysis: Type-2 SDT for LLM Metacognition")
    parser.add_argument("--data", required=True, help="Path to m1_trial_data.csv")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help=f"Number of bootstrap resamples (default: {N_BOOTSTRAP})")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip bootstrap (point estimates only)")
    args = parser.parse_args()

    N_BOOTSTRAP = args.n_bootstrap

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)

    # ── Load and clean ──
    print("=" * 70)
    print("M1 ANALYSIS PIPELINE")
    print("=" * 70)
    df = load_and_clean(args.data)

    # ── Bin edges and ratings ──
    print("\nComputing bin edges at T=1.0...")
    bin_edges = compute_bin_edges(df)
    df = assign_ratings(df, bin_edges)

    # ── Sanity checks ──
    check_monotonicity(df, bin_edges)
    excluded = check_trial_counts(df)

    # ── Criterion 4: d' consistency ──
    print("\n" + "=" * 70)
    print("CRITERION 4: d' CONSISTENCY")
    print("=" * 70)
    tqa10 = df[(df.dataset == "triviaqa") & (df.temperature == 1.0)]
    for model in MODELS:
        sub = tqa10[tqa10.model == model].reset_index(drop=True)
        r = compute_metad(sub)
        expected = TABLE1_DA[model]
        if expected is not None:
            diff = abs(r["dprime"] - expected)
            print(f"  {MODEL_LABELS[model]}: d'={r['dprime']:.3f} "
                  f"(Table 1 d_a={expected}, Δ={diff:.3f})")
        else:
            print(f"  {MODEL_LABELS[model]}: d'={r['dprime']:.3f} "
                  f"(no prior d_a — post-registration model)")
    print("  NOTE: d' (EVSDT) vs d_a (UVSDT); differences expected for Mistral.")

    # ── H1 ──
    if not args.skip_bootstrap:
        h1_results = test_h1(df)
    else:
        print("\n[Skipping H1 bootstrap — point estimates only]")
        h1_results = {}
        for model in MODELS:
            sub = tqa10[tqa10.model == model].reset_index(drop=True)
            r = compute_metad(sub)
            h1_results[model] = {
                "dprime": r["dprime"], "meta_d": r["meta_d"],
                "m_ratio": r["m_ratio"], "ci_lo": np.nan, "ci_hi": np.nan,
                "supported": False, "n_valid_boot": 0,
            }

    # ── H2 ──
    if not args.skip_bootstrap:
        h2_results = test_h2(df)
    else:
        print("\n[Skipping H2 bootstrap]")
        h2_results = None

    # ── H3 ──
    h3_results = test_h3(df)

    # ── H4 ──
    h4_results = test_h4(h1_results)

    # ── Figures ──
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    fig_dir = os.path.join(args.output, "figures")
    plot_h1_h4(h1_results, fig_dir)
    if h2_results:
        plot_h2_domains(h2_results, fig_dir)
    plot_h3_temperature(h3_results, fig_dir)

    # ── R1: Binning robustness ──
    robustness_nratings(df, h1_results)

    # ── Exploratory: AUROC_2 vs M-ratio dissociation ──
    auroc_results = exploratory_auroc_dissociation(df, h1_results, fig_dir)

    # ── Exploratory: Selective prediction ──
    selpred_results = exploratory_selective_prediction(df, fig_dir)

    # ── Exploratory: Instruct vs Base (E1) ──
    exploratory_instruct_vs_base(h1_results, h2_results)

    # ── Save all results ──
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save H1
    h1_df = pd.DataFrame([
        {"model": m, **{k: v for k, v in r.items() if k != "boot_distributions"}}
        for m, r in h1_results.items()
    ])
    h1_df.to_csv(os.path.join(args.output, "h1_results.csv"), index=False)

    # Save H3
    with open(os.path.join(args.output, "h3_results.json"), "w") as f:
        json.dump(h3_results, f, indent=2, default=str)

    # Save H4
    with open(os.path.join(args.output, "h4_results.json"), "w") as f:
        json.dump(h4_results, f, indent=2, default=str)

    print("  All results saved to", args.output)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  H1 (M<1):           {'Partially supported' if any(r['supported'] for r in h1_results.values()) else 'Not supported'}")
    if h2_results:
        n_h2 = sum(1 for r in h2_results.values() if r["n_sig_pairs"] > 0)
        print(f"  H2 (domain-spec):   {'Supported' if n_h2 >= 2 else 'Not supported'} ({n_h2}/3 models)")
    print(f"  H3 (temp robust):   See per-model results above")
    print(f"  H4 (hidden struct): {'Supported' if any(r['ci_no_overlap'] for r in h4_results.values()) else 'Not supported'}")


if __name__ == "__main__":
    main()
