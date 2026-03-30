"""
Domain-Conditional Metacognitive Training — Task 2.2
Diagnostic Profile: M1 M-ratio → Domain Prescription Table

Reads M1 results (per-domain d′, meta-d′, M-ratio, AUROC₂, mean NLP)
and applies the v1.2 §2.2 classification thresholds to produce a
prescription table for each model.

Classification rules (pre-registered):
  Under-monitoring:  d′ > median AND M-ratio < 0.85 AND AUROC₂ > 0.55
  Over-confident:    d′ < median AND mean NLP in top quartile
  Well-calibrated:   M-ratio ∈ [0.85, 1.15]

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
Date: 30 March 2026

Usage:
  python diagnostic_profile.py
  
  Or with custom M1 results:
  python diagnostic_profile.py --m1-results path/to/m1_results.json

  If you don't have a pre-built JSON of M1 results, the script includes
  hardcoded values from the M1 paper (arXiv: 2603.25112) as defaults.
"""

import json
import argparse
from pathlib import Path

# ============================================================
# M1 RESULTS — hardcoded from arXiv: 2603.25112
# Update these if you recompute M1 metrics with different parameters
# ============================================================

M1_RESULTS = {
    # Actual values from arXiv: 2603.25112 and m1_trial_data.csv
    # M-ratio values are from the paper (Table X).
    # mean_NLP and accuracy from trial data (T=1.0, TriviaQA).
    # d_prime, meta_d_prime, AUROC2: extract from m1_analysis.py output
    #   and fill in here. Classification uses M-ratio as primary signal,
    #   so None values for d'/meta-d'/AUROC2 are handled gracefully.
    "Llama-3-8B-Instruct": {
        "Science & Technology": {
            "d_prime": None,        # TODO: from m1_analysis.py
            "meta_d_prime": None,   # TODO: from m1_analysis.py
            "M_ratio": 0.788,       # PAPER VALUE
            "AUROC2": None,         # TODO: from m1_analysis.py
            "mean_NLP": -0.331,     # TRIAL DATA
            "accuracy": 0.544,      # TRIAL DATA (N=634)
        },
        "History & Politics": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 0.962,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.353,
            "accuracy": 0.534,      # (N=1248)
        },
        "Geography": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 1.198,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.280,
            "accuracy": 0.625,      # (N=667)
        },
        "Arts & Literature": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 1.130,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.406,
            "accuracy": 0.509,      # (N=1167)
        },
    },
    "Mistral-7B-Instruct-v0.3": {
        "Science & Technology": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 1.068,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.256,
            "accuracy": 0.385,      # (N=634)
        },
        "History & Politics": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 0.805,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.205,
            "accuracy": 0.413,      # (N=1248)
        },
        "Geography": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 0.812,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.180,
            "accuracy": 0.538,      # (N=667)
        },
        "Arts & Literature": {
            "d_prime": None,
            "meta_d_prime": None,
            "M_ratio": 0.677,       # PAPER VALUE
            "AUROC2": None,
            "mean_NLP": -0.246,
            "accuracy": 0.410,      # (N=1167)
        },
    },
}


# ============================================================
# CLASSIFICATION THRESHOLDS (v1.2 §2.2)
# ============================================================

M_RATIO_LOWER = 0.85   # Below this = potential under-monitoring
M_RATIO_UPPER = 1.15   # Above this = potential over-monitoring
AUROC2_MIN = 0.55      # Minimum AUROC₂ for under-monitoring classification


def classify_domain(domain_metrics, median_d_prime, nlp_top_quartile_threshold):
    """
    Classify a single domain according to v1.2 §2.2 rules.
    
    Under-monitoring:  d′ > median AND M-ratio < 0.85 AND AUROC₂ > 0.55
      → The model performs well (high d′) but doesn't know it (low M-ratio).
      → Intervention: confidence amplification on correct answers.
    
    Over-confident:    d′ < median AND mean NLP in top quartile
      → The model performs poorly but expresses high confidence.
      → Intervention: abstention training on incorrect answers.
    
    Well-calibrated:   M-ratio ∈ [0.85, 1.15]
      → No intervention needed.
    
    When d′ or AUROC₂ are not available (None), classification falls back
    to M-ratio only. This is a conservative approach — the full criteria
    should be applied once all metrics are extracted from m1_analysis.py.
    
    Returns: (classification, rationale)
    """
    d = domain_metrics["d_prime"]
    m = domain_metrics["M_ratio"]
    auroc = domain_metrics["AUROC2"]
    nlp = domain_metrics["mean_NLP"]
    
    # Check well-calibrated first (takes priority if M-ratio is in range)
    if M_RATIO_LOWER <= m <= M_RATIO_UPPER:
        return "well-calibrated", (
            f"M-ratio = {m:.3f} is within [{M_RATIO_LOWER}, {M_RATIO_UPPER}] band. "
            f"No intervention needed."
        )
    
    # If d′ and AUROC₂ are available, apply full criteria
    if d is not None and auroc is not None and median_d_prime is not None:
        # Under-monitoring: good performance, poor monitoring
        if d > median_d_prime and m < M_RATIO_LOWER and auroc > AUROC2_MIN:
            return "under-monitoring", (
                f"d′ = {d:.3f} > median ({median_d_prime:.3f}), "
                f"M-ratio = {m:.3f} < {M_RATIO_LOWER}, "
                f"AUROC₂ = {auroc:.3f} > {AUROC2_MIN}. "
                f"Model performs well but monitors poorly. "
                f"→ Confidence amplification on correct answers."
            )
        
        # Over-confident: poor performance, high confidence
        if d < median_d_prime and nlp > nlp_top_quartile_threshold:
            return "over-confident", (
                f"d′ = {d:.3f} < median ({median_d_prime:.3f}), "
                f"mean NLP = {nlp:.3f} > top-quartile threshold ({nlp_top_quartile_threshold:.3f}). "
                f"Model performs poorly but is highly confident. "
                f"→ Abstention training on incorrect answers."
            )
    
    # Fallback: classify by M-ratio deviation alone
    if m < M_RATIO_LOWER:
        return "under-monitoring", (
            f"M-ratio = {m:.3f} < {M_RATIO_LOWER}. "
            f"Classified by M-ratio alone (d′/AUROC₂ not yet available). "
            f"→ Confidence amplification."
        )
    elif m > M_RATIO_UPPER:
        return "well-calibrated", (
            f"M-ratio = {m:.3f} > {M_RATIO_UPPER}. "
            f"Slightly over-monitoring (conservative), treated as well-calibrated."
        )
    
    return "well-calibrated", f"Default: M-ratio = {m:.3f}. No clear deficit."


def build_prescription_table(model_name, model_results):
    """
    Build the full prescription table for a model.
    
    Returns a dict: domain → {classification, intervention, rationale, metrics}
    """
    domains = list(model_results.keys())
    
    # Compute median d′ across domains (None-safe)
    d_primes = [model_results[d]["d_prime"] for d in domains 
                if model_results[d]["d_prime"] is not None]
    if d_primes:
        d_primes_sorted = sorted(d_primes)
        n = len(d_primes_sorted)
        if n % 2 == 0:
            median_d = (d_primes_sorted[n//2 - 1] + d_primes_sorted[n//2]) / 2
        else:
            median_d = d_primes_sorted[n//2]
    else:
        median_d = None
        print("  WARNING: d′ values not available. Using M-ratio-only classification.")
    
    # Compute NLP top-quartile threshold (75th percentile = least negative)
    nlps = sorted([model_results[d]["mean_NLP"] for d in domains])
    # Top quartile = highest NLP = least negative
    # For 4 domains, top quartile threshold = value at position 3 (0-indexed)
    nlp_q75 = nlps[-1]  # With only 4 domains, top quartile = highest value
    # More robust: use the value above which 25% of domains fall
    q75_idx = int(0.75 * len(nlps))
    nlp_q75 = nlps[q75_idx] if q75_idx < len(nlps) else nlps[-1]
    
    prescription = {}
    
    for domain in domains:
        classification, rationale = classify_domain(
            model_results[domain], median_d, nlp_q75
        )
        
        intervention_map = {
            "under-monitoring": "confidence_amplification",
            "over-confident": "abstention_training",
            "well-calibrated": "none",
        }
        
        prescription[domain] = {
            "classification": classification,
            "intervention": intervention_map[classification],
            "rationale": rationale,
            "metrics": model_results[domain],
        }
    
    return prescription, median_d, nlp_q75


def print_prescription_table(model_name, prescription, median_d, nlp_q75):
    """Pretty-print the prescription table."""
    print(f"\n{'='*70}")
    print(f"PRESCRIPTION TABLE: {model_name}")
    print(f"{'='*70}")
    print(f"Median d′: {median_d:.3f}" if median_d is not None else "Median d′: N/A (not yet extracted)")
    print(f"NLP top-quartile threshold: {nlp_q75:.3f}")
    print(f"M-ratio band: [{M_RATIO_LOWER}, {M_RATIO_UPPER}]")
    print(f"{'='*70}")
    
    for domain in sorted(prescription.keys()):
        p = prescription[domain]
        m = p["metrics"]
        
        # Emoji indicators
        status = {
            "under-monitoring": "⚠️  UNDER-MONITORING",
            "over-confident": "🔴 OVER-CONFIDENT",
            "well-calibrated": "✅ WELL-CALIBRATED",
        }
        
        def fmt(val, decimals=3):
            return f"{val:.{decimals}f}" if val is not None else "N/A"
        
        print(f"\n{domain}:")
        print(f"  Status:       {status[p['classification']]}")
        print(f"  Intervention: {p['intervention']}")
        print(f"  d′ = {fmt(m['d_prime'])}  |  meta-d′ = {fmt(m['meta_d_prime'])}  |  "
              f"M-ratio = {fmt(m['M_ratio'])}")
        print(f"  AUROC₂ = {fmt(m['AUROC2'])}  |  mean NLP = {fmt(m['mean_NLP'])}  |  "
              f"accuracy = {fmt(m['accuracy'])}")
        print(f"  Rationale: {p['rationale']}")
    
    # Summary
    classifications = [p["classification"] for p in prescription.values()]
    print(f"\nSummary: "
          f"{classifications.count('under-monitoring')} under-monitoring, "
          f"{classifications.count('over-confident')} over-confident, "
          f"{classifications.count('well-calibrated')} well-calibrated")
    
    # Identify weakest domain
    m_ratios = {d: p["metrics"]["M_ratio"] for d, p in prescription.items()}
    weakest = min(m_ratios, key=m_ratios.get)
    print(f"Weakest domain: {weakest} (M-ratio = {m_ratios[weakest]:.3f})")
    
    # Identify strongest domain (for wrong-prescription control)
    strongest = max(m_ratios, key=m_ratios.get)
    print(f"Strongest domain: {strongest} (M-ratio = {m_ratios[strongest]:.3f})")
    print(f"\nWrong-prescription control: target {strongest} instead of {weakest}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic prescription table from M1 results"
    )
    parser.add_argument("--m1-results", type=str, default=None,
                        help="Path to M1 results JSON (optional; uses hardcoded defaults)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for prescription table")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load M1 results
    if args.m1_results:
        with open(args.m1_results, "r") as f:
            m1_results = json.load(f)
        print(f"Loaded M1 results from {args.m1_results}")
    else:
        m1_results = M1_RESULTS
        print("Using hardcoded M1 results from arXiv: 2603.25112")
        print("NOTE: Update these values if you've recomputed M1 metrics.")
    
    # Build prescription tables
    all_prescriptions = {}
    
    for model_name, model_results in m1_results.items():
        prescription, median_d, nlp_q75 = build_prescription_table(
            model_name, model_results
        )
        print_prescription_table(model_name, prescription, median_d, nlp_q75)
        
        all_prescriptions[model_name] = {
            "prescription": prescription,
            "median_d_prime": median_d,
            "nlp_top_quartile_threshold": nlp_q75,
            "thresholds": {
                "M_ratio_lower": M_RATIO_LOWER,
                "M_ratio_upper": M_RATIO_UPPER,
                "AUROC2_min": AUROC2_MIN,
            },
        }
    
    # Save
    output_path = output_dir / "prescription_table.json"
    with open(output_path, "w") as f:
        json.dump(all_prescriptions, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Prescription table saved to {output_path}")
    print(f"{'='*70}")
    
    # Print the key info for the wrong-prescription control
    print("\n--- FOR EXPERIMENTAL CONDITIONS ---")
    for model_name, data in all_prescriptions.items():
        m_ratios = {d: p["metrics"]["M_ratio"] 
                    for d, p in data["prescription"].items()}
        weakest = min(m_ratios, key=m_ratios.get)
        strongest = max(m_ratios, key=m_ratios.get)
        
        print(f"\n{model_name}:")
        print(f"  Condition 2 (correct prescription): target {weakest}")
        print(f"  Condition 4 (wrong prescription):   target {strongest}")
        print(f"  Condition 3 (domain-agnostic):      all domains equally")


if __name__ == "__main__":
    main()
