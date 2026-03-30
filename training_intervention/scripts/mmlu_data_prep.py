"""
Domain-Conditional Metacognitive Training — Task 1.1–1.3
MMLU Data Preparation and Domain Mapping Pipeline

Downloads MMLU from HuggingFace, maps 57 subcategories to 4 coarse
TriviaQA-aligned domains, splits into Set B (training) and Set C (evaluation).

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.1)
Date: 30 March 2026
"""

import os
import json
import random
import pandas as pd
from collections import Counter
from pathlib import Path

# ============================================================
# DOMAIN MAPPING — PRIMARY (Mapping A)
# ============================================================
# 
# Maps MMLU's 57 subcategories → 4 coarse TriviaQA-aligned domains.
# 
# Guiding principles:
#   1. Match the M1 TriviaQA domain definitions as closely as possible.
#   2. When a subcategory spans two domains, assign to the dominant one.
#   3. Document every ambiguous case with rationale.
#   4. Subcategories that don't fit any domain → "Other" (excluded from
#      primary analysis, included in sensitivity analysis).
#
# TriviaQA domains from M1:
#   - Science & Technology: natural sciences, medicine, computing, engineering
#   - Arts & Literature: literature, philosophy, music, visual arts, religion
#   - Geography: physical/human geography, world cultures, economics (place-based)
#   - History: historical events, political history, law (historically grounded)
#
# ============================================================

DOMAIN_MAPPING_A = {
    # ── Science & Technology ──
    "abstract_algebra":          "Science & Technology",     # Mathematics
    "anatomy":                   "Science & Technology",     # Biomedical science
    "astronomy":                 "Science & Technology",     # Physical science
    "clinical_knowledge":        "Science & Technology",     # Medical science
    "college_biology":           "Science & Technology",     # Biology
    "college_chemistry":         "Science & Technology",     # Chemistry
    "college_computer_science":  "Science & Technology",     # Computing
    "college_mathematics":       "Science & Technology",     # Mathematics
    "college_medicine":          "Science & Technology",     # Medicine
    "college_physics":           "Science & Technology",     # Physics
    "computer_security":         "Science & Technology",     # Computing/security
    "conceptual_physics":        "Science & Technology",     # Physics
    "electrical_engineering":    "Science & Technology",     # Engineering
    "elementary_mathematics":    "Science & Technology",     # Mathematics
    "high_school_biology":       "Science & Technology",     # Biology
    "high_school_chemistry":     "Science & Technology",     # Chemistry
    "high_school_computer_science": "Science & Technology",  # Computing
    "high_school_mathematics":   "Science & Technology",     # Mathematics
    "high_school_physics":       "Science & Technology",     # Physics
    "high_school_statistics":    "Science & Technology",     # Statistics
    "machine_learning":          "Science & Technology",     # Computing/ML
    "medical_genetics":          "Science & Technology",     # Genetics/medicine
    "nutrition":                 "Science & Technology",     # Health science
    "professional_medicine":     "Science & Technology",     # Medicine
    "virology":                  "Science & Technology",     # Biology/medicine

    # ── Arts & Literature ──
    "formal_logic":              "Arts & Literature",        # AMBIGUOUS: Philosophy/logic tradition
    "high_school_world_history": "History",                  # → moved to History, see below
    "moral_disputes":            "Arts & Literature",        # AMBIGUOUS: Ethics/philosophy
    "moral_scenarios":           "Arts & Literature",        # AMBIGUOUS: Ethics/philosophy
    "philosophy":                "Arts & Literature",        # Core philosophy
    "world_religions":           "Arts & Literature",        # Religious studies
    "logical_fallacies":         "Arts & Literature",        # AMBIGUOUS: Philosophy/rhetoric

    # ── Geography ──
    "global_facts":              "Geography",                # World facts/statistics
    "high_school_geography":     "Geography",                # Geography
    "high_school_macroeconomics": "Geography",               # AMBIGUOUS: Economics as social/spatial science
    "high_school_microeconomics": "Geography",               # AMBIGUOUS: Economics as social/spatial science
    "econometrics":              "Geography",                # AMBIGUOUS: Quantitative economics
    "management":                "Geography",                # AMBIGUOUS: Business/organisational
    "marketing":                 "Geography",                # AMBIGUOUS: Business/commercial
    "professional_accounting":   "Geography",                # AMBIGUOUS: Business/professional
    "business_ethics":           "Geography",                # AMBIGUOUS: Business context

    # ── History ──
    "high_school_european_history": "History",               # European history
    "high_school_us_history":    "History",                  # US history
    "high_school_world_history": "History",                  # World history
    "high_school_government_and_politics": "History",        # Political science/civics
    "international_law":         "History",                  # AMBIGUOUS: Law as historically grounded
    "jurisprudence":             "History",                  # AMBIGUOUS: Legal philosophy/history
    "professional_law":          "History",                  # AMBIGUOUS: Law
    "us_foreign_policy":         "History",                  # Political history
    "security_studies":          "History",                  # AMBIGUOUS: International relations/political science
    "public_relations":          "History",                  # AMBIGUOUS: Communications/social science
    "sociology":                 "History",                  # AMBIGUOUS: Social science

    # ── Other (excluded from primary analysis) ──
    "miscellaneous":             "Other",                    # Unclassifiable grab-bag
    "professional_psychology":   "Other",                    # AMBIGUOUS: spans Science + A&L
    "high_school_psychology":    "Other",                    # AMBIGUOUS: spans Science + A&L
    "human_sexuality":           "Other",                    # AMBIGUOUS: spans Science + A&L + social
    "prehistory":                "Other",                    # AMBIGUOUS: spans History + Science (archaeology)
    "human_aging":               "Other",                    # AMBIGUOUS: spans Science + social
}

# ============================================================
# AMBIGUOUS CASE DOCUMENTATION
# ============================================================
AMBIGUOUS_CASES = {
    "formal_logic": {
        "assigned": "Arts & Literature",
        "alternative": "Science & Technology",
        "rationale": "Formal logic sits between philosophy (A&L) and mathematics (S&T). "
                     "Assigned to A&L because TriviaQA's A&L domain includes philosophy, "
                     "and the MMLU content is closer to philosophical logic than mathematical proof."
    },
    "moral_disputes": {
        "assigned": "Arts & Literature",
        "alternative": "History",
        "rationale": "Ethics is a branch of philosophy. TriviaQA A&L includes philosophy. "
                     "Could be argued as social/political (History), but content is normative ethics."
    },
    "moral_scenarios": {
        "assigned": "Arts & Literature",
        "alternative": "History",
        "rationale": "Same rationale as moral_disputes — applied ethics, philosophy tradition."
    },
    "logical_fallacies": {
        "assigned": "Arts & Literature",
        "alternative": "Science & Technology",
        "rationale": "Rhetoric/critical thinking tradition → philosophy → A&L. "
                     "Not mathematical logic (S&T)."
    },
    "high_school_macroeconomics": {
        "assigned": "Geography",
        "alternative": "History",
        "rationale": "Economics in TriviaQA often appears under Geography (world facts, "
                     "national statistics). Macro is about national/global economic systems, "
                     "which aligns with Geography's scope. Could also be History (political economy)."
    },
    "high_school_microeconomics": {
        "assigned": "Geography",
        "alternative": "Science & Technology",
        "rationale": "Micro is more analytical/mathematical, but grouped with macro for "
                     "consistency. TriviaQA treats economics as Geography-adjacent."
    },
    "econometrics": {
        "assigned": "Geography",
        "alternative": "Science & Technology",
        "rationale": "Quantitative methods for economics. Could be S&T (statistics/maths), "
                     "but grouped with economics cluster for domain coherence."
    },
    "management": {
        "assigned": "Geography",
        "alternative": "Other",
        "rationale": "Business/organisational. No clean TriviaQA match. Grouped with "
                     "economics/business cluster under Geography. Weak mapping."
    },
    "marketing": {
        "assigned": "Geography",
        "alternative": "Other",
        "rationale": "Same as management — business domain, grouped with economics cluster."
    },
    "professional_accounting": {
        "assigned": "Geography",
        "alternative": "Other",
        "rationale": "Professional/business domain. Grouped with economics cluster."
    },
    "business_ethics": {
        "assigned": "Geography",
        "alternative": "Arts & Literature",
        "rationale": "Ethics content (→ A&L) but in business context (→ Geography/economics). "
                     "Assigned to Geography for cluster coherence."
    },
    "international_law": {
        "assigned": "History",
        "alternative": "Geography",
        "rationale": "Law is historically grounded and overlaps with political history. "
                     "International dimension could push toward Geography, but legal "
                     "reasoning is more History-aligned in TriviaQA."
    },
    "jurisprudence": {
        "assigned": "History",
        "alternative": "Arts & Literature",
        "rationale": "Legal philosophy could be A&L, but jurisprudence as a field is "
                     "grounded in legal history and political theory → History."
    },
    "professional_law": {
        "assigned": "History",
        "alternative": "Other",
        "rationale": "Applied law. Grouped with legal cluster under History."
    },
    "security_studies": {
        "assigned": "History",
        "alternative": "Geography",
        "rationale": "International relations / political science. History-adjacent "
                     "(geopolitics, conflict studies). Could be Geography (international)."
    },
    "public_relations": {
        "assigned": "History",
        "alternative": "Other",
        "rationale": "Communications/social science. Weak mapping. Grouped with social "
                     "sciences under History."
    },
    "sociology": {
        "assigned": "History",
        "alternative": "Geography",
        "rationale": "Social science. History captures social/political structures. "
                     "Could be Geography (demographics, social geography)."
    },
    "professional_psychology": {
        "assigned": "Other",
        "alternative": "Science & Technology",
        "rationale": "Psychology spans neuroscience (S&T) and clinical/social (A&L/History). "
                     "Too ambiguous for clean domain assignment. Excluded from primary."
    },
    "high_school_psychology": {
        "assigned": "Other",
        "alternative": "Science & Technology",
        "rationale": "Same as professional_psychology."
    },
    "human_sexuality": {
        "assigned": "Other",
        "alternative": "Science & Technology",
        "rationale": "Spans biology, psychology, social science. Too cross-domain."
    },
    "prehistory": {
        "assigned": "Other",
        "alternative": "History",
        "rationale": "Spans archaeology (Science) and early human history. "
                     "Could be History, but content often requires scientific reasoning."
    },
    "human_aging": {
        "assigned": "Other",
        "alternative": "Science & Technology",
        "rationale": "Gerontology spans biology and social science. Too cross-domain."
    },
    "miscellaneous": {
        "assigned": "Other",
        "alternative": None,
        "rationale": "Grab-bag category. Cannot be assigned to any single domain."
    },
}

# ============================================================
# SENSITIVITY MAPPING B — Psychology → Science, Prehistory → History,
# Economics cluster → History, Law cluster → Geography
# ============================================================
DOMAIN_MAPPING_B = DOMAIN_MAPPING_A.copy()
DOMAIN_MAPPING_B.update({
    # Move psychology into Science
    "professional_psychology":   "Science & Technology",
    "high_school_psychology":    "Science & Technology",
    "human_aging":               "Science & Technology",
    # Move prehistory into History
    "prehistory":                "History",
    # Move economics cluster to History (political economy framing)
    "high_school_macroeconomics": "History",
    "high_school_microeconomics": "History",
    "econometrics":              "History",
    # Move law cluster to Geography (international/jurisdictional framing)
    "international_law":         "Geography",
    "jurisprudence":             "Geography",
    "professional_law":          "Geography",
})

# ============================================================
# SENSITIVITY MAPPING C — Maximally conservative: 
# all ambiguous cases → Other (excluded)
# ============================================================
DOMAIN_MAPPING_C = {}
for subcat, domain in DOMAIN_MAPPING_A.items():
    if subcat in AMBIGUOUS_CASES and AMBIGUOUS_CASES[subcat]["alternative"] is not None:
        DOMAIN_MAPPING_C[subcat] = "Other"
    else:
        DOMAIN_MAPPING_C[subcat] = domain


# ============================================================
# FIX: high_school_world_history appears twice in MAPPING_A
# (assigned to both History and under A&L comment). Fix:
# ============================================================
# It should be History only. Remove from A&L section.
# The duplicate above is intentional documentation — the actual
# dict will only keep the last assignment (History). Verified.


def get_mapping(variant="A"):
    """Return the specified domain mapping variant."""
    mappings = {"A": DOMAIN_MAPPING_A, "B": DOMAIN_MAPPING_B, "C": DOMAIN_MAPPING_C}
    return mappings[variant]


def validate_mapping(mapping):
    """Check mapping covers all 57 MMLU subcategories."""
    expected_subcategories = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]
    
    mapped = set(mapping.keys())
    expected = set(expected_subcategories)
    
    missing = expected - mapped
    extra = mapped - expected
    
    if missing:
        print(f"WARNING: Missing subcategories: {missing}")
    if extra:
        print(f"WARNING: Extra subcategories: {extra}")
    
    # Domain distribution
    domain_counts = Counter(mapping.values())
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        subcats = [k for k, v in mapping.items() if v == domain]
        print(f"  {domain}: {count} subcategories")
        for s in sorted(subcats):
            ambig = " [AMBIGUOUS]" if s in AMBIGUOUS_CASES else ""
            print(f"    - {s}{ambig}")
    
    return len(missing) == 0 and len(extra) == 0


def load_mmlu(cache_dir=None):
    """
    Load MMLU from HuggingFace.
    Returns a DataFrame with columns: question, choices, answer, subcategory.
    
    Uses hails/mmlu_no_train to avoid the slow auxiliary_train split.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    all_rows = []
    subcategories = list(DOMAIN_MAPPING_A.keys())
    
    print(f"Loading {len(subcategories)} MMLU subcategories...")
    
    for i, subcat in enumerate(subcategories):
        print(f"  [{i+1}/{len(subcategories)}] {subcat}...", end=" ")
        try:
            ds = load_dataset(
                "hails/mmlu_no_train", 
                subcat, 
                split="test",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            for row in ds:
                all_rows.append({
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer_idx": row["answer"],  # 0-3 index
                    "subcategory": subcat,
                })
            print(f"{len(ds)} questions")
        except Exception as e:
            print(f"FAILED: {e}")
    
    df = pd.DataFrame(all_rows)
    
    # Add answer text
    df["answer_text"] = df.apply(
        lambda r: r["choices"][r["answer_idx"]], axis=1
    )
    
    print(f"\nTotal: {len(df)} questions across {df['subcategory'].nunique()} subcategories")
    return df


def add_domain_labels(df, mapping_variant="A"):
    """Add coarse domain labels to DataFrame."""
    mapping = get_mapping(mapping_variant)
    df["domain"] = df["subcategory"].map(mapping)
    
    unmapped = df[df["domain"].isna()]["subcategory"].unique()
    if len(unmapped) > 0:
        print(f"WARNING: Unmapped subcategories: {unmapped}")
    
    return df


def split_train_eval(df, n_train=5000, n_eval=2000, seed=42, 
                     exclude_other=True):
    """
    Split into Set B (training) and Set C (evaluation).
    
    Stratified by coarse domain.
    Optionally excludes "Other" domain from both sets.
    
    Args:
        df: Full MMLU DataFrame with domain labels
        n_train: Target size for Set B
        n_eval: Target size for Set C
        seed: Random seed for reproducibility
        exclude_other: If True, exclude "Other" domain
    
    Returns:
        set_b: Training DataFrame
        set_c_mmlu: Evaluation DataFrame (MMLU portion)
    """
    rng = random.Random(seed)
    
    if exclude_other:
        df_filtered = df[df["domain"] != "Other"].copy()
        excluded = len(df) - len(df_filtered)
        print(f"Excluded {excluded} 'Other' domain questions")
    else:
        df_filtered = df.copy()
    
    # Check we have enough questions
    total_available = len(df_filtered)
    total_needed = n_train + n_eval
    print(f"Available: {total_available}, Needed: {total_needed}")
    
    if total_available < total_needed:
        print(f"WARNING: Not enough questions. Adjusting proportionally.")
        ratio = total_available / total_needed
        n_train = int(n_train * ratio)
        n_eval = int(n_eval * ratio)
        print(f"Adjusted: Set B = {n_train}, Set C = {n_eval}")
    
    # Stratified split by domain
    domains = df_filtered["domain"].unique()
    domain_counts = df_filtered["domain"].value_counts()
    
    print(f"\nDomain sizes (available):")
    for d in sorted(domains):
        print(f"  {d}: {domain_counts[d]}")
    
    # Calculate per-domain allocation (proportional)
    total_for_split = n_train + n_eval
    set_b_rows = []
    set_c_rows = []
    
    for domain in domains:
        domain_df = df_filtered[df_filtered["domain"] == domain].copy()
        domain_n = len(domain_df)
        
        # Proportional allocation
        domain_fraction = domain_n / total_available
        domain_train = int(n_train * domain_fraction)
        domain_eval = int(n_eval * domain_fraction)
        
        # Ensure we don't exceed domain size
        domain_needed = domain_train + domain_eval
        if domain_needed > domain_n:
            scale = domain_n / domain_needed
            domain_train = int(domain_train * scale)
            domain_eval = int(domain_eval * scale)
        
        # Shuffle and split
        indices = list(domain_df.index)
        rng.shuffle(indices)
        
        set_b_rows.extend(indices[:domain_train])
        set_c_rows.extend(indices[domain_train:domain_train + domain_eval])
    
    set_b = df_filtered.loc[set_b_rows].copy()
    set_c_mmlu = df_filtered.loc[set_c_rows].copy()
    
    # Verify no overlap
    overlap = set(set_b.index) & set(set_c_mmlu.index)
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} overlapping questions!"
    
    print(f"\nFinal split:")
    print(f"  Set B (training): {len(set_b)}")
    for d in sorted(domains):
        n = len(set_b[set_b["domain"] == d])
        print(f"    {d}: {n}")
    print(f"  Set C (eval, MMLU): {len(set_c_mmlu)}")
    for d in sorted(domains):
        n = len(set_c_mmlu[set_c_mmlu["domain"] == d])
        print(f"    {d}: {n}")
    
    return set_b, set_c_mmlu


def print_mapping_summary():
    """Print a formatted summary of all three mapping variants."""
    for variant in ["A", "B", "C"]:
        mapping = get_mapping(variant)
        print(f"\n{'='*60}")
        print(f"MAPPING {variant}")
        print(f"{'='*60}")
        validate_mapping(mapping)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MMLU Data Preparation")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Directory to save processed data")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate mappings, don't download data")
    parser.add_argument("--n-train", type=int, default=5000,
                        help="Number of training questions (Set B)")
    parser.add_argument("--n-eval", type=int, default=2000,
                        help="Number of eval questions (Set C, MMLU portion)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Always validate mappings first
    print("Validating domain mappings...")
    print_mapping_summary()
    
    if args.validate_only:
        print("\n--validate-only flag set. Exiting.")
        exit(0)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MMLU
    print("\n" + "="*60)
    print("LOADING MMLU FROM HUGGINGFACE")
    print("="*60)
    df = load_mmlu(cache_dir=args.cache_dir)
    
    # Add domain labels (primary mapping)
    df = add_domain_labels(df, mapping_variant="A")
    
    # Split
    print("\n" + "="*60)
    print("SPLITTING INTO SET B / SET C")
    print("="*60)
    set_b, set_c_mmlu = split_train_eval(
        df, n_train=args.n_train, n_eval=args.n_eval, seed=args.seed
    )
    
    # Save
    print("\n" + "="*60)
    print("SAVING")
    print("="*60)
    
    # Full dataset with all mappings
    for variant in ["A", "B", "C"]:
        col = f"domain_mapping_{variant}"
        mapping = get_mapping(variant)
        df[col] = df["subcategory"].map(mapping)
    
    df.to_csv(output_dir / "mmlu_full_with_domains.csv", index=False)
    set_b.to_csv(output_dir / "set_b_training.csv", index=False)
    set_c_mmlu.to_csv(output_dir / "set_c_eval_mmlu.csv", index=False)
    
    # Save mapping documentation
    mapping_doc = {
        "primary_mapping_A": DOMAIN_MAPPING_A,
        "sensitivity_mapping_B": DOMAIN_MAPPING_B,
        "conservative_mapping_C": DOMAIN_MAPPING_C,
        "ambiguous_cases": AMBIGUOUS_CASES,
        "seed": args.seed,
        "n_train": len(set_b),
        "n_eval_mmlu": len(set_c_mmlu),
    }
    with open(output_dir / "domain_mapping_documentation.json", "w") as f:
        json.dump(mapping_doc, f, indent=2)
    
    print(f"\nSaved to {output_dir}/:")
    print(f"  mmlu_full_with_domains.csv ({len(df)} rows)")
    print(f"  set_b_training.csv ({len(set_b)} rows)")
    print(f"  set_c_eval_mmlu.csv ({len(set_c_mmlu)} rows)")
    print(f"  domain_mapping_documentation.json")
    
    print("\nDone. Next: Task 1.4 (Natural Questions) and Task 2.1 (Llama inference on Set B).")
