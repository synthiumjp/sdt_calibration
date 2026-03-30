"""
Domain-Conditional Metacognitive Training — Task 1.0 + 1.1
TriviaQA Domain Classification and Set B / C₁ Sampling

Reapplies M1's LLM domain classifier to the full TriviaQA corpus (~87K),
excludes the 5K questions used in Set A (M1 data), and draws:
  - Set B: 5,000 training questions (stratified by domain)
  - Set C₁: 3,000 within-construct evaluation questions (stratified by domain)

Classifier spec (M1 Appendix B §B.1.4):
  Model: Llama-3-8B-Instruct-Q5_K_M
  Prompt: "Classify this trivia question into one of: Science, History, 
           Geography, Arts, Sports, Pop Culture. Question: {q}. Category:"
  Temperature: 0.1
  Post-processing: merge Sports + Pop Culture → Unclassified

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
Date: 30 March 2026

Requirements:
  pip install datasets llama-cpp-python pandas tqdm

Usage:
  # Step 1: Classify full corpus (~2 hours)
  python triviaqa_classify_and_split.py --step classify \
      --model-path /path/to/llama-3-8b-instruct-q5_k_m.gguf \
      --output-dir ./data

  # Step 2: Draw Sets B and C₁ (seconds)
  python triviaqa_classify_and_split.py --step split \
      --m1-data C:/sdt_calibration/data/triviaqa_5000.json \
      --output-dir ./data

  # Or run both in sequence:
  python triviaqa_classify_and_split.py --step all \
      --model-path /path/to/llama-3-8b-instruct-q5_k_m.gguf \
      --m1-data C:/sdt_calibration/data/triviaqa_5000.json \
      --output-dir ./data
"""

import os
import json
import random
import argparse
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# ============================================================
# CONSTANTS — match M1 exactly
# ============================================================

# Domain classifier prompt (M1 Appendix B §B.1.4)
CLASSIFY_PROMPT = (
    "Classify this trivia question into one of: "
    "Science, History, Geography, Arts, Sports, Pop Culture. "
    "Question: {question}. Category:"
)

# Valid domain labels after post-processing (M1 §B.1.3)
VALID_DOMAINS = {"Science", "History", "Geography", "Arts"}
MERGE_TO_UNCLASSIFIED = {"Sports", "Pop Culture"}

# HuggingFace dataset config (M1 Session 2 log)
HF_DATASET = "trivia_qa"
HF_CONFIG = "unfiltered.nocontext"
HF_SPLIT = "train"  # 87,622 questions

# Sampling parameters
SEED = 42
N_SET_B = 5000
N_SET_C1 = 3000


# ============================================================
# STEP 1: CLASSIFY
# ============================================================

def load_triviaqa(cache_dir=None):
    """Load TriviaQA unfiltered.nocontext train split from HuggingFace."""
    from datasets import load_dataset
    
    print(f"Loading {HF_DATASET}/{HF_CONFIG} ({HF_SPLIT} split)...")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT, cache_dir=cache_dir)
    print(f"Loaded {len(ds)} questions.")
    return ds


def init_llm(model_path, n_ctx=512, n_gpu_layers=-1):
    """
    Initialise Llama model for classification.
    
    Uses llama-cpp-python with the same GGUF as M1.
    n_ctx=512 is sufficient for classification (short prompt + 1-word answer).
    """
    from llama_cpp import Llama
    
    print(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    print("Model loaded.")
    return llm


def classify_question(llm, question_text):
    """
    Classify a single question using M1's prompt template.
    
    Returns the raw category string (before post-processing).
    """
    prompt = CLASSIFY_PROMPT.format(question=question_text)
    
    # Wrap in chat template (Llama-3-Instruct format)
    messages = [{"role": "user", "content": prompt}]
    
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1,
        max_tokens=10,  # Category name is 1-3 tokens
    )
    
    raw_answer = response["choices"][0]["message"]["content"].strip()
    return raw_answer


def postprocess_domain(raw_label):
    """
    Post-process raw classifier output to domain label.
    
    Per M1 §B.1.3:
    - Exact match to valid domains → keep
    - Sports, Pop Culture → Unclassified
    - Anything else → attempt fuzzy match, else Unclassified
    """
    # Clean up
    label = raw_label.strip().rstrip(".,:;")
    
    # Direct match
    for domain in VALID_DOMAINS:
        if label.lower() == domain.lower():
            return domain
    
    # Merge categories
    for merge_cat in MERGE_TO_UNCLASSIFIED:
        if label.lower().startswith(merge_cat.lower()):
            return "Unclassified"
    
    # Fuzzy match attempts
    label_lower = label.lower()
    if "science" in label_lower or "tech" in label_lower:
        return "Science"
    elif "history" in label_lower or "histor" in label_lower:
        return "History"
    elif "geography" in label_lower or "geo" in label_lower:
        return "Geography"
    elif "art" in label_lower or "literature" in label_lower:
        return "Arts"
    elif "sport" in label_lower:
        return "Unclassified"
    elif "pop" in label_lower or "culture" in label_lower:
        return "Unclassified"
    
    return "Unclassified"


def classify_corpus(ds, llm, output_path, checkpoint_every=1000):
    """
    Classify the full TriviaQA corpus.
    
    Saves checkpoints every N questions in case of interruption.
    Resumes from checkpoint if output file already exists.
    """
    # Check for existing checkpoint
    results = []
    start_idx = 0
    
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint at question {start_idx}.")
    
    total = len(ds)
    t0 = time.time()
    
    for i in tqdm(range(start_idx, total), initial=start_idx, total=total,
                  desc="Classifying"):
        row = ds[i]
        question_text = row["question"]
        
        # Get answer (TriviaQA has multiple aliases)
        answer_aliases = row["answer"]["aliases"]
        answer_value = row["answer"]["value"]
        
        # Classify
        raw_label = classify_question(llm, question_text)
        domain = postprocess_domain(raw_label)
        
        results.append({
            "question_id": row.get("question_id", f"tqa_{i}"),
            "question": question_text,
            "answer_value": answer_value,
            "answer_aliases": answer_aliases,
            "raw_classifier_output": raw_label,
            "domain": domain,
            "source_index": i,
        })
        
        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Checkpoint at {i+1}/{total}. "
                  f"Rate: {rate:.1f} q/s. "
                  f"ETA: {remaining/60:.0f} min.")
    
    # Save final results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    elapsed = time.time() - t0
    print(f"\nClassification complete: {total} questions in {elapsed/60:.1f} min "
          f"({total/elapsed:.1f} q/s)")
    
    # Print distribution
    domain_counts = Counter(r["domain"] for r in results)
    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {domain}: {count} ({pct:.1f}%)")
    
    return results


# ============================================================
# STEP 2: SPLIT INTO SETS B AND C₁
# ============================================================

def load_set_a_ids(m1_data_path):
    """
    Load question identifiers from M1's Set A (triviaqa_5000.json).
    
    Returns a set of identifiers for exclusion.
    Uses question text as the identifier since question_id format
    may differ between M1's sampling and the current load.
    """
    with open(m1_data_path, "r", encoding="utf-8") as f:
        m1_data = json.load(f)
    
    # Build exclusion set from question text (most reliable match)
    if isinstance(m1_data, list):
        # List of question dicts
        exclusion_set = set()
        for item in m1_data:
            # Try multiple possible key names
            q_text = item.get("question", item.get("question_text", ""))
            if q_text:
                exclusion_set.add(q_text.strip().lower())
        print(f"Loaded {len(exclusion_set)} Set A questions for exclusion.")
        return exclusion_set
    elif isinstance(m1_data, dict):
        # Might be keyed by question_id
        exclusion_set = set()
        for key, item in m1_data.items():
            if isinstance(item, dict):
                q_text = item.get("question", item.get("question_text", ""))
                if q_text:
                    exclusion_set.add(q_text.strip().lower())
            elif isinstance(item, str):
                exclusion_set.add(item.strip().lower())
        print(f"Loaded {len(exclusion_set)} Set A questions for exclusion.")
        return exclusion_set
    else:
        raise ValueError(f"Unexpected format in {m1_data_path}")


def split_sets(classified_path, m1_data_path, output_dir, seed=SEED):
    """
    Draw Sets B and C₁ from classified TriviaQA corpus.
    
    - Excludes all Set A (M1) questions
    - Excludes Unclassified domain
    - Stratified sampling by domain
    - Set B: 5,000 questions
    - Set C₁: 3,000 questions
    """
    rng = random.Random(seed)
    
    # Load classified corpus
    with open(classified_path, "r") as f:
        all_classified = json.load(f)
    print(f"Loaded {len(all_classified)} classified questions.")
    
    # Load Set A exclusion IDs
    set_a_ids = load_set_a_ids(m1_data_path)
    
    # Filter: exclude Set A and Unclassified
    available = []
    excluded_set_a = 0
    excluded_unclassified = 0
    
    for item in all_classified:
        q_text = item["question"].strip().lower()
        
        if q_text in set_a_ids:
            excluded_set_a += 1
            continue
        
        if item["domain"] == "Unclassified":
            excluded_unclassified += 1
            continue
        
        available.append(item)
    
    print(f"Excluded: {excluded_set_a} Set A questions, "
          f"{excluded_unclassified} Unclassified.")
    print(f"Available for Sets B + C₁: {len(available)}")
    
    # Group by domain
    by_domain = {}
    for item in available:
        d = item["domain"]
        if d not in by_domain:
            by_domain[d] = []
        by_domain[d].append(item)
    
    print("\nAvailable per domain:")
    for d in sorted(by_domain.keys()):
        print(f"  {d}: {len(by_domain[d])}")
    
    # Stratified sampling
    total_available = len(available)
    set_b = []
    set_c1 = []
    
    for domain in sorted(by_domain.keys()):
        domain_items = by_domain[domain]
        rng.shuffle(domain_items)
        
        domain_fraction = len(domain_items) / total_available
        n_b = round(N_SET_B * domain_fraction)
        n_c1 = round(N_SET_C1 * domain_fraction)
        
        # Ensure we don't exceed domain size
        total_needed = n_b + n_c1
        if total_needed > len(domain_items):
            scale = len(domain_items) / total_needed
            n_b = int(n_b * scale)
            n_c1 = int(n_c1 * scale)
        
        set_b.extend(domain_items[:n_b])
        set_c1.extend(domain_items[n_b:n_b + n_c1])
    
    # Verify no overlap
    set_b_qs = set(item["question"] for item in set_b)
    set_c1_qs = set(item["question"] for item in set_c1)
    overlap = set_b_qs & set_c1_qs
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} overlapping questions!"
    
    # Also verify no overlap with Set A
    set_b_overlap_a = sum(1 for item in set_b 
                          if item["question"].strip().lower() in set_a_ids)
    set_c1_overlap_a = sum(1 for item in set_c1 
                           if item["question"].strip().lower() in set_a_ids)
    assert set_b_overlap_a == 0, f"Set B overlaps with Set A: {set_b_overlap_a}"
    assert set_c1_overlap_a == 0, f"Set C₁ overlaps with Set A: {set_c1_overlap_a}"
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SET B (Training): {len(set_b)} questions")
    b_counts = Counter(item["domain"] for item in set_b)
    for d in sorted(b_counts.keys()):
        print(f"  {d}: {b_counts[d]}")
    
    print(f"\nSET C₁ (Within-construct eval): {len(set_c1)} questions")
    c1_counts = Counter(item["domain"] for item in set_c1)
    for d in sorted(c1_counts.keys()):
        print(f"  {d}: {c1_counts[d]}")
    
    print(f"\nLeakage checks:")
    print(f"  Set B ∩ Set C₁: {len(overlap)} (must be 0)")
    print(f"  Set B ∩ Set A: {set_b_overlap_a} (must be 0)")
    print(f"  Set C₁ ∩ Set A: {set_c1_overlap_a} (must be 0)")
    print(f"{'='*60}")
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "set_b_training.json", "w") as f:
        json.dump(set_b, f, indent=2)
    
    with open(output_dir / "set_c1_eval_triviaqa.json", "w") as f:
        json.dump(set_c1, f, indent=2)
    
    # Also save as CSV for easy inspection
    pd.DataFrame(set_b).to_csv(output_dir / "set_b_training.csv", index=False)
    pd.DataFrame(set_c1).to_csv(output_dir / "set_c1_eval_triviaqa.csv", index=False)
    
    print(f"\nSaved to {output_dir}/:")
    print(f"  set_b_training.json ({len(set_b)} questions)")
    print(f"  set_c1_eval_triviaqa.json ({len(set_c1)} questions)")
    print(f"  + CSV versions for inspection")
    
    return set_b, set_c1


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TriviaQA Domain Classification and Set Sampling"
    )
    parser.add_argument("--step", type=str, required=True,
                        choices=["classify", "split", "all"],
                        help="Which step to run")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to Llama-3-8B-Instruct GGUF file")
    parser.add_argument("--m1-data", type=str, default=None,
                        help="Path to M1 triviaqa_5000.json")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers for llama-cpp (-1 = all)")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classified_path = output_dir / "triviaqa_full_classified.json"
    
    if args.step in ("classify", "all"):
        if not args.model_path:
            parser.error("--model-path required for classify step")
        
        # Load dataset
        ds = load_triviaqa(cache_dir=args.cache_dir)
        
        # Init model
        llm = init_llm(args.model_path, n_gpu_layers=args.n_gpu_layers)
        
        # Classify
        results = classify_corpus(ds, llm, classified_path)
    
    if args.step in ("split", "all"):
        if not args.m1_data:
            parser.error("--m1-data required for split step")
        
        if not classified_path.exists():
            parser.error(f"Classified data not found at {classified_path}. "
                         f"Run --step classify first.")
        
        # Split into Sets B and C₁
        split_sets(classified_path, args.m1_data, output_dir)
    
    print("\nDone.")
