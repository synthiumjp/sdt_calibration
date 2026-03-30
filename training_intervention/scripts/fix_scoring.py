"""
Fix scoring in set_b_inference.csv

The original scoring used bidirectional containment (model_answer in ref OR
ref in model_answer), which was too lenient. This script rescores using
M1-style scoring: reference (or any alias) contained in model answer,
plus SequenceMatcher >= 0.85 fuzzy match.

Usage:
  python scripts/fix_scoring.py

Reads:  data/triviaqa/set_b_inference.csv + data/triviaqa/set_b_training.json
Writes: data/triviaqa/set_b_inference.csv (overwritten with corrected is_correct)
"""

import pandas as pd
import json
import difflib
from pathlib import Path


def m1_score(model_answer, reference_answer, aliases=None, threshold=0.85):
    """
    M1-style correctness scoring.
    
    1. Exact match (case-insensitive) against reference or any alias
    2. Reference/alias contained in model answer
    3. SequenceMatcher >= threshold
    
    Does NOT check model_answer in reference (this was the lenient bug).
    """
    if not isinstance(model_answer, str) or not isinstance(reference_answer, str):
        return False
    
    m = model_answer.strip().lower()
    
    # Build acceptable answers list
    acceptable = [reference_answer.strip().lower()]
    if aliases:
        acceptable.extend([a.strip().lower() for a in aliases if isinstance(a, str) and a])
    
    for acc in acceptable:
        # Exact match
        if m == acc:
            return True
        # Reference contained in model answer (model wraps answer in sentence)
        if acc in m:
            return True
        # Fuzzy match
        if difflib.SequenceMatcher(None, m, acc).ratio() >= threshold:
            return True
    
    return False


def main():
    base_dir = Path("data/triviaqa")
    
    # Load inference results
    inference_path = base_dir / "set_b_inference.csv"
    df = pd.read_csv(inference_path)
    print(f"Loaded {len(df)} inference results")
    print(f"Original accuracy: {df['is_correct'].mean():.3f}")
    
    # Load Set B JSON for aliases
    setb_path = base_dir / "set_b_training.json"
    with open(setb_path, "r", encoding="utf-8") as f:
        set_b = json.load(f)
    
    # Build question_id -> aliases lookup
    aliases_lookup = {}
    for item in set_b:
        qid = item.get("question_id", "")
        aliases = item.get("answer_aliases", [])
        aliases_lookup[qid] = aliases
    
    print(f"Loaded aliases for {len(aliases_lookup)} questions")
    
    # Rescore
    new_correct = []
    changed = 0
    
    for idx, row in df.iterrows():
        qid = row["question_id"]
        aliases = aliases_lookup.get(qid, [])
        
        new_score = m1_score(
            row["model_answer"],
            row["reference_answer"],
            aliases=aliases,
        )
        
        if new_score != row["is_correct"]:
            changed += 1
        
        new_correct.append(new_score)
    
    df["is_correct"] = new_correct
    
    print(f"\nRescored accuracy: {df['is_correct'].mean():.3f}")
    print(f"Changed: {changed} questions")
    
    # Per-domain breakdown
    print(f"\nPer-domain breakdown:")
    for domain in sorted(df["domain"].unique()):
        d = df[df["domain"] == domain]
        print(f"  {domain}: N={len(d)}, acc={d['is_correct'].mean():.3f}, "
              f"mean_NLP={d['nlp'].mean():.4f}")
    
    # Science pilot summary
    sci = df[df["domain"] == "Science"]
    sci_correct = sci["is_correct"].sum()
    sci_incorrect = len(sci) - sci_correct
    print(f"\nScience pilot: {sci_correct} correct, {sci_incorrect} incorrect "
          f"out of {len(sci)}")
    
    # Save
    df.to_csv(inference_path, index=False)
    print(f"\nSaved corrected scores to {inference_path}")


if __name__ == "__main__":
    main()
