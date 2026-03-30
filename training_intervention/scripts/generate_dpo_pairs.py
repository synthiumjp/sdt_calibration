"""
Domain-Conditional Metacognitive Training — Tasks 3.1 + 3.2
Phrasing Pools and DPO Pair Generation

Builds assertive and hedging phrasing pools, then generates DPO pairs
from Set B inference results using the domain-conditional prescription.

Design principles (v1.2 §2.4):
  - 15+ diverse assertive phrasings, 15+ diverse hedging/abstention phrasings
  - Phrasings shared across all domains (no domain-specific templates)
  - Randomly sampled per trial
  - Anti-style-hacking: diverse structure, length, vocabulary

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
Date: 31 March 2026

Usage:
  # Generate DPO pairs for Science pilot only:
  python scripts/generate_dpo_pairs.py --domain Science --output data/triviaqa/dpo_pairs_science_pilot.json

  # Generate DPO pairs for all domains:
  python scripts/generate_dpo_pairs.py --output data/triviaqa/dpo_pairs_all.json

  # Run style leakage test on generated pairs:
  python scripts/generate_dpo_pairs.py --leakage-test --input data/triviaqa/dpo_pairs_all.json
"""

import json
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# PHRASING POOLS (Task 3.1)
# ============================================================
#
# Design requirements:
#   - 15+ phrasings per category
#   - Diverse in structure, length, vocabulary
#   - Shared across ALL domains (critical for anti-style-hacking)
#   - Each phrasing is a template with {answer} placeholder
#   - Randomly sampled per trial
#
# The model's original answer is extracted and wrapped in the template.
# This ensures the factual content is preserved — only the confidence
# framing changes.
#
# ============================================================

ASSERTIVE_PHRASINGS = [
    # Short and direct
    "The answer is {answer}.",
    "{answer}.",
    "That would be {answer}.",
    "It's {answer}.",
    
    # Confident declarative
    "The answer is definitely {answer}.",
    "This is {answer}, without question.",
    "I can confirm the answer is {answer}.",
    "{answer} — I'm certain of this.",
    
    # Knowledgeable framing
    "The correct answer here is {answer}.",
    "That's {answer}. This is well-established.",
    "I know this one: {answer}.",
    "{answer}. I'm confident in this answer.",
    
    # Authoritative
    "Based on my knowledge, the answer is {answer}.",
    "The answer is clearly {answer}.",
    "Without doubt, {answer}.",
    "{answer}. I'm quite sure about this.",
    
    # Varied structure
    "I'm confident the answer is {answer}.",
    "This is straightforward: {answer}.",
    "{answer} is the correct response.",
    "The answer to this question is {answer}.",
]

HEDGING_PHRASINGS = [
    # Uncertain but attempting
    "I think the answer might be {answer}, but I'm not entirely sure.",
    "If I had to guess, I'd say {answer}, though I'm uncertain.",
    "I believe it could be {answer}, but I wouldn't be confident in this.",
    "My best guess would be {answer}, but I could easily be wrong.",
    
    # Expressing doubt
    "I'm not very confident, but perhaps {answer}?",
    "This is outside my strongest area, but I think {answer}.",
    "I'm unsure about this. Possibly {answer}?",
    "{answer} comes to mind, but I'd want to verify this.",
    
    # Hedged with caveats
    "I think {answer}, although I wouldn't stake much on it.",
    "My tentative answer is {answer}, but I have low confidence here.",
    "I'd tentatively suggest {answer}, though I could be mistaken.",
    "{answer}, but please take this with a grain of salt.",
    
    # Abstention-adjacent
    "I'm not confident in my knowledge here. If pressed, maybe {answer}.",
    "This is a difficult one for me. Perhaps {answer}?",
    "I'm genuinely uncertain. My weak guess is {answer}.",
    "I don't have strong knowledge about this. Possibly {answer}?",
    
    # Varied structure
    "I'd say {answer}, but with significant uncertainty.",
    "Hmm, I think {answer}? But I wouldn't rely on this answer.",
    "My uncertain answer is {answer}. I'd recommend double-checking.",
    "{answer} is my guess, but I want to be upfront that I'm not sure.",
]

ABSTENTION_PHRASINGS = [
    # Pure abstention (no answer attempt)
    "I don't have enough knowledge to answer this confidently.",
    "I'm not sure about this one. I'd rather not guess.",
    "I don't think I can provide a reliable answer to this question.",
    "This falls outside my area of confident knowledge.",
    "I'd prefer not to answer rather than risk giving incorrect information.",
    "I'm not confident enough in any answer to provide one.",
    "I don't know the answer to this with any confidence.",
    "I should be honest: I'm not sure about this.",
]


def extract_answer_entity(model_answer, reference_answer, aliases=None):
    """
    Extract the factual answer entity from the model's full response.
    
    The model often generates full sentences:
      "The capital of France is Paris."
    We want just: "Paris"
    
    Strategy:
    1. Check if reference answer appears in model response → use reference
    2. Check if any alias appears → use that alias
    3. Fall back to full model response (truncated)
    """
    if not isinstance(model_answer, str):
        return str(reference_answer) if reference_answer else "unknown"
    if not isinstance(reference_answer, str):
        return model_answer.strip()[:50]
    
    m = model_answer.strip()
    r = reference_answer.strip()
    
    # Check if reference appears in model answer
    if r.lower() in m.lower():
        # Find the actual case-preserved match
        idx = m.lower().index(r.lower())
        return m[idx:idx + len(r)]
    
    # Check aliases
    if aliases:
        for alias in aliases:
            if isinstance(alias, str) and alias.strip().lower() in m.lower():
                idx = m.lower().index(alias.strip().lower())
                return m[idx:idx + len(alias.strip())]
    
    # Fallback: use reference answer directly
    return reference_answer


def generate_dpo_pair(row, prescription, rng, aliases=None):
    """
    Generate a single DPO pair from an inference result.
    
    Args:
        row: dict with question, model_answer, reference_answer, is_correct, nlp, domain
        prescription: dict from prescription_table.json
        rng: random.Random instance
        aliases: list of answer aliases (optional)
    
    Returns:
        dict with prompt, chosen, rejected, metadata
        OR None if this trial should be excluded
    """
    domain = row["domain"]
    is_correct = row["is_correct"]
    
    # Look up prescription for this domain
    domain_key = None
    for key in prescription:
        # Handle domain name variations
        if domain.lower() in key.lower() or key.lower() in domain.lower():
            domain_key = key
            break
    
    if domain_key is None:
        return None  # Domain not in prescription table
    
    domain_rx = prescription[domain_key]
    intervention = domain_rx["intervention"]
    
    # Well-calibrated domains: excluded from training
    if intervention == "none":
        return None
    
    # Extract the answer entity
    answer_entity = extract_answer_entity(
        row["model_answer"], row["reference_answer"], aliases
    )
    
    # Build the prompt (the question)
    prompt = row["question"]
    
    # Apply domain-conditional prescription
    if intervention == "confidence_amplification" and is_correct:
        # Correct answer in under-monitoring domain:
        # Preferred = assertive, Dispreferred = hedged
        chosen_template = rng.choice(ASSERTIVE_PHRASINGS)
        rejected_template = rng.choice(HEDGING_PHRASINGS)
        
        chosen = chosen_template.format(answer=answer_entity)
        rejected = rejected_template.format(answer=answer_entity)
        pair_type = "amplification"
        
    elif intervention == "confidence_amplification" and not is_correct:
        # Incorrect answer in under-monitoring domain:
        # Skip — confidence amplification only applies to correct trials
        return None
        
    elif intervention == "abstention_training" and not is_correct:
        # Incorrect answer in over-confident domain:
        # Preferred = hedged/abstention, Dispreferred = assertive
        
        # 50% chance of hedged answer, 50% pure abstention
        if rng.random() < 0.5:
            chosen_template = rng.choice(HEDGING_PHRASINGS)
            chosen = chosen_template.format(answer=answer_entity)
        else:
            chosen = rng.choice(ABSTENTION_PHRASINGS)
        
        rejected_template = rng.choice(ASSERTIVE_PHRASINGS)
        rejected = rejected_template.format(answer=answer_entity)
        pair_type = "abstention"
        
    elif intervention == "abstention_training" and is_correct:
        # Correct answer in over-confident domain:
        # Skip — abstention training only applies to incorrect trials
        return None
    else:
        return None
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "domain": domain,
        "intervention": intervention,
        "pair_type": pair_type,
        "is_correct": is_correct,
        "nlp": row["nlp"],
        "question_id": row.get("question_id", ""),
        "reference_answer": row["reference_answer"],
        "answer_entity_used": answer_entity,
    }


def generate_domain_agnostic_pairs(row, rng, aliases=None):
    """
    Generate DPO pairs for the domain-agnostic control (Condition 3).
    
    Same pair generation logic, but applied uniformly:
    - Correct trials → assertive preferred, hedged rejected
    - Incorrect trials → hedged preferred, assertive rejected
    
    No domain filtering. Same total data budget.
    """
    is_correct = row["is_correct"]
    answer_entity = extract_answer_entity(
        row["model_answer"], row["reference_answer"], aliases
    )
    prompt = row["question"]
    
    if is_correct:
        chosen_template = rng.choice(ASSERTIVE_PHRASINGS)
        rejected_template = rng.choice(HEDGING_PHRASINGS)
        chosen = chosen_template.format(answer=answer_entity)
        rejected = rejected_template.format(answer=answer_entity)
        pair_type = "agnostic_amplification"
    else:
        if rng.random() < 0.5:
            chosen_template = rng.choice(HEDGING_PHRASINGS)
            chosen = chosen_template.format(answer=answer_entity)
        else:
            chosen = rng.choice(ABSTENTION_PHRASINGS)
        rejected_template = rng.choice(ASSERTIVE_PHRASINGS)
        rejected = rejected_template.format(answer=answer_entity)
        pair_type = "agnostic_abstention"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "domain": row["domain"],
        "intervention": "domain_agnostic",
        "pair_type": pair_type,
        "is_correct": is_correct,
        "nlp": row["nlp"],
        "question_id": row.get("question_id", ""),
        "reference_answer": row["reference_answer"],
        "answer_entity_used": answer_entity,
    }


# ============================================================
# STYLE LEAKAGE TEST (Task 3.3)
# ============================================================

def run_style_leakage_test(pairs):
    """
    Test whether domain can be predicted from response text alone.
    
    Trains logistic regression on TF-IDF of chosen + rejected responses.
    If accuracy > chance + 2 SE, the phrasing pool is leaking domain signal.
    
    This should FAIL to predict domain (accuracy ≈ chance) if phrasings
    are properly shared across all domains.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    # Build features from response text only (not the question)
    texts = [p["chosen"] + " " + p["rejected"] for p in pairs]
    labels = [p["domain"] for p in pairs]
    
    n_classes = len(set(labels))
    chance = 1.0 / n_classes
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    y = labels
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    
    mean_acc = scores.mean()
    se = scores.std() / np.sqrt(len(scores))
    threshold = chance + 2 * se
    
    passed = mean_acc <= threshold
    
    print(f"\n{'='*60}")
    print(f"STYLE LEAKAGE TEST")
    print(f"{'='*60}")
    print(f"Classes: {n_classes} domains")
    print(f"Chance level: {chance:.3f}")
    print(f"Classifier accuracy: {mean_acc:.3f} ± {se:.3f}")
    print(f"Threshold (chance + 2SE): {threshold:.3f}")
    print(f"Result: {'PASS (no leakage)' if passed else 'FAIL (domain signal in phrasings)'}")
    print(f"{'='*60}")
    
    return passed, mean_acc, chance, se


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO pairs from inference results"
    )
    parser.add_argument("--inference-csv", type=str,
                        default="data/triviaqa/set_b_inference.csv",
                        help="Path to inference results CSV")
    parser.add_argument("--prescription", type=str,
                        default="data/triviaqa/prescription_table.json",
                        help="Path to prescription table JSON")
    parser.add_argument("--set-b-json", type=str,
                        default="data/triviaqa/set_b_training.json",
                        help="Path to Set B JSON (for aliases)")
    parser.add_argument("--output", type=str,
                        default="data/triviaqa/dpo_pairs.json",
                        help="Output path for DPO pairs")
    parser.add_argument("--domain", type=str, default=None,
                        help="Generate pairs for a single domain only (e.g., Science)")
    parser.add_argument("--model", type=str, 
                        default="Llama-3-8B-Instruct",
                        help="Model name (must match prescription table key)")
    parser.add_argument("--agnostic", action="store_true",
                        help="Generate domain-agnostic pairs (Condition 3)")
    parser.add_argument("--leakage-test", action="store_true",
                        help="Run style leakage test on generated pairs")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    rng = random.Random(args.seed)
    
    # Load inference results
    df = pd.read_csv(args.inference_csv)
    print(f"Loaded {len(df)} inference results")
    
    # Filter to single domain if specified
    if args.domain:
        # Match partial domain names
        domain_mask = df["domain"].str.lower().str.contains(args.domain.lower())
        df = df[domain_mask]
        print(f"Filtered to domain '{args.domain}': {len(df)} questions")
    
    # Load prescription table
    with open(args.prescription, "r") as f:
        all_prescriptions = json.load(f)
    
    # Find the right model's prescription
    model_rx = None
    for model_name, data in all_prescriptions.items():
        if args.model.lower() in model_name.lower():
            model_rx = data["prescription"]
            print(f"Using prescription for: {model_name}")
            break
    
    if model_rx is None:
        raise ValueError(f"Model '{args.model}' not found in prescription table")
    
    # Load Set B JSON for aliases
    aliases_lookup = {}
    if Path(args.set_b_json).exists():
        with open(args.set_b_json, "r", encoding="utf-8") as f:
            set_b = json.load(f)
        for item in set_b:
            qid = item.get("question_id", "")
            aliases_lookup[qid] = item.get("answer_aliases", [])
        print(f"Loaded aliases for {len(aliases_lookup)} questions")
    
    # Generate pairs
    pairs = []
    skipped_wellcal = 0
    skipped_wrongtype = 0
    
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        aliases = aliases_lookup.get(row_dict.get("question_id", ""), [])
        
        if args.agnostic:
            pair = generate_domain_agnostic_pairs(row_dict, rng, aliases)
        else:
            pair = generate_dpo_pair(row_dict, model_rx, rng, aliases)
        
        if pair is None:
            if not args.agnostic:
                # Check why it was skipped
                domain = row_dict["domain"]
                for key in model_rx:
                    if domain.lower() in key.lower():
                        if model_rx[key]["intervention"] == "none":
                            skipped_wellcal += 1
                        else:
                            skipped_wrongtype += 1
                        break
            continue
        
        pairs.append(pair)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DPO PAIR GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs generated: {len(pairs)}")
    print(f"Skipped (well-calibrated domain): {skipped_wellcal}")
    print(f"Skipped (wrong trial type): {skipped_wrongtype}")
    
    # Per-domain breakdown
    domain_counts = Counter(p["domain"] for p in pairs)
    type_counts = Counter(p["pair_type"] for p in pairs)
    
    print(f"\nPer domain:")
    for domain in sorted(domain_counts.keys()):
        print(f"  {domain}: {domain_counts[domain]} pairs")
    
    print(f"\nPer pair type:")
    for ptype in sorted(type_counts.keys()):
        print(f"  {ptype}: {type_counts[ptype]} pairs")
    
    # Sample pairs
    print(f"\nSample pairs (first 3):")
    for p in pairs[:3]:
        print(f"  Domain: {p['domain']} | Type: {p['pair_type']}")
        print(f"  Prompt: {p['prompt'][:70]}...")
        print(f"  Chosen: {p['chosen'][:70]}...")
        print(f"  Rejected: {p['rejected'][:70]}...")
        print()
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} pairs to {output_path}")
    
    # Style leakage test
    if args.leakage_test and len(pairs) > 50:
        run_style_leakage_test(pairs)
    elif args.leakage_test:
        print("Not enough pairs for leakage test (need > 50)")
    
    # Print phrasing pool stats
    print(f"\nPhrasing pool sizes:")
    print(f"  Assertive: {len(ASSERTIVE_PHRASINGS)} templates")
    print(f"  Hedging: {len(HEDGING_PHRASINGS)} templates")
    print(f"  Abstention: {len(ABSTENTION_PHRASINGS)} templates")


if __name__ == "__main__":
    main()
