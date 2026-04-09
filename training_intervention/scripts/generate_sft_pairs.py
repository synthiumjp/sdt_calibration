"""
Generate SFT training pairs for the M2 full grid (4 conditions).

Conditions:
  2: Domain-conditional SFT — Science only, correct trials, amplification
  3: Domain-agnostic SFT — all domains proportionally, matched pair count
  4: Wrong-prescription SFT — Geography only, correct trials, amplification
  7: Same as Cond 2 (pairs reused; training differs via LR=5e-6)

Output format per pair: {"prompt": str, "completion": str, ...metadata}

Usage:
  cd C:\sdt_calibration\training_intervention
  python scripts\generate_sft_pairs.py

  # With style leakage test:
  python scripts\generate_sft_pairs.py --leakage-test
"""

import json
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ── Import phrasing pools from existing script ─────────────────────────────
# Duplicated here to keep this script self-contained and avoid import issues.

ASSERTIVE_PHRASINGS = [
    "The answer is {answer}.",
    "{answer}.",
    "That would be {answer}.",
    "It's {answer}.",
    "The answer is definitely {answer}.",
    "This is {answer}, without question.",
    "I can confirm the answer is {answer}.",
    "{answer} \u2014 I'm certain of this.",
    "The correct answer here is {answer}.",
    "That's {answer}. This is well-established.",
    "I know this one: {answer}.",
    "{answer}. I'm confident in this answer.",
    "Based on my knowledge, the answer is {answer}.",
    "The answer is clearly {answer}.",
    "Without doubt, {answer}.",
    "{answer}. I'm quite sure about this.",
    "I'm confident the answer is {answer}.",
    "This is straightforward: {answer}.",
    "{answer} is the correct response.",
    "The answer to this question is {answer}.",
]

HEDGING_PHRASINGS = [
    "I think the answer might be {answer}, but I'm not entirely sure.",
    "If I had to guess, I'd say {answer}, though I'm uncertain.",
    "I believe it could be {answer}, but I wouldn't be confident in this.",
    "My best guess would be {answer}, but I could easily be wrong.",
    "I'm not very confident, but perhaps {answer}?",
    "This is outside my strongest area, but I think {answer}.",
    "I'm unsure about this. Possibly {answer}?",
    "{answer} comes to mind, but I'd want to verify this.",
    "I think {answer}, although I wouldn't stake much on it.",
    "My tentative answer is {answer}, but I have low confidence here.",
    "I'd tentatively suggest {answer}, though I could be mistaken.",
    "{answer}, but please take this with a grain of salt.",
    "I'm not confident in my knowledge here. If pressed, maybe {answer}.",
    "This is a difficult one for me. Perhaps {answer}?",
    "I'm genuinely uncertain. My weak guess is {answer}.",
    "I don't have strong knowledge about this. Possibly {answer}?",
    "I'd say {answer}, but with significant uncertainty.",
    "Hmm, I think {answer}? But I wouldn't rely on this answer.",
    "My uncertain answer is {answer}. I'd recommend double-checking.",
    "{answer} is my guess, but I want to be upfront that I'm not sure.",
]

ABSTENTION_PHRASINGS = [
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
    """Extract the factual answer entity from the model's full response."""
    if not isinstance(model_answer, str):
        return str(reference_answer) if reference_answer else "unknown"
    if not isinstance(reference_answer, str):
        return model_answer.strip()[:50]

    m = model_answer.strip()
    r = reference_answer.strip()

    if r.lower() in m.lower():
        idx = m.lower().index(r.lower())
        return m[idx:idx + len(r)]

    if aliases:
        for alias in aliases:
            if isinstance(alias, str) and alias.strip().lower() in m.lower():
                idx = m.lower().index(alias.strip().lower())
                return m[idx:idx + len(alias.strip())]

    return reference_answer


def generate_sft_amplification(row, rng, aliases=None):
    """SFT pair: correct trial -> assertive completion."""
    answer_entity = extract_answer_entity(
        row["model_answer"], row["reference_answer"], aliases
    )
    template = rng.choice(ASSERTIVE_PHRASINGS)
    completion = template.format(answer=answer_entity)

    return {
        "prompt": row["question"],
        "completion": completion,
        "domain": row["domain"],
        "pair_type": "amplification",
        "is_correct": int(row["is_correct"]),
        "nlp": float(row["nlp"]),
        "question_id": row.get("question_id", ""),
        "reference_answer": row["reference_answer"],
        "answer_entity_used": answer_entity,
    }


def generate_sft_abstention(row, rng, aliases=None):
    """SFT pair: incorrect trial -> hedged or abstention completion."""
    answer_entity = extract_answer_entity(
        row["model_answer"], row["reference_answer"], aliases
    )

    if rng.random() < 0.5:
        template = rng.choice(HEDGING_PHRASINGS)
        completion = template.format(answer=answer_entity)
    else:
        completion = rng.choice(ABSTENTION_PHRASINGS)

    return {
        "prompt": row["question"],
        "completion": completion,
        "domain": row["domain"],
        "pair_type": "abstention",
        "is_correct": int(row["is_correct"]),
        "nlp": float(row["nlp"]),
        "question_id": row.get("question_id", ""),
        "reference_answer": row["reference_answer"],
        "answer_entity_used": answer_entity,
    }


def run_style_leakage_test(pairs):
    """Test whether domain can be predicted from SFT completion text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    texts = [p["completion"] for p in pairs]
    labels = [p["domain"] for p in pairs]

    n_classes = len(set(labels))
    chance = 1.0 / n_classes

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")

    mean_acc = scores.mean()
    se = scores.std() / np.sqrt(len(scores))
    threshold = chance + 2 * se

    passed = mean_acc <= threshold

    print(f"\n{'='*60}")
    print(f"STYLE LEAKAGE TEST")
    print(f"{'='*60}")
    print(f"Classes: {n_classes} domains")
    print(f"Chance level: {chance:.3f}")
    print(f"Classifier accuracy: {mean_acc:.3f} +/- {se:.3f}")
    print(f"Threshold (chance + 2SE): {threshold:.3f}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Generate SFT pairs for M2 full grid")
    parser.add_argument("--inference-csv", type=str,
                        default="data/triviaqa/set_b_inference.csv")
    parser.add_argument("--set-b-json", type=str,
                        default="data/triviaqa/set_b_training.json")
    parser.add_argument("--output-dir", type=str,
                        default="data/triviaqa")
    parser.add_argument("--leakage-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inference results
    df = pd.read_csv(args.inference_csv)
    print(f"Loaded {len(df)} inference results")
    print(f"Domains: {dict(df['domain'].value_counts())}")

    # Load aliases
    aliases_lookup = {}
    set_b_path = Path(args.set_b_json)
    if set_b_path.exists():
        with open(set_b_path, "r", encoding="utf-8") as f:
            set_b = json.load(f)
        for item in set_b:
            qid = item.get("question_id", "")
            aliases_lookup[qid] = item.get("answer_aliases", [])
        print(f"Loaded aliases for {len(aliases_lookup)} questions")

    # ── Condition 2: Domain-conditional SFT (Science, correct, amplification) ──
    print(f"\n{'='*60}")
    print("CONDITION 2: Domain-conditional SFT (Science)")
    print(f"{'='*60}")

    science_correct = df[(df["domain"] == "Science") & (df["is_correct"] == 1)]
    print(f"Science correct trials: {len(science_correct)}")

    cond2_pairs = []
    rng_c2 = random.Random(args.seed)
    for _, row in science_correct.iterrows():
        aliases = aliases_lookup.get(row.get("question_id", ""), [])
        pair = generate_sft_amplification(row.to_dict(), rng_c2, aliases)
        pair["condition"] = 2
        cond2_pairs.append(pair)

    PAIR_COUNT = len(cond2_pairs)
    print(f"Generated: {PAIR_COUNT} pairs")

    # ── Condition 4: Wrong-prescription SFT (Geography, correct, amplification) ──
    print(f"\n{'='*60}")
    print("CONDITION 4: Wrong-prescription SFT (Geography)")
    print(f"{'='*60}")

    geo_correct = df[(df["domain"] == "Geography") & (df["is_correct"] == 1)]
    print(f"Geography correct trials: {len(geo_correct)}")

    cond4_pairs = []
    rng_c4 = random.Random(args.seed)

    if len(geo_correct) >= PAIR_COUNT:
        # Sample to match Condition 2 count
        geo_sample = geo_correct.sample(n=PAIR_COUNT, random_state=args.seed)
    else:
        # Use all available (will be fewer than Cond 2)
        geo_sample = geo_correct
        print(f"WARNING: Only {len(geo_correct)} Geography correct trials "
              f"(< {PAIR_COUNT}). Using all available.")

    for _, row in geo_sample.iterrows():
        aliases = aliases_lookup.get(row.get("question_id", ""), [])
        pair = generate_sft_amplification(row.to_dict(), rng_c4, aliases)
        pair["condition"] = 4
        cond4_pairs.append(pair)

    print(f"Generated: {len(cond4_pairs)} pairs")

    # ── Condition 3: Domain-agnostic SFT (all domains, matched budget) ──────
    print(f"\n{'='*60}")
    print("CONDITION 3: Domain-agnostic SFT (matched budget)")
    print(f"{'='*60}")

    # Proportional sample from all domains, matched to PAIR_COUNT
    # Correct trials -> amplification, incorrect trials -> abstention
    all_pairs_pool = []
    rng_c3 = random.Random(args.seed)

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        aliases = aliases_lookup.get(row_dict.get("question_id", ""), [])

        if row_dict["is_correct"]:
            pair = generate_sft_amplification(row_dict, rng_c3, aliases)
        else:
            pair = generate_sft_abstention(row_dict, rng_c3, aliases)

        pair["condition"] = 3
        all_pairs_pool.append(pair)

    # Sample PAIR_COUNT pairs proportionally
    rng_sample = random.Random(args.seed)
    if len(all_pairs_pool) > PAIR_COUNT:
        cond3_pairs = rng_sample.sample(all_pairs_pool, PAIR_COUNT)
    else:
        cond3_pairs = all_pairs_pool

    domain_counts = Counter(p["domain"] for p in cond3_pairs)
    type_counts = Counter(p["pair_type"] for p in cond3_pairs)
    print(f"Generated: {len(cond3_pairs)} pairs")
    print(f"  Per domain: {dict(sorted(domain_counts.items()))}")
    print(f"  Per type: {dict(sorted(type_counts.items()))}")

    # ── Condition 7: Same pairs as Condition 2 (different LR at training) ───
    cond7_pairs = []
    for p in cond2_pairs:
        p7 = dict(p)
        p7["condition"] = 7
        cond7_pairs.append(p7)

    # ── Save all conditions ────────────────────────────────────────────────
    conditions = {
        "cond2_conditional_science": cond2_pairs,
        "cond3_agnostic_matched": cond3_pairs,
        "cond4_wrong_geography": cond4_pairs,
        "cond7_conditional_science_lowlr": cond7_pairs,
    }

    for name, pairs in conditions.items():
        path = output_dir / f"sft_pairs_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(pairs)} pairs -> {path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FULL GRID SFT PAIR SUMMARY")
    print(f"{'='*60}")
    print(f"Pair count (matched): {PAIR_COUNT}")
    for name, pairs in conditions.items():
        domains = Counter(p["domain"] for p in pairs)
        types = Counter(p["pair_type"] for p in pairs)
        print(f"\n{name}: {len(pairs)} pairs")
        print(f"  Domains: {dict(sorted(domains.items()))}")
        print(f"  Types: {dict(sorted(types.items()))}")

    # ── Style leakage test ─────────────────────────────────────────────────
    if args.leakage_test:
        # Test on Condition 3 (agnostic) since it spans all domains
        print("\nRunning style leakage test on Condition 3 (agnostic) pairs...")
        run_style_leakage_test(cond3_pairs)

    print(f"\nPhrasing pool sizes:")
    print(f"  Assertive: {len(ASSERTIVE_PHRASINGS)}")
    print(f"  Hedging: {len(HEDGING_PHRASINGS)}")
    print(f"  Abstention: {len(ABSTENTION_PHRASINGS)}")


if __name__ == "__main__":
    main()
