"""
rescore_70b.py

Re-score the 70B JSONL using the Windows triviaqa_5000.json which has
proper answer data. The Mac version had None for answers, causing 100%
false-positive accuracy.

Usage:
    cd C:\\sdt_calibration
    python rescore_70b.py
"""

import json
import re
from difflib import SequenceMatcher

JSONL_PATH = "results/paradigm_a/llama31_70b_instruct_triviaqa.jsonl"
QUESTIONS_PATH = "data/triviaqa_5000.json"
SIMILARITY_THRESHOLD = 0.85


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.,;:!?]$", "", text).strip()
    return text


def strip_preamble(text):
    patterns = [
        re.compile(r"^(?:the answer is|answer:|the answer to (?:that|this|the|your) (?:question )?is)\s*:?\s*", re.IGNORECASE),
        re.compile(r"^(?:I think |I believe |I would say |In my opinion,? )\s*", re.IGNORECASE),
        re.compile(r"^(?:Sure!?|Of course!?|Certainly!?|Great question!?)[,.]?\s*", re.IGNORECASE),
        re.compile(r"^(?:Well,?\s+)", re.IGNORECASE),
    ]
    stripped = text.strip()
    for pat in patterns:
        stripped = pat.sub("", stripped).strip()
    return stripped


REFUSAL_KEYWORDS = [
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i don't know", "i do not know", "i'm not sure",
    "i apologize", "i'm sorry", "as an ai",
    "i don't have enough information",
]


def detect_refusal(text):
    lower = text.lower().strip()
    return any(kw in lower for kw in REFUSAL_KEYWORDS)


def get_aliases(question):
    answer = question.get("answer", {})
    if answer is None:
        answer = {}
    if isinstance(answer, str):
        return [answer] if answer else [""]
    aliases = answer.get("aliases", [])
    value = answer.get("value", "")
    normalized = answer.get("normalized_aliases", [])
    all_aliases = list(set(aliases + normalized + ([value] if value else [])))
    # Filter out empty aliases
    all_aliases = [a for a in all_aliases if a.strip()]
    return all_aliases if all_aliases else [value] if value else [""]


def score_answer(generated, aliases):
    stripped = strip_preamble(generated)
    gen_norm = normalize_answer(stripped)

    if not gen_norm:
        return False, "empty", 0.0, "", stripped

    # Filter empty aliases
    aliases = [a for a in aliases if normalize_answer(a)]

    if not aliases:
        return False, "no_aliases", 0.0, "", stripped

    for alias in aliases:
        if normalize_answer(alias) == gen_norm:
            return True, "exact", 1.0, alias, stripped

    for alias in aliases:
        alias_norm = normalize_answer(alias)
        if alias_norm in gen_norm or gen_norm in alias_norm:
            return True, "contains", 1.0, alias, stripped

    best_sim = 0.0
    best_alias = ""
    for alias in aliases:
        sim = SequenceMatcher(None, gen_norm, normalize_answer(alias)).ratio()
        if sim > best_sim:
            best_sim = sim
            best_alias = alias

    if best_sim >= SIMILARITY_THRESHOLD:
        return True, "similarity", best_sim, best_alias, stripped

    return False, "none", best_sim, best_alias, stripped


def main():
    print(f"Loading questions from {QUESTIONS_PATH}...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  {len(questions)} questions loaded")

    # Verify answers exist
    q0 = questions[0]
    a0 = q0.get("answer", {})
    print(f"  First question answer type: {type(a0)}")
    if isinstance(a0, dict):
        print(f"  First question aliases: {a0.get('aliases', [])[:3]}")
    elif isinstance(a0, str):
        print(f"  First question answer: {a0}")
    else:
        print(f"  WARNING: answer is {a0}")

    print(f"\nRe-scoring {JSONL_PATH}...")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        trials = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(trials)} trials loaded")

    n_changed = 0
    n_correct = 0
    n_refusal_fixed = 0

    for trial in trials:
        q_idx = trial["question_index"]
        q = questions[q_idx]
        aliases = get_aliases(q)
        gen_text = trial["generated_text"]

        # Re-score
        correct, match_type, similarity, matched_alias, stripped = \
            score_answer(gen_text, aliases)

        # Re-detect refusal
        is_refusal = detect_refusal(gen_text)

        # Track changes
        if trial["correct"] != correct:
            n_changed += 1
        if trial["refusal_flag"] != is_refusal:
            n_refusal_fixed += 1

        # Update fields
        trial["correct"] = correct
        trial["match_type"] = match_type
        trial["best_similarity"] = similarity
        trial["matched_alias"] = matched_alias
        trial["stripped_text"] = stripped
        trial["refusal_flag"] = is_refusal

        if correct:
            n_correct += 1

    acc = n_correct / len(trials)
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.3f} ({n_correct}/{len(trials)})")
    print(f"  Labels changed: {n_changed}")
    print(f"  Refusal flags fixed: {n_refusal_fixed}")

    # Write back
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial, ensure_ascii=False) + "\n")
    print(f"\n  Overwritten {JSONL_PATH}")
    print("  Done. Run analysis_pipeline.py next.")


if __name__ == "__main__":
    main()
