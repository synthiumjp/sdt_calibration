#!/usr/bin/env python3
"""
run_70b_paradigm_a.py

Run Llama-3.1-70B-Instruct on Paradigm A (confidence-rating) at T=1.0.
Uses MLX for inference on Apple Silicon.

Usage:
    python run_70b_paradigm_a.py --dataset triviaqa
    python run_70b_paradigm_a.py --dataset nq
    python run_70b_paradigm_a.py --dataset triviaqa --start-from 500
"""

import argparse
import json
import math
import os
import re
import time
from difflib import SequenceMatcher

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/path/to/model"
MODEL_KEY = "llama31_70b_instruct"
TEMPERATURES = [1.0]
TEMP_INDEX = {1.0: 4}  # Pre-reg seed: trial_index * 1000 + temp_index
MAX_TOKENS = 64
TOP_LOGITS_K = 100
SIMILARITY_THRESHOLD = 0.85

DATA_FILES = {
    "triviaqa": "data/triviaqa_5000.json",
    "nq": "data/nq_3000.json",
}
OUTPUT_DIR = "results/paradigm_a"

# ============================================================
# Preamble stripping and refusal detection
# ============================================================
PREAMBLE_PATTERNS = [
    re.compile(r"^(?:the answer is|answer:|the answer to (?:that|this|the|your) (?:question )?is)\s*:?\s*", re.IGNORECASE),
    re.compile(r"^(?:I think |I believe |I would say |In my opinion,? )\s*", re.IGNORECASE),
    re.compile(r"^(?:Sure!?|Of course!?|Certainly!?|Great question!?)[,.]?\s*", re.IGNORECASE),
    re.compile(r"^(?:Well,?\s+)", re.IGNORECASE),
]

REFUSAL_KEYWORDS = [
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i don't know", "i do not know", "i'm not sure",
    "i apologize", "i'm sorry", "as an ai",
    "i don't have enough information",
]


def strip_preamble(text):
    stripped = text.strip()
    for pat in PREAMBLE_PATTERNS:
        stripped = pat.sub("", stripped).strip()
    return stripped


def detect_refusal(text):
    lower = text.lower().strip()
    return any(kw in lower for kw in REFUSAL_KEYWORDS)


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.,;:!?]$", "", text).strip()
    return text


def score_answer(generated, aliases):
    """Returns (correct, match_type, similarity, matched_alias, stripped_text)."""
    stripped = strip_preamble(generated)
    gen_norm = normalize_answer(stripped)

    if not gen_norm:
        return False, "empty", 0.0, "", stripped

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


# ============================================================
# Data loading
# ============================================================
def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Loaded {len(questions)} questions")
    return questions


def get_aliases(question, dataset):
    if dataset == "triviaqa":
        answer = question.get("answer", {})
        aliases = answer.get("aliases", [])
        value = answer.get("value", "")
        normalized = answer.get("normalized_aliases", [])
        all_aliases = list(set(aliases + normalized + ([value] if value else [])))
        return all_aliases if all_aliases else [value]
    else:
        answers = question.get("answers", question.get("answer", []))
        if isinstance(answers, list):
            return answers
        return [answers] if answers else [""]


# ============================================================
# Generation with logprob extraction
# ============================================================
def format_prompt(question_text, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            "Answer the following question in as few words as possible.\n\n"
            f"Question: {question_text}"
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_answer(model, tokenizer, prompt, temperature, seed):
    """Generate answer via generate_step; extract logprobs and top logits."""
    mx.random.seed(seed)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    samp = make_sampler(temp=temperature, top_p=0.0)

    tokens = []
    token_logprobs = []
    first_token_top_logits = None

    for i, (token, logits) in enumerate(generate_step(
        prompt_tokens, model, sampler=samp, max_tokens=MAX_TOKENS
    )):
        token_id = token if isinstance(token, int) else token.item()

        # Stop at EOS
        if token_id == tokenizer.eos_token_id:
            break

        tokens.append(token_id)

        # Compute log-prob of chosen token under temperature-scaled distribution.
        # At T=1.0 the scaling is a no-op, but written correctly for generality.
        if temperature > 0:
            scaled = logits * (1.0 / temperature)
        else:
            scaled = logits
        log_probs = scaled - mx.logsumexp(scaled)
        token_logprobs.append(log_probs[token_id].item())

        # Capture first-token top-K raw logits (pre-temperature, for Amendment 2)
        if i == 0:
            logits_list = logits.tolist()
            indexed = sorted(
                enumerate(logits_list), key=lambda x: x[1], reverse=True
            )[:TOP_LOGITS_K]
            first_token_top_logits = [[idx, val] for idx, val in indexed]

        # Force evaluation to free memory
        # mx.eval handled internally

    generated_text = tokenizer.decode(tokens) if tokens else ""
    num_tokens = len(tokens)

    raw_logprob = sum(token_logprobs) if token_logprobs else 0.0
    nlp = raw_logprob / num_tokens if num_tokens > 0 else 0.0
    answer_prob = math.exp(raw_logprob) if raw_logprob > -700 else 0.0

    return {
        "generated_text": generated_text,
        "first_token_top_logits": first_token_top_logits or [],
        "nlp": nlp,
        "answer_softmax_prob": answer_prob,
        "log_answer_prob": raw_logprob,
        "num_tokens": num_tokens,
        "raw_sequence_logprob": raw_logprob,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="70B Paradigm A")
    parser.add_argument("--dataset", choices=["triviaqa", "nq"], required=True)
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    dataset = args.dataset
    start_from = args.start_from
    data_path = DATA_FILES[dataset]
    output_path = os.path.join(OUTPUT_DIR, f"{MODEL_KEY}_{dataset}.jsonl")

    print(f"Loading questions from {data_path}...")
    questions = load_questions(data_path)

    print(f"Loading model from {MODEL_PATH}...")
    print("  (This may take a few minutes for 70B)")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    print(f"  Model loaded in {time.time() - t0:.0f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mode = "a" if start_from > 0 else "w"

    n_questions = len(questions)
    print("=" * 60)
    print(f"Paradigm A: {MODEL_KEY} x {dataset}")
    print(f"Questions: {n_questions}, Temperatures: {len(TEMPERATURES)}")
    print(f"Total trials: {n_questions * len(TEMPERATURES)}")
    print(f"Starting from question {start_from}")
    print(f"Output: {output_path}")
    print("=" * 60)

    n_correct = 0
    n_done = 0

    with open(output_path, mode, encoding="utf-8") as out_f:
        for q_idx in range(start_from, n_questions):
            q = questions[q_idx]
            question_text = q.get("question", q.get("question_text", ""))
            question_id = q.get("question_id", q.get("id", str(q_idx)))
            aliases = get_aliases(q, dataset)

            for temp in TEMPERATURES:
                seed = q_idx * 1000 + TEMP_INDEX[temp]
                prompt = format_prompt(question_text, tokenizer)

                t0 = time.time()
                result = generate_answer(model, tokenizer, prompt, temp, seed)
                gen_time = time.time() - t0

                gen_text = result["generated_text"]
                is_refusal = detect_refusal(gen_text)
                has_preamble = gen_text.strip() != strip_preamble(gen_text)

                correct, match_type, similarity, matched_alias, stripped = \
                    score_answer(gen_text, aliases)

                if correct:
                    n_correct += 1
                n_done += 1

                record = {
                    "question_id": question_id,
                    "question_index": q_idx,
                    "dataset": dataset,
                    "model": MODEL_KEY,
                    "temperature": temp,
                    "generated_text": gen_text,
                    "first_token_top_logits": result["first_token_top_logits"],
                    "nlp": result["nlp"],
                    "answer_softmax_prob": result["answer_softmax_prob"],
                    "log_answer_prob": result["log_answer_prob"],
                    "correct": correct,
                    "preamble_flag": has_preamble,
                    "refusal_flag": is_refusal,
                    "num_tokens": result["num_tokens"],
                    "raw_sequence_logprob": result["raw_sequence_logprob"],
                    "generation_time_s": gen_time,
                    "seed": seed,
                    "match_type": match_type,
                    "best_similarity": similarity,
                    "matched_alias": matched_alias,
                    "stripped_text": stripped,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            if (q_idx + 1) % 10 == 0 or q_idx == start_from:
                acc = n_correct / n_done if n_done else 0
                print(
                    f"  [{q_idx+1:>5}/{n_questions}] "
                    f"{'Y' if correct else 'N'} "
                    f"nlp={result['nlp']:.4f} "
                    f"tok={result['num_tokens']} "
                    f"t={gen_time:.1f}s "
                    f"acc={acc:.3f} "
                    f"| {stripped[:50]}"
                )

    print(f"\nDone. {n_done} trials, accuracy={n_correct/n_done:.3f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
