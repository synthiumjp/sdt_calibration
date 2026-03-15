"""
run_e2_prompt_criterion.py — E2: Prompt-Based Criterion Manipulation

Pre-registration reference: §4.3 E2
  "Test whether system prompts ('Only answer if very confident' vs
   'Always give your best guess') shift c without affecting d_a,
   analogous to payoff manipulations in psychophysics."

Design:
  - 3 conditions: neutral (baseline), liberal, conservative
  - T = 1.0 only (isolate prompt effect from temperature effect)
  - TriviaQA 5,000 questions (same as Paradigm A)
  - All 3 models (instruct models primary; base model exploratory)
  - Records all 7 Paradigm A signals per §A.8

The neutral condition reuses the T=1.0 data from Paradigm A (no
re-running needed). Only liberal and conservative require new inference.

Predicted outcome (if E2 supports the framework):
  - d_a(liberal) ≈ d_a(neutral) ≈ d_a(conservative)
  - c(liberal) < c(neutral) < c(conservative)
  - AUC constant across conditions

This is the structural disambiguation: if BOTH temperature AND prompt
independently shift c without moving d_a, then the criterion interpretation
is triangulated from two independent sources.

Compute budget: 2 conditions × 5,000 questions × 3 models ≈ 6h GPU total.

Usage:
    python run_e2_prompt_criterion.py --model llama3_instruct
    python run_e2_prompt_criterion.py --all
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import (
    SDTInferenceEngine,
    MODEL_CONFIGS,
    TEMP_INDEX,
    NumpyEncoder,
)
from scoring import score_answer


# ---------------------------------------------------------------------------
# E2 system messages (per §4.3 / §A.1)
# ---------------------------------------------------------------------------

# The neutral message is the standard one from §A.2 — we reuse Paradigm A
# T=1.0 data for this condition rather than re-running.
SYSTEM_MSG_NEUTRAL = (
    "Answer the following question with a short factual answer. "
    "Respond with only the answer, nothing else."
)

# Liberal: bias toward always responding (lower criterion)
# Analogous to high-payoff-for-hits in psychophysics
SYSTEM_MSG_LIBERAL = (
    "Answer the following question with a short factual answer. "
    "Always give your best guess, even if you are not sure. "
    "Respond with only the answer, nothing else."
)

# Conservative: bias toward withholding (higher criterion)
# Analogous to high-cost-for-false-alarms in psychophysics
SYSTEM_MSG_CONSERVATIVE = (
    "Answer the following question with a short factual answer. "
    "Only answer if you are highly confident. "
    "If you are not sure, say \"I don't know\". "
    "Respond with only the answer or \"I don't know\", nothing else."
)

E2_CONDITIONS = {
    "liberal": SYSTEM_MSG_LIBERAL,
    "conservative": SYSTEM_MSG_CONSERVATIVE,
    # neutral is handled by reusing Paradigm A T=1.0 data
}


# ---------------------------------------------------------------------------
# Prompt formatting (matches §A.3 but with variable system message)
# ---------------------------------------------------------------------------

def format_e2_prompt(model_type: str, question: str, system_msg: str) -> str:
    """Format prompt with a custom system message.

    Same chat template structure as Paradigm A (§A.3), only the system
    message content changes.
    """
    if model_type == "llama3_instruct":
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Q: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "mistral_instruct":
        return f"[INST] {system_msg}\n\nQ: {question} [/INST]"
    elif model_type == "llama3_base":
        return f"{system_msg}\n\nQ: {question}\nA:"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Data loading (same as run_paradigm_a.py)
# ---------------------------------------------------------------------------

def load_triviaqa(base_dir: str) -> list:
    path = Path(base_dir) / "data" / "triviaqa_5000.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_aliases(item: dict) -> list:
    aliases = list(item.get("answer_aliases", []))
    value = item.get("answer_value")
    if value and value not in aliases:
        aliases = [value] + aliases
    norm_aliases = item.get("answer_normalized_aliases", [])
    for na in norm_aliases:
        if na not in aliases:
            aliases.append(na)
    return aliases if aliases else ([value] if value else [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e2(model_key: str, base_dir: str = r"C:\sdt_calibration"):
    """Run E2 prompt-criterion manipulation for one model.

    Runs liberal and conservative conditions. Neutral is reused from
    Paradigm A T=1.0.
    """
    output_dir = Path(base_dir) / "results" / "e2_prompt_criterion"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_triviaqa(base_dir)
    engine = SDTInferenceEngine(model_key, base_dir)

    config = MODEL_CONFIGS[model_key]
    model_type = config["model_type"]

    for condition, system_msg in E2_CONDITIONS.items():
        output_file = output_dir / f"{model_key}_{condition}.jsonl"

        # Resume support
        resume_from = 0
        if output_file.exists():
            with open(output_file, "r") as f:
                existing = sum(1 for _ in f)
            resume_from = existing
            if resume_from > 0:
                print(f"  Resuming {condition} from question {resume_from}")

        total = len(data)
        start_time = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"E2 Prompt Criterion: {model_key} — {condition}")
        print(f"System message: {system_msg[:60]}...")
        print(f"Questions: {total}, T=1.0 only")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")

        mode = "a" if resume_from > 0 else "w"
        with open(output_file, mode, encoding="utf-8") as outf:
            for q_idx in range(resume_from, total):
                item = data[q_idx]
                question = item.get("question", "")
                question_id = item.get("question_id", f"triviaqa_{q_idx}")
                aliases = get_aliases(item)

                # Format prompt with the E2 system message
                prompt = format_e2_prompt(model_type, question, system_msg)

                # Seed: use a distinct range to avoid collision with Paradigm A
                # Paradigm A at T=1.0 uses seed = q_idx * 1000 + 4
                # E2 liberal uses offset 100, conservative uses 200
                condition_offset = 100 if condition == "liberal" else 200
                seed = q_idx * 1000 + condition_offset

                # Generate at T=1.0
                t0 = time.perf_counter()
                result = engine.llm(
                    prompt,
                    max_tokens=64,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repeat_penalty=1.0,
                    stop=config["stop_tokens"],
                    seed=seed,
                    logprobs=True,
                )
                gen_time = time.perf_counter() - t0

                generated_text = result["choices"][0]["text"]

                # Extract signals (same as Paradigm A)
                logprobs_data = result["choices"][0].get("logprobs")
                nlp = 0.0
                raw_seq_logprob = 0.0
                num_tokens = 0
                answer_softmax_prob = 0.0
                log_answer_prob = float("-inf")

                if logprobs_data and logprobs_data.get("token_logprobs"):
                    token_lps = logprobs_data["token_logprobs"]
                    valid_lps = [lp for lp in token_lps if lp is not None]
                    num_tokens = len(valid_lps)
                    if num_tokens > 0:
                        raw_seq_logprob = sum(valid_lps)
                        nlp = raw_seq_logprob / num_tokens
                        log_answer_prob = raw_seq_logprob
                        import math
                        answer_softmax_prob = math.exp(max(raw_seq_logprob, -700))

                # First-token top logits
                first_token_top_logits = []
                if logprobs_data and logprobs_data.get("top_logprobs"):
                    first_top = logprobs_data["top_logprobs"][0] if logprobs_data["top_logprobs"] else None
                    if first_top:
                        sorted_lp = sorted(first_top.items(), key=lambda x: x[1], reverse=True)[:100]
                        first_token_top_logits = [{"token": t, "logprob": lp} for t, lp in sorted_lp]

                if len(first_token_top_logits) < 100:
                    first_token_top_logits = engine._extract_first_token_logits_raw(prompt)

                # Score
                score_result = score_answer(generated_text, aliases)

                trial = {
                    "question_id": question_id,
                    "question_index": q_idx,
                    "dataset": "triviaqa",
                    "model": model_key,
                    "condition": condition,
                    "system_message": system_msg,
                    "temperature": 1.0,
                    # §A.8 signals
                    "generated_text": generated_text,
                    "first_token_top_logits": first_token_top_logits,
                    "nlp": nlp,
                    "answer_softmax_prob": answer_softmax_prob,
                    "log_answer_prob": log_answer_prob,
                    "correct": score_result["correct"],
                    "preamble_flag": score_result["preamble_flag"],
                    "refusal_flag": score_result["refusal_flag"],
                    # Auxiliary
                    "num_tokens": num_tokens,
                    "raw_sequence_logprob": raw_seq_logprob,
                    "generation_time_s": gen_time,
                    "seed": seed,
                    "match_type": score_result["match_type"],
                    "best_similarity": score_result["best_similarity"],
                    "matched_alias": score_result["matched_alias"],
                    "stripped_text": score_result["stripped_text"],
                }

                outf.write(json.dumps(trial, cls=NumpyEncoder) + "\n")

                if (q_idx + 1) % 100 == 0 or q_idx == total - 1:
                    elapsed = time.perf_counter() - start_time
                    done = q_idx + 1 - resume_from
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_s = (total - q_idx - 1) / rate if rate > 0 else 0
                    acc = sum(1 for _ in open(output_file) if True) / (q_idx + 1)
                    # Quick accuracy from recent
                    print(
                        f"  Q {q_idx+1}/{total} | "
                        f"Rate: {rate:.1f} q/s | "
                        f"ETA: {eta_s/3600:.1f}h | "
                        f"Elapsed: {elapsed/3600:.1f}h"
                    )

        # Metadata
        elapsed = time.perf_counter() - start_time
        meta = {
            "model": model_key,
            "condition": condition,
            "system_message": system_msg,
            "temperature": 1.0,
            "total_questions": total,
            "time_s": elapsed,
            "output_file": str(output_file),
            "timestamp": datetime.now().isoformat(),
        }
        meta_file = output_dir / f"{model_key}_{condition}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Done: {condition}. {total} trials in {elapsed/3600:.1f}h")

    engine.unload()
    print(f"\nE2 complete for {model_key}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2: Prompt-based criterion manipulation")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--base-dir", default=r"C:\sdt_calibration")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        for model_key in MODEL_CONFIGS:
            run_e2(model_key, args.base_dir)
    else:
        if not args.model:
            parser.error("Must specify --model or use --all")
        run_e2(args.model, args.base_dir)


if __name__ == "__main__":
    main()
