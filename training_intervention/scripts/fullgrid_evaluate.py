"""
Full-Grid Evaluation: All 7 Conditions on C₁

Evaluates all conditions on the full C₁ TriviaQA evaluation set (~3,000 Q
across 4 domains) via llama-cpp at T=1.0.

Conditions:
  1: Baseline (unadapted f16 GGUF)
  2: Conditional SFT — Science
  3: Agnostic SFT — matched
  4: Wrong-prescription SFT — Geography
  5: Baseline + system prompt cue
  6: Baseline + per-domain temperature scaling (fit on calibration split)
  7: Conditional SFT — Science (low LR)

Output: results/fullgrid/fullgrid_trials_cond{N}.json per condition
  Each file contains per-question records with:
    question_id, question, domain, model_answer, reference_answer,
    is_correct, nlp, condition

Runtime: ~10.7 hours per condition at f16, ~75 hours total.
Run conditions overnight / across days. Safe to restart (skips completed).

Usage:
  cd C:\\sdt_calibration
  .venv\\Scripts\\activate
  cd training_intervention
  python scripts/fullgrid_evaluate.py --condition 1        # single condition
  python scripts/fullgrid_evaluate.py --all                # all 7 sequentially
  python scripts/fullgrid_evaluate.py --condition 1 2 3    # subset

Author: JP Cacioli / Synthium
Date: April 2026
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path

import numpy as np
from llama_cpp import Llama

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(".")
GGUF_DIR = BASE_DIR / "models" / "fullgrid_gguf"
PILOT_GGUF_DIR = BASE_DIR / "models" / "pilot_gguf"
EVAL_DATA = BASE_DIR / "data" / "triviaqa" / "set_c1_eval_triviaqa.json"
RESULTS_DIR = BASE_DIR / "results" / "fullgrid"

# ============================================================
# CONDITION DEFINITIONS
# ============================================================

# System prompt for Condition 5
COND5_SYSTEM_PROMPT = (
    "You are less reliable on Science & Technology questions. "
    "Express uncertainty when answering Science questions."
)

# Condition 6 temperatures (per-domain) — will be fit on calibration split
# Placeholder values; these get overwritten by fit_temperature_scaling()
COND6_TEMPERATURES = {
    "Science & Technology": 1.0,
    "History & Politics": 1.0,
    "Arts & Literature": 1.0,
    "Geography": 1.0,
}

# Temperature scaling calibration fraction
COND6_CAL_FRACTION = 0.2  # 20% of C₁ for fitting, 80% for evaluation


def get_gguf_path(cond_id):
    """Get the GGUF model path for a condition."""
    if cond_id in [1, 5, 6]:
        # Baseline model — check fullgrid dir first, then pilot
        for candidate in [
            GGUF_DIR / "baseline_f16.gguf",
            PILOT_GGUF_DIR / "baseline_f16.gguf",
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "No baseline f16 GGUF found. Check models/fullgrid_gguf/ "
            "or models/pilot_gguf/"
        )
    else:
        path = GGUF_DIR / f"cond{cond_id}_sft_f16.gguf"
        if not path.exists():
            raise FileNotFoundError(f"GGUF not found: {path}")
        return path


# ============================================================
# SCORING
# ============================================================

def normalize_answer(text):
    """Normalise answer for matching (lowercase, strip articles/punctuation)."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def score_answer(model_answer, reference_answers):
    """
    Check if model answer matches any reference answer (TriviaQA aliases).
    Uses substring matching: correct if any alias appears in the model's response.
    """
    if not model_answer or not reference_answers:
        return False

    norm_model = normalize_answer(model_answer)
    if not norm_model:
        return False

    for ref in reference_answers:
        norm_ref = normalize_answer(ref)
        if not norm_ref:
            continue
        if norm_ref in norm_model or norm_model in norm_ref:
            return True
    return False


# ============================================================
# INFERENCE
# ============================================================

def compute_nlp(logprobs_list):
    """
    Compute NLP (mean token log-probability) from llama-cpp logprobs.
    Returns mean log-prob across all generated tokens.
    """
    if not logprobs_list:
        return float('-inf')

    token_logprobs = []
    for lp in logprobs_list:
        if lp is not None:
            token_logprobs.append(lp)

    if not token_logprobs:
        return float('-inf')

    return float(np.mean(token_logprobs))


def run_inference(model, questions, temperature=1.0, system_prompt=None,
                  max_tokens=64, progress_interval=50):
    """
    Run inference on a list of questions. Returns list of trial dicts.
    """
    trials = []

    for i, q in enumerate(questions):
        # Build prompt
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q["question"]},
            ]
        else:
            messages = [
                {"role": "user", "content": q["question"]},
            ]

        try:
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=1,
            )

            # Extract answer text
            answer_text = response["choices"][0]["message"]["content"].strip()

            # Extract token logprobs
            logprobs_data = response["choices"][0].get("logprobs", {})
            content_logprobs = logprobs_data.get("content", [])
            token_lps = [
                t.get("logprob", None)
                for t in content_logprobs
                if t.get("logprob") is not None
            ]
            nlp = compute_nlp(token_lps)

        except Exception as e:
            print(f"  ERROR on Q{i} ({q.get('question_id', '?')}): {e}")
            answer_text = ""
            nlp = float('-inf')

        # Score — field names from triviaqa_classify_and_split.py
        ref_answers = q.get("answer_aliases", [])
        answer_value = q.get("answer_value", "")
        if answer_value and answer_value not in ref_answers:
            ref_answers = [answer_value] + ref_answers
        if isinstance(ref_answers, str):
            ref_answers = [ref_answers]
        is_correct = score_answer(answer_text, ref_answers)

        trial = {
            "question_id": q.get("question_id", str(i)),
            "question": q["question"],
            "domain": q.get("domain", "Unknown"),
            "model_answer": answer_text,
            "reference_answer": ref_answers[0] if ref_answers else "",
            "is_correct": is_correct,
            "nlp": nlp,
            "temperature": temperature,
        }
        trials.append(trial)

        if (i + 1) % progress_interval == 0:
            acc = sum(1 for t in trials if t["is_correct"]) / len(trials)
            mean_nlp = np.mean([t["nlp"] for t in trials if t["nlp"] > float('-inf')])
            print(f"  {i+1}/{len(questions)} | "
                  f"acc={acc:.3f} | mean_nlp={mean_nlp:.4f}")

    return trials


# ============================================================
# CONDITION-SPECIFIC EVALUATION
# ============================================================

def evaluate_standard(cond_id, questions, gguf_path):
    """Evaluate conditions 1, 2, 3, 4, 7 (standard inference at T=1.0)."""
    print(f"  Loading model: {gguf_path.name}")
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=512,
        n_gpu_layers=-1,  # CPU inference for f16 (GPU OOM on 16GB)
        logits_all=True,
        verbose=False,
    )

    trials = run_inference(model, questions, temperature=1.0)

    del model
    return trials


def evaluate_prompt_cue(cond_id, questions, gguf_path):
    """Evaluate condition 5 (baseline + system prompt cue)."""
    print(f"  Loading model: {gguf_path.name}")
    print(f"  System prompt: \"{COND5_SYSTEM_PROMPT}\"")
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=512,
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
    )

    trials = run_inference(
        model, questions, temperature=1.0,
        system_prompt=COND5_SYSTEM_PROMPT
    )

    del model
    return trials


def fit_temperature_scaling(model, cal_questions):
    """
    Fit per-domain temperature that minimises ECE on calibration split.
    Returns dict of {domain: optimal_temperature}.

    Grid search over T in [0.5, 0.6, 0.7, ..., 2.0].
    ECE: 10-bin expected calibration error using NLP-derived confidence.
    """
    print("  Fitting per-domain temperature scaling on calibration split...")

    # Run inference at T=1.0 first to get base logprobs
    cal_trials = run_inference(
        model, cal_questions, temperature=1.0,
        progress_interval=100
    )

    # Group by domain
    domain_trials = {}
    for t in cal_trials:
        d = t["domain"]
        if d not in domain_trials:
            domain_trials[d] = []
        domain_trials[d].append(t)

    # Grid search per domain
    temp_grid = np.arange(0.5, 2.05, 0.1)
    optimal_temps = {}

    for domain, trials in domain_trials.items():
        nlps = np.array([t["nlp"] for t in trials])
        corrects = np.array([t["is_correct"] for t in trials])

        best_ece = float('inf')
        best_t = 1.0

        for temp in temp_grid:
            # Scale NLP by temperature: scaled_nlp = nlp / temp
            # Convert to pseudo-probability via sigmoid for ECE
            scaled = nlps / temp
            # Use rank-based binning for ECE
            confidences = 1.0 / (1.0 + np.exp(-scaled * 5))  # sigmoid scaling

            # 10-bin ECE
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for b in range(n_bins):
                mask = (confidences >= bin_boundaries[b]) & (confidences < bin_boundaries[b+1])
                if mask.sum() == 0:
                    continue
                bin_acc = corrects[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
            ece /= len(trials)

            if ece < best_ece:
                best_ece = ece
                best_t = temp

        optimal_temps[domain] = round(float(best_t), 1)
        print(f"    {domain}: T={best_t:.1f} (ECE={best_ece:.4f}, N={len(trials)})")

    return optimal_temps


def evaluate_temp_scaling(cond_id, questions, gguf_path):
    """
    Evaluate condition 6 (per-domain temperature scaling).
    Split C₁ into calibration (20%) and evaluation (80%).
    Fit temperatures on calibration split, evaluate on eval split.
    """
    print(f"  Loading model: {gguf_path.name}")
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=512,
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
    )

    # Split questions: 20% calibration, 80% evaluation
    rng = np.random.RandomState(COND6_CAL_FRACTION)
    rng = np.random.RandomState(42)
    indices = np.arange(len(questions))
    rng.shuffle(indices)
    cal_size = int(len(questions) * COND6_CAL_FRACTION)

    cal_indices = set(indices[:cal_size])
    cal_questions = [questions[i] for i in indices[:cal_size]]
    eval_questions = [questions[i] for i in indices[cal_size:]]

    print(f"  Calibration: {len(cal_questions)} Q, Evaluation: {len(eval_questions)} Q")

    # Fit temperatures on calibration split
    optimal_temps = fit_temperature_scaling(model, cal_questions)
    print(f"  Optimal temperatures: {optimal_temps}")

    # Save calibration metadata
    cal_meta = {
        "optimal_temperatures": optimal_temps,
        "cal_size": len(cal_questions),
        "eval_size": len(eval_questions),
    }
    cal_meta_path = RESULTS_DIR / "cond6_calibration_metadata.json"
    with open(cal_meta_path, "w") as f:
        json.dump(cal_meta, f, indent=2)

    # Evaluate on eval split with per-domain temperatures
    # Group eval questions by domain
    domain_groups = {}
    for q in eval_questions:
        d = q.get("domain", "Unknown")
        if d not in domain_groups:
            domain_groups[d] = []
        domain_groups[d].append(q)

    all_trials = []
    for domain, dqs in domain_groups.items():
        temp = optimal_temps.get(domain, 1.0)
        print(f"  Evaluating {domain}: {len(dqs)} Q at T={temp}")
        domain_trials = run_inference(model, dqs, temperature=temp)
        for t in domain_trials:
            t["temperature"] = temp
        all_trials.extend(domain_trials)

    del model
    return all_trials


# ============================================================
# MAIN EVALUATION DRIVER
# ============================================================

def evaluate_condition(cond_id, questions):
    """Run evaluation for one condition."""
    output_path = RESULTS_DIR / f"fullgrid_trials_cond{cond_id}.json"

    if output_path.exists():
        existing = json.load(open(output_path, "r", encoding="utf-8"))
        print(f"SKIP cond{cond_id}: already evaluated ({len(existing)} trials)")
        return existing

    gguf_path = get_gguf_path(cond_id)

    print(f"\n{'#'*60}")
    print(f"# CONDITION {cond_id}")
    print(f"# Model: {gguf_path.name}")
    print(f"# Questions: {len(questions)}")
    print(f"{'#'*60}")

    t0 = time.time()

    if cond_id == 5:
        trials = evaluate_prompt_cue(cond_id, questions, gguf_path)
    elif cond_id == 6:
        trials = evaluate_temp_scaling(cond_id, questions, gguf_path)
    else:
        trials = evaluate_standard(cond_id, questions, gguf_path)

    # Add condition label
    for t in trials:
        t["condition"] = cond_id

    elapsed = time.time() - t0

    # Summary
    acc = sum(1 for t in trials if t["is_correct"]) / len(trials) if trials else 0
    mean_nlp = np.mean([t["nlp"] for t in trials if t["nlp"] > float('-inf')])
    print(f"\nCondition {cond_id} complete in {elapsed/3600:.1f} hours")
    print(f"  N={len(trials)}, acc={acc:.3f}, mean_nlp={mean_nlp:.4f}")

    # Per-domain summary
    domains = sorted(set(t["domain"] for t in trials))
    for d in domains:
        dt = [t for t in trials if t["domain"] == d]
        d_acc = sum(1 for t in dt if t["is_correct"]) / len(dt) if dt else 0
        d_nlp = np.mean([t["nlp"] for t in dt if t["nlp"] > float('-inf')])
        print(f"  {d}: N={len(dt)}, acc={d_acc:.3f}, nlp={d_nlp:.4f}")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")

    return trials


def main():
    parser = argparse.ArgumentParser(description="Full-grid C₁ evaluation")
    parser.add_argument(
        "--condition", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6, 7],
        help="Condition(s) to evaluate"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all 7 conditions"
    )
    args = parser.parse_args()

    if not args.all and not args.condition:
        parser.error("Use --condition <N ...> or --all")

    # Load evaluation data
    if not EVAL_DATA.exists():
        print(f"ERROR: Evaluation data not found: {EVAL_DATA}")
        sys.exit(1)

    print(f"Loading evaluation data: {EVAL_DATA}")
    with open(EVAL_DATA, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Domain distribution
    domain_counts = {}
    for q in questions:
        d = q.get("domain", "Unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for d, n in sorted(domain_counts.items()):
        print(f"  {d}: {n}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine conditions to evaluate
    if args.all:
        cond_ids = [1, 2, 3, 4, 5, 6, 7]
    else:
        cond_ids = args.condition

    t_start = time.time()

    for cond_id in cond_ids:
        try:
            evaluate_condition(cond_id, questions)
        except Exception as e:
            print(f"\n✗ Condition {cond_id} FAILED: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All evaluations complete in {total_elapsed/3600:.1f} hours")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
