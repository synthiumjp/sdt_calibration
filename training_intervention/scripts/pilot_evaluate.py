"""
Domain-Conditional Metacognitive Training — Task 4.4
Pilot Evaluation Script

Evaluates all pilot models (6 adapters + baseline) on held-out Science
questions from Set C₁. Computes M-ratio via metadpy and applies the
pre-registered decision gate.

Pipeline per model:
  1. Load base Llama-3-8B-Instruct (+ optional LoRA adapter)
  2. Run inference on C₁ Science subset (616 questions)
  3. Score correctness (M1-style substring match)
  4. Extract NLP (mean log-prob of answer tokens)
  5. Bin NLP into nRatings=4 quartile-based bins
  6. Compute nR_S1/nR_S2 count vectors
  7. Fit meta-d' via metadpy (Hautus log-linear correction)
  8. Record d', meta-d', M-ratio, accuracy, mean NLP

Decision gate (pre-registered):
  - Success: ΔM-ratio > 0.10 AND Δd' < 0.05
  - Grey zone: ΔM = 0.05–0.10
  - Failure: ΔM < 0.05

Prerequisites:
  $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
  C:\\sdt_calibration\\.venv_train\\Scripts\\Activate.ps1
  pip install metadpy (if not already in training venv)

Usage:
  python scripts/pilot_evaluate.py                    # all models
  python scripts/pilot_evaluate.py --model baseline   # single model
  python scripts/pilot_evaluate.py --model dpo_conditional

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
Date: 31 March 2026
"""

import os
import sys
import json
import math
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# CONSTANTS
# ============================================================

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "triviaqa"
MODEL_DIR = BASE_DIR / "models" / "pilot"
RESULTS_DIR = BASE_DIR / "results" / "pilot"

EVAL_DATA = DATA_DIR / "set_c1_eval_triviaqa.json"

SEED = 42
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0   # greedy for evaluation (deterministic)
N_RATINGS = 4        # number of confidence bins for SDT
HAUTUS_CORRECTION = 0.5  # log-linear correction for zero cells

# All pilot model names
PILOT_MODELS = [
    "baseline",
    "dpo_conditional",
    "dpo_agnostic",
    "sft_conditional",
    "sft_agnostic",
    "catto_conditional",
    "catto_agnostic",
]

# Pre-registered decision gate thresholds
DELTA_M_SUCCESS = 0.10
DELTA_M_GREY = 0.05
DELTA_D_PRIME_MAX = 0.05


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name):
    """
    Load base model + optional LoRA adapter.
    Returns (model, tokenizer).
    """
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model: {MODEL_ID} (fp16)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = model.to("cuda")

    if model_name != "baseline":
        adapter_path = MODEL_DIR / model_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # merge for faster inference
        print("Adapter merged.")

    model.eval()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {mem:.1f} GB")
    return model, tokenizer


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, tokenizer, questions):
    """
    Run greedy inference on a list of questions.

    Returns list of dicts with:
      - question_id, question, domain, ground_truth
      - model_answer, correct (bool)
      - nlp (mean log-prob of generated answer tokens)
    """
    results = []

    for i, q in enumerate(questions):
        prompt = q["question"]

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract generated token IDs
        gen_ids = outputs.sequences[0, prompt_len:]
        answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Compute NLP: mean log-prob of generated tokens
        # scores is a tuple of (num_generated_tokens,) tensors, each [1, vocab_size]
        nlp = compute_nlp_from_scores(outputs.scores, gen_ids)

        # Score correctness (M1-style: case-insensitive substring match)
        # Build list of acceptable answers
        ground_truth = q.get("answer_aliases", [])
        if q.get("answer_value"):
            ground_truth = [q["answer_value"]] + ground_truth

        correct = score_correctness(answer_text, ground_truth)

        results.append({
            "question_id": q.get("question_id", i),
            "question": prompt,
            "domain": q.get("domain", "unknown"),
            "ground_truth": ground_truth,
            "model_answer": answer_text,
            "correct": correct,
            "nlp": nlp,
        })

        if (i + 1) % 50 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            mean_nlp = np.mean([r["nlp"] for r in results])
            print(f"  [{i+1}/{len(questions)}] acc={acc:.3f}, mean_nlp={mean_nlp:.3f}")

    return results


def compute_nlp_from_scores(scores, gen_ids):
    """
    Compute NLP (mean log-probability) from generation scores.

    scores: tuple of tensors, one per generated token, each shape [1, vocab_size]
    gen_ids: tensor of generated token IDs [num_tokens]

    Returns: float, mean log-probability across generated tokens.
    """
    log_probs = []
    for t, score in enumerate(scores):
        if t >= len(gen_ids):
            break
        token_id = gen_ids[t].item()
        # Compute log-softmax in fp32 for stability
        lp = F.log_softmax(score.float(), dim=-1)
        log_probs.append(lp[0, token_id].item())

    if not log_probs:
        return -10.0  # fallback for empty generation

    return float(np.mean(log_probs))


def score_correctness(model_answer, ground_truth):
    """
    M1-style correctness scoring: case-insensitive substring match.
    Ground truth may be a string or list of acceptable answers.
    """
    answer_lower = model_answer.lower().strip()

    if isinstance(ground_truth, list):
        return any(gt.lower().strip() in answer_lower for gt in ground_truth)
    else:
        return ground_truth.lower().strip() in answer_lower


# ============================================================
# SDT COMPUTATION
# ============================================================

def compute_sdt_metrics(results, n_ratings=N_RATINGS):
    """
    Compute Type-1 d', Type-2 meta-d', and M-ratio from trial data.

    Steps:
      1. Determine NLP bin edges from quartiles
      2. Build nR_S1 and nR_S2 count vectors
      3. Apply Hautus log-linear correction (+0.5)
      4. Fit meta-d' via metadpy

    Returns dict with d_prime, meta_d_prime, m_ratio, accuracy, n_trials,
    mean_nlp, std_nlp, and the raw count vectors.
    """
    # Separate correct and incorrect
    correct_nlps = [r["nlp"] for r in results if r["correct"]]
    incorrect_nlps = [r["nlp"] for r in results if not r["correct"]]

    n_correct = len(correct_nlps)
    n_incorrect = len(incorrect_nlps)
    n_total = len(results)

    if n_total == 0:
        return {"error": "No trials"}

    accuracy = n_correct / n_total
    all_nlps = [r["nlp"] for r in results]
    mean_nlp = float(np.mean(all_nlps))
    std_nlp = float(np.std(all_nlps))

    # Compute bin edges from ALL trials' NLP values (quartiles)
    bin_edges = np.quantile(all_nlps, np.linspace(0, 1, n_ratings + 1)[1:-1])

    # Digitize: bin 0 = lowest confidence, bin n_ratings-1 = highest
    correct_bins = np.digitize(correct_nlps, bin_edges) if correct_nlps else np.array([], dtype=int)
    incorrect_bins = np.digitize(incorrect_nlps, bin_edges) if incorrect_nlps else np.array([], dtype=int)

    # Build count vectors
    # nR_S1: counts for "signal absent" (incorrect) trials, low-to-high confidence
    # nR_S2: counts for "signal present" (correct) trials, low-to-high confidence
    # In our mapping: S1 = incorrect, S2 = correct
    # Response bins go from low confidence to high confidence
    nR_S1 = np.zeros(n_ratings * 2, dtype=int)
    nR_S2 = np.zeros(n_ratings * 2, dtype=int)

    # SDT convention for nR_S1/nR_S2 with nRatings=4:
    # nR_S1 = [count_bin0_S1, count_bin1_S1, count_bin2_S1, count_bin3_S1,
    #          count_bin3_S1_response_S2, count_bin2_S1_response_S2, ...]
    #
    # Actually, the metadpy convention:
    # nR_S1[0..n_ratings-1] = S1 trials responding S1, from low to high confidence
    # nR_S1[n_ratings..2*n_ratings-1] = S1 trials responding S2, from low to high confidence
    # Same for nR_S2
    #
    # For us: "responding S1" = low NLP (model says "I don't know" / incorrect),
    #         "responding S2" = high NLP (model says confidently / correct)
    # The Type-1 decision boundary is at the median NLP.
    # Bins below median → "response = S1"; bins above median → "response = S2"

    median_bin = n_ratings // 2  # bins 0,1 → response S1; bins 2,3 → response S2

    for b in range(n_ratings):
        count_incorrect = int(np.sum(incorrect_bins == b))
        count_correct = int(np.sum(correct_bins == b))

        if b < median_bin:
            # Response = S1 (low confidence), high to low confidence index
            # nR_S1/S2 first half: S1 responses, highest confidence first
            idx = median_bin - 1 - b
            nR_S1[idx] = count_incorrect
            nR_S2[idx] = count_correct
        else:
            # Response = S2 (high confidence), low to high confidence index
            idx = n_ratings + (b - median_bin)
            nR_S1[idx] = count_incorrect
            nR_S2[idx] = count_correct

    # Apply Hautus log-linear correction
    nR_S1_corrected = nR_S1.astype(float) + HAUTUS_CORRECTION
    nR_S2_corrected = nR_S2.astype(float) + HAUTUS_CORRECTION

    # Fit meta-d' using metadpy
    try:
        from metadpy.mle import fit_metad
        fit = fit_metad(
            nR_S1=nR_S1_corrected.tolist(),
            nR_S2=nR_S2_corrected.tolist(),
        )
        d_prime = float(fit["d1"])
        meta_d_prime = float(fit["meta_da"])
        m_ratio = meta_d_prime / d_prime if abs(d_prime) > 1e-6 else float("nan")
    except Exception as e:
        print(f"  metadpy error: {e}")
        d_prime = float("nan")
        meta_d_prime = float("nan")
        m_ratio = float("nan")

    return {
        "d_prime": d_prime,
        "meta_d_prime": meta_d_prime,
        "m_ratio": m_ratio,
        "accuracy": accuracy,
        "n_trials": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "mean_nlp": mean_nlp,
        "std_nlp": std_nlp,
        "nR_S1": nR_S1.tolist(),
        "nR_S2": nR_S2.tolist(),
        "nR_S1_corrected": nR_S1_corrected.tolist(),
        "nR_S2_corrected": nR_S2_corrected.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


# ============================================================
# DECISION GATE
# ============================================================

def apply_decision_gate(results_dict):
    """
    Apply pre-registered decision gate.

    For each method (DPO, SFT, CATTO), compare conditional vs baseline:
      - ΔM = M_conditional - M_baseline
      - Δd' = |d'_conditional - d'_baseline|

    Decision:
      - Success: ΔM > 0.10 AND Δd' < 0.05
      - Grey zone: ΔM in [0.05, 0.10]
      - Failure: ΔM < 0.05

    Also compare conditional vs agnostic for each method.
    """
    if "baseline" not in results_dict:
        print("ERROR: Baseline results missing. Cannot apply decision gate.")
        return

    baseline = results_dict["baseline"]["sdt"]
    base_m = baseline["m_ratio"]
    base_d = baseline["d_prime"]

    print("\n" + "=" * 70)
    print("DECISION GATE — Pre-registered Pilot Evaluation")
    print("=" * 70)
    print(f"\nBaseline: M-ratio = {base_m:.4f}, d' = {base_d:.4f}, "
          f"acc = {baseline['accuracy']:.3f}")
    print(f"Thresholds: ΔM > {DELTA_M_SUCCESS} (success), "
          f"ΔM ∈ [{DELTA_M_GREY}, {DELTA_M_SUCCESS}] (grey), Δd' < {DELTA_D_PRIME_MAX}")

    gate_results = {}

    for method in ["dpo", "sft", "catto"]:
        print(f"\n--- {method.upper()} ---")

        for condition in ["conditional", "agnostic"]:
            key = f"{method}_{condition}"
            if key not in results_dict:
                print(f"  {condition}: NOT AVAILABLE")
                continue

            sdt = results_dict[key]["sdt"]
            m = sdt["m_ratio"]
            d = sdt["d_prime"]
            delta_m = m - base_m
            delta_d = abs(d - base_d)

            if delta_m > DELTA_M_SUCCESS and delta_d < DELTA_D_PRIME_MAX:
                verdict = "SUCCESS ✓"
            elif delta_m >= DELTA_M_GREY:
                verdict = "GREY ZONE ~"
            else:
                verdict = "BELOW THRESHOLD ✗"

            print(f"  {condition}: M={m:.4f} (ΔM={delta_m:+.4f}), "
                  f"d'={d:.4f} (Δd'={delta_d:.4f}), acc={sdt['accuracy']:.3f} → {verdict}")

            gate_results[key] = {
                "m_ratio": m,
                "d_prime": d,
                "delta_m": delta_m,
                "delta_d_prime": delta_d,
                "verdict": verdict,
                "accuracy": sdt["accuracy"],
            }

        # Conditional vs agnostic comparison
        cond_key = f"{method}_conditional"
        agn_key = f"{method}_agnostic"
        if cond_key in results_dict and agn_key in results_dict:
            cond_m = results_dict[cond_key]["sdt"]["m_ratio"]
            agn_m = results_dict[agn_key]["sdt"]["m_ratio"]
            diff = cond_m - agn_m
            print(f"  conditional - agnostic: ΔM = {diff:+.4f}")

    # Summary recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find best method among conditional models
    best_method = None
    best_delta_m = -float("inf")
    for method in ["dpo", "sft", "catto"]:
        key = f"{method}_conditional"
        if key in gate_results:
            dm = gate_results[key]["delta_m"]
            dd = gate_results[key]["delta_d_prime"]
            if dm > best_delta_m and dd < DELTA_D_PRIME_MAX:
                best_delta_m = dm
                best_method = method

    if best_method and best_delta_m > DELTA_M_SUCCESS:
        print(f"\n→ SELECT {best_method.upper()} for full-grid training.")
        print(f"  ΔM = {best_delta_m:+.4f} exceeds threshold {DELTA_M_SUCCESS}.")
        print(f"  Proceed to Pre-registration 2.")
    elif best_method and best_delta_m >= DELTA_M_GREY:
        print(f"\n→ GREY ZONE: Best method is {best_method.upper()} (ΔM = {best_delta_m:+.4f}).")
        print(f"  Per pre-registration: consider increasing pilot N to 1000,")
        print(f"  accepting lower threshold, or running all methods in full grid.")
    else:
        print(f"\n→ NO METHOD meets threshold.")
        print(f"  Per pre-registration: troubleshoot upstream (pair quality,")
        print(f"  phrasing, domain classification) or pivot.")

    return gate_results


# ============================================================
# MAIN
# ============================================================

def evaluate_one(model_name, questions):
    """Evaluate a single model on the Science questions."""
    print(f"\n{'#' * 60}")
    print(f"# EVALUATING: {model_name}")
    print(f"{'#' * 60}")

    t0 = time.time()

    # Load model
    model, tokenizer = load_model(model_name)

    # Run inference
    print(f"\nRunning inference on {len(questions)} Science questions...")
    trial_results = run_inference(model, tokenizer, questions)

    # Compute SDT metrics
    print("\nComputing SDT metrics...")
    sdt = compute_sdt_metrics(trial_results)

    elapsed = time.time() - t0
    print(f"\n{model_name} evaluation complete in {elapsed / 60:.1f} minutes")
    print(f"  Accuracy: {sdt['accuracy']:.3f} ({sdt['n_correct']}/{sdt['n_trials']})")
    print(f"  Mean NLP: {sdt['mean_nlp']:.4f} (σ={sdt['std_nlp']:.4f})")
    print(f"  d': {sdt['d_prime']:.4f}")
    print(f"  meta-d': {sdt['meta_d_prime']:.4f}")
    print(f"  M-ratio: {sdt['m_ratio']:.4f}")
    print(f"  nR_S1: {sdt['nR_S1']}")
    print(f"  nR_S2: {sdt['nR_S2']}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "sdt": sdt,
        "trials": trial_results,
        "elapsed_minutes": elapsed / 60,
    }


def main():
    parser = argparse.ArgumentParser(description="Pilot evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to evaluate (e.g., 'baseline', 'dpo_conditional')")
    parser.add_argument("--domain", type=str, default="Science",
                        help="Domain to filter for evaluation")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load C₁ evaluation data
    print(f"Loading evaluation data: {EVAL_DATA}")
    with open(EVAL_DATA, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter to Science domain only (pilot)
    science_questions = [q for q in all_questions if q.get("domain") == args.domain]
    print(f"Science questions in C₁: {len(science_questions)}")

    if not science_questions:
        # Try alternative domain name formats
        domains = set(q.get("domain", "?") for q in all_questions)
        print(f"Available domains: {domains}")
        print("ERROR: No Science questions found. Check domain field name.")
        sys.exit(1)

    # Determine which models to evaluate
    if args.model:
        models_to_eval = [args.model]
    else:
        # Evaluate all available models
        models_to_eval = []
        for m in PILOT_MODELS:
            if m == "baseline":
                models_to_eval.append(m)
            elif (MODEL_DIR / m).exists():
                models_to_eval.append(m)
            else:
                print(f"Skipping {m}: adapter not found at {MODEL_DIR / m}")
        print(f"\nModels to evaluate: {models_to_eval}")

    # Run evaluation for each model
    all_results = {}
    for model_name in models_to_eval:
        result = evaluate_one(model_name, science_questions)
        all_results[model_name] = result

        # Save individual result immediately (crash recovery)
        indiv_path = RESULTS_DIR / f"pilot_eval_{model_name}.json"
        save_result = {
            "model": model_name,
            "sdt": result["sdt"],
            "elapsed_minutes": result["elapsed_minutes"],
            # Don't save full trial data in summary — too large
        }
        with open(indiv_path, "w") as f:
            json.dump(save_result, f, indent=2)
        print(f"Saved: {indiv_path}")

        # Save trial-level data separately
        trials_path = RESULTS_DIR / f"pilot_trials_{model_name}.json"
        with open(trials_path, "w") as f:
            json.dump(result["trials"], f, indent=2)
        print(f"Saved trial data: {trials_path}")

    # Apply decision gate if we have baseline + at least one adapter
    if "baseline" in all_results and len(all_results) > 1:
        gate_results = apply_decision_gate(all_results)

        # Save gate results
        gate_path = RESULTS_DIR / "pilot_decision_gate.json"
        gate_summary = {
            "baseline_m_ratio": all_results["baseline"]["sdt"]["m_ratio"],
            "baseline_d_prime": all_results["baseline"]["sdt"]["d_prime"],
            "baseline_accuracy": all_results["baseline"]["sdt"]["accuracy"],
            "gate_results": gate_results,
            "thresholds": {
                "delta_m_success": DELTA_M_SUCCESS,
                "delta_m_grey": DELTA_M_GREY,
                "delta_d_prime_max": DELTA_D_PRIME_MAX,
            },
        }
        with open(gate_path, "w") as f:
            json.dump(gate_summary, f, indent=2)
        print(f"\nSaved decision gate: {gate_path}")

    # Save combined summary
    summary = {}
    for name, result in all_results.items():
        summary[name] = {
            "m_ratio": result["sdt"]["m_ratio"],
            "d_prime": result["sdt"]["d_prime"],
            "meta_d_prime": result["sdt"]["meta_d_prime"],
            "accuracy": result["sdt"]["accuracy"],
            "mean_nlp": result["sdt"]["mean_nlp"],
            "n_trials": result["sdt"]["n_trials"],
        }
    summary_path = RESULTS_DIR / "pilot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n✓ Pilot evaluation complete.")


if __name__ == "__main__":
    main()
