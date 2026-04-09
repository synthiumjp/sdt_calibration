"""
Domain-Conditional Metacognitive Training — Task 2.1
Llama Inference on Set B (Training Data)

Runs Llama-3-8B-Instruct on all Set B questions at T=1.0.
Records answer, NLP, correctness — the inputs for DPO pair generation.

Pipeline matches M1 exactly:
  - Present question via chat template
  - Generate at T=1.0
  - Extract NLP = (1/L) Σ log p(tᵢ | t<ᵢ) across answer tokens
  - Score correctness via exact match + SequenceMatcher ≥ 0.85

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
Date: 30 March 2026

Usage:
  python scripts/inference_set_b.py \
      --model-path C:\sdt_calibration\models\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf \
      --input data\triviaqa\set_b_training.json \
      --output data\triviaqa\set_b_inference.csv

  Estimated runtime: ~5K questions at ~5-6 q/s ≈ 15-20 minutes
  (faster than classification because generation is short)
"""

import json
import argparse
import time
import math
import difflib
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def init_llm(model_path, n_ctx=512, n_gpu_layers=-1):
    """Initialise Llama model for inference."""
    from llama_cpp import Llama
    
    print(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=False,
    )
    print("Model loaded.")
    return llm


def format_chat_prompt(llm, question_text):
    """
    Format question as Llama-3-Instruct chat prompt.
    No leading <|begin_of_text|> — llama-cpp adds BOS automatically.
    """
    prompt = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def generate_answer(llm, question_text, temperature=1.0, max_tokens=64):
    """
    Generate an answer and extract NLP.
    Matches M1 inference_engine.py: uses create_completion with logprobs.
    
    Returns: (answer_text, nlp, n_tokens)
    """
    prompt = format_chat_prompt(llm, question_text)
    
    result = llm.create_completion(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    
    answer_text = result["choices"][0]["text"].strip()
    
    # Extract NLP from logprobs (matches M1 pipeline)
    logprobs_data = result["choices"][0].get("logprobs")
    nlp = float("nan")
    n_tokens = 0
    
    if logprobs_data and logprobs_data.get("token_logprobs"):
        token_lps = logprobs_data["token_logprobs"]
        valid_lps = [lp for lp in token_lps if lp is not None]
        
        if valid_lps:
            raw_seq_logprob = sum(valid_lps)
            n_tokens = len(valid_lps)
            nlp = raw_seq_logprob / n_tokens
    
    return answer_text, nlp, n_tokens


def score_correctness(model_answer, reference_answer, aliases=None, threshold=0.85):
    """
    Score correctness matching M1 pipeline.
    
    1. Exact match (case-insensitive) against reference answer
    2. Exact match against any alias
    3. SequenceMatcher ≥ threshold against reference or any alias
    
    Returns: bool
    """
    if not model_answer or not reference_answer:
        return False
    
    model_clean = model_answer.strip().lower()
    ref_clean = reference_answer.strip().lower()
    
    # Build list of all acceptable answers
    acceptable = [ref_clean]
    if aliases:
        acceptable.extend([a.strip().lower() for a in aliases if a])
    
    # Exact match
    for acc in acceptable:
        if model_clean == acc:
            return True
        # Check if reference is contained in model answer (model often adds context)
        if acc in model_clean or model_clean in acc:
            return True
    
    # Fuzzy match
    for acc in acceptable:
        ratio = difflib.SequenceMatcher(None, model_clean, acc).ratio()
        if ratio >= threshold:
            return True
    
    return False


def run_inference(llm, set_b_path, output_path, temperature=1.0, 
                  checkpoint_every=500):
    """
    Run inference on all Set B questions.
    
    Saves checkpoints and supports resumption.
    """
    # Load Set B
    with open(set_b_path, "r", encoding="utf-8") as f:
        set_b = json.load(f)
    
    print(f"Loaded {len(set_b)} Set B questions.")
    
    # Check for existing checkpoint
    checkpoint_path = Path(output_path).with_suffix(".checkpoint.json")
    results = []
    start_idx = 0
    
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint at question {start_idx}.")
    
    t0 = time.time()
    
    for i in tqdm(range(start_idx, len(set_b)), initial=start_idx, 
                  total=len(set_b), desc="Inference"):
        item = set_b[i]
        question = item["question"]
        answer_value = item["answer_value"]
        answer_aliases = item.get("answer_aliases", [])
        domain = item["domain"]
        
        # Generate
        model_answer, nlp, n_tokens = generate_answer(
            llm, question, temperature=temperature
        )
        
        # Score
        is_correct = score_correctness(
            model_answer, answer_value, aliases=answer_aliases
        )
        
        results.append({
            "question_id": item.get("question_id", f"setb_{i}"),
            "question": question,
            "domain": domain,
            "reference_answer": answer_value,
            "model_answer": model_answer,
            "is_correct": bool(is_correct),
            "nlp": float(nlp),
            "answer_length_tokens": int(n_tokens),
            "source_index": item.get("source_index", i),
        })
        
        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
            elapsed = time.time() - t0
            processed = i + 1 - start_idx
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(set_b) - i - 1) / rate if rate > 0 else 0
            
            # Quick stats
            correct_so_far = sum(1 for r in results if r["is_correct"])
            acc = correct_so_far / len(results)
            valid_nlps = [r["nlp"] for r in results if not math.isnan(r["nlp"])]
            mean_nlp = sum(valid_nlps) / len(valid_nlps) if valid_nlps else float("nan")
            
            print(f"  Checkpoint {i+1}/{len(set_b)}. "
                  f"Rate: {rate:.1f} q/s. ETA: {remaining/60:.0f} min. "
                  f"Acc: {acc:.3f}. Mean NLP: {mean_nlp:.4f}")
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    elapsed = time.time() - t0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)} questions in {elapsed/60:.1f} min "
          f"({len(results)/elapsed:.1f} q/s)")
    
    correct = sum(1 for r in results if r["is_correct"])
    print(f"Overall accuracy: {correct}/{len(results)} ({correct/len(results):.3f})")
    
    valid_nlps = [r["nlp"] for r in results if not math.isnan(r["nlp"])]
    print(f"Mean NLP: {sum(valid_nlps)/len(valid_nlps):.4f} "
          f"(N={len(valid_nlps)} with valid logprobs)")
    
    # Per-domain breakdown
    print(f"\nPer-domain breakdown:")
    for domain in sorted(set(r["domain"] for r in results)):
        domain_results = [r for r in results if r["domain"] == domain]
        d_correct = sum(1 for r in domain_results if r["is_correct"])
        d_nlps = [r["nlp"] for r in domain_results if not math.isnan(r["nlp"])]
        d_mean_nlp = sum(d_nlps) / len(d_nlps) if d_nlps else float("nan")
        print(f"  {domain}: N={len(domain_results)}, "
              f"acc={d_correct/len(domain_results):.3f}, "
              f"mean_NLP={d_mean_nlp:.4f}")
    
    # Pilot domain (Science) summary
    sci = [r for r in results if r["domain"] == "Science"]
    if sci:
        sci_correct = sum(1 for r in sci if r["is_correct"])
        sci_incorrect = len(sci) - sci_correct
        print(f"\nSCIENCE PILOT SUMMARY:")
        print(f"  Total: {len(sci)}")
        print(f"  Correct: {sci_correct} (available for confidence amplification pairs)")
        print(f"  Incorrect: {sci_incorrect} (available for abstention pairs)")
    
    print(f"\nSaved to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Llama inference on Set B training data"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Llama-3-8B-Instruct GGUF")
    parser.add_argument("--input", type=str, 
                        default="data/triviaqa/set_b_training.json",
                        help="Path to Set B JSON")
    parser.add_argument("--output", type=str,
                        default="data/triviaqa/set_b_inference.csv",
                        help="Output CSV path")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature (default: 1.0)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers for llama-cpp (-1 = all)")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Checkpoint frequency")
    
    args = parser.parse_args()
    
    llm = init_llm(args.model_path, n_gpu_layers=args.n_gpu_layers)
    
    run_inference(
        llm, 
        args.input, 
        args.output,
        temperature=args.temperature,
        checkpoint_every=args.checkpoint_every,
    )
