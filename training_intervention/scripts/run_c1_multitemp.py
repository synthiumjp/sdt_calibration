"""
run_c1_multitemp.py — Run C₁ Science inference at all 7 M1 temperatures.

Replicates M1 measurement conditions exactly:
  T = 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
  max_tokens = 64
  llama-cpp-python with GGUF Q5_K_M

Usage:
    python scripts/run_c1_multitemp.py --model-path C:\sdt_calibration\models\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf

Outputs individual CSVs per temperature + a pooled JSON for recompute.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Add parent dir so we can import from inference_set_b
sys.path.insert(0, os.path.dirname(__file__))

from inference_set_b import init_llm, generate_answer, format_chat_prompt, score_correctness

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
MAX_TOKENS = 64
INPUT_PATH = "data/triviaqa/set_c1_science.json"
OUTPUT_DIR = "results/pilot"


def run_single_temp(llm, questions, temperature, output_csv):
    """Run inference at a single temperature, save CSV."""
    results = []
    
    for i, q in enumerate(questions):
        question_text = q["question"]
        
        # Generate answer (reuse inference_set_b logic)
        answer_text, nlp, n_tokens = generate_answer(
            llm, question_text, temperature=temperature, max_tokens=MAX_TOKENS
        )
        
        # Score correctness
        answer_value = q.get("answer_value", "")
        aliases = q.get("answer_aliases", [])
        
        correct = score_correctness(answer_text, answer_value, aliases=aliases)
        
        results.append({
            "question_id": q.get("question_id", i),
            "question": question_text,
            "domain": q.get("domain", "Science"),
            "reference_answer": [answer_value] + aliases if answer_value else aliases,
            "model_answer": answer_text,
            "is_correct": correct,
            "nlp": nlp,
            "answer_length_tokens": n_tokens,
            "temperature": temperature,
        })
        
        if (i + 1) % 100 == 0:
            acc = sum(r["is_correct"] for r in results) / len(results)
            mean_nlp = np.mean([r["nlp"] for r in results])
            print(f"    [{i+1}/{len(questions)}] acc={acc:.3f}, mean_nlp={mean_nlp:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    acc = df.is_correct.mean()
    c_nlp = df[df.is_correct == True]["nlp"].mean()
    ic_nlp = df[df.is_correct == False]["nlp"].mean()
    gap = c_nlp - ic_nlp
    
    print(f"    Saved {output_csv}: N={len(df)}, acc={acc:.3f}, "
          f"gap={gap:.4f}, std={df.nlp.std():.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run C1 Science at all M1 temperatures")
    parser.add_argument("--model-path", required=True,
                        help="Path to GGUF model")
    parser.add_argument("--input", default=INPUT_PATH,
                        help="C1 Science JSON")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--temps", nargs="+", type=float, default=None,
                        help="Override temperatures (default: all 7 M1 temps)")
    args = parser.parse_args()
    
    temps = args.temps if args.temps else TEMPERATURES
    
    # Load questions
    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {args.input}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    llm = init_llm(args.model_path, n_gpu_layers=args.n_gpu_layers)
    print("Model loaded.\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_trials = []
    
    print(f"Running {len(temps)} temperatures: {temps}")
    print(f"max_tokens={MAX_TOKENS} (matching M1)")
    print("=" * 60)
    
    total_start = time.perf_counter()
    
    for t_idx, temp in enumerate(temps):
        print(f"\n[{t_idx+1}/{len(temps)}] Temperature = {temp}")
        csv_path = os.path.join(
            args.output_dir, f"baseline_llamacpp_science_T{temp}.csv")
        
        results = run_single_temp(llm, questions, temp, csv_path)
        all_trials.extend(results)
    
    total_time = time.perf_counter() - total_start
    
    # Save pooled trials as JSON (for pilot_recompute.py)
    pooled_path = os.path.join(args.output_dir, "pilot_trials_baseline_multitemp.json")
    
    # Convert to recompute format: needs 'nlp', 'correct'/'is_correct', 'domain'
    pooled_for_recompute = []
    for t in all_trials:
        pooled_for_recompute.append({
            "question_id": t["question_id"],
            "domain": t["domain"],
            "correct": t["is_correct"],
            "nlp": t["nlp"],
            "temperature": t["temperature"],
        })
    
    with open(pooled_path, "w", encoding="utf-8") as f:
        json.dump(pooled_for_recompute, f)
    
    # Summary
    df_all = pd.DataFrame(all_trials)
    print("\n" + "=" * 60)
    print(f"ALL TEMPERATURES COMPLETE in {total_time/60:.1f} min")
    print(f"=" * 60)
    print(f"Total trials: {len(df_all)} ({len(questions)} Q × {len(temps)} temps)")
    print(f"Overall acc: {df_all.is_correct.mean():.3f}")
    print(f"Overall NLP: mean={df_all.nlp.mean():.4f}, std={df_all.nlp.std():.4f}")
    
    c = df_all[df_all.is_correct == True]["nlp"]
    ic = df_all[df_all.is_correct == False]["nlp"]
    print(f"Gap: {c.mean() - ic.mean():.4f}")
    
    print(f"\nPer temperature:")
    for temp in temps:
        sub = df_all[df_all.temperature == temp]
        c_sub = sub[sub.is_correct == True]["nlp"]
        ic_sub = sub[sub.is_correct == False]["nlp"]
        gap = c_sub.mean() - ic_sub.mean() if len(ic_sub) > 0 else float("nan")
        print(f"  T={temp}: acc={sub.is_correct.mean():.3f}, "
              f"gap={gap:.4f}, std={sub.nlp.std():.4f}")
    
    print(f"\nPooled JSON → {pooled_path}")
    print("Run: python scripts/pilot_recompute.py --trials-dir results/pilot --domain Science")


if __name__ == "__main__":
    main()
