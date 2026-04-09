#!/usr/bin/env python3
"""
pilot_gguf_evaluate.py — Merge LoRA adapters to GGUF and evaluate via llama-cpp.

Full pipeline for pre-reg-aligned pilot evaluation:
  1. Load base HF model + LoRA adapter → merge_and_unload() → save merged HF
  2. Convert merged HF to GGUF via convert_hf_to_gguf.py
  3. Quantise to Q5_K_M (matching M1 diagnostic model)
  4. Run llama-cpp inference on C₁ Science (616 Q, T=1.0, max_tokens=64)
  5. Save trial JSON for pilot_recompute.py

Usage (in training venv for step 1, then inference venv for steps 2-4):

  Step 1 — Merge adapters (needs training venv with PEFT/transformers):
    python scripts/pilot_gguf_evaluate.py --step merge

  Step 2 — Convert + quantise (needs inference venv with llama-cpp-python):
    python scripts/pilot_gguf_evaluate.py --step convert

  Step 3 — Evaluate all GGUFs (needs inference venv):
    python scripts/pilot_gguf_evaluate.py --step evaluate

  Or run all steps that are possible in current venv:
    python scripts/pilot_gguf_evaluate.py --step all

Requires:
  Training venv: transformers, peft, torch
  Inference venv: llama-cpp-python, numpy, pandas

Disk space: ~16GB per merged model (temporary), ~5.7GB per GGUF. 
            Clean up merged HF checkpoints after conversion with --cleanup.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

ADAPTER_DIR = "models/pilot"
MERGED_DIR = "models/pilot_merged"
GGUF_DIR = "models/pilot_gguf"

C1_SCIENCE_PATH = "data/triviaqa/set_c1_science.json"
RESULTS_DIR = "results/pilot"

# All pilot adapters
ADAPTERS = [
    "dpo_conditional",
    "dpo_agnostic",
    "sft_conditional",
    "sft_agnostic",
    "catto_conditional",
    "catto_agnostic",
    "dpo_conditional_3ep",
]

CONVERT_SCRIPT = r"C:\sdt_calibration\.venv\Lib\site-packages\bin\convert_hf_to_gguf.py"

# Inference settings (matching M1)
TEMPERATURE = 1.0
MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# Step 1: Merge LoRA adapters into base model
# ---------------------------------------------------------------------------
def step_merge(adapters=None):
    """Merge LoRA adapters into base HF model and save."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("ERROR: This step requires the training venv (transformers, peft, torch).")
        print("  Activate: C:\\sdt_calibration\\.venv_train\\Scripts\\activate")
        print("  Set: $env:HSA_OVERRIDE_GFX_VERSION='11.0.0'")
        sys.exit(1)

    adapters = adapters or ADAPTERS
    os.makedirs(MERGED_DIR, exist_ok=True)

    print("Loading base model (this takes a few minutes)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load on CPU in fp16 (no bitsandbytes on Windows ROCm)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    print(f"Base model loaded: {BASE_MODEL}\n")

    for adapter_name in adapters:
        adapter_path = os.path.join(ADAPTER_DIR, adapter_name)
        merged_path = os.path.join(MERGED_DIR, adapter_name)

        if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
            print(f"  SKIP {adapter_name}: adapter not found at {adapter_path}")
            continue

        if os.path.exists(merged_path) and os.path.exists(
            os.path.join(merged_path, "model.safetensors")
        ):
            print(f"  SKIP {adapter_name}: merged model already exists at {merged_path}")
            continue

        print(f"  Merging {adapter_name}...")
        t0 = time.perf_counter()

        # Load adapter on top of base model
        model_with_adapter = PeftModel.from_pretrained(
            base_model, adapter_path
        )

        # Merge LoRA weights into base weights
        merged_model = model_with_adapter.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        elapsed = time.perf_counter() - t0
        print(f"    Saved to {merged_path} ({elapsed:.0f}s)")

        # Free memory before next adapter
        del model_with_adapter, merged_model

    print("\nMerge step complete.")


# ---------------------------------------------------------------------------
# Step 2: Convert merged HF models to GGUF + quantise
# ---------------------------------------------------------------------------
def step_convert(adapters=None):
    """Convert merged HF models to GGUF and quantise to Q5_K_M."""
    adapters = adapters or ADAPTERS
    os.makedirs(GGUF_DIR, exist_ok=True)

    # Check for llama-quantize
    quantize_cmd = find_quantize_binary()

    for adapter_name in adapters:
        merged_path = os.path.join(MERGED_DIR, adapter_name)
        gguf_f16_path = os.path.join(GGUF_DIR, f"{adapter_name}_f16.gguf")
        gguf_q5_path = os.path.join(GGUF_DIR, f"{adapter_name}_Q5_K_M.gguf")

        if not os.path.exists(merged_path):
            print(f"  SKIP {adapter_name}: no merged model at {merged_path}")
            continue

        if os.path.exists(gguf_q5_path):
            print(f"  SKIP {adapter_name}: GGUF already exists at {gguf_q5_path}")
            continue

        # Convert HF → GGUF (f16)
        if not os.path.exists(gguf_f16_path):
            print(f"  Converting {adapter_name} → GGUF f16...")
            t0 = time.perf_counter()

            cmd = [
                sys.executable, CONVERT_SCRIPT,
                merged_path,
                "--outfile", gguf_f16_path,
                "--outtype", "f16",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"    ERROR converting {adapter_name}:")
                print(result.stderr[-500:] if result.stderr else "No stderr")
                continue

            elapsed = time.perf_counter() - t0
            size_gb = os.path.getsize(gguf_f16_path) / 1e9
            print(f"    f16 GGUF: {size_gb:.1f}GB ({elapsed:.0f}s)")

        # Quantise f16 → Q5_K_M
        if quantize_cmd:
            print(f"  Quantising {adapter_name} → Q5_K_M...")
            t0 = time.perf_counter()

            cmd = [quantize_cmd, gguf_f16_path, gguf_q5_path, "Q5_K_M"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"    ERROR quantising {adapter_name}:")
                print(result.stderr[-500:] if result.stderr else "No stderr")
                continue

            elapsed = time.perf_counter() - t0
            size_gb = os.path.getsize(gguf_q5_path) / 1e9
            print(f"    Q5_K_M GGUF: {size_gb:.1f}GB ({elapsed:.0f}s)")

            # Optionally remove f16 to save space
            os.remove(gguf_f16_path)
            print(f"    Removed f16 intermediate")
        else:
            print(f"    WARNING: llama-quantize not found. f16 GGUF saved but not quantised.")
            print(f"    You can quantise manually or use the f16 GGUF directly (slower inference).")

    print("\nConvert step complete.")


def find_quantize_binary():
    """Find llama-quantize or llama-cpp quantize binary."""
    candidates = [
        "llama-quantize",
        "llama-quantize.exe",
        r"C:\sdt_calibration\.venv\Lib\site-packages\bin\llama-quantize.exe",
        r"C:\sdt_calibration\.venv\Scripts\llama-quantize.exe",
    ]

    for c in candidates:
        try:
            result = subprocess.run([c, "--help"], capture_output=True, text=True)
            if result.returncode in (0, 1):  # some versions return 1 for --help
                print(f"  Found quantize binary: {c}")
                return c
        except FileNotFoundError:
            continue

    # Also check if llama-cpp-python ships with it
    try:
        import llama_cpp
        pkg_dir = os.path.dirname(llama_cpp.__file__)
        for name in ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]:
            candidate = os.path.join(pkg_dir, name)
            if os.path.exists(candidate):
                print(f"  Found quantize binary: {candidate}")
                return candidate
    except ImportError:
        pass

    print("  WARNING: Could not find llama-quantize binary.")
    return None


# ---------------------------------------------------------------------------
# Step 3: Evaluate each GGUF via llama-cpp on C₁
# ---------------------------------------------------------------------------
def step_evaluate(adapters=None):
    """Run llama-cpp inference on C₁ Science for each GGUF model."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: This step requires the inference venv (llama-cpp-python).")
        print("  Activate from C:\\sdt_calibration: .venv\\Scripts\\activate")
        sys.exit(1)

    # Import inference helpers
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from inference_set_b import init_llm, generate_answer, score_correctness

    adapters = adapters or ADAPTERS

    # Load C₁ Science questions
    with open(C1_SCIENCE_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} C₁ Science questions\n")

    for adapter_name in adapters:
        # Look for Q5_K_M first, fall back to f16
        gguf_q5 = os.path.join(GGUF_DIR, f"{adapter_name}_Q5_K_M.gguf")
        gguf_f16 = os.path.join(GGUF_DIR, f"{adapter_name}_f16.gguf")

        if os.path.exists(gguf_q5):
            model_path = gguf_q5
        elif os.path.exists(gguf_f16):
            model_path = gguf_f16
            print(f"  NOTE: Using f16 GGUF for {adapter_name} (no Q5_K_M found)")
        else:
            print(f"  SKIP {adapter_name}: no GGUF found in {GGUF_DIR}")
            continue

        output_json = os.path.join(RESULTS_DIR, f"pilot_trials_{adapter_name}_gguf.json")
        if os.path.exists(output_json):
            print(f"  SKIP {adapter_name}: results already exist at {output_json}")
            continue

        print(f"  Evaluating {adapter_name}...")
        print(f"    Model: {model_path}")
        t0 = time.perf_counter()

        # Load model
        llm = init_llm(model_path, n_gpu_layers=0)

        results = []
        for i, q in enumerate(questions):
            answer_text, nlp, n_tokens = generate_answer(
                llm, q["question"],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            answer_value = q.get("answer_value", "")
            aliases = q.get("answer_aliases", [])
            correct = score_correctness(answer_text, answer_value, aliases=aliases)

            results.append({
                "question_id": q.get("question_id", i),
                "question": q["question"],
                "domain": q.get("domain", "Science"),
                "ground_truth": [answer_value] + aliases if answer_value else aliases,
                "model_answer": answer_text,
                "correct": correct,
                "nlp": float(nlp),
                "answer_length_tokens": n_tokens,
            })

            if (i + 1) % 100 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                mean_nlp = np.mean([r["nlp"] for r in results])
                print(f"    [{i+1}/{len(questions)}] acc={acc:.3f}, nlp={mean_nlp:.4f}")

        elapsed = time.perf_counter() - t0

        # Save
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        acc = sum(r["correct"] for r in results) / len(results)
        c_nlps = [r["nlp"] for r in results if r["correct"]]
        ic_nlps = [r["nlp"] for r in results if not r["correct"]]
        gap = np.mean(c_nlps) - np.mean(ic_nlps) if ic_nlps else float("nan")

        print(f"    Done: {len(results)} trials, acc={acc:.3f}, "
              f"gap={gap:.4f}, time={elapsed/60:.1f}min")
        print(f"    Saved → {output_json}")

        # Free model
        del llm

    print("\nEvaluate step complete.")
    print(f"Run: python scripts/pilot_recompute.py --trials-dir {RESULTS_DIR} "
          f"--baseline-csv {RESULTS_DIR}/baseline_llamacpp_science.csv --domain Science")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def step_cleanup():
    """Remove merged HF checkpoints to free disk space."""
    import shutil
    if os.path.exists(MERGED_DIR):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(MERGED_DIR)
            for f in filenames
        )
        print(f"Removing {MERGED_DIR} ({size/1e9:.1f}GB)...")
        shutil.rmtree(MERGED_DIR)
        print("Done.")
    else:
        print(f"Nothing to clean: {MERGED_DIR} doesn't exist")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA → GGUF → evaluate via llama-cpp")
    parser.add_argument("--step", required=True,
                        choices=["merge", "convert", "evaluate", "all", "cleanup"],
                        help="Which step to run")
    parser.add_argument("--adapters", nargs="+", default=None,
                        help="Specific adapters to process (default: all)")
    args = parser.parse_args()

    adapters = args.adapters

    # Filter to only adapters that actually exist
    if adapters is None:
        adapters = [a for a in ADAPTERS
                    if os.path.exists(os.path.join(ADAPTER_DIR, a, "adapter_model.safetensors"))]
        print(f"Found {len(adapters)} adapters: {adapters}\n")

    if args.step == "merge":
        step_merge(adapters)
    elif args.step == "convert":
        step_convert(adapters)
    elif args.step == "evaluate":
        step_evaluate(adapters)
    elif args.step == "cleanup":
        step_cleanup()
    elif args.step == "all":
        print("Running all steps...\n")
        try:
            step_merge(adapters)
        except SystemExit:
            print("\nMerge failed (wrong venv?). Trying convert + evaluate...\n")
        if any(os.path.exists(os.path.join(MERGED_DIR, a)) for a in adapters):
            step_convert(adapters)
        if any(
            os.path.exists(os.path.join(GGUF_DIR, f"{a}_Q5_K_M.gguf")) or
            os.path.exists(os.path.join(GGUF_DIR, f"{a}_f16.gguf"))
            for a in adapters
        ):
            step_evaluate(adapters)


if __name__ == "__main__":
    main()
