"""
Full-Grid GGUF Pipeline: Merge LoRA → f16 GGUF

Step 1 (merge): Run in training venv
  - Loads base model + each LoRA adapter → merge_and_unload() → save safetensors
  - Output: models/fullgrid_merged/cond{N}_sft/

Step 2 (convert): Run in inference venv
  - Converts each merged safetensors → f16 GGUF
  - Output: models/fullgrid_gguf/cond{N}_sft_f16.gguf

Both steps skip already-completed models (safe for restart).

Usage:
  # Step 1 — training venv
  cd C:\\sdt_calibration
  .venv_train\\Scripts\\activate
  $env:HSA_OVERRIDE_GFX_VERSION="11.0.0"
  cd training_intervention
  python scripts/fullgrid_gguf_pipeline.py --step merge

  # Step 2 — inference venv
  cd C:\\sdt_calibration
  .venv\\Scripts\\activate
  cd training_intervention
  python scripts/fullgrid_gguf_pipeline.py --step convert

  # Cleanup (optional, after GGUF conversion confirmed)
  python scripts/fullgrid_gguf_pipeline.py --step cleanup

Author: JP Cacioli / Synthium
Date: April 2026
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import time
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_DIR = Path(".")
ADAPTER_DIR = BASE_DIR / "models" / "fullgrid_adapters"
MERGED_DIR = BASE_DIR / "models" / "fullgrid_merged"
GGUF_DIR = BASE_DIR / "models" / "fullgrid_gguf"

# Also need the baseline f16 GGUF — reuse from pilot if it exists
PILOT_GGUF_DIR = BASE_DIR / "models" / "pilot_gguf"
BASELINE_GGUF = PILOT_GGUF_DIR / "baseline_f16.gguf"

# Conditions that have trained adapters
TRAINED_CONDITIONS = [2, 3, 4, 7]

# GGUF converter location (in inference venv)
CONVERTER_CANDIDATES = [
    Path(sys.prefix) / "Lib" / "site-packages" / "bin" / "convert_hf_to_gguf.py",
    Path(r"C:\sdt_calibration\.venv\Lib\site-packages\bin\convert_hf_to_gguf.py"),
]


# ============================================================
# STEP 1: MERGE LoRA → SAFETENSORS
# ============================================================

def step_merge():
    """Merge each LoRA adapter with base model. Requires training venv."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Check which conditions need merging
    to_merge = []
    for cond_id in TRAINED_CONDITIONS:
        adapter_dir = ADAPTER_DIR / f"cond{cond_id}_sft"
        merged_dir = MERGED_DIR / f"cond{cond_id}_sft"

        if not (adapter_dir / "adapter_config.json").exists():
            print(f"SKIP cond{cond_id}: no adapter found at {adapter_dir}")
            continue

        if (merged_dir / "model.safetensors.index.json").exists() or \
           (merged_dir / "model.safetensors").exists():
            print(f"SKIP cond{cond_id}: already merged at {merged_dir}")
            continue

        to_merge.append((cond_id, adapter_dir, merged_dir))

    if not to_merge:
        print("Nothing to merge. All conditions already done.")
        return

    print(f"\nMerging {len(to_merge)} conditions: {[c[0] for c in to_merge]}")

    # Load base model once (CPU only — no GPU needed for merge)
    print(f"\nLoading base model: {MODEL_ID} (CPU, fp16)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for cond_id, adapter_dir, merged_dir in to_merge:
        print(f"\n{'='*50}")
        print(f"Merging condition {cond_id}: {adapter_dir.name}")
        print(f"{'='*50}")

        t0 = time.time()

        # Load base model fresh each time (merge_and_unload modifies in-place)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )

        # Load and merge LoRA
        print(f"  Loading LoRA from {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        print("  Merging...")
        model = model.merge_and_unload()

        # Save merged model
        merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving to {merged_dir}")
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

        elapsed = time.time() - t0
        size_gb = sum(f.stat().st_size for f in merged_dir.glob("*.safetensors")) / 1e9
        print(f"  Done in {elapsed:.0f}s ({size_gb:.1f} GB)")

        # Free memory
        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll merges complete. Output: {MERGED_DIR}")


# ============================================================
# STEP 2: CONVERT SAFETENSORS → f16 GGUF
# ============================================================

def find_converter():
    """Find convert_hf_to_gguf.py."""
    for p in CONVERTER_CANDIDATES:
        if p.exists():
            return p
    # Try to find it in the current Python environment
    import importlib.util
    spec = importlib.util.find_spec("gguf")
    if spec and spec.origin:
        candidate = Path(spec.origin).parent.parent / "bin" / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
    return None


def step_convert():
    """Convert merged safetensors → f16 GGUF. Requires inference venv."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    converter = find_converter()
    if converter is None:
        print("ERROR: Cannot find convert_hf_to_gguf.py")
        print("Install: pip install gguf")
        sys.exit(1)
    print(f"Using converter: {converter}")

    to_convert = []
    for cond_id in TRAINED_CONDITIONS:
        merged_dir = MERGED_DIR / f"cond{cond_id}_sft"
        gguf_path = GGUF_DIR / f"cond{cond_id}_sft_f16.gguf"

        if gguf_path.exists():
            size_gb = gguf_path.stat().st_size / 1e9
            print(f"SKIP cond{cond_id}: GGUF exists ({size_gb:.1f} GB)")
            continue

        if not merged_dir.exists():
            print(f"SKIP cond{cond_id}: no merged model at {merged_dir}")
            continue

        to_convert.append((cond_id, merged_dir, gguf_path))

    # Also convert baseline if not already done
    baseline_merged = MERGED_DIR / "baseline"
    baseline_gguf = GGUF_DIR / "baseline_f16.gguf"
    if BASELINE_GGUF.exists() and not baseline_gguf.exists():
        # Copy existing pilot baseline
        print(f"Copying pilot baseline GGUF to {baseline_gguf}")
        shutil.copy2(BASELINE_GGUF, baseline_gguf)
    elif not baseline_gguf.exists():
        print("NOTE: No baseline f16 GGUF found. You may need to convert one from")
        print(f"  the HF cache, or copy from {BASELINE_GGUF}")

    if not to_convert:
        print("Nothing to convert. All GGUFs already exist.")
        return

    print(f"\nConverting {len(to_convert)} models: {[c[0] for c in to_convert]}")

    for cond_id, merged_dir, gguf_path in to_convert:
        print(f"\n{'='*50}")
        print(f"Converting condition {cond_id}: {merged_dir.name} → {gguf_path.name}")
        print(f"{'='*50}")

        t0 = time.time()

        cmd = [
            sys.executable, str(converter),
            str(merged_dir),
            "--outfile", str(gguf_path),
            "--outtype", "f16",
        ]
        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")
            continue

        elapsed = time.time() - t0
        if gguf_path.exists():
            size_gb = gguf_path.stat().st_size / 1e9
            print(f"  Done in {elapsed:.0f}s ({size_gb:.1f} GB)")
        else:
            print(f"  WARNING: GGUF file not found after conversion")

    print(f"\nAll conversions complete. Output: {GGUF_DIR}")

    # Report disk usage
    total_gb = sum(
        f.stat().st_size for f in GGUF_DIR.glob("*.gguf")
    ) / 1e9
    print(f"Total GGUF size: {total_gb:.1f} GB")


# ============================================================
# CLEANUP
# ============================================================

def step_cleanup():
    """Remove merged safetensors (large intermediate files)."""
    if not MERGED_DIR.exists():
        print("Nothing to clean up.")
        return

    # Verify GGUFs exist before deleting
    missing = []
    for cond_id in TRAINED_CONDITIONS:
        gguf_path = GGUF_DIR / f"cond{cond_id}_sft_f16.gguf"
        if not gguf_path.exists():
            missing.append(cond_id)

    if missing:
        print(f"WARNING: GGUFs missing for conditions {missing}")
        print("Aborting cleanup. Run --step convert first.")
        return

    size_gb = sum(
        f.stat().st_size for f in MERGED_DIR.rglob("*") if f.is_file()
    ) / 1e9
    print(f"Removing {MERGED_DIR} ({size_gb:.1f} GB)")

    response = input("Confirm deletion? [y/N] ")
    if response.lower() == "y":
        shutil.rmtree(MERGED_DIR)
        print("Deleted.")
    else:
        print("Cancelled.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full-grid GGUF pipeline")
    parser.add_argument(
        "--step", required=True,
        choices=["merge", "convert", "cleanup"],
        help="Pipeline step to run"
    )
    args = parser.parse_args()

    if args.step == "merge":
        step_merge()
    elif args.step == "convert":
        step_convert()
    elif args.step == "cleanup":
        step_cleanup()


if __name__ == "__main__":
    main()
