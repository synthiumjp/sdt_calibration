"""
Domain-Conditional Metacognitive Training — Project Setup
=========================================================

Creates the directory structure at:
  C:\sdt_calibration\training_intervention\

Usage:
  cd C:\sdt_calibration
  mkdir training_intervention
  cd training_intervention
  python setup_project.py

Then follow the printed instructions.
"""

import os
from pathlib import Path

# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

PROJECT_ROOT = Path(r"C:\sdt_calibration\training_intervention")

DIRS = [
    "data",                    # Raw and processed datasets
    "data/triviaqa",           # Classified TriviaQA corpus, Sets B, C₁
    "data/mmlu",               # MMLU for C₂ near-transfer eval
    "data/nq",                 # Natural Questions for C₃ far-transfer eval
    "models",                  # LoRA adapters and checkpoints
    "models/pilot",            # Week 1 pilot models (DPO, SFT, CATTO)
    "models/full_grid",        # Full experimental grid (conditions 1-7)
    "results",                 # Evaluation outputs, metrics, figures
    "results/pilot",           # Pilot evaluation results
    "results/full",            # Full grid evaluation results
    "scripts",                 # Pipeline scripts
    "configs",                 # Training configs, hyperparameters
    "preregistration",         # OSF pre-registration documents
    "paper",                   # Manuscript drafts, figures
    "logs",                    # Training logs, run logs
]


def setup():
    print(f"Setting up project at: {PROJECT_ROOT}\n")
    
    for d in DIRS:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {path}")
    
    print(f"\n{'='*60}")
    print("SETUP COMPLETE")
    print(f"{'='*60}")
    
    print(f"""
PROJECT STRUCTURE:
  {PROJECT_ROOT}
  ├── data/
  │   ├── triviaqa/          # Sets A (ref), B (train), C₁ (eval)
  │   ├── mmlu/              # C₂ near-transfer eval
  │   └── nq/                # C₃ far-transfer eval
  ├── models/
  │   ├── pilot/             # Week 1: DPO, SFT, CATTO on Science
  │   └── full_grid/         # Conditions 1-7, both models
  ├── results/
  │   ├── pilot/             # Pilot M-ratio, d′, decision gate
  │   └── full/              # Full evaluation battery
  ├── scripts/               # Pipeline code
  ├── configs/               # Training hyperparameters
  ├── preregistration/       # OSF documents
  ├── paper/                 # Manuscript
  └── logs/                  # Run logs

CROSS-REFERENCES:
  M1 trial data:    C:\\sdt_calibration\\data\\triviaqa_5000.json
  M1 analysis:      C:\\sdt_calibration\\m1_type2\\m1_analysis.py
  GGUF model:       [UPDATE PATH BELOW]

{'='*60}
NEXT STEPS
{'='*60}

1. Copy the pipeline scripts into scripts/:
   copy triviaqa_classify_and_split.py scripts\\
   copy diagnostic_profile.py scripts\\
   copy mmlu_data_prep.py scripts\\

2. Verify your GGUF model path. Find it with:
   dir /s C:\\*.gguf
   (or wherever you store your GGUF files)

3. Run the diagnostic profile (instant, no GPU needed):
   cd {PROJECT_ROOT}
   python scripts\\diagnostic_profile.py --output-dir data\\triviaqa

   >>> This prints the prescription table. VERIFY the M1 values
   >>> match your actual results before proceeding.

4. Run TriviaQA classification (~2 hours, needs GPU):
   python scripts\\triviaqa_classify_and_split.py --step classify ^
       --model-path [YOUR_GGUF_PATH] ^
       --output-dir data\\triviaqa ^
       --cache-dir data\\triviaqa\\.hf_cache

5. After classification completes, draw Sets B and C₁ (seconds):
   python scripts\\triviaqa_classify_and_split.py --step split ^
       --m1-data C:\\sdt_calibration\\data\\triviaqa_5000.json ^
       --output-dir data\\triviaqa

6. Verify the output:
   - data\\triviaqa\\triviaqa_full_classified.json  (~87K questions with domains)
   - data\\triviaqa\\set_b_training.json            (5K training questions)
   - data\\triviaqa\\set_c1_eval_triviaqa.json      (3K eval questions)
   - data\\triviaqa\\prescription_table.json         (domain prescriptions)

   CHECK: No overlap between Set A, Set B, and Set C₁.
   CHECK: Domain distribution is reasonable (no domain < 500 questions).
   CHECK: Prescription table matches your M1 paper values.

7. Then: Task 2.1 (Llama inference on Set B) — overnight run.
""")


if __name__ == "__main__":
    setup()
