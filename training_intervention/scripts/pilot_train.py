"""
Domain-Conditional Metacognitive Training — Task 4.1
Pilot Training: DPO, SFT, and CATTO on Science Domain

Trains 6 LoRA adapters (3 methods × 2 conditions) on the Science pilot data.
Saves adapters to models/pilot/ for evaluation.

Hardware constraints (AMD 7900 GRE, ROCm):
  - No device_map='auto' (causes HIP kernel errors)
  - No bitsandbytes (incompatible with Windows ROCm)
  - autocast_adapter_dtype=False required for PEFT
  - Load on CPU then .to('cuda')
  - fp16 training (no 4-bit quantisation)

Prerequisites:
  - Activate training venv: C:\\sdt_calibration\\.venv_train\\Scripts\\Activate.ps1
  - Set: $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"

Usage:
  # Train all 6 models:
  python scripts/pilot_train.py --all

  # Train a specific method + condition:
  python scripts/pilot_train.py --method dpo --condition conditional
  python scripts/pilot_train.py --method sft --condition agnostic
  python scripts/pilot_train.py --method catto --condition conditional

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
Date: 31 March 2026
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

# Add scripts dir to path for catto_loss import
sys.path.insert(0, str(Path(__file__).parent))
from catto_loss import catto_token_calibration_loss

# ============================================================
# CONSTANTS
# ============================================================

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "triviaqa"
MODEL_DIR = BASE_DIR / "models" / "pilot"
LOG_DIR = BASE_DIR / "logs"

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training hyperparameters (from v1.2 §2.5)
SEED = 42
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # effective batch = 16
MAX_LENGTH = 256  # prompt + response (trivia answers are short)
MAX_PROMPT_LENGTH = 128
DPO_BETA = 0.1
CATTO_LAMBDA = 0.1

# Data files
PAIRS_FILES = {
    "conditional": DATA_DIR / "dpo_pairs_science_pilot.json",
    "agnostic": DATA_DIR / "dpo_pairs_agnostic_science_matched.json",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_dpo_dataset(pairs_path):
    """
    Load DPO pairs from JSON and convert to HuggingFace Dataset.
    TRL expects: prompt, chosen, rejected columns.
    """
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    dataset_dict = {
        "prompt": [p["prompt"] for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    }
    
    ds = Dataset.from_dict(dataset_dict)
    print(f"Loaded {len(ds)} DPO pairs from {pairs_path}")
    return ds


def load_sft_dataset(pairs_path):
    """
    Load SFT data from DPO pairs — uses only the preferred (chosen) responses.
    TRL SFTTrainer expects a 'text' column with the full formatted text,
    or prompt + completion columns.
    """
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    dataset_dict = {
        "prompt": [p["prompt"] for p in pairs],
        "completion": [p["chosen"] for p in pairs],
    }
    
    ds = Dataset.from_dict(dataset_dict)
    print(f"Loaded {len(ds)} SFT examples from {pairs_path}")
    return ds


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer():
    """
    Load Llama-3-8B-Instruct with LoRA.
    
    Critical ROCm constraints:
      - Load on CPU first, then .to('cuda')
      - Do NOT use device_map='auto'
      - autocast_adapter_dtype=False
    """
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model: {MODEL_ID} (fp16, CPU first)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    
    print("Moving model to GPU...")
    model = model.to("cuda")
    
    print(f"GPU memory after model load: "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapter with ROCm-compatible settings."""
    print("Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    return model


# ============================================================
# DPO TRAINING
# ============================================================

def train_dpo(model, tokenizer, dataset, output_dir, run_name):
    """Standard DPO training via TRL."""
    print(f"\n{'='*60}")
    print(f"DPO TRAINING: {run_name}")
    print(f"{'='*60}")
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        seed=SEED,
        logging_steps=10,
        save_strategy="epoch",
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting DPO training...")
    train_result = trainer.train()
    
    # Save adapter
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training metrics
    metrics = train_result.metrics
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete. Final loss: {metrics.get('train_loss', 'N/A')}")
    return metrics


# ============================================================
# SFT TRAINING
# ============================================================

def train_sft(model, tokenizer, dataset, output_dir, run_name):
    """SFT training on preferred responses only."""
    print(f"\n{'='*60}")
    print(f"SFT TRAINING: {run_name}")
    print(f"{'='*60}")
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        max_seq_length=MAX_LENGTH,
        seed=SEED,
        logging_steps=10,
        save_strategy="epoch",
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field=None,  # Using prompt/completion format
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting SFT training...")
    train_result = trainer.train()
    
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    metrics = train_result.metrics
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete. Final loss: {metrics.get('train_loss', 'N/A')}")
    return metrics


# ============================================================
# CATTO TRAINING (Custom DPO + Calibration Loss)
# ============================================================

class CATTODPOTrainer(DPOTrainer):
    """
    DPO Trainer with CATTO calibration loss.
    
    Overrides the loss computation to add per-token calibration loss:
      L_total = L_DPO + λ * (L_cal_chosen + L_cal_rejected)
    
    Implementation follows Parikh et al. (2026), arXiv:2601.23096.
    """
    
    def __init__(self, *args, catto_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.catto_lambda = catto_lambda
        self._catto_losses = []  # Track calibration losses for logging
    
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """
        Override DPO's loss computation to add CATTO calibration loss.
        
        The parent class computes the standard DPO loss. We intercept
        to also compute the calibration loss on the logits.
        """
        # Get the standard DPO metrics from parent
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        
        # Now compute calibration loss on chosen and rejected sequences
        # We need to do a forward pass to get logits (parent may not expose them)
        try:
            # Extract input IDs and labels for chosen and rejected
            chosen_ids = batch["chosen_input_ids"]
            rejected_ids = batch["rejected_input_ids"]
            chosen_mask = batch.get("chosen_attention_mask", 
                                    torch.ones_like(chosen_ids))
            rejected_mask = batch.get("rejected_attention_mask",
                                      torch.ones_like(rejected_ids))
            
            # Forward pass to get logits
            chosen_outputs = model(
                input_ids=chosen_ids, 
                attention_mask=chosen_mask
            )
            rejected_outputs = model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask
            )
            
            # Compute calibration losses
            # Labels are the input_ids shifted by 1 (next token prediction)
            chosen_labels = chosen_ids[:, 1:].contiguous()
            rejected_labels = rejected_ids[:, 1:].contiguous()
            chosen_logits = chosen_outputs.logits[:, :-1, :].contiguous()
            rejected_logits = rejected_outputs.logits[:, :-1, :].contiguous()
            chosen_cal_mask = chosen_mask[:, 1:].contiguous()
            rejected_cal_mask = rejected_mask[:, 1:].contiguous()
            
            loss_cal_chosen = catto_token_calibration_loss(
                chosen_logits, chosen_labels, chosen_cal_mask, 
                invert_target=False
            )
            loss_cal_rejected = catto_token_calibration_loss(
                rejected_logits, rejected_labels, rejected_cal_mask,
                invert_target=True
            )
            
            catto_loss = self.catto_lambda * (loss_cal_chosen + loss_cal_rejected)
            loss = loss + catto_loss
            
            # Log calibration losses
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}catto_cal_chosen"] = loss_cal_chosen.detach().item()
            metrics[f"{prefix}catto_cal_rejected"] = loss_cal_rejected.detach().item()
            metrics[f"{prefix}catto_total_cal"] = catto_loss.detach().item()
            
        except Exception as e:
            # If CATTO computation fails, fall back to pure DPO
            print(f"WARNING: CATTO calibration loss failed: {e}")
            print("Falling back to pure DPO loss for this batch.")
        
        return loss, metrics


def train_catto(model, tokenizer, dataset, output_dir, run_name):
    """CATTO training: DPO + per-token calibration loss."""
    print(f"\n{'='*60}")
    print(f"CATTO TRAINING: {run_name}")
    print(f"{'='*60}")
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        seed=SEED,
        logging_steps=10,
        save_strategy="epoch",
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    trainer = CATTODPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        catto_lambda=CATTO_LAMBDA,
    )
    
    print(f"Starting CATTO training (λ={CATTO_LAMBDA})...")
    train_result = trainer.train()
    
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    metrics = train_result.metrics
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete. Final loss: {metrics.get('train_loss', 'N/A')}")
    return metrics


# ============================================================
# MAIN
# ============================================================

TRAIN_FUNCS = {
    "dpo": (train_dpo, load_dpo_dataset),
    "sft": (train_sft, load_sft_dataset),
    "catto": (train_catto, load_dpo_dataset),
}


def run_one(method, condition):
    """Train one method × condition combination."""
    run_name = f"{method}_{condition}"
    output_dir = MODEL_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_path = PAIRS_FILES[condition]
    train_func, load_func = TRAIN_FUNCS[method]
    
    print(f"\n{'#'*60}")
    print(f"# PILOT TRAINING: {run_name}")
    print(f"# Method: {method.upper()}")
    print(f"# Condition: {condition}")
    print(f"# Data: {pairs_path}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")
    
    # Load data
    dataset = load_func(pairs_path)
    
    # Load model fresh for each run (clean LoRA each time)
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    
    # Train
    metrics = train_func(model, tokenizer, dataset, output_dir, run_name)
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Pilot training")
    parser.add_argument("--method", choices=["dpo", "sft", "catto"], default=None)
    parser.add_argument("--condition", choices=["conditional", "agnostic"], default=None)
    parser.add_argument("--all", action="store_true", help="Train all 6 combinations")
    
    args = parser.parse_args()
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        all_metrics = {}
        for method in ["dpo", "sft", "catto"]:
            for condition in ["conditional", "agnostic"]:
                run_name = f"{method}_{condition}"
                try:
                    metrics = run_one(method, condition)
                    all_metrics[run_name] = metrics
                    print(f"\n✓ {run_name} complete")
                except Exception as e:
                    print(f"\n✗ {run_name} FAILED: {e}")
                    all_metrics[run_name] = {"error": str(e)}
        
        # Save summary
        summary_path = MODEL_DIR / "pilot_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n{'='*60}")
        print(f"ALL TRAINING COMPLETE")
        print(f"Summary saved to {summary_path}")
        print(f"{'='*60}")
        
    elif args.method and args.condition:
        run_one(args.method, args.condition)
    else:
        parser.error("Specify --method and --condition, or use --all")


if __name__ == "__main__":
    main()
