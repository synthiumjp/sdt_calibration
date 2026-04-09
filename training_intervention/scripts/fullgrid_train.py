"""
Domain-Conditional Metacognitive Training — Full Grid SFT Training
Custom Training Loop: SFT only (method selected by pilot decision gate)

No TRL dependency — uses raw PyTorch + PEFT.

Trains 4 conditions sequentially:
  Cond 2: conditional Science (LR=1e-5)
  Cond 3: agnostic matched  (LR=1e-5)
  Cond 4: wrong Geography   (LR=1e-5)
  Cond 7: conditional Science low-LR (LR=5e-6)

Prerequisites:
  $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
  C:\\sdt_calibration\\.venv_train\\Scripts\\Activate.ps1

Usage:
  python scripts/fullgrid_train.py --condition 2        # single condition
  python scripts/fullgrid_train.py --all                # all 4 sequentially

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" — Full Grid (Post Pre-reg 2)
Date: April 2026
"""

import os
import sys
import json
import math
import argparse
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ============================================================
# CONSTANTS
# ============================================================

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "triviaqa"
MODEL_DIR = BASE_DIR / "models" / "fullgrid_adapters"
LOG_DIR = BASE_DIR / "logs"

SEED = 42
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # effective batch = 16
MAX_LENGTH = 256
WARMUP_RATIO = 0.10  # 10% of optimizer steps for linear warmup

# Per-condition configuration
# Format: { condition_id: (pair_file, learning_rate, description) }
CONDITIONS = {
    2: (
        DATA_DIR / "sft_pairs_cond2_conditional_science.json",
        1e-5,
        "Conditional SFT — Science (correct prescription)"
    ),
    3: (
        DATA_DIR / "sft_pairs_cond3_agnostic_matched.json",
        1e-5,
        "Agnostic SFT — all domains (matched budget)"
    ),
    4: (
        DATA_DIR / "sft_pairs_cond4_wrong_geography.json",
        1e-5,
        "Wrong-prescription SFT — Geography"
    ),
    7: (
        DATA_DIR / "sft_pairs_cond7_conditional_science_lowlr.json",
        5e-6,
        "Conditional SFT — Science (low LR variant)"
    ),
}


# ============================================================
# DATASET
# ============================================================

class SFTDataset(Dataset):
    """
    Dataset for SFT training — returns tokenised prompt + completion.
    
    Pair format: {"prompt": <question>, "completion": <target_response>}
    (from generate_sft_pairs.py)
    """
    
    def __init__(self, pairs_path, tokenizer, max_length=MAX_LENGTH):
        with open(pairs_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Detect field names: new format uses "completion", pilot used "chosen"
        sample = self.pairs[0]
        if "completion" in sample:
            self.response_key = "completion"
        elif "chosen" in sample:
            self.response_key = "chosen"
        else:
            raise ValueError(
                f"Unrecognised pair format. Keys: {list(sample.keys())}. "
                f"Expected 'completion' or 'chosen'."
            )
        print(f"  SFTDataset: {len(self.pairs)} pairs, response key='{self.response_key}'")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        response = pair[self.response_key]
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        enc = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        
        # Prompt length for masking loss to response only
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_enc = self.tokenizer(
            prompt_text, max_length=self.max_length, truncation=True,
            return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_length": prompt_len,
        }


# ============================================================
# LOSS FUNCTION
# ============================================================

def sft_loss_fn(model, input_ids, attention_mask, prompt_length):
    """Cross-entropy loss on response tokens only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift: predict token t from position t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Mask prompt tokens
    response_mask = shift_mask.clone()
    for b in range(input_ids.shape[0]):
        pl = prompt_length[b] if isinstance(prompt_length, (list, torch.Tensor)) else prompt_length
        if isinstance(pl, torch.Tensor):
            pl = pl.item()
        response_mask[b, :max(0, pl - 1)] = 0
    
    # Cross-entropy (fp32 for stability)
    loss_per_token = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none"
    ).view(shift_logits.size(0), -1)
    
    masked_loss = (loss_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)
    return masked_loss


# ============================================================
# TRAINING LOOP
# ============================================================

def train_sft(model, tokenizer, dataset, output_dir, run_name, learning_rate):
    """SFT training loop with per-condition learning rate."""
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=0.01
    )
    
    total_steps = NUM_EPOCHS * math.ceil(len(dataset) / BATCH_SIZE)
    num_optimizer_steps = total_steps // GRADIENT_ACCUMULATION
    warmup_steps = max(1, int(num_optimizer_steps * WARMUP_RATIO))
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_optimizer_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )
    
    print(f"Training {run_name}: {len(dataset)} examples, {NUM_EPOCHS} epochs, "
          f"{total_steps} micro-steps, {num_optimizer_steps} optimizer steps, "
          f"warmup={warmup_steps}, grad_accum={GRADIENT_ACCUMULATION}, "
          f"LR={learning_rate}")
    
    model.train()
    global_step = 0
    log_entries = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            prompt_len = batch["prompt_length"]
            
            loss = sft_loss_fn(model, input_ids, attention_mask, prompt_len)
            
            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [NaN/Inf at step {global_step}] loss={loss.item()}")
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1
                continue
            
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            epoch_steps += 1
            global_step += 1
            
            if global_step % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if global_step % 50 == 0:
                avg = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{total_steps} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e}")
                log_entries.append({
                    "step": global_step, "epoch": epoch,
                    "loss": avg, "lr": lr
                })
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} complete. "
              f"Avg loss: {avg_epoch_loss:.4f}")
    
    save_model(model, tokenizer, output_dir, log_entries, run_name)
    return log_entries


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def load_base_model():
    """Load Llama-3-8B-Instruct. CPU first, then .to('cuda')."""
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model: {MODEL_ID} (fp16)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    print("Moving to GPU...")
    model = model.to("cuda")
    
    mem = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {mem:.1f} / {total:.1f} GB")
    
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA and cast adapter weights to fp32 (ROCm workaround)."""
    print("Applying LoRA (r=16, alpha=32, q_proj + v_proj)...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    
    # Cast LoRA weights to fp32 via CPU detour (ROCm HIP kernel workaround)
    # Base model stays fp16 (frozen), only ~6.8M LoRA params become fp32
    print("Casting LoRA weights to fp32 (CPU detour)...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.cpu().float().to("cuda")
    
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    
    return model


def save_model(model, tokenizer, output_dir, log_entries, run_name):
    """Save LoRA adapter and training log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    
    # Save run metadata
    meta = {
        "run_name": run_name,
        "model_id": MODEL_ID,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "final_loss": log_entries[-1]["loss"] if log_entries else None,
    }
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved {run_name} to {output_dir}")


# ============================================================
# RUN ONE CONDITION
# ============================================================

def run_condition(cond_id):
    """Train one condition. Loads model fresh each time (clean LoRA init)."""
    if cond_id not in CONDITIONS:
        raise ValueError(f"Unknown condition: {cond_id}. Valid: {list(CONDITIONS.keys())}")
    
    pairs_path, learning_rate, description = CONDITIONS[cond_id]
    run_name = f"cond{cond_id}_sft"
    output_dir = MODEL_DIR / run_name
    
    # Check if already trained
    if (output_dir / "adapter_config.json").exists():
        print(f"\n{'='*60}")
        print(f"SKIP: {run_name} already exists at {output_dir}")
        print(f"Delete the directory to retrain.")
        print(f"{'='*60}")
        return None
    
    # Check pair file exists
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair file not found: {pairs_path}")
    
    print(f"\n{'#'*60}")
    print(f"# CONDITION {cond_id}: {run_name}")
    print(f"# {description}")
    print(f"# Pairs: {pairs_path.name}")
    print(f"# LR: {learning_rate}")
    print(f"{'#'*60}")
    
    torch.manual_seed(SEED)
    
    # Load model fresh (clean base weights, no residual LoRA)
    model, tokenizer = load_base_model()
    model = apply_lora(model)
    
    # Load dataset
    dataset = SFTDataset(pairs_path, tokenizer)
    
    t0 = time.time()
    log = train_sft(model, tokenizer, dataset, output_dir, run_name, learning_rate)
    elapsed = time.time() - t0
    
    print(f"\n{run_name} complete in {elapsed/60:.1f} minutes")
    if log:
        print(f"Final loss: {log[-1]['loss']:.4f}")
    
    # Free GPU memory before next condition
    del model
    torch.cuda.empty_cache()
    
    return log


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-grid SFT training (Conditions 2, 3, 4, 7)"
    )
    parser.add_argument(
        "--condition", type=int, choices=[2, 3, 4, 7],
        help="Train a single condition"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all 4 conditions sequentially"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override NUM_EPOCHS (default: 3)"
    )
    
    args = parser.parse_args()
    
    if args.epochs is not None:
        global NUM_EPOCHS
        NUM_EPOCHS = args.epochs
        print(f"[override] NUM_EPOCHS = {NUM_EPOCHS}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        all_results = {}
        train_order = [2, 3, 4, 7]
        
        print(f"\n{'='*60}")
        print(f"FULL GRID: Training {len(train_order)} conditions sequentially")
        print(f"Order: {train_order}")
        print(f"{'='*60}")
        
        t_start = time.time()
        
        for cond_id in train_order:
            try:
                log = run_condition(cond_id)
                status = "complete" if log else "skipped"
                final_loss = log[-1]["loss"] if log else None
                all_results[f"cond{cond_id}"] = {
                    "status": status,
                    "final_loss": final_loss,
                    "description": CONDITIONS[cond_id][2],
                    "learning_rate": CONDITIONS[cond_id][1],
                }
                print(f"\n✓ Condition {cond_id} {status}")
            except Exception as e:
                print(f"\n✗ Condition {cond_id} FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"cond{cond_id}"] = {
                    "status": "failed",
                    "error": str(e),
                }
        
        total_elapsed = time.time() - t_start
        
        # Save summary
        summary = {
            "total_time_minutes": total_elapsed / 60,
            "conditions": all_results,
        }
        summary_path = MODEL_DIR / "fullgrid_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"ALL DONE in {total_elapsed/60:.1f} minutes")
        for k, v in all_results.items():
            loss_str = f"loss={v['final_loss']:.4f}" if v.get('final_loss') else ""
            print(f"  {k}: {v['status']} {loss_str}")
        print(f"Summary: {summary_path}")
        print(f"{'='*60}")
        
    elif args.condition:
        run_condition(args.condition)
    else:
        parser.error("Use --condition <2|3|4|7> or --all")


if __name__ == "__main__":
    main()
