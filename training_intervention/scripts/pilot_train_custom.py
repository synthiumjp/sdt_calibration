"""
Domain-Conditional Metacognitive Training — Task 4.1
Custom Pilot Training Loop: DPO, SFT, and CATTO

No TRL dependency — uses raw PyTorch + PEFT.
TRL is incompatible with the ROCm PyTorch build (missing torch.distributed).

Components:
  - Model: Llama-3-8B-Instruct (fp16, LoRA r=16)
  - DPO: standard log-sigmoid loss on chosen vs rejected
  - SFT: cross-entropy on preferred responses only
  - CATTO: DPO + per-token calibration loss (λ=0.1)

Prerequisites:
  $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
  C:\\sdt_calibration\\.venv_train\\Scripts\\Activate.ps1

Usage:
  python scripts/pilot_train_custom.py --method dpo --condition conditional
  python scripts/pilot_train_custom.py --method sft --condition conditional
  python scripts/pilot_train_custom.py --method catto --condition conditional
  python scripts/pilot_train_custom.py --all

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
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Import CATTO loss
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

SEED = 42
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # effective batch = 16
MAX_LENGTH = 256
DPO_BETA = 0.1
CATTO_LAMBDA = 0.1
WARMUP_RATIO = 0.10  # 10% of optimizer steps for linear warmup

PAIRS_FILES = {
    "conditional": DATA_DIR / "dpo_pairs_science_pilot.json",
    "agnostic": DATA_DIR / "dpo_pairs_agnostic_science_matched.json",
}


# ============================================================
# DATASET
# ============================================================

class DPODataset(Dataset):
    """Dataset for DPO/CATTO training — returns tokenised prompt+chosen and prompt+rejected."""
    
    def __init__(self, pairs_path, tokenizer, max_length=MAX_LENGTH):
        with open(pairs_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        # Format as chat messages
        prompt_messages = [{"role": "user", "content": prompt}]
        
        chosen_messages = prompt_messages + [{"role": "assistant", "content": chosen}]
        rejected_messages = prompt_messages + [{"role": "assistant", "content": rejected}]
        
        # Tokenise full sequences
        chosen_text = self.tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False
        )
        rejected_text = self.tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False
        )
        
        chosen_enc = self.tokenizer(
            chosen_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        
        # Also tokenise prompt alone to know where the response starts
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_enc = self.tokenizer(
            prompt_text, max_length=self.max_length, truncation=True,
            return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "prompt_length": prompt_len,
        }


class SFTDataset(Dataset):
    """Dataset for SFT training — returns tokenised prompt+chosen only."""
    
    def __init__(self, pairs_path, tokenizer, max_length=MAX_LENGTH):
        with open(pairs_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
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
# LOSS FUNCTIONS
# ============================================================

def compute_sequence_logprobs(model, input_ids, attention_mask, prompt_length):
    """
    Compute mean log-probability of the response tokens (after prompt).
    
    Returns: ([B] tensor of per-sequence mean log-probs, model outputs)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits  # [B, T, V]
    
    # Shift: predict token t from position t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Compute log-probs in fp32 for numerical stability
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask: only count response tokens (after prompt) and non-padding
    response_mask = shift_mask.clone().float()
    for b in range(input_ids.shape[0]):
        pl = prompt_length[b] if isinstance(prompt_length, (list, torch.Tensor)) else prompt_length
        if isinstance(pl, torch.Tensor):
            pl = pl.item()
        # Zero out prompt positions (we only want response log-probs)
        response_mask[b, :max(0, pl - 1)] = 0
    
    # Mean log-prob over response tokens
    masked_log_probs = token_log_probs * response_mask
    num_response_tokens = response_mask.sum(dim=-1).clamp(min=1)
    seq_log_probs = masked_log_probs.sum(dim=-1) / num_response_tokens
    
    return seq_log_probs, outputs


def dpo_loss_fn(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta=DPO_BETA):
    """Standard DPO loss."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def sft_loss_fn(model, input_ids, attention_mask, prompt_length):
    """Cross-entropy loss on response tokens only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift
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
    
    # Cross-entropy
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none"
    ).view(shift_logits.size(0), -1)
    
    masked_loss = (loss_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)
    return masked_loss


# ============================================================
# TRAINING LOOPS
# ============================================================

def train_dpo_loop(model, ref_model, tokenizer, dataset, output_dir, run_name):
    """DPO training loop."""
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED))
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    
    total_steps = NUM_EPOCHS * math.ceil(len(dataset) / BATCH_SIZE)
    num_optimizer_steps = total_steps // GRADIENT_ACCUMULATION
    warmup_steps = max(1, int(num_optimizer_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_optimizer_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )
    
    print(f"Training {run_name}: {len(dataset)} pairs, {NUM_EPOCHS} epochs, "
          f"{total_steps} micro-steps, {num_optimizer_steps} optimizer steps, "
          f"warmup={warmup_steps}, grad_accum={GRADIENT_ACCUMULATION}, LR={LEARNING_RATE}")
    
    model.train()
    global_step = 0
    log_entries = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to GPU
            chosen_ids = batch["chosen_input_ids"].to("cuda")
            chosen_mask = batch["chosen_attention_mask"].to("cuda")
            rejected_ids = batch["rejected_input_ids"].to("cuda")
            rejected_mask = batch["rejected_attention_mask"].to("cuda")
            prompt_len = batch["prompt_length"]
            
            # Policy forward pass
            policy_chosen_logps, _ = compute_sequence_logprobs(
                model, chosen_ids, chosen_mask, prompt_len
            )
            policy_rejected_logps, _ = compute_sequence_logprobs(
                model, rejected_ids, rejected_mask, prompt_len
            )
            
            # Reference forward pass (no grad)
            with torch.no_grad():
                ref_chosen_logps, _ = compute_sequence_logprobs(
                    ref_model, chosen_ids, chosen_mask, prompt_len
                )
                ref_rejected_logps, _ = compute_sequence_logprobs(
                    ref_model, rejected_ids, rejected_mask, prompt_len
                )
            
            # Debug: print first 20 steps
            if global_step < 20:
                print(f"  [step {global_step}] "
                      f"p_c={policy_chosen_logps.item():.4f} "
                      f"p_r={policy_rejected_logps.item():.4f} "
                      f"r_c={ref_chosen_logps.item():.4f} "
                      f"r_r={ref_rejected_logps.item():.4f} "
                      f"diff_c={policy_chosen_logps.item()-ref_chosen_logps.item():.6f} "
                      f"diff_r={policy_rejected_logps.item()-ref_rejected_logps.item():.6f}")
            
            loss = dpo_loss_fn(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps
            )
            
            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [NaN/Inf at step {global_step}] "
                      f"policy_c={policy_chosen_logps.item():.4f} "
                      f"policy_r={policy_rejected_logps.item():.4f} "
                      f"ref_c={ref_chosen_logps.item():.4f} "
                      f"ref_r={ref_rejected_logps.item():.4f} "
                      f"loss={loss.item()}")
                # Skip this batch
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
                print(f"  Step {global_step}/{total_steps} | Loss: {avg:.4f} | LR: {lr:.2e}")
                log_entries.append({
                    "step": global_step, "epoch": epoch,
                    "loss": avg, "lr": lr
                })
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} complete. Avg loss: {avg_epoch_loss:.4f}")
    
    # Save
    save_model(model, tokenizer, output_dir, log_entries, run_name)
    return log_entries


def train_sft_loop(model, tokenizer, dataset, output_dir, run_name):
    """SFT training loop."""
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED))
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    
    total_steps = NUM_EPOCHS * math.ceil(len(dataset) / BATCH_SIZE)
    num_optimizer_steps = total_steps // GRADIENT_ACCUMULATION
    warmup_steps = max(1, int(num_optimizer_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_optimizer_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )
    
    print(f"Training {run_name}: {len(dataset)} examples, {NUM_EPOCHS} epochs, "
          f"{total_steps} micro-steps, {num_optimizer_steps} optimizer steps, "
          f"warmup={warmup_steps}, grad_accum={GRADIENT_ACCUMULATION}, LR={LEARNING_RATE}")
    
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
                print(f"  Step {global_step}/{total_steps} | Loss: {avg:.4f} | LR: {lr:.2e}")
                log_entries.append({
                    "step": global_step, "epoch": epoch,
                    "loss": avg, "lr": lr
                })
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} complete. Avg loss: {avg_epoch_loss:.4f}")
    
    save_model(model, tokenizer, output_dir, log_entries, run_name)
    return log_entries


def train_catto_loop(model, ref_model, tokenizer, dataset, output_dir, run_name):
    """CATTO training loop: DPO + per-token calibration loss."""
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED))
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    
    total_steps = NUM_EPOCHS * math.ceil(len(dataset) / BATCH_SIZE)
    num_optimizer_steps = total_steps // GRADIENT_ACCUMULATION
    warmup_steps = max(1, int(num_optimizer_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_optimizer_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )
    
    print(f"Training {run_name}: {len(dataset)} pairs, {NUM_EPOCHS} epochs, "
          f"{total_steps} micro-steps, {num_optimizer_steps} optimizer steps, "
          f"warmup={warmup_steps}, grad_accum={GRADIENT_ACCUMULATION}, "
          f"LR={LEARNING_RATE}, λ={CATTO_LAMBDA}")
    
    model.train()
    global_step = 0
    log_entries = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_dpo_loss = 0
        epoch_cal_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to("cuda")
            chosen_mask = batch["chosen_attention_mask"].to("cuda")
            rejected_ids = batch["rejected_input_ids"].to("cuda")
            rejected_mask = batch["rejected_attention_mask"].to("cuda")
            prompt_len = batch["prompt_length"]
            
            # Policy forward pass (need logits for CATTO)
            chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            
            # Compute sequence log-probs from logits
            def logps_from_logits(logits, ids, mask, pl):
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = ids[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_lps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                resp_mask = shift_mask.clone()
                for b in range(ids.shape[0]):
                    p = pl[b] if isinstance(pl, (list, torch.Tensor)) else pl
                    if isinstance(p, torch.Tensor):
                        p = p.item()
                    resp_mask[b, :max(0, p - 1)] = 0
                return (token_lps * resp_mask).sum(dim=-1) / resp_mask.sum(dim=-1).clamp(min=1)
            
            policy_chosen_logps = logps_from_logits(
                chosen_outputs.logits, chosen_ids, chosen_mask, prompt_len
            )
            policy_rejected_logps = logps_from_logits(
                rejected_outputs.logits, rejected_ids, rejected_mask, prompt_len
            )
            
            # Reference forward pass
            with torch.no_grad():
                ref_chosen_out = ref_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                ref_rejected_out = ref_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                ref_chosen_logps = logps_from_logits(
                    ref_chosen_out.logits, chosen_ids, chosen_mask, prompt_len
                )
                ref_rejected_logps = logps_from_logits(
                    ref_rejected_out.logits, rejected_ids, rejected_mask, prompt_len
                )
            
            # DPO loss
            loss_dpo = dpo_loss_fn(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps
            )
            
            # CATTO calibration loss
            # Cast logits to fp32 — fp16 softmax overflows on Llama's 128K vocab
            chosen_labels = chosen_ids[:, 1:].contiguous()
            rejected_labels = rejected_ids[:, 1:].contiguous()
            chosen_logits = chosen_outputs.logits[:, :-1, :].contiguous().float()
            rejected_logits = rejected_outputs.logits[:, :-1, :].contiguous().float()
            chosen_cal_mask = chosen_mask[:, 1:].contiguous()
            rejected_cal_mask = rejected_mask[:, 1:].contiguous()
            
            loss_cal_chosen = catto_token_calibration_loss(
                chosen_logits, chosen_labels, chosen_cal_mask, invert_target=False
            )
            loss_cal_rejected = catto_token_calibration_loss(
                rejected_logits, rejected_labels, rejected_cal_mask, invert_target=True
            )
            
            loss_cal = CATTO_LAMBDA * (loss_cal_chosen + loss_cal_rejected)
            loss = loss_dpo + loss_cal
            
            # NaN guard with diagnostics
            if torch.isnan(loss) or torch.isinf(loss):
                if global_step < 5:
                    print(f"  [CATTO NaN at step {global_step}] "
                          f"dpo={loss_dpo.item():.4f} "
                          f"cal_c={loss_cal_chosen.item():.4f} "
                          f"cal_r={loss_cal_rejected.item():.4f} "
                          f"logits_max_c={chosen_logits.max().item():.1f} "
                          f"logits_max_r={rejected_logits.max().item():.1f}")
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1
                continue
            
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            epoch_dpo_loss += loss_dpo.item()
            epoch_cal_loss += loss_cal.item()
            epoch_steps += 1
            global_step += 1
            
            if global_step % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if global_step % 50 == 0:
                avg = epoch_loss / epoch_steps
                avg_dpo = epoch_dpo_loss / epoch_steps
                avg_cal = epoch_cal_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{total_steps} | "
                      f"Loss: {avg:.4f} (DPO: {avg_dpo:.4f}, Cal: {avg_cal:.4f}) | "
                      f"LR: {lr:.2e}")
                log_entries.append({
                    "step": global_step, "epoch": epoch,
                    "loss": avg, "loss_dpo": avg_dpo, "loss_cal": avg_cal, "lr": lr
                })
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} complete. Avg loss: {avg_epoch_loss:.4f}")
    
    save_model(model, tokenizer, output_dir, log_entries, run_name)
    return log_entries


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def load_base_model(tokenizer_only=False):
    """Load Llama-3-8B-Instruct. CPU first, then .to('cuda')."""
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if tokenizer_only:
        return None, tokenizer
    
    print(f"Loading model: {MODEL_ID} (fp16)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    print("Moving to GPU...")
    model = model.to("cuda")
    
    mem = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {mem:.1f} / {total:.1f} GB")
    
    return model, tokenizer


def create_ref_model():
    """
    Create reference model for DPO/CATTO.
    
    With PEFT, the standard approach is to use the same model with
    adapters disabled. But since we need both policy and ref forward
    passes, we load a second copy of the base model (frozen).
    
    This doubles VRAM usage (~32GB for two 8B fp16 models).
    If OOM, we'll switch to sequential ref computation or adapter toggling.
    """
    print("Loading reference model (frozen copy)...")
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    ref_model = ref_model.to("cuda")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after ref model: {mem:.1f} GB")
    return ref_model


def create_ref_model_peft(model):
    """
    Alternative: use PEFT's adapter disable for reference.
    
    Instead of loading a second model, we disable the LoRA adapter
    to get reference model behavior. This saves VRAM but requires
    careful adapter toggling during training.
    """
    # PEFT DPO pattern: disable adapter for ref forward pass
    class PEFTRefWrapper:
        def __init__(self, peft_model):
            self.model = peft_model
            self.training = False
        
        def __call__(self, *args, **kwargs):
            with self.model.disable_adapter():
                return self.model(*args, **kwargs)
        
        def eval(self):
            pass
        
        def parameters(self):
            return []
    
    return PEFTRefWrapper(model)


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
    
    print(f"Saved {run_name} to {output_dir}")


# ============================================================
# MAIN
# ============================================================

def run_one(method, condition):
    """Train one method × condition."""
    run_name = f"{method}_{condition}"
    output_dir = MODEL_DIR / run_name
    
    print(f"\n{'#'*60}")
    print(f"# PILOT: {run_name}")
    print(f"# Method: {method.upper()}")
    print(f"# Condition: {condition}")
    print(f"{'#'*60}")
    
    torch.manual_seed(SEED)
    
    pairs_path = PAIRS_FILES[condition]
    
    # Load model
    model, tokenizer = load_base_model()
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    
    # Manually cast LoRA adapter weights to fp32 for numerical stability
    # (autocast_adapter_dtype=False prevents PEFT from doing this, but the
    # ROCm .to(float32) kernel fails on GPU. So we do it on CPU first.)
    print("Casting LoRA weights to fp32...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.cpu().float().to("cuda")
    
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    
    # Reference model: use PEFT adapter disable (saves VRAM)
    # Instead of loading a second 16GB model
    ref_model = create_ref_model_peft(model)
    
    t0 = time.time()
    
    if method == "dpo":
        dataset = DPODataset(pairs_path, tokenizer)
        log = train_dpo_loop(model, ref_model, tokenizer, dataset, output_dir, run_name)
    elif method == "sft":
        dataset = SFTDataset(pairs_path, tokenizer)
        log = train_sft_loop(model, tokenizer, dataset, output_dir, run_name)
    elif method == "catto":
        dataset = DPODataset(pairs_path, tokenizer)
        log = train_catto_loop(model, ref_model, tokenizer, dataset, output_dir, run_name)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time.time() - t0
    print(f"\n{run_name} complete in {elapsed/60:.1f} minutes")
    
    # Free memory
    del model
    if hasattr(ref_model, 'model'):
        del ref_model
    torch.cuda.empty_cache()
    
    return log


def main():
    parser = argparse.ArgumentParser(description="Custom pilot training")
    parser.add_argument("--method", choices=["dpo", "sft", "catto"])
    parser.add_argument("--condition", choices=["conditional", "agnostic"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS (default: use constant)")
    
    args = parser.parse_args()
    
    if args.epochs is not None:
        global NUM_EPOCHS
        NUM_EPOCHS = args.epochs
        print(f"[override] NUM_EPOCHS = {NUM_EPOCHS}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        all_results = {}
        for method in ["dpo", "sft", "catto"]:
            for condition in ["conditional", "agnostic"]:
                run_name = f"{method}_{condition}"
                try:
                    log = run_one(method, condition)
                    all_results[run_name] = {"status": "complete", "final_loss": log[-1]["loss"] if log else None}
                    print(f"\n✓ {run_name} complete")
                except Exception as e:
                    print(f"\n✗ {run_name} FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[run_name] = {"status": "failed", "error": str(e)}
        
        summary_path = MODEL_DIR / "pilot_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll done. Summary: {summary_path}")
        
    elif args.method and args.condition:
        run_one(args.method, args.condition)
    else:
        parser.error("Use --method and --condition, or --all")


if __name__ == "__main__":
    main()
