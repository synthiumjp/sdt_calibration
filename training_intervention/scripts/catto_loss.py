"""
CATTO Loss Implementation
Calibration-Aware Token-level Training Objective (Parikh et al., 2026)
arXiv: 2601.23096

Combines standard DPO loss with a per-token calibration loss that aligns
predicted confidence with empirical correctness.

L_total = L_DPO + λ · (L_cal_chosen + L_cal_rejected)

Reviewed against the paper. Key corrections from draft:
  - Uses BCE (not L1) for calibration loss, matching paper §3.1
  - Added numerical clamping for log stability
  - pred_conf clamped to [eps, 1-eps] before log

Author: JP Cacioli / Synthium
Project: "Prescribe, Don't Average" (v1.2)
"""

import torch
import torch.nn.functional as F


def catto_token_calibration_loss(logits, labels, attention_mask=None, 
                                  invert_target=False, eps=1e-7):
    """
    Per-token calibration loss from CATTO (§3.1).
    
    Aligns the model's predicted confidence c_θ(x_t) with a differentiable
    correctness surrogate z̃(x_t).
    
    Args:
        logits: [B, T, V] — raw logits from the model
        labels: [B, T] — ground-truth token ids
        attention_mask: [B, T] — 1 for valid positions, 0 for padding
        invert_target: False for preferred (chosen) response, 
                       True for rejected response
        eps: numerical stability constant
    
    Returns:
        scalar calibration loss
    """
    probs = logits.softmax(dim=-1)  # [B, T, V]
    
    # c_θ(x_t): model's confidence in its own top prediction
    pred_ids = probs.argmax(dim=-1)  # [B, T]
    pred_conf = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
    pred_conf = pred_conf.clamp(eps, 1.0 - eps)  # numerical stability for log
    
    # p_y*(x_t): probability assigned to the ground-truth token
    p_true = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    
    # p_ȳ(x_t): highest probability among NON-ground-truth tokens
    masked_probs = probs.clone()
    masked_probs.scatter_(-1, labels.unsqueeze(-1), 0.0)  # zero out true token
    p_best_wrong = masked_probs.max(dim=-1).values  # [B, T]
    
    # z̃(x_t) = σ(p_true - p_best_wrong): differentiable correctness surrogate
    z_tilde = torch.sigmoid(p_true - p_best_wrong)  # [B, T], range (0, 1)
    
    # For rejected responses, invert the target:
    # we want the model to be LESS calibrated on dispreferred outputs
    if invert_target:
        z_tilde = 1.0 - z_tilde
    
    # BCE calibration loss (paper §3.1, Eq. 3)
    # L_cal = -[z̃ · log(c) + (1 - z̃) · log(1 - c)]
    per_token = -(z_tilde * torch.log(pred_conf) + 
                  (1.0 - z_tilde) * torch.log(1.0 - pred_conf))
    
    # Mask and average
    if attention_mask is not None:
        per_token = per_token * attention_mask
        denom = attention_mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(per_token.numel(), dtype=per_token.dtype, 
                             device=per_token.device)
    
    return per_token.sum() / denom


def dpo_loss(policy_logp_chosen, policy_logp_rejected,
             ref_logp_chosen, ref_logp_rejected, beta=0.1):
    """
    Standard DPO loss (Rafailov et al., 2023).
    
    Args:
        policy_logp_chosen: [B] sequence log-probs for chosen responses
        policy_logp_rejected: [B] sequence log-probs for rejected responses
        ref_logp_chosen: [B] reference model log-probs for chosen
        ref_logp_rejected: [B] reference model log-probs for rejected
        beta: temperature parameter (default 0.1)
    
    Returns:
        scalar DPO loss
    """
    chosen_reward = beta * (policy_logp_chosen - ref_logp_chosen)
    rejected_reward = beta * (policy_logp_rejected - ref_logp_rejected)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def catto_total_loss(
    chosen_logits, chosen_labels, chosen_mask,
    rejected_logits, rejected_labels, rejected_mask,
    policy_logp_chosen, policy_logp_rejected,
    ref_logp_chosen, ref_logp_rejected,
    beta=0.1, lambda_catto=0.1
):
    """
    Combined CATTO loss: L_total = L_DPO + λ · (L_cal_chosen + L_cal_rejected)
    
    Args:
        chosen_logits: [B, T, V] logits for chosen responses
        chosen_labels: [B, T] token ids for chosen responses
        chosen_mask: [B, T] attention mask for chosen responses
        rejected_logits: [B, T, V] logits for rejected responses
        rejected_labels: [B, T] token ids for rejected responses
        rejected_mask: [B, T] attention mask for rejected responses
        policy_logp_chosen: [B] sequence log-probs (policy, chosen)
        policy_logp_rejected: [B] sequence log-probs (policy, rejected)
        ref_logp_chosen: [B] sequence log-probs (reference, chosen)
        ref_logp_rejected: [B] sequence log-probs (reference, rejected)
        beta: DPO temperature (default 0.1)
        lambda_catto: calibration loss weight (default 0.1, from paper)
    
    Returns:
        dict with loss, loss_dpo, loss_cal_chosen, loss_cal_rejected
    """
    loss_dpo = dpo_loss(
        policy_logp_chosen, policy_logp_rejected,
        ref_logp_chosen, ref_logp_rejected,
        beta=beta
    )
    
    loss_cal_chosen = catto_token_calibration_loss(
        chosen_logits, chosen_labels, chosen_mask, invert_target=False
    )
    
    loss_cal_rejected = catto_token_calibration_loss(
        rejected_logits, rejected_labels, rejected_mask, invert_target=True
    )
    
    loss_total = loss_dpo + lambda_catto * (loss_cal_chosen + loss_cal_rejected)
    
    return {
        "loss": loss_total,
        "loss_dpo": loss_dpo.detach(),
        "loss_cal_chosen": loss_cal_chosen.detach(),
        "loss_cal_rejected": loss_cal_rejected.detach(),
    }


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":
    """Quick test with random tensors to verify shapes and gradients."""
    B, T, V = 2, 16, 32000  # batch, seq_len, vocab
    
    # Random inputs
    torch.manual_seed(42)
    chosen_logits = torch.randn(B, T, V, requires_grad=True)
    rejected_logits = torch.randn(B, T, V, requires_grad=True)
    chosen_labels = torch.randint(0, V, (B, T))
    rejected_labels = torch.randint(0, V, (B, T))
    chosen_mask = torch.ones(B, T)
    rejected_mask = torch.ones(B, T)
    
    # Sequence log-probs (scalar per sequence)
    policy_logp_chosen = torch.tensor([-1.5, -2.0], requires_grad=True)
    policy_logp_rejected = torch.tensor([-3.0, -2.5], requires_grad=True)
    ref_logp_chosen = torch.tensor([-1.6, -2.1])
    ref_logp_rejected = torch.tensor([-3.1, -2.6])
    
    # Compute loss
    result = catto_total_loss(
        chosen_logits, chosen_labels, chosen_mask,
        rejected_logits, rejected_labels, rejected_mask,
        policy_logp_chosen, policy_logp_rejected,
        ref_logp_chosen, ref_logp_rejected,
    )
    
    print("CATTO Loss Smoke Test:")
    print(f"  L_total:       {result['loss'].item():.4f}")
    print(f"  L_DPO:         {result['loss_dpo'].item():.4f}")
    print(f"  L_cal_chosen:  {result['loss_cal_chosen'].item():.4f}")
    print(f"  L_cal_rejected:{result['loss_cal_rejected'].item():.4f}")
    
    # Check gradients flow
    result["loss"].backward()
    print(f"  Grad flows to chosen_logits: {chosen_logits.grad is not None}")
    print(f"  Grad flows to policy_logp:   {policy_logp_chosen.grad is not None}")
    print("\nSmoke test passed.")
