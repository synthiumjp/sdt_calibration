# SDT Calibration: Signal Detection Theory for LLM Evaluation

## Overview

This repository contains code, data, and analysis scripts for two pre-registered studies applying Signal Detection Theory (SDT) to large language models (LLMs) on factual question-answering tasks.

**Paper 1 (Type-1 SDT):** Tests whether temperature scaling functions as a criterion shift—changing response bias without affecting sensitivity—analogous to payoff manipulations in human psychophysics. Five models (7B–70B), 229,000 trials. Submitted to EMNLP 2026 via ACL Rolling Review. Pre-registered at [osf.io/qpk9a](https://osf.io/qpk9a). Preprint: [arXiv:2603.14893](https://arxiv.org/abs/2603.14893).

**Paper 2 (Type-2 SDT / M1):** Applies SDT at the Type-2 level to the LLM confidence signal—characterising its Type-2 ROC, unequal-variance (z-ROC) structure, and metacognitive efficiency. Efficiency is quantified with a **model-free information measure** (normalised metacognitive information, meta-I₂ᵣ; Dayan 2023) rather than meta-d′/M-ratio, because open-ended QA has no two-alternative Type-1 decision (see the v2 note below). Four models, 224,000 trials. Pre-registered at [osf.io/5q7mt](https://osf.io/5q7mt). Preprint: [arXiv:2603.25112](https://arxiv.org/abs/2603.25112).

> ### ⚠️ Paper 2 correction (v2)
> An earlier version of Paper 2 (arXiv v1) estimated metacognitive efficiency with **meta-d′/M-ratio**. That estimator requires a two-alternative Type-1 detection decision, which open-ended factual QA does not provide. Mapping correctness onto S1/S2 makes d′ and meta-d′ functions of the **same** correctness-by-confidence table, so M-ratio is pinned near 1 by construction and the cross-model M-ratio differences reflect departures from the equal-variance assumption, not metacognitive efficiency. **v2 replaces M-ratio throughout with meta-I₂ᵣ** (model-free) and reports the Type-2 SDT structure (AUROC₂, z-ROC slope, d_a) directly. The direction of the cross-model finding **reverses** under the corrected measure. See `CHANGELOG_v2.md` and `paper_v2/`.

## Key Findings

### Type-1 SDT (Paper 1)

- **Temperature is not a pure criterion manipulation.** It simultaneously changes sensitivity (AUC) and criterion (c).
- **LLMs exhibit unequal-variance evidence distributions.** z-ROC slopes range from 0.56 to 0.78, with asymmetry intensifying with scale within the Meta family (8B: 0.63, 70B: 0.56).
- **The SDT decomposition reveals structure invisible to ECE.** Models with different sensitivity and bias profiles cannot be distinguished by calibration metrics alone.

### Type-2 SDT (Paper 2 / M1) — v2, model-free measures

- **Metacognitive information varies more than two-fold across models and is decoupled from accuracy.** Mistral has the **lowest** accuracy (0.427) yet the **highest** metacognitive information (meta-I₂ᵣ = 0.328); Gemma-2, the most accurate (0.600), has the **lowest** (0.143). This ordering replicates on Natural Questions. *(This reverses the v1 M-ratio claim, which was an artifact of the circular mapping.)*
- **The confidence signal has model-specific unequal-variance structure.** Type-2 z-ROC slopes range 0.81–1.18 (Mistral 0.81, Llama-3-Instruct 0.86, Gemma-2 0.99, Llama-3-Base 1.18); CIs exclude 1 for three of four models; ordering replicates on NQ. Invisible to ECE and AUROC₂.
- **Metacognitive information is domain-specific.** Arts & Literature is the strongest domain for every model; the weakest is Science & Technology or Geography. Within-model range up to 0.16.
- **Temperature dissociates accuracy from metacognitive information.** Accuracy falls monotonically with temperature (ρ = −1.00) while meta-I₂ᵣ stays near-flat for three of four models (range ≤ 0.025); Llama-3-Base is the exception.
- **Metacognitive information predicts selective-prediction gain.** meta-I₂ᵣ tracks the accuracy gain from confidence-based abstention (Spearman ρ = +0.80), not the absolute accuracy level (which follows base accuracy).

All Paper 2 estimates carry permutation nulls (guarding the small-sample bias of plug-in mutual information) and trial-level bootstrap confidence intervals.

## Pre-Registration

- **Type-1 (Paper 1):** [OSF](https://osf.io/qpk9a)
- **Type-2 (Paper 2 / M1):** [OSF](https://osf.io/5q7mt). Hypotheses were pre-registered in meta-d′/M-ratio terms; v2 tests the identical conceptual claims with meta-I₂ᵣ (see `CHANGELOG_v2.md` and the OSF addendum).

## Models

| Model                    | Parameters | Quantisation | Family     | Paper           |
| ------------------------ | ---------- | ------------ | ---------- | --------------- |
| Llama-3-8B-Instruct      | 8B         | Q5_K_M       | Meta       | 1, 2            |
| Mistral-7B-Instruct-v0.3 | 7B         | Q5_K_M       | Mistral AI | 1, 2            |
| Llama-3-8B-Base          | 8B         | Q5_K_M       | Meta       | 1, 2            |
| Gemma-2-9B-Instruct      | 9B         | Q5_K_M       | Google     | 1, 2            |
| Llama-3.1-70B-Instruct   | 70B        | bf16         | Meta       | 1 (scale probe) |

Inference: 7–9B models via llama-cpp-python 0.3.16 (Vulkan backend) on AMD RX 7900 GRE (16GB VRAM). 70B model via MLX on Apple M3 Ultra.

## Datasets

- **TriviaQA:** 5,000 questions (unfiltered, seed=42), 4 knowledge domains + Unclassified.
- **Natural Questions:** 3,000 short-answer questions (NQ-Open subset).

## Design (Paper 2 / M1) — v2

- **Data:** 4 models × 2 datasets × 7 temperatures = 224,000 trials.
- **Confidence:** NLP → 8 quantile bins.
- **Measures (model-free):**
  - meta-I₂ᵣ = I(correct; confidence) / H(correct), bias-corrected against a permutation null.
  - Type-2 AUROC₂ (non-parametric).
  - z-ROC slope s and d_a from linear regression on the empirical Type-2 z-ROC.
- **Inference:** 2,000 trial-level bootstrap resamples, seed=42; 2,000-shuffle permutation null.
- **Robustness:** binning K ∈ {4,6,8,10,16} (ordering stable), NQ replication.

> Note: the v1 pipeline (`m1_analysis.py`, `m1_type2/`) computed meta-d′/M-ratio via
> `NLP → nR_S1/nR_S2 → MLE meta-d′`. It is retained for provenance but **superseded** by
> `m1_type2/v2_analysis/` (see the v2 note above). Do not use M-ratio outputs as findings.

## Reproduction

### Requirements
```
pip install numpy scipy matplotlib seaborn pandas scikit-learn metadpy pymc arviz
```

### Type-2 v2 analysis (Paper 2, current)
```
cd m1_type2/v2_analysis
python analysis_v2.py       # meta-I₂ᵣ, AUROC₂, bootstrap CIs, permutation nulls
python sdt_structure.py     # z-ROC slopes, d_a, Type-2 ROC figure
# both read ../m1_trial_data.csv (224,000 trials)
```

### Type-2 v1 analysis (superseded, provenance only)
```
cd m1_type2
python m1_analysis.py --data ../results/m1_trial_data.csv --output results_4model/
# Produces meta-d′/M-ratio. See v2 note: M-ratio is not a valid efficiency measure here.
```

## Pre-Registration Deviations

### Paper 1 (Type-1 SDT)
Ten deviations documented in the paper's Supplementary Materials (domain classification fallback, model sources, MLE initialisation, etc.).

### Paper 2 (Type-2 SDT / M1)
1. **Gemma-2-9B-Instruct** added post-registration (cross-family generalisability).
2. **Measure change (v2):** metacognitive efficiency is reported with meta-I₂ᵣ rather than the pre-registered meta-d′/M-ratio, because meta-d′ is undefined for open-ended QA (no Type-1 decision). The conceptual hypotheses (cross-model variation, domain-specificity, temperature dissociation) are unchanged. Documented in `CHANGELOG_v2.md` and the OSF addendum.

## Citation

```
@article{cacioli2026llms,
  author  = {Cacioli, Jon-Paul},
  title   = {{LLMs} as Signal Detectors: Sensitivity, Bias, and the Temperature--Criterion Analogy},
  journal = {arXiv preprint arXiv:2603.14893},
  year    = {2026}
}

@article{cacioli2026metacognition,
  author  = {Cacioli, Jon-Paul},
  title   = {Do {LLMs} Know What They Know? {M}easuring Metacognitive Efficiency with Signal Detection Theory},
  journal = {arXiv preprint arXiv:2603.25112},
  year    = {2026}
}
```

## License

MIT
