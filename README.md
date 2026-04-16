# SDT Calibration: Signal Detection Theory for LLM Evaluation

## Overview

This repository contains code, data, and analysis scripts for two pre-registered studies applying Signal Detection Theory (SDT) to large language models (LLMs) on factual question-answering tasks.

**Paper 1 (Type-1 SDT):** Tests whether temperature scaling functions as a criterion shift—changing response bias without affecting sensitivity—analogous to payoff manipulations in human psychophysics. Three models, 168,000 trials. Under review at *Computational Brain & Behavior*. Pre-registered at [osf.io/qpk9a](https://osf.io/qpk9a). Preprint: [arXiv:2603.14893](https://arxiv.org/abs/2603.14893).

**Paper 2 (Type-2 SDT / M1):** Extends the framework to metacognitive efficiency using meta-d′ and M-ratio (Maniscalco & Lau, 2012; Fleming, 2017). Tests whether LLMs "know what they know" by measuring how well their internal confidence (NLP) monitors their own correctness, controlling for Type-1 sensitivity. Four models, 224,000 trials. Pre-registered at [osf.io/5q7mt](https://osf.io/5q7mt). Preprint: [arXiv:2604.XXXXX](https://arxiv.org/abs/2604.XXXXX) *(update with final arXiv ID once moderation clears)*. Submitted to NeurIPS 2026 Evaluations & Datasets Track.

## Key Findings

### Type-1 SDT (Paper 1)
- **Temperature is not a pure criterion manipulation.** It simultaneously changes sensitivity (AUC) and criterion (c).
- **LLMs exhibit unequal-variance evidence distributions.** z-ROC slopes range from 0.52 to 0.84.
- **The SDT decomposition reveals structure invisible to ECE.** Models with different sensitivity and bias profiles cannot be distinguished by calibration metrics alone.

### Type-2 SDT (Paper 2 / M1)
- **Metacognitive efficiency varies across models.** Mistral achieves the highest d′ (1.597) but the lowest M-ratio (0.852)—best discriminator, worst metacognitive monitor.
- **AUROC₂ and M-ratio produce fully inverted model rankings.** These metrics answer fundamentally different evaluation questions.
- **Metacognitive efficiency is domain-specific.** Different models have different weakest domains (range 0.31–0.70 within models), invisible to aggregate metrics.
- **Temperature dissociates confidence policy from metacognitive capacity** for instruction-tuned models (Mistral, Gemma) but not the base model.

## Pre-Registration

- **Type-1 (Paper 1):** [OSF Pre-Registration](https://osf.io/qpk9a)
- **Type-2 (Paper 2 / M1):** [OSF Pre-Registration](https://osf.io/5q7mt)

## Repository Structure

```
├── README.md
├── .gitignore
│
├── # Data Preparation
├── prepare_datasets.py          # Download and filter TriviaQA (5K) and NQ (3K)
├── classify_domains.py          # Domain classification for TriviaQA questions
├── build_4afc.py                # 4AFC distractor pipeline (embedding-based)
│
├── # Inference
├── inference_engine.py          # llama-cpp-python wrapper with logit extraction
├── run_paradigm_a.py            # Paradigm A: generation at 7 temperatures
├── run_paradigm_b.py            # Paradigm B: 4AFC forced choice
├── run_analysis_a.py            # Analysis A: force-decode
├── run_e2_prompt_criterion.py   # E2: prompt-based criterion manipulation
│
├── # Type-1 Analysis (Paper 1)
├── scoring.py                   # Exact match + string similarity scoring
├── analysis_pipeline.py         # ROC construction, UVSD fitting, bootstrap CIs
├── scoring_robustness.py        # Robustness across similarity thresholds
├── secondary_analyses.py        # H4-H6, E1, E5 analyses
├── sdt_equivalence_simulation.py # Monte Carlo equivalence bounds
├── quantile_bins_robustness.py  # Equal-count bin robustness check
├── build_spotcheck.py           # Human spot-check sampling tool
├── generate_figures.py          # 8 publication figures
│
├── # Type-2 Analysis (Paper 2 / M1)
├── m1_type2/
│   ├── m1_analysis.py           # Full meta-d′ pipeline: H1–H4, bootstrap, figures
│   └── results_4model/          # Four-model results
│       ├── h1_results.csv       # Aggregate M-ratio with bootstrap CIs
│       ├── h3_results.json      # Temperature analysis
│       ├── h4_results.json      # Hidden structure pairwise comparisons
│       └── figures/             # Publication figures
│           ├── fig1_dprime_vs_metad.png
│           ├── fig2_domain_mratio.png
│           ├── fig3_temperature.png
│           ├── fig4_auroc_vs_mratio.png
│           └── fig5_selective_prediction.png
│
├── # Spot-check
├── spotcheck_final.xlsx         # 1,200 human-scored judgments
│
├── data/                        # Prepared datasets (not tracked)
│   ├── triviaqa_5000.json
│   ├── nq_3000.json
│   └── 4afc_2000.json
│
├── results/                     # Raw outputs and analysis results
│   ├── m1_trial_data.csv        # 224,000 trial-level data (4 models)
│   ├── paradigm_a/              # Raw generation outputs per model × temperature
│   ├── paradigm_b/              # 4AFC outputs
│   ├── analysis_a/              # Force-decode outputs
│   └── analysis/                # Type-1 analysis results
│       ├── full_results.json
│       ├── bootstrap_results.json
│       ├── roc_data.json
│       ├── scoring_robustness.json
│       ├── secondary_analyses.json
│       ├── quantile_bins_robustness.json
│       └── figures/
│
└── simulation_results/
    └── equivalence_bounds.json
```

## Models

| Model | Parameters | Quantisation | Family | Paper |
|---|---|---|---|---|
| Llama-3-8B-Instruct | 8B | Q5_K_M | Meta | 1, 2 |
| Mistral-7B-Instruct-v0.3 | 7B | Q5_K_M | Mistral AI | 1, 2 |
| Llama-3-8B-Base | 8B | Q5_K_M | Meta | 1, 2 |
| Gemma-2-9B-Instruct | 9B | Q5_K_M | Google | 2 |

Inference via llama-cpp-python 0.3.16 (Vulkan backend) on AMD RX 7900 GRE (16GB VRAM).

## Datasets

- **TriviaQA:** 5,000 questions (unfiltered set, seed=42), classified into 4 knowledge domains + Unclassified
- **Natural Questions:** 3,000 short-answer questions (NQ-Open subset)

## Design

### Paper 1 (Type-1 SDT)
- **Paradigm A:** 3 models × 2 datasets × 7 temperatures × 5,000/3,000 questions = 168,000 trials
- **Paradigm B:** 3 models × 2,000 TriviaQA questions × 4AFC at T=1.0 = 6,000 trials
- **Analysis A:** Force-decode at T=1.0 for all models × both datasets

### Paper 2 (Type-2 SDT / M1)
- **Paradigm A extended:** 4 models × 2 datasets × 7 temperatures = 224,000 trials
- **Type-2 pipeline:** NLP → 8 quantile bins → nR_S1/nR_S2 count arrays → MLE meta-d′ → M-ratio
- **Bootstrap:** 10,000 resamples per condition, seed=42
- **Robustness:** nRatings ∈ {3, 4, 6}, UVSDT, equal-width bins, NQ replication

## Reproduction

### Requirements
```bash
# Paper 1
pip install numpy scipy matplotlib seaborn

# Paper 2 (additional)
pip install metadpy pymc arviz scikit-learn
```

### Type-1 Figures
```bash
python generate_figures.py
```

### Type-2 Analysis (Paper 2)
```bash
cd m1_type2

# Quick run (point estimates, no bootstrap)
python m1_analysis.py --data ../results/m1_trial_data.csv --output results_4model/ --skip-bootstrap

# Full run (10,000 bootstrap resamples — several hours)
python m1_analysis.py --data ../results/m1_trial_data.csv --output results_4model/ --n-bootstrap 10000
```

### Full Inference Pipeline

The full inference pipeline requires local GPU access and model files. Scripts are provided for transparency and reproducibility. Key dependencies:

- `llama-cpp-python >= 0.3.16` (with Vulkan or CUDA backend)
- `nomic-ai/nomic-embed-text-v1.5` (for 4AFC distractor pipeline)
- `difflib` (standard library, for scoring)

## Pre-Registration Deviations

### Paper 1 (Type-1 SDT)
Seven deviations from the pre-registered plan are documented in the paper's Supplementary Materials:

1. **Domain classification:** LLM fallback after Wikipedia API failure (93% entity resolution failure)
2. **Llama-3-Base source:** QuantFactory instead of bartowski repository
3. **NQ dataset:** `nq_open` subset instead of full NQ filtering
4. **MLE optimisation:** z-ROC regression initialisation (11 total fits) instead of 50 random restarts
5. **Paradigm B implementation:** minor adjustments to 4AFC format
6. **NLL vectorisation:** computational optimisation (no analytical change)
7. **Scoring pipeline:** missed-match rate 30.1% (exceeds 3% threshold; documented, not revised)

### Paper 2 (Type-2 SDT / M1)
One deviation: **Gemma-2-9B-Instruct was added post-registration** to test cross-family generalisability. All analysis procedures follow the pre-registered protocol. Domain collapse (Pop Culture & Entertainment and Sports merged into Unclassified) matches the Type-1 paper's domain structure.

## Citation

```bibtex
@article{cacioli2026llms,
  author  = {Cacioli, Jon-Paul},
  title   = {{LLMs} as Signal Detectors: Sensitivity, Bias, and the Temperature--Criterion Analogy},
  journal = {arXiv preprint arXiv:2603.14893},
  year    = {2026}
}

@article{cacioli2026metacognition,
  author  = {Cacioli, Jon-Paul},
  title   = {Do {LLMs} Know What They Know? {M}easuring Metacognitive Efficiency with Signal Detection Theory},
  journal = {arXiv preprint arXiv:2604.XXXXX},
  year    = {2026},
  note    = {Submitted to NeurIPS 2026 Evaluations \& Datasets Track}
}
```

## License

MIT
