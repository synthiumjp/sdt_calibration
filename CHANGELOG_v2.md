# v2 — metacognition measure correction

## Summary
v1 estimated metacognitive efficiency with meta-d'/M-ratio. That estimator needs a
two-alternative Type-1 detection decision, which open-ended QA does not have: forcing
correctness onto S1/S2 makes d' and meta-d' functions of the same
correctness-by-confidence table, so M-ratio is ~1 by construction. v2 removes it and
uses a model-free measure (normalised metacognitive information, meta-I_2r; Dayan 2023)
plus the directly-estimated Type-2 SDT structure (AUROC2, z-ROC slope, d_a).

## What changed in results
- Cross-model finding REVERSES: Mistral has the HIGHEST metacognitive information
  (0.328), Gemma the lowest (0.143). v1 reported the opposite via M-ratio.
- z-ROC slopes reported directly: Mistral 0.81, Llama-3-Instruct 0.86, Gemma 0.99,
  Llama-3-Base 1.18 (TriviaQA, T=1.0); ordering replicates on NQ.
- Domain-specificity SURVIVES (Arts & Lit strongest for all models).
- Temperature dissociation SURVIVES (meta-I_2r flat while accuracy falls monotonically).
- Selective prediction + NLP monotonicity unchanged.

## New / changed files
- m1_type2/v2_analysis/analysis_v2.py   meta-I_2r, AUROC2, bootstrap CI, permutation null
- m1_type2/v2_analysis/sdt_structure.py z-ROC slopes, d_a, Type-2 ROC figure
- m1_type2/results_v2/*.csv             regenerated results (aggregate, domain, temperature, sdt_structure)
- m1_type2/results_v2/figures/*.png     regenerated figures
- paper_v2/                             arXiv v2 source (tex, bib, tables, figures)

## Not changed
- m1_trial_data.csv (224,000 trials) — same raw data, re-analysed.
- v1 analysis files retained for provenance.
