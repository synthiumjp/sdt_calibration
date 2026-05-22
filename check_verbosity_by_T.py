"""
check_verbosity_by_T.py

Computes answer-length statistics per (model, dataset, temperature) from
paradigm-A JSONL files.

Usage (from C:\\sdt_calibration\\):
    python check_verbosity_by_T.py
"""

import os
import json
import glob
from collections import defaultdict

import pandas as pd

JSONL_DIR = os.path.join("results", "paradigm_a")
OUT_CSV   = os.path.join("results", "verbosity_by_T.csv")

rows = []

for path in sorted(glob.glob(os.path.join(JSONL_DIR, "*.jsonl"))):
    print(f"[read] {path}")
    by_T = defaultdict(lambda: {"tokens": [], "chars": [], "words": [],
                                "n_correct": 0, "n": 0})
    model = None
    dataset = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            T = round(r["temperature"], 1)
            txt = r.get("stripped_text") or r.get("generated_text") or ""
            by_T[T]["tokens"].append(r.get("num_tokens", 0))
            by_T[T]["chars"].append(len(txt))
            by_T[T]["words"].append(len(txt.split()))
            by_T[T]["n_correct"] += int(r.get("correct", False))
            by_T[T]["n"] += 1
            model = r.get("model", model)
            dataset = r.get("dataset", dataset)

    for T, d in sorted(by_T.items()):
        rows.append({
            "model": model,
            "dataset": dataset,
            "T": T,
            "n": d["n"],
            "median_tokens": pd.Series(d["tokens"]).median(),
            "mean_tokens":   pd.Series(d["tokens"]).mean(),
            "median_chars":  pd.Series(d["chars"]).median(),
            "median_words":  pd.Series(d["words"]).median(),
            "p90_tokens":    pd.Series(d["tokens"]).quantile(0.9),
            "acc":           d["n_correct"] / d["n"] if d["n"] else 0.0,
        })

if not rows:
    raise SystemExit("No data extracted -- check JSONL_DIR path.")

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\n[saved] {OUT_CSV}\n")

print("=== Median TOKENS by (model, dataset, T) ===")
print(df.pivot_table(index=["model", "dataset"], columns="T",
                     values="median_tokens").round(1))

print("\n=== Mean TOKENS by (model, dataset, T) ===")
print(df.pivot_table(index=["model", "dataset"], columns="T",
                     values="mean_tokens").round(1))

print("\n=== P90 TOKENS by (model, dataset, T) ===")
print(df.pivot_table(index=["model", "dataset"], columns="T",
                     values="p90_tokens").round(1))

print("\n=== Verbosity ratio T=2.0 / T=0.1 (mean tokens) ===")
for (m, ds), sub in df.groupby(["model", "dataset"]):
    sub = sub.set_index("T")
    if 0.1 in sub.index and 2.0 in sub.index:
        r = sub.loc[2.0, "mean_tokens"] / max(sub.loc[0.1, "mean_tokens"], 1e-9)
        print(f"  {m:>20s} / {ds:>8s}: {r:5.2f}x")
