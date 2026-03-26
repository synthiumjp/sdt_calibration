"""
Run this on your local machine at C:\sdt_calibration\

Reads all 6 Paradigm A JSONL files, trims to M1-required columns,
joins domain from the frozen item set, and outputs a single CSV.

Usage:
    python prep_m1_data.py

Output:
    results/m1_trial_data.csv
"""

import json
import csv
from pathlib import Path

RESULTS_DIR = Path("results/paradigm_a")
OUTPUT_PATH = Path("results/m1_trial_data.csv")

FILES = {
    ("llama3_instruct", "triviaqa"): "llama3_instruct_triviaqa.jsonl",
    ("llama3_instruct", "nq"):       "llama3_instruct_nq.jsonl",
    ("mistral_instruct", "triviaqa"): "mistral_instruct_triviaqa.jsonl",
    ("mistral_instruct", "nq"):       "mistral_instruct_nq.jsonl",
    ("llama3_base", "triviaqa"):      "llama3_base_triviaqa.jsonl",
    ("llama3_base", "nq"):            "llama3_base_nq.jsonl",
}

DOMAIN_FILE = Path("data/triviaqa_5000.json")


def load_domains(path):
    """Load trial_index -> domain mapping from frozen TriviaQA items."""
    domains = {}
    if not path.exists():
        print(f"WARNING: Domain file not found at {path}")
        return domains
    
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    for item in items:
        idx = item["trial_index"]
        dom = item["domain"]
        domains[idx] = dom
    
    print(f"Loaded {len(domains)} domain mappings from {path}")
    return domains


def detect_nlp_field(trial):
    """Figure out which field holds the NLP value."""
    for candidate in ["nlp", "log_prob", "norm_log_prob", "normalized_log_prob", "NLP"]:
        if candidate in trial:
            return candidate
    print(f"  Available fields: {list(trial.keys())}")
    return None


def main():
    domains = load_domains(DOMAIN_FILE)
    
    nlp_field = None
    rows = []
    
    for (model, dataset), filename in FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"MISSING: {path} — skipping")
            continue
        
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                trial = json.loads(line)
                
                # Auto-detect NLP field on first trial
                if nlp_field is None:
                    nlp_field = detect_nlp_field(trial)
                    if nlp_field is None:
                        print("ERROR: Cannot find NLP field.")
                        print(f"  Fields: {list(trial.keys())}")
                        return
                    print(f"Using NLP field: '{nlp_field}'")
                
                q_idx = trial.get("question_index")
                domain = domains.get(q_idx, "") if dataset == "triviaqa" else ""
                
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "temperature": trial["temperature"],
                    "question_index": q_idx,
                    "nlp": trial[nlp_field],
                    "correct": trial["correct"],
                    "domain": domain,
                })
                count += 1
        
        print(f"  {model} x {dataset}: {count} trials")
    
    # Write CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "temperature", 
                                                "question_index", "nlp", "correct", "domain"])
        writer.writeheader()
        writer.writerows(rows)
    
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone. {len(rows)} trials written to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
