"""
run_analysis_a.py — Analysis A (force-decode) for SDT Calibration Project 4.1

Force-decode correct answers and incorrect generations at T=1.0 to compute
temperature-invariant sensitivity estimates. Records 3 signals per trial (§A.8).

Also implements Amendment 1 (force-decode noise robustness: random incorrect
answer from same domain as alternative noise).

Pre-registration references:
  - Analysis A: §5.3.3, Appendix A §A.4
  - Signals: Appendix A §A.8
  - Amendment 1: Addendum

Depends on Paradigm A T=1.0 results (for incorrect generations).

Usage:
    python run_analysis_a.py --model llama3_instruct --dataset triviaqa
    python run_analysis_a.py --all

~15 min total for all models.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import SDTInferenceEngine, MODEL_CONFIGS, NumpyEncoder


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str, base_dir: str) -> list:
    """Load TriviaQA or NQ dataset."""
    if dataset_name == "triviaqa":
        path = Path(base_dir) / "data" / "triviaqa_5000.json"
    elif dataset_name == "nq":
        path = Path(base_dir) / "data" / "nq_3000.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_aliases(item: dict, dataset_name: str) -> list:
    """Extract answer aliases.

    Data format (from prepare_datasets.py):
      TriviaQA: answer_value, answer_aliases, answer_normalized_aliases
      NQ: answer_value, answer_aliases
    """
    if dataset_name == "triviaqa":
        aliases = list(item.get("answer_aliases", []))
        value = item.get("answer_value")
        if value and value not in aliases:
            aliases = [value] + aliases
        norm = item.get("answer_normalized_aliases", [])
        for na in norm:
            if na not in aliases:
                aliases.append(na)
        return aliases if aliases else ([value] if value else [])
    elif dataset_name == "nq":
        aliases = list(item.get("answer_aliases", []))
        value = item.get("answer_value")
        if value and value not in aliases:
            aliases = [value] + aliases
        return aliases if aliases else ([value] if value else [])
    return []


def load_paradigm_a_t10_results(
    model_key: str,
    dataset_name: str,
    base_dir: str,
) -> dict:
    """Load Paradigm A results at T=1.0 for this model × dataset.

    Returns dict: question_index -> {generated_text, correct, stripped_text}
    """
    result_file = (
        Path(base_dir) / "results" / "paradigm_a" / f"{model_key}_{dataset_name}.jsonl"
    )
    if not result_file.exists():
        raise FileNotFoundError(
            f"Paradigm A results not found: {result_file}\n"
            "Run Paradigm A first."
        )

    t10_results = {}
    with open(result_file, "r") as f:
        for line in f:
            trial = json.loads(line)
            if abs(trial["temperature"] - 1.0) < 0.01:
                t10_results[trial["question_index"]] = {
                    "generated_text": trial["generated_text"],
                    "correct": trial["correct"],
                    "stripped_text": trial.get("stripped_text", trial["generated_text"]),
                }

    print(f"Loaded {len(t10_results)} T=1.0 results from Paradigm A")
    return t10_results


# ---------------------------------------------------------------------------
# Main force-decode analysis
# ---------------------------------------------------------------------------

def run_analysis_a(
    model_key: str,
    dataset_name: str,
    base_dir: str = r"C:\sdt_calibration",
):
    """Run Analysis A for one model × one dataset.

    For each question:
      - Signal: force-decode each correct alias, take max NLP
      - Noise: force-decode the incorrect generation from T=1.0
      - Amendment 1: force-decode a random incorrect answer from same domain
    """
    output_dir = Path(base_dir) / "results" / "analysis_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_key}_{dataset_name}.jsonl"

    # Load data
    data = load_dataset(dataset_name, base_dir)
    t10_results = load_paradigm_a_t10_results(model_key, dataset_name, base_dir)

    # Build domain-indexed answer pool for Amendment 1
    domain_answers = defaultdict(list)
    for idx, item in enumerate(data):
        domain = item.get("domain", "Unclassified")
        aliases = get_aliases(item, dataset_name)
        if aliases:
            domain_answers[domain].append(aliases[0])  # primary answer

    # Load model
    engine = SDTInferenceEngine(model_key, base_dir)

    # Tracking
    total = len(data)
    signal_count = 0
    noise_count = 0
    start_time = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Analysis A (force-decode): {model_key} × {dataset_name}")
    print(f"Questions: {total}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    rng = random.Random(42)  # for Amendment 1 random selection

    with open(output_file, "w", encoding="utf-8") as outf:
        for q_idx in range(total):
            item = data[q_idx]
            question = item.get("question", "")
            question_id = item.get("question_id", f"{dataset_name}_{q_idx}")
            aliases = get_aliases(item, dataset_name)
            domain = item.get("domain", "Unclassified")

            # --- Signal evidence: force-decode each alias, take max NLP ---
            alias_nlps = {}
            for alias in aliases:
                if not alias.strip():
                    continue
                nlp = engine.force_decode_nlp(question, alias)
                alias_nlps[alias] = nlp

            if alias_nlps:
                best_alias = max(alias_nlps, key=alias_nlps.get)
                nlp_correct = alias_nlps[best_alias]
            else:
                best_alias = ""
                nlp_correct = float("-inf")

            signal_count += 1

            # --- Noise evidence: force-decode incorrect generation from T=1.0 ---
            # §A.4 says "force-decode the model's actual generated answer".
            # This is ambiguous regarding preamble. We force-decode both:
            #   (a) stripped_text (answer content only — primary)
            #   (b) generated_text (raw including preamble — robustness check)
            # Primary analysis uses stripped; difference is reported as metadata.
            nlp_incorrect = None
            nlp_incorrect_raw = None
            incorrect_text = None
            incorrect_text_raw = None
            t10 = t10_results.get(q_idx)
            if t10 and not t10["correct"]:
                incorrect_text = t10["stripped_text"]
                incorrect_text_raw = t10["generated_text"]
                if incorrect_text and incorrect_text.strip():
                    nlp_incorrect = engine.force_decode_nlp(question, incorrect_text)
                    noise_count += 1
                # Also force-decode the raw text if it differs from stripped
                if (incorrect_text_raw and incorrect_text_raw.strip()
                        and incorrect_text_raw.strip() != (incorrect_text or "").strip()):
                    nlp_incorrect_raw = engine.force_decode_nlp(
                        question, incorrect_text_raw
                    )

            # --- Amendment 1: random incorrect answer from same domain ---
            nlp_random_incorrect = None
            random_incorrect_text = None
            pool = domain_answers.get(domain, [])
            # Filter out the correct answer
            correct_set = set(a.lower().strip() for a in aliases)
            candidates = [a for a in pool if a.lower().strip() not in correct_set]
            if candidates:
                random_incorrect_text = rng.choice(candidates)
                nlp_random_incorrect = engine.force_decode_nlp(
                    question, random_incorrect_text
                )

            # Build trial record per §A.8
            trial = {
                "question_id": question_id,
                "question_index": q_idx,
                "dataset": dataset_name,
                "model": model_key,
                # §A.8 Analysis A signals
                "nlp_correct": nlp_correct,
                "nlp_incorrect": nlp_incorrect,  # primary: stripped text
                "best_alias": best_alias,
                # Auxiliary
                "all_alias_nlps": alias_nlps,
                "incorrect_text": incorrect_text,
                "incorrect_text_raw": incorrect_text_raw,
                "nlp_incorrect_raw": nlp_incorrect_raw,  # robustness: raw text
                "t10_was_correct": t10["correct"] if t10 else None,
                # Amendment 1
                "nlp_random_incorrect": nlp_random_incorrect,
                "random_incorrect_text": random_incorrect_text,
            }

            outf.write(json.dumps(trial, cls=NumpyEncoder) + "\n")

            # Progress
            if (q_idx + 1) % 500 == 0 or q_idx == total - 1:
                elapsed = time.perf_counter() - start_time
                rate = (q_idx + 1) / elapsed if elapsed > 0 else 0
                eta_s = (total - q_idx - 1) / rate if rate > 0 else 0
                print(
                    f"  Q {q_idx+1}/{total} | "
                    f"Signal: {signal_count} | Noise: {noise_count} | "
                    f"Rate: {rate:.1f} q/s | "
                    f"ETA: {eta_s:.0f}s"
                )

    engine.unload()

    # Metadata
    elapsed = time.perf_counter() - start_time
    meta = {
        "model": model_key,
        "dataset": dataset_name,
        "total_questions": total,
        "signal_trials": signal_count,
        "noise_trials": noise_count,
        "noise_fraction": noise_count / total if total > 0 else 0,
        "time_s": elapsed,
        "output_file": str(output_file),
        "timestamp": datetime.now().isoformat(),
    }
    meta_file = output_dir / f"{model_key}_{dataset_name}_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {total} questions in {elapsed:.1f}s")
    print(f"Signal trials: {signal_count}, Noise trials: {noise_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analysis A (force-decode)")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--dataset", choices=["triviaqa", "nq"])
    parser.add_argument("--base-dir", default=r"C:\sdt_calibration")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        for model_key in MODEL_CONFIGS:
            for dataset in ["triviaqa", "nq"]:
                print(f"\n{'#'*60}")
                print(f"# Analysis A: {model_key} × {dataset}")
                print(f"{'#'*60}")
                run_analysis_a(model_key, dataset, args.base_dir)
    else:
        if not args.model or not args.dataset:
            parser.error("Must specify --model and --dataset, or use --all")
        run_analysis_a(args.model, args.dataset, args.base_dir)


if __name__ == "__main__":
    main()
