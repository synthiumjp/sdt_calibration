"""
run_paradigm_a.py — Paradigm A data collection for SDT Calibration Project 4.1

Confidence-rating yes/no paradigm: generate answers at T = {0.1, 0.3, 0.5,
0.7, 1.0, 1.5, 2.0} for all 3 models × 2 datasets. Records all 7 signals
per trial (§A.8).

Pre-registration references:
  - Paradigm A: §5.3.1, Appendix A §A.3
  - Generation parameters: Appendix A §A.6
  - Scoring: Appendix A §A.7
  - Signals: Appendix A §A.8

Usage:
    python run_paradigm_a.py --model llama3_instruct --dataset triviaqa
    python run_paradigm_a.py --model llama3_instruct --dataset nq
    python run_paradigm_a.py --model mistral_instruct --dataset triviaqa
    ...

    # Or run all:
    python run_paradigm_a.py --all

Run one model at a time to fit in VRAM. Each model × dataset takes ~5h
for TriviaQA (5000 × 7 temps) or ~3h for NQ (3000 × 7 temps).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import (
    SDTInferenceEngine,
    TEMPERATURES,
    TEMP_INDEX,
    MODEL_CONFIGS,
    NumpyEncoder,
    ParadigmAResult,
)
from scoring import score_answer, strip_preamble, detect_refusal


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str, base_dir: str = r"C:\sdt_calibration") -> list:
    """Load TriviaQA or NQ dataset."""
    if dataset_name == "triviaqa":
        path = Path(base_dir) / "data" / "triviaqa_5000.json"
    elif dataset_name == "nq":
        path = Path(base_dir) / "data" / "nq_3000.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {dataset_name}")
    return data


def get_aliases(item: dict, dataset_name: str) -> list:
    """Extract answer aliases from a dataset item.

    Data format (from prepare_datasets.py):
      TriviaQA: answer_value, answer_aliases, answer_normalized_aliases
      NQ: answer_value, answer_aliases
    """
    if dataset_name == "triviaqa":
        aliases = list(item.get("answer_aliases", []))
        value = item.get("answer_value")
        if value and value not in aliases:
            aliases = [value] + aliases
        # Include normalized_aliases for broader matching
        norm_aliases = item.get("answer_normalized_aliases", [])
        for na in norm_aliases:
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


def get_question_text(item: dict, dataset_name: str) -> str:
    """Extract question text from a dataset item."""
    if dataset_name == "triviaqa":
        return item.get("question", "")
    elif dataset_name == "nq":
        return item.get("question", "")
    return ""


def get_question_id(item: dict, dataset_name: str, index: int) -> str:
    """Get a unique question ID."""
    if dataset_name == "triviaqa":
        return item.get("question_id", f"triviaqa_{index}")
    elif dataset_name == "nq":
        return item.get("question_id", f"nq_{index}")
    return f"{dataset_name}_{index}"


# ---------------------------------------------------------------------------
# Main data collection
# ---------------------------------------------------------------------------

def run_paradigm_a(
    model_key: str,
    dataset_name: str,
    base_dir: str = r"C:\sdt_calibration",
    resume_from: int = 0,
):
    """Run Paradigm A data collection for one model × one dataset.

    Saves results incrementally (every 100 questions) to handle crashes.
    """
    # Output directory
    output_dir = Path(base_dir) / "results" / "paradigm_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_key}_{dataset_name}.jsonl"

    # Load dataset
    data = load_dataset(dataset_name, base_dir)

    # Load model
    engine = SDTInferenceEngine(model_key, base_dir)

    # Resume support: count existing lines
    existing_lines = 0
    if output_file.exists() and resume_from == 0:
        with open(output_file, "r") as f:
            existing_lines = sum(1 for _ in f)
        # Each question produces len(TEMPERATURES) lines
        resume_from = existing_lines // len(TEMPERATURES)
        if resume_from > 0:
            print(f"Resuming from question {resume_from} ({existing_lines} lines exist)")

    # Open output file
    mode = "a" if resume_from > 0 else "w"
    outf = open(output_file, mode, encoding="utf-8")

    # Tracking
    total_questions = len(data)
    total_trials = total_questions * len(TEMPERATURES)
    completed = resume_from * len(TEMPERATURES)
    start_time = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Paradigm A: {model_key} × {dataset_name}")
    print(f"Questions: {total_questions}, Temperatures: {len(TEMPERATURES)}")
    print(f"Total trials: {total_trials}")
    print(f"Starting from question {resume_from}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    try:
        for q_idx in range(resume_from, total_questions):
            item = data[q_idx]
            question = get_question_text(item, dataset_name)
            question_id = get_question_id(item, dataset_name, q_idx)
            aliases = get_aliases(item, dataset_name)

            for temp in TEMPERATURES:
                # Generate
                result = engine.generate_paradigm_a(
                    question=question,
                    temperature=temp,
                    trial_index=q_idx,
                )

                # Score
                score_result = score_answer(result["generated_text"], aliases)

                # Build trial record
                trial = {
                    "question_id": question_id,
                    "question_index": q_idx,
                    "dataset": dataset_name,
                    "model": model_key,
                    "temperature": temp,
                    # §A.8 signals
                    "generated_text": result["generated_text"],
                    "first_token_top_logits": result["first_token_top_logits"],
                    "nlp": result["nlp"],
                    "answer_softmax_prob": result["answer_softmax_prob"],
                    "log_answer_prob": result["log_answer_prob"],
                    "correct": score_result["correct"],
                    "preamble_flag": score_result["preamble_flag"],
                    "refusal_flag": score_result["refusal_flag"],
                    # Auxiliary
                    "num_tokens": result["num_tokens"],
                    "raw_sequence_logprob": result["raw_sequence_logprob"],
                    "generation_time_s": result["generation_time_s"],
                    "seed": result["seed"],
                    "match_type": score_result["match_type"],
                    "best_similarity": score_result["best_similarity"],
                    "matched_alias": score_result["matched_alias"],
                    "stripped_text": score_result["stripped_text"],
                }

                # Write immediately (JSONL for streaming + crash safety)
                outf.write(json.dumps(trial, cls=NumpyEncoder) + "\n")
                completed += 1

            # Flush every question
            outf.flush()

            # Progress reporting every 100 questions
            if (q_idx + 1) % 100 == 0 or q_idx == total_questions - 1:
                elapsed = time.perf_counter() - start_time
                rate = (completed - resume_from * len(TEMPERATURES)) / elapsed if elapsed > 0 else 0
                eta_s = (total_trials - completed) / rate if rate > 0 else 0
                eta_h = eta_s / 3600

                # Quick accuracy check at T=1.0
                print(
                    f"  Q {q_idx+1}/{total_questions} | "
                    f"Trials: {completed}/{total_trials} | "
                    f"Rate: {rate:.1f} trials/s | "
                    f"ETA: {eta_h:.1f}h | "
                    f"Elapsed: {elapsed/3600:.1f}h"
                )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at question {q_idx}. Progress saved.")
        print(f"Resume with: --resume-from {q_idx}")
    finally:
        outf.close()
        engine.unload()

    # Write metadata
    meta = {
        "model": model_key,
        "dataset": dataset_name,
        "total_questions": total_questions,
        "temperatures": TEMPERATURES,
        "total_trials": total_trials,
        "completed_trials": completed,
        "output_file": str(output_file),
        "timestamp": datetime.now().isoformat(),
    }
    meta_file = output_dir / f"{model_key}_{dataset_name}_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {completed} trials saved to {output_file}")
    print(f"Metadata saved to {meta_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paradigm A data collection")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to run",
    )
    parser.add_argument(
        "--dataset",
        choices=["triviaqa", "nq"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--base-dir",
        default=r"C:\sdt_calibration",
        help="Base project directory",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume from question index",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all model × dataset combinations sequentially",
    )

    args = parser.parse_args()

    if args.all:
        # Run all combinations (one model at a time for VRAM)
        for model_key in MODEL_CONFIGS:
            for dataset in ["triviaqa", "nq"]:
                print(f"\n{'#'*60}")
                print(f"# Starting: {model_key} × {dataset}")
                print(f"{'#'*60}")
                run_paradigm_a(model_key, dataset, args.base_dir)
    else:
        if not args.model or not args.dataset:
            parser.error("Must specify --model and --dataset, or use --all")
        run_paradigm_a(args.model, args.dataset, args.base_dir, args.resume_from)


if __name__ == "__main__":
    main()
