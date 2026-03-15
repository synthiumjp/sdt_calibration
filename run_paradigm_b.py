"""
run_paradigm_b.py — Paradigm B data collection for SDT Calibration Project 4.1

4AFC forced choice at T=1.0 for all 3 models. Extracts first-token
log-probabilities for {A, B, C, D}. Records all 4 signals per trial (§A.8).

Pre-registration references:
  - Paradigm B: §5.3.2, Appendix A §A.5
  - Base model compliance: §A.5.1
  - Option randomisation: seed=42, uniform across {A,B,C,D}
  - Signals: Appendix A §A.8

Usage:
    python run_paradigm_b.py --model llama3_instruct
    python run_paradigm_b.py --all

~20 min total for all 3 models.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference_engine import SDTInferenceEngine, MODEL_CONFIGS, NumpyEncoder


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_4afc_data(base_dir: str = r"C:\sdt_calibration") -> list:
    """Load frozen 4AFC distractor set."""
    path = Path(base_dir) / "data" / "4afc_2000.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} 4AFC questions")
    return data


# ---------------------------------------------------------------------------
# Main data collection
# ---------------------------------------------------------------------------

def run_paradigm_b(
    model_key: str,
    base_dir: str = r"C:\sdt_calibration",
):
    """Run Paradigm B (4AFC) for one model.

    The 4AFC data is pre-frozen with randomised option positions (seed=42).
    """
    output_dir = Path(base_dir) / "results" / "paradigm_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_key}_4afc.jsonl"

    # Load data
    data = load_4afc_data(base_dir)

    # Load model
    engine = SDTInferenceEngine(model_key, base_dir)

    # Tracking
    total = len(data)
    correct_count = 0
    compliance_failures = 0
    start_time = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Paradigm B (4AFC): {model_key}")
    print(f"Questions: {total}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    with open(output_file, "w", encoding="utf-8") as outf:
        for q_idx, item in enumerate(data):
            question = item["question"]
            question_id = item.get("question_id", f"4afc_{q_idx}")

            # The 4AFC set should have pre-randomised options
            # Data format from build_4afc.py: options dict with A/B/C/D keys, correct_label
            options = item.get("options", {})
            correct_position = item.get("correct_label", "")

            # If options are stored as a list, convert
            if isinstance(options, list):
                labels = ["A", "B", "C", "D"]
                options = {labels[i]: options[i] for i in range(4)}

            # Extract 4AFC choice
            try:
                logit_result = engine.extract_4afc_logits(question, options)
            except Exception as e:
                print(f"  Error on Q{q_idx}: {e}")
                logit_result = {
                    "model_choice": None,
                    "generated_text": "",
                    "generated_logprob": None,
                    "is_compliant": False,
                }

            # Diagnostic on first trial
            if q_idx == 0:
                print(f"  [Diagnostic] Generated:  {logit_result['generated_text']!r}")
                print(f"  [Diagnostic] Choice:     {logit_result['model_choice']}")
                print(f"  [Diagnostic] Compliant:  {logit_result['is_compliant']}")
                print(f"  [Diagnostic] Correct:    {correct_position}")

            model_choice = logit_result["model_choice"]

            # Correctness
            is_correct = model_choice == correct_position
            if is_correct:
                correct_count += 1

            # Compliance check (§A.5.1): did the model produce a valid label?
            if not logit_result["is_compliant"]:
                compliance_failures += 1

            # Build trial record
            trial = {
                "question_id": question_id,
                "question_index": q_idx,
                "model": model_key,
                # §A.8 Paradigm B signals (adapted — see deviation note)
                "model_choice": model_choice,
                "correct": is_correct,
                "correct_position": correct_position,
                # Auxiliary
                "generated_text": logit_result["generated_text"],
                "generated_logprob": logit_result["generated_logprob"],
                "is_compliant": logit_result["is_compliant"],
            }

            outf.write(json.dumps(trial, cls=NumpyEncoder) + "\n")

            # Progress
            if (q_idx + 1) % 200 == 0 or q_idx == total - 1:
                elapsed = time.perf_counter() - start_time
                acc = correct_count / (q_idx + 1) * 100
                compliance_rate = 1.0 - compliance_failures / (q_idx + 1)
                print(
                    f"  Q {q_idx+1}/{total} | "
                    f"Acc: {acc:.1f}% | "
                    f"Compliance: {compliance_rate:.3f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    engine.unload()

    # Metadata
    elapsed = time.perf_counter() - start_time
    accuracy = correct_count / total
    compliance_rate = 1.0 - compliance_failures / total

    meta = {
        "model": model_key,
        "total_questions": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "compliance_failures": compliance_failures,
        "compliance_rate": compliance_rate,
        "time_s": elapsed,
        "output_file": str(output_file),
        "timestamp": datetime.now().isoformat(),
    }

    # Base model compliance check per §A.5.1
    # Adapted: compliance = model generated a valid label {A,B,C,D}
    if model_key == "llama3_base":
        if compliance_failures / total > 0.50:
            meta["base_model_excluded"] = True
            meta["exclusion_reason"] = (
                f"Model failed to generate a valid label on {compliance_failures}/{total} "
                f"({compliance_failures/total:.1%}) trials, exceeding 50% threshold"
            )
            print(f"\n  WARNING: Base model FAILS compliance ({compliance_failures/total:.1%} non-label)")
        else:
            meta["base_model_excluded"] = False

    meta_file = output_dir / f"{model_key}_4afc_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {total} trials in {elapsed:.1f}s")
    print(f"Accuracy: {accuracy:.3f} ({correct_count}/{total})")
    print(f"Compliance rate: {compliance_rate:.3f}")
    print(f"Results: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paradigm B (4AFC) data collection")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to run",
    )
    parser.add_argument(
        "--base-dir",
        default=r"C:\sdt_calibration",
        help="Base project directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models sequentially",
    )

    args = parser.parse_args()

    if args.all:
        for model_key in MODEL_CONFIGS:
            print(f"\n{'#'*60}")
            print(f"# Paradigm B: {model_key}")
            print(f"{'#'*60}")
            run_paradigm_b(model_key, args.base_dir)
    else:
        if not args.model:
            parser.error("Must specify --model or use --all")
        run_paradigm_b(args.model, args.base_dir)


if __name__ == "__main__":
    main()
