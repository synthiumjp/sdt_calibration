"""
run_all.py — Master orchestration script for SDT Calibration Project 4.1

Runs the complete data collection and analysis pipeline in the correct order:
  1. Paradigm A (generation) — ~30h GPU total, one model at a time
  2. Paradigm B (4AFC) — ~20 min total
  3. Analysis A (force-decode) — ~15 min total
  4. Analysis pipeline — ~1h CPU (plus ~4h for bootstrap)

Usage:
    # Run everything sequentially:
    python run_all.py

    # Run specific phases:
    python run_all.py --phase paradigm_a
    python run_all.py --phase paradigm_b
    python run_all.py --phase analysis_a
    python run_all.py --phase analysis
    python run_all.py --phase bootstrap

    # Run a single model (for overnight runs):
    python run_all.py --phase paradigm_a --model llama3_instruct
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def phase_paradigm_a(model=None, base_dir=r"C:\sdt_calibration"):
    """Run Paradigm A data collection."""
    from run_paradigm_a import run_paradigm_a
    from inference_engine import MODEL_CONFIGS

    models = [model] if model else list(MODEL_CONFIGS.keys())
    datasets = ["triviaqa", "nq"]

    for m in models:
        for d in datasets:
            print(f"\n{'#'*60}")
            print(f"# Paradigm A: {m} × {d}")
            print(f"# Started: {datetime.now().isoformat()}")
            print(f"{'#'*60}")
            run_paradigm_a(m, d, base_dir)


def phase_paradigm_b(model=None, base_dir=r"C:\sdt_calibration"):
    """Run Paradigm B data collection."""
    from run_paradigm_b import run_paradigm_b
    from inference_engine import MODEL_CONFIGS

    models = [model] if model else list(MODEL_CONFIGS.keys())

    for m in models:
        print(f"\n{'#'*60}")
        print(f"# Paradigm B: {m}")
        print(f"{'#'*60}")
        run_paradigm_b(m, base_dir)


def phase_analysis_a(model=None, base_dir=r"C:\sdt_calibration"):
    """Run Analysis A (force-decode)."""
    from run_analysis_a import run_analysis_a
    from inference_engine import MODEL_CONFIGS

    models = [model] if model else list(MODEL_CONFIGS.keys())
    datasets = ["triviaqa", "nq"]

    for m in models:
        for d in datasets:
            print(f"\n{'#'*60}")
            print(f"# Analysis A: {m} × {d}")
            print(f"{'#'*60}")
            run_analysis_a(m, d, base_dir)


def phase_analysis(base_dir=r"C:\sdt_calibration"):
    """Run full analysis pipeline."""
    from analysis_pipeline import run_full_analysis
    run_full_analysis(base_dir)


def phase_bootstrap(base_dir=r"C:\sdt_calibration"):
    """Run bootstrap CIs (CPU-intensive)."""
    from analysis_pipeline import run_bootstrap
    run_bootstrap(base_dir)


def main():
    parser = argparse.ArgumentParser(description="SDT Calibration master runner")
    parser.add_argument(
        "--phase",
        choices=["paradigm_a", "paradigm_b", "analysis_a", "analysis", "bootstrap", "all"],
        default="all",
    )
    parser.add_argument("--model", default=None, help="Specific model to run")
    parser.add_argument("--base-dir", default=r"C:\sdt_calibration")
    args = parser.parse_args()

    start = time.perf_counter()
    print(f"SDT Calibration — Run started at {datetime.now().isoformat()}")
    print(f"Base directory: {args.base_dir}")
    print(f"Phase: {args.phase}")

    if args.phase in ("paradigm_a", "all"):
        phase_paradigm_a(args.model, args.base_dir)

    if args.phase in ("paradigm_b", "all"):
        phase_paradigm_b(args.model, args.base_dir)

    if args.phase in ("analysis_a", "all"):
        phase_analysis_a(args.model, args.base_dir)

    if args.phase in ("analysis", "all"):
        phase_analysis(args.base_dir)

    if args.phase in ("bootstrap",):
        phase_bootstrap(args.base_dir)

    elapsed = time.perf_counter() - start
    print(f"\nTotal time: {elapsed/3600:.1f} hours")
    print(f"Finished at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
