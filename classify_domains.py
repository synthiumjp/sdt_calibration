"""
Domain classification fallback — Appendix B §B.1.4

The Wikipedia API primary method yielded <3 viable domains (93% of answer_value
lookups returned no Wikipedia page). This script invokes the pre-registered
fallback: Llama-3-8B-Instruct at T=0.1 with the exact prompt from §B.1.4.

Usage:
    python classify_domains.py [--input data/triviaqa_5000.json] [--output data/triviaqa_5000.json]

Requires: Llama-3-8B-Instruct Q5_K_M loaded via llama-cpp-python (Vulkan GPU).
Estimated time: ~25 minutes (5,000 questions × ~0.3s/question).
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

# ── Pre-registered constants ──────────────────────────────────────────
VALID_DOMAINS = [
    "Science",
    "History",
    "Geography",
    "Arts",
    "Sports",
    "Pop Culture",
]

# Mapping from short LLM labels to the full macro-domain names used in the study
DOMAIN_MAP = {
    "science": "Science & Technology",
    "history": "History & Politics",
    "geography": "Geography",
    "arts": "Arts & Literature",
    "sports": "Sports",
    "pop culture": "Pop Culture & Entertainment",
}

MODEL_PATH = Path(r"C:\sdt_calibration\models\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf")

# Exact prompt from Appendix B §B.1.4
CLASSIFY_PROMPT_TEMPLATE = (
    "Classify this trivia question into one of: "
    "Science, History, Geography, Arts, Sports, Pop Culture. "
    "Question: {q}. Category:"
)


def build_prompt(question: str) -> str:
    """Build Llama-3 instruct prompt for classification."""
    system_msg = (
        "You are a question classifier. Respond with exactly one word: "
        "Science, History, Geography, Arts, Sports, or Pop Culture. "
        "Nothing else."
    )
    user_msg = CLASSIFY_PROMPT_TEMPLATE.format(q=question)

    # Llama-3 instruct format (llama-cpp-python adds <|begin_of_text|> automatically)
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def parse_domain(generated: str) -> str:
    """Parse LLM output into a valid domain label."""
    text = generated.strip().lower().rstrip(".").strip()

    # Direct match
    if text in DOMAIN_MAP:
        return DOMAIN_MAP[text]

    # Partial match (e.g., "pop culture" from "pop culture.")
    for key, domain in DOMAIN_MAP.items():
        if key in text:
            return domain

    # Check for the full domain names too
    for domain in DOMAIN_MAP.values():
        if domain.lower() in text:
            return domain

    return "Unclassified"


def main():
    parser = argparse.ArgumentParser(description="Classify TriviaQA questions by domain (fallback LLM method)")
    parser.add_argument("--input", type=str, default=r"C:\sdt_calibration\data\triviaqa_5000.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: overwrites input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print("=" * 70)
    print("DOMAIN CLASSIFICATION — Fallback Method (Appendix B §B.1.4)")
    print("=" * 70)
    print(f"  Method: Llama-3-8B-Instruct at T=0.1")
    print(f"  Reason: Wikipedia API primary method yielded <3 viable domains")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    # Load questions
    with open(input_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Questions: {len(questions)}")

    # Load model
    from llama_cpp import Llama

    print(f"\n  Loading Llama-3-8B-Instruct...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=512,
        logits_all=False,  # Don't need logits for classification
        verbose=False,
    )
    print(f"  Model loaded.")

    # Classify
    print(f"\n  Classifying {len(questions)} questions...\n")
    t_start = time.time()
    domain_counts = Counter()
    parse_failures = 0

    for i, q in enumerate(questions):
        prompt = build_prompt(q["question"])

        output = llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.0,
            seed=42,  # Fixed seed for reproducibility
        )

        raw = output["choices"][0]["text"]
        domain = parse_domain(raw)

        q["domain"] = domain
        q["domain_raw_llm_output"] = raw.strip()
        q["domain_method"] = "llm_fallback_b1.4"

        domain_counts[domain] += 1

        if domain == "Unclassified":
            parse_failures += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(questions) - i - 1) / rate
            print(f"    {i+1}/{len(questions)} ({rate:.1f} q/s, ~{remaining/60:.0f} min remaining)")

    elapsed_total = time.time() - t_start

    # Clean up model
    del llm

    # Report
    print(f"\n  {'─' * 50}")
    print(f"  DOMAIN DISTRIBUTION (N={len(questions)})")
    print(f"  {'─' * 50}")
    all_domains = [
        "Science & Technology", "History & Politics", "Geography",
        "Arts & Literature", "Sports", "Pop Culture & Entertainment",
        "Unclassified",
    ]
    viable_count = 0
    for domain in all_domains:
        count = domain_counts.get(domain, 0)
        pct = 100 * count / len(questions)
        meets = "✓" if count >= 500 else "✗"
        if count >= 500:
            viable_count += 1
        print(f"    {domain:<30} {count:>5} ({pct:5.1f}%) {meets} ≥500")

    print(f"\n  Parse failures (Unclassified): {parse_failures} ({100*parse_failures/len(questions):.1f}%)")
    print(f"  Viable domains (≥500): {viable_count}")
    print(f"  H5 status: {'viable' if viable_count >= 3 else 'underpowered'}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {output_path}")

    # Save classification report
    report_path = output_path.parent / "domain_classification_report.json"
    report = {
        "method": "llm_fallback_b1.4",
        "model": "Llama-3-8B-Instruct-Q5_K_M",
        "temperature": 0.1,
        "reason": "Wikipedia API primary method yielded <3 viable domains (93% lookup failure rate)",
        "wiki_api_stats": {
            "entities_attempted": 4000,
            "entities_resolved": 275,
            "failure_rate": 0.931,
        },
        "domain_distribution": dict(domain_counts),
        "viable_domains": viable_count,
        "parse_failures": parse_failures,
        "total_questions": len(questions),
        "time_seconds": round(elapsed_total, 1),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report to {report_path}")


if __name__ == "__main__":
    main()
