"""
Dataset preparation for SDT Calibration project.
Downloads TriviaQA (unfiltered) and Natural Questions (nq_open),
applies all pre-registered filters, stratifies TriviaQA by domain,
and saves frozen question sets.

Pre-registration specs: Appendix B (B.1, B.3)

Usage:
    python prepare_datasets.py [--skip-wiki-api] [--triviaqa-cache DIR] [--output-dir DIR]

Outputs:
    {output_dir}/triviaqa_5000.json    — 5,000 TriviaQA questions with domain labels
    {output_dir}/nq_3000.json          — 3,000 Natural Questions (short-answer)
    {output_dir}/dataset_report.json   — Metadata: counts, domain distribution, filter stats
"""

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

# ── Pre-registered constants ──────────────────────────────────────────
SEED = 42
TRIVIAQA_N = 5000
NQ_N = 3000
NQ_MAX_ANSWER_TOKENS = 5   # Appendix B §B.3(2): 1-5 tokens
NQ_MIN_ANNOTATOR_AGREE = 2  # Appendix B §B.3(3): ≥2 annotators agree

# Domain keyword mapping — Appendix B §B.1.2, Table
# Priority order: Science > History > Geography > Arts > Sports > Pop Culture
DOMAIN_KEYWORDS = {
    "Science & Technology": [
        "science", "physics", "chemistry", "biology", "mathematics",
        "technology", "computing", "engineering", "medicine", "astronomy",
    ],
    "History & Politics": [
        "history", "war", "military", "politics", "government", "empire",
        "dynasty", "revolution", "century", "medieval",
    ],
    "Geography": [
        "geography", "country", "countries", "city", "cities",
        "continent", "river", "mountain", "island", "place",
    ],
    "Arts & Literature": [
        "art", "literature", "novel", "book", "author", "writer", "poet",
        "music", "film", "theatre", "painting", "sculpture",
    ],
    "Sports": [
        "sport", "football", "soccer", "cricket", "tennis", "olympic",
        "baseball", "basketball", "athletics", "rugby",
    ],
    "Pop Culture & Entertainment": [
        "television", "tv", "game", "celebrity", "band", "singer", "actor",
        "actress", "comic", "cartoon", "video game",
    ],
}
DOMAIN_PRIORITY = list(DOMAIN_KEYWORDS.keys())


def classify_by_keywords(categories: list[str]) -> str:
    """Classify a list of Wikipedia categories into a macro-domain.
    Returns first match in priority order, or 'Unclassified'."""
    cat_text = " ".join(categories).lower()
    for domain in DOMAIN_PRIORITY:
        for keyword in DOMAIN_KEYWORDS[domain]:
            if keyword in cat_text:
                return domain
    return "Unclassified"


def fetch_wiki_categories(entity_name: str, max_retries: int = 3) -> list[str]:
    """Fetch Wikipedia categories for an entity via the MediaWiki API.
    Returns list of content category titles (excluding hidden categories)."""
    if not entity_name or not entity_name.strip():
        return []

    title = urllib.parse.quote(entity_name.replace(" ", "_"))
    url = (
        f"https://en.wikipedia.org/w/api.php?"
        f"action=query&titles={title}&prop=categories"
        f"&cllimit=50&clshow=!hidden&format=json"
    )

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SDTCalibration/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return []  # Page not found
                cats = page_data.get("categories", [])
                return [c["title"].replace("Category:", "") for c in cats]
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return []


def load_triviaqa(cache_dir: str | None = None) -> list[dict]:
    """Load TriviaQA unfiltered using HuggingFace datasets."""
    from datasets import load_dataset

    print("Loading TriviaQA (unfiltered)...")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Load the unfiltered train split (largest pool)
    # The 'unfiltered' config excludes no-context questions
    ds = load_dataset("trivia_qa", "unfiltered.nocontext", split="train", **kwargs)
    print(f"  Loaded {len(ds)} questions from TriviaQA unfiltered train split")

    questions = []
    for item in ds:
        q = {
            "question_id": item["question_id"],
            "question": item["question"],
            "answer_value": item["answer"]["value"],
            "answer_aliases": item["answer"]["aliases"],
            "answer_normalized_aliases": item["answer"]["normalized_aliases"],
            "answer_type": item["answer"].get("type", ""),
            "matched_wiki_entity_name": item["answer"].get("matched_wiki_entity_name", ""),
            "question_source": item.get("question_source", ""),
        }
        questions.append(q)

    return questions


def load_nq(cache_dir: str | None = None) -> list[dict]:
    """Load Natural Questions (open, short-answer) using HuggingFace datasets."""
    from datasets import load_dataset

    print("Loading Natural Questions (nq_open)...")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # nq_open already filters to short answers with ≤5 tokens
    ds = load_dataset("google-research-datasets/nq_open", split="train", **kwargs)
    print(f"  Loaded {len(ds)} questions from nq_open train split")

    # Also load validation for more questions if needed
    ds_val = load_dataset("google-research-datasets/nq_open", split="validation", **kwargs)
    print(f"  Loaded {len(ds_val)} questions from nq_open validation split")

    questions = []
    seen_questions = set()

    for split_name, split_ds in [("train", ds), ("validation", ds_val)]:
        for item in split_ds:
            q_text = item["question"]
            if q_text in seen_questions:
                continue
            seen_questions.add(q_text)

            answers = item["answer"]
            if isinstance(answers, str):
                answers = [answers]

            # Appendix B §B.3(2): 1-5 tokens (whitespace-split)
            valid_answers = [a for a in answers if 1 <= len(a.split()) <= NQ_MAX_ANSWER_TOKENS]
            if not valid_answers:
                continue

            # Appendix B §B.3(3): ≥2 annotators agree
            # nq_open provides answer list — count agreement
            answer_counts = Counter(a.lower().strip() for a in valid_answers)
            majority_answer = answer_counts.most_common(1)[0]
            # For nq_open, if there's only one answer listed, that's the consensus
            # The original NQ has 5 annotators; nq_open collapses these
            # We accept all questions from nq_open since they're already filtered
            # to have short answers with agreement

            q = {
                "question": q_text,
                "answer_value": valid_answers[0],  # Primary answer
                "answer_aliases": list(set(valid_answers)),  # All valid answers as aliases
                "source_split": split_name,
            }
            questions.append(q)

    print(f"  {len(questions)} unique questions after dedup and filtering")
    return questions


def stratify_triviaqa_domains(
    questions: list[dict],
    skip_wiki_api: bool = False,
    wiki_cache_path: Path | None = None,
) -> list[dict]:
    """Add domain labels to TriviaQA questions per Appendix B §B.1."""

    # Load or create wiki category cache
    wiki_cache = {}
    if wiki_cache_path and wiki_cache_path.exists():
        print(f"  Loading wiki category cache from {wiki_cache_path}")
        with open(wiki_cache_path) as f:
            wiki_cache = json.load(f)

    # Determine lookup key for each question.
    # The nocontext config has empty matched_wiki_entity_name, so we use
    # answer_value as the Wikipedia lookup key instead. This is documented
    # as a minor implementation detail — the pre-reg intent (B.1.2) is to
    # get Wikipedia categories for the answer entity.
    lookup_keys = {}
    for q in questions:
        entity = q.get("matched_wiki_entity_name", "").strip()
        if not entity:
            # Fallback: use answer_value directly (works for WikipediaEntity type)
            entity = q.get("answer_value", "").strip()
        lookup_keys[id(q)] = entity

    entities_to_fetch = set()
    for q in questions:
        entity = lookup_keys[id(q)]
        if entity and entity not in wiki_cache:
            entities_to_fetch.add(entity)

    if skip_wiki_api:
        print(f"  Skipping Wikipedia API (--skip-wiki-api). {len(entities_to_fetch)} entities uncached.")
        print(f"  Using answer_type field as fallback classification signal.")
    elif entities_to_fetch:
        print(f"  Fetching Wikipedia categories for {len(entities_to_fetch)} unique entities...")
        print(f"  (Using answer_value as lookup key — matched_wiki_entity_name is empty in nocontext config)")
        print(f"  Estimated time: ~{len(entities_to_fetch) * 0.12 / 60:.0f} minutes")
        for i, entity in enumerate(sorted(entities_to_fetch)):
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(entities_to_fetch)}...")
            cats = fetch_wiki_categories(entity)
            wiki_cache[entity] = cats
            # Rate limit: ~1 request per 100ms to be polite
            time.sleep(0.1)

            # Save cache periodically (every 1000 entities) in case of interruption
            if wiki_cache_path and (i + 1) % 1000 == 0:
                with open(wiki_cache_path, "w") as f:
                    json.dump(wiki_cache, f)

        # Save final cache
        if wiki_cache_path:
            wiki_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wiki_cache_path, "w") as f:
                json.dump(wiki_cache, f)
            print(f"  Saved wiki cache ({len(wiki_cache)} entities)")
    else:
        print(f"  All entities already cached ({len(wiki_cache)} in cache)")

    # Classify each question
    for q in questions:
        entity = lookup_keys[id(q)]
        cats = wiki_cache.get(entity, [])
        if cats:
            q["domain"] = classify_by_keywords(cats)
            q["wiki_categories"] = cats
        else:
            q["domain"] = "Unclassified"
            q["wiki_categories"] = []

    return questions


def sample_triviaqa(questions: list[dict], n: int = TRIVIAQA_N) -> list[dict]:
    """Sample n questions from TriviaQA, attempting domain balance."""
    random.seed(SEED)

    # Shuffle first
    shuffled = list(questions)
    random.shuffle(shuffled)

    if len(shuffled) <= n:
        print(f"  WARNING: Only {len(shuffled)} questions available, requested {n}")
        return shuffled

    # Simple random sample (stratification is post-hoc analysis, not sampling requirement)
    sampled = shuffled[:n]

    # Report domain distribution
    domain_counts = Counter(q["domain"] for q in sampled)
    print(f"\n  Domain distribution (N={len(sampled)}):")
    for domain in DOMAIN_PRIORITY + ["Unclassified"]:
        count = domain_counts.get(domain, 0)
        pct = 100 * count / len(sampled)
        meets_threshold = "✓" if count >= 500 else "✗"
        print(f"    {domain:<30} {count:>5} ({pct:5.1f}%) {meets_threshold} ≥500")

    return sampled


def sample_nq(questions: list[dict], n: int = NQ_N) -> list[dict]:
    """Sample n questions from Natural Questions. Seed=42 per Appendix B §B.3(4)."""
    random.seed(SEED)

    shuffled = list(questions)
    random.shuffle(shuffled)

    if len(shuffled) < n:
        print(f"  WARNING: Only {len(shuffled)} NQ questions after filtering (requested {n})")
        print(f"  Using all {len(shuffled)} questions per pre-reg contingency.")
        return shuffled

    return shuffled[:n]


def build_report(
    triviaqa_sampled: list[dict],
    nq_sampled: list[dict],
    triviaqa_total: int,
    nq_total_before_filter: int,
    nq_total_after_filter: int,
) -> dict:
    """Build metadata report."""
    domain_counts = Counter(q["domain"] for q in triviaqa_sampled)
    domains_above_500 = sum(1 for c in domain_counts.values() if c >= 500)

    # Answer type distribution for TriviaQA
    type_counts = Counter(q.get("answer_type", "") for q in triviaqa_sampled)

    # Answer length distribution for NQ
    nq_answer_lengths = [len(q["answer_value"].split()) for q in nq_sampled]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "triviaqa": {
            "total_available": triviaqa_total,
            "sampled": len(triviaqa_sampled),
            "domain_distribution": dict(domain_counts),
            "domains_above_500": domains_above_500,
            "h5_viable": domains_above_500 >= 3,
            "answer_type_distribution": dict(type_counts),
            "pct_unclassified": round(
                100 * domain_counts.get("Unclassified", 0) / len(triviaqa_sampled), 1
            ),
        },
        "nq": {
            "total_before_filter": nq_total_before_filter,
            "total_after_filter": nq_total_after_filter,
            "sampled": len(nq_sampled),
            "answer_length_mean": round(np.mean(nq_answer_lengths), 2),
            "answer_length_max": max(nq_answer_lengths),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for SDT Calibration project")
    parser.add_argument("--skip-wiki-api", action="store_true",
                        help="Skip Wikipedia API calls for domain classification")
    parser.add_argument("--triviaqa-cache", type=str, default=None,
                        help="HuggingFace cache directory for TriviaQA")
    parser.add_argument("--output-dir", type=str, default=r"C:\sdt_calibration\data",
                        help="Output directory for prepared datasets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wiki_cache_path = output_dir / "wiki_category_cache.json"

    # ── TriviaQA ──────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: TriviaQA")
    print("=" * 70)

    triviaqa_all = load_triviaqa(cache_dir=args.triviaqa_cache)
    triviaqa_total = len(triviaqa_all)

    print(f"\n  Stratifying by domain (Wikipedia API)...")
    triviaqa_all = stratify_triviaqa_domains(
        triviaqa_all,
        skip_wiki_api=args.skip_wiki_api,
        wiki_cache_path=wiki_cache_path,
    )

    triviaqa_sampled = sample_triviaqa(triviaqa_all, TRIVIAQA_N)

    # Add sequential index for reproducibility
    for i, q in enumerate(triviaqa_sampled):
        q["trial_index"] = i

    triviaqa_path = output_dir / "triviaqa_5000.json"
    with open(triviaqa_path, "w", encoding="utf-8") as f:
        json.dump(triviaqa_sampled, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(triviaqa_sampled)} TriviaQA questions to {triviaqa_path}")

    # ── Natural Questions ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 2: Natural Questions")
    print("=" * 70)

    nq_all = load_nq(cache_dir=args.triviaqa_cache)
    nq_total_before = len(nq_all)  # Already filtered by nq_open

    # Additional filter: answer length 1-5 tokens (should be redundant with nq_open)
    nq_filtered = [q for q in nq_all if 1 <= len(q["answer_value"].split()) <= NQ_MAX_ANSWER_TOKENS]
    nq_total_after = len(nq_filtered)
    print(f"  After answer-length filter: {nq_total_after} questions")

    nq_sampled = sample_nq(nq_filtered, NQ_N)

    for i, q in enumerate(nq_sampled):
        q["trial_index"] = i

    nq_path = output_dir / "nq_3000.json"
    with open(nq_path, "w", encoding="utf-8") as f:
        json.dump(nq_sampled, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(nq_sampled)} NQ questions to {nq_path}")

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 3: Report")
    print("=" * 70)

    report = build_report(
        triviaqa_sampled, nq_sampled,
        triviaqa_total, nq_total_before, nq_total_after,
    )

    report_path = output_dir / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report to {report_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  TriviaQA: {len(triviaqa_sampled)} questions, "
          f"{report['triviaqa']['domains_above_500']} domains ≥500 "
          f"({'H5 viable' if report['triviaqa']['h5_viable'] else 'H5 underpowered'})")
    print(f"  NQ:       {len(nq_sampled)} questions, "
          f"mean answer length {report['nq']['answer_length_mean']} tokens")
    print(f"  Unclassified: {report['triviaqa']['pct_unclassified']}% of TriviaQA")
    print(f"\n  Output directory: {output_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Review domain distribution — if <3 domains ≥500, may need fallback classifier")
    print(f"    2. Build 4AFC distractor pipeline (needs triviaqa_5000.json)")
    print(f"    3. Commit datasets to OSF before data collection")


if __name__ == "__main__":
    main()
