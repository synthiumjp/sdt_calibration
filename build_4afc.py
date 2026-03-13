"""
4AFC Distractor Pipeline — Appendix B §B.2

Samples 2,000 questions from the 5,000 TriviaQA set and generates 3 distractors
per question using:
  1. Domain-matched candidate pool
  2. Token-length filter (±30%)
  3. Cosine similarity filter (0.15–0.65) via nomic-embed-text-v1.5
  4. Random selection (seed = 42 + question_index)

Usage:
    python build_4afc.py [--input data/triviaqa_5000.json] [--output-dir data]

Outputs:
    {output_dir}/4afc_2000.json       — 2,000 questions with 4 options each
    {output_dir}/4afc_report.json     — Pipeline stats, rejection rates, edge cases
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Pre-registered constants ──────────────────────────────────────────
SEED = 42
N_4AFC = 2000
N_DISTRACTORS = 3
MIN_POOL_SIZE = 50           # §B.2.2(1): expand to full pool if domain pool < 50
LENGTH_TOLERANCE = 0.30      # §B.2.2(2): ±30% token length
COS_LOW = 0.15               # §B.2.2(3): minimum cosine similarity
COS_HIGH = 0.65              # §B.2.2(3): maximum cosine similarity
COS_LOW_RELAXED = 0.10       # §B.2.2(5): relaxed lower bound
COS_HIGH_RELAXED = 0.70      # §B.2.2(5): relaxed upper bound


def load_questions(path: Path) -> list[dict]:
    """Load the classified TriviaQA question set."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_4afc_questions(questions: list[dict], n: int = N_4AFC) -> list[dict]:
    """Randomly sample n questions for 4AFC. Seed = 42 per §B.2.2."""
    random.seed(SEED)
    shuffled = list(questions)
    random.shuffle(shuffled)
    sampled = shuffled[:n]
    # Preserve trial_index from the original set for traceability
    return sampled


def build_answer_pools(questions: list[dict]) -> tuple[dict[str, list[str]], list[str]]:
    """Build domain-specific and global answer pools from ALL 5,000 questions."""
    domain_pools = defaultdict(set)
    global_pool = set()

    for q in questions:
        answer = q["answer_value"]
        domain = q.get("domain", "Unclassified")
        domain_pools[domain].add(answer)
        global_pool.add(answer)

    # Convert to sorted lists for reproducibility
    domain_pools = {k: sorted(v) for k, v in domain_pools.items()}
    global_pool = sorted(global_pool)

    return domain_pools, global_pool


def token_length(s: str) -> int:
    """Whitespace-split token count. Minimum 1."""
    return max(1, len(s.split()))


def length_filter(candidate: str, correct: str, tolerance: float = LENGTH_TOLERANCE) -> bool:
    """Check if candidate is within ±tolerance of correct answer's token length."""
    correct_len = token_length(correct)
    candidate_len = token_length(candidate)
    lower = max(1, correct_len * (1 - tolerance))
    upper = correct_len * (1 + tolerance)
    return lower <= candidate_len <= upper


def load_embedding_model():
    """Load nomic-embed-text-v1.5 via sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print("  Loading nomic-embed-text-v1.5...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # Record model revision for pre-reg compliance (§B.2.1)
    try:
        revision = model._model_config.get("model_revision", "unknown")
    except Exception:
        revision = "unknown"

    print(f"  Model loaded. Revision: {revision}")
    return model, revision


def compute_embeddings(model, texts: list[str], prefix: str = "search_query: ") -> np.ndarray:
    """Compute embeddings with nomic's required prefix."""
    # nomic-embed-text-v1.5 requires a task prefix
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def select_distractors(
    correct_answer: str,
    correct_embedding: np.ndarray,
    candidate_answers: list[str],
    candidate_embeddings: np.ndarray,
    question_index: int,
    cos_low: float = COS_LOW,
    cos_high: float = COS_HIGH,
) -> tuple[list[str], dict]:
    """Select 3 distractors from candidates using length + cosine filters.

    Returns (distractors, stats_dict).
    """
    stats = {
        "n_candidates_initial": len(candidate_answers),
        "n_after_length": 0,
        "n_after_cosine": 0,
        "n_rejected_too_similar": 0,
        "n_rejected_too_dissimilar": 0,
        "method": "standard",
    }

    # Step 1: Remove the correct answer itself
    filtered_answers = []
    filtered_embeddings = []
    for i, ans in enumerate(candidate_answers):
        if ans.lower().strip() != correct_answer.lower().strip():
            filtered_answers.append(ans)
            filtered_embeddings.append(candidate_embeddings[i])

    # Step 2: Length filter (§B.2.2(2))
    length_passed = []
    for i, ans in enumerate(filtered_answers):
        if length_filter(ans, correct_answer):
            length_passed.append((ans, filtered_embeddings[i]))
    stats["n_after_length"] = len(length_passed)

    # Step 3: Cosine filter (§B.2.2(3))
    cosine_passed = []
    for ans, emb in length_passed:
        sim = cosine_similarity(correct_embedding, emb)
        if sim > cos_high:
            stats["n_rejected_too_similar"] += 1
        elif sim < cos_low:
            stats["n_rejected_too_dissimilar"] += 1
        else:
            cosine_passed.append((ans, sim))
    stats["n_after_cosine"] = len(cosine_passed)

    # Step 4: Random selection (§B.2.2(4))
    if len(cosine_passed) >= N_DISTRACTORS:
        random.seed(SEED + question_index)
        selected = random.sample(cosine_passed, N_DISTRACTORS)
        return [s[0] for s in selected], stats

    # Not enough — return what we have for now (caller handles edge cases)
    return [s[0] for s in cosine_passed], stats


def main():
    parser = argparse.ArgumentParser(description="Build 4AFC distractor set (Appendix B §B.2)")
    parser.add_argument("--input", type=str, default=r"C:\sdt_calibration\data\triviaqa_5000.json")
    parser.add_argument("--output-dir", type=str, default=r"C:\sdt_calibration\data")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("4AFC DISTRACTOR PIPELINE — Appendix B §B.2")
    print("=" * 70)

    # Load questions
    all_questions = load_questions(input_path)
    print(f"  Loaded {len(all_questions)} TriviaQA questions")

    # Sample 2,000 for 4AFC
    afc_questions = sample_4afc_questions(all_questions, N_4AFC)
    print(f"  Sampled {len(afc_questions)} for 4AFC (seed={SEED})")

    # Build answer pools
    domain_pools, global_pool = build_answer_pools(all_questions)
    print(f"  Global answer pool: {len(global_pool)} unique answers")
    for domain, pool in sorted(domain_pools.items()):
        print(f"    {domain:<30} {len(pool)} answers")

    # Load embedding model
    embed_model, model_revision = load_embedding_model()

    # Pre-compute embeddings for all unique answers
    print(f"\n  Computing embeddings for {len(global_pool)} unique answers...")
    t0 = time.time()
    all_embeddings = compute_embeddings(embed_model, global_pool)
    embed_time = time.time() - t0
    print(f"  Embeddings computed in {embed_time:.1f}s")

    # Build answer -> embedding index
    answer_to_idx = {ans: i for i, ans in enumerate(global_pool)}

    # ── Main pipeline ─────────────────────────────────────────────────
    print(f"\n  Building distractors for {len(afc_questions)} questions...")
    t0 = time.time()

    results = []
    all_stats = []
    edge_case_relaxed = 0
    edge_case_bypass = 0

    for qi, q in enumerate(afc_questions):
        correct = q["answer_value"]
        domain = q.get("domain", "Unclassified")
        q_idx = q.get("trial_index", qi)

        # Get correct answer embedding
        if correct in answer_to_idx:
            correct_emb = all_embeddings[answer_to_idx[correct]]
        else:
            # Answer not in pool (shouldn't happen, but handle gracefully)
            correct_emb = compute_embeddings(embed_model, [correct])[0]

        # Step 1: Domain-matched candidate pool (§B.2.2(1))
        pool = domain_pools.get(domain, [])
        if len(pool) < MIN_POOL_SIZE:
            pool = global_pool
            pool_source = "global (domain pool < 50)"
        else:
            pool_source = f"domain:{domain}"

        # Get embeddings for pool
        pool_embeddings = np.array([all_embeddings[answer_to_idx[a]] for a in pool])

        # Try standard thresholds
        distractors, stats = select_distractors(
            correct, correct_emb, pool, pool_embeddings, q_idx,
            cos_low=COS_LOW, cos_high=COS_HIGH,
        )
        stats["pool_source"] = pool_source

        # Edge case handling (§B.2.2(5))
        if len(distractors) < N_DISTRACTORS:
            # Try relaxed thresholds
            distractors_relaxed, stats_relaxed = select_distractors(
                correct, correct_emb, pool, pool_embeddings, q_idx,
                cos_low=COS_LOW_RELAXED, cos_high=COS_HIGH_RELAXED,
            )
            if len(distractors_relaxed) >= N_DISTRACTORS:
                distractors = distractors_relaxed
                stats = stats_relaxed
                stats["method"] = "relaxed"
                stats["pool_source"] = pool_source
                edge_case_relaxed += 1
            else:
                # Bypass: draw from full pool without cosine filtering
                random.seed(SEED + q_idx)
                fallback_candidates = [
                    a for a in global_pool
                    if a.lower().strip() != correct.lower().strip()
                    and length_filter(a, correct)
                ]
                if len(fallback_candidates) >= N_DISTRACTORS:
                    distractors = random.sample(fallback_candidates, N_DISTRACTORS)
                else:
                    # Last resort: no length filter either
                    all_candidates = [
                        a for a in global_pool
                        if a.lower().strip() != correct.lower().strip()
                    ]
                    distractors = random.sample(all_candidates, min(N_DISTRACTORS, len(all_candidates)))
                stats["method"] = "bypass"
                stats["pool_source"] = "global (bypass)"
                edge_case_bypass += 1

        # Construct the 4AFC item
        options = [correct] + distractors[:N_DISTRACTORS]
        # Shuffle options (fixed seed for reproducibility)
        random.seed(SEED + q_idx + 10000)
        random.shuffle(options)
        correct_label = chr(65 + options.index(correct))  # A, B, C, or D

        result = {
            "trial_index": q_idx,
            "question_id": q.get("question_id", ""),
            "question": q["question"],
            "correct_answer": correct,
            "options": {chr(65 + i): opt for i, opt in enumerate(options)},
            "correct_label": correct_label,
            "domain": domain,
            "distractor_method": stats["method"],
        }
        results.append(result)
        all_stats.append(stats)

        if (qi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {qi+1}/{len(afc_questions)} ({elapsed:.0f}s)")

    total_time = time.time() - t0

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  4AFC PIPELINE REPORT")
    print(f"  {'─' * 50}")
    print(f"  Questions: {len(results)}")
    print(f"  Standard method: {sum(1 for s in all_stats if s['method'] == 'standard')}")
    print(f"  Relaxed thresholds: {edge_case_relaxed}")
    print(f"  Bypass (no cosine): {edge_case_bypass}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Rejection rate stats
    total_rejected_similar = sum(s["n_rejected_too_similar"] for s in all_stats)
    total_rejected_dissimilar = sum(s["n_rejected_too_dissimilar"] for s in all_stats)
    total_after_length = sum(s["n_after_length"] for s in all_stats)
    print(f"\n  Cosine rejection stats (across all questions):")
    print(f"    Rejected too similar (>{COS_HIGH}):     {total_rejected_similar}")
    print(f"    Rejected too dissimilar (<{COS_LOW}):  {total_rejected_dissimilar}")
    if total_after_length > 0:
        print(f"    Overall cosine pass rate: "
              f"{sum(s['n_after_cosine'] for s in all_stats) / total_after_length:.1%}")

    # Domain distribution of 4AFC set
    afc_domain_counts = Counter(r["domain"] for r in results)
    print(f"\n  4AFC domain distribution:")
    for domain, count in sorted(afc_domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:<30} {count}")

    # Correct answer position distribution (should be ~uniform)
    label_counts = Counter(r["correct_label"] for r in results)
    print(f"\n  Correct answer position distribution:")
    for label in "ABCD":
        print(f"    {label}: {label_counts.get(label, 0)}")

    # ── Save ──────────────────────────────────────────────────────────
    afc_path = output_dir / "4afc_2000.json"
    with open(afc_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved 4AFC set to {afc_path}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "spec": "Appendix B §B.2",
        "n_questions": len(results),
        "n_standard": sum(1 for s in all_stats if s["method"] == "standard"),
        "n_relaxed": edge_case_relaxed,
        "n_bypass": edge_case_bypass,
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "embedding_revision": model_revision,
        "cosine_thresholds": {
            "standard": [COS_LOW, COS_HIGH],
            "relaxed": [COS_LOW_RELAXED, COS_HIGH_RELAXED],
        },
        "length_tolerance": LENGTH_TOLERANCE,
        "rejection_stats": {
            "too_similar": total_rejected_similar,
            "too_dissimilar": total_rejected_dissimilar,
            "cosine_pass_rate": round(
                sum(s["n_after_cosine"] for s in all_stats) / max(1, total_after_length), 4
            ),
        },
        "domain_distribution": dict(afc_domain_counts),
        "correct_label_distribution": dict(label_counts),
        "time_seconds": round(total_time, 1),
    }
    report_path = output_dir / "4afc_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report to {report_path}")

    # Git reminder
    print(f"\n  Next steps:")
    print(f"    1. Review 4afc_2000.json — spot-check a few questions")
    print(f"    2. git add build_4afc.py classify_domains.py")
    print(f"    3. git commit -m 'Phase 2: domain classification and 4AFC pipeline'")
    print(f"    4. Commit 4afc_2000.json + triviaqa_5000.json + nq_3000.json to OSF")


if __name__ == "__main__":
    main()
