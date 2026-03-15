"""
scoring.py — Scoring pipeline for SDT Calibration Project 4.1

Implements the pre-registered scoring pipeline (Appendix A §A.7):
  1. Preamble stripping (§A.2)
  2. Refusal detection (§A.7)
  3. Exact match against aliases (case-insensitive, articles removed)
  4. String similarity fallback (difflib.SequenceMatcher ≥ 0.85)

Also provides robustness-check thresholds {0.80, 0.85, 0.90}.
"""

import re
from difflib import SequenceMatcher
from typing import Optional


# ---------------------------------------------------------------------------
# Preamble stripping (§A.2, pre-registered regex)
# ---------------------------------------------------------------------------

PREAMBLE_PATTERN = re.compile(
    r"^(The answer is|Sure[,!.]?|I think|I believe|It'?s|Here'?s)\s*[:.]?\s*",
    re.IGNORECASE,
)


def strip_preamble(text: str) -> tuple[str, bool]:
    """Strip preamble from generated text per §A.2.

    Returns:
        (stripped_text, preamble_was_present)
    """
    stripped = PREAMBLE_PATTERN.sub("", text)
    stripped = stripped.strip()
    preamble_present = stripped != text.strip()
    return stripped, preamble_present


# ---------------------------------------------------------------------------
# Refusal detection (§A.7, pre-registered regex)
# ---------------------------------------------------------------------------

REFUSAL_PATTERN = re.compile(
    r"^(I don'?t know|I'?m not sure|I cannot|I can'?t|Sorry|I don'?t have)",
    re.IGNORECASE,
)


def detect_refusal(text: str) -> bool:
    """Detect refusal per §A.7.

    Returns True if the text matches any refusal pattern.
    """
    return bool(REFUSAL_PATTERN.match(text.strip()))


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

# Articles to remove per §A.7 Step 1
ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Normalise answer for comparison: lowercase, remove articles, normalise whitespace."""
    text = text.lower().strip()
    text = ARTICLES.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Scoring pipeline (§A.7)
# ---------------------------------------------------------------------------

def score_answer(
    generated_text: str,
    aliases: list[str],
    similarity_threshold: float = 0.85,
) -> dict:
    """Score a generated answer against ground-truth aliases.

    Pipeline per §A.7:
      Step 1: Preamble stripping
      Step 2: Refusal detection
      Step 3: Exact match (case-insensitive, articles removed)
      Step 4: String similarity fallback (SequenceMatcher ≥ threshold)

    Returns dict with:
        correct: bool
        preamble_flag: bool
        refusal_flag: bool
        match_type: str ('exact', 'similarity', 'refusal', 'incorrect')
        best_similarity: float (max similarity score across aliases)
        matched_alias: str or None (which alias was matched)
        stripped_text: str (after preamble stripping)
    """
    # Step 1: Preamble stripping
    stripped, preamble_flag = strip_preamble(generated_text)

    # Step 2: Refusal detection
    refusal_flag = detect_refusal(generated_text)  # check original text too
    if not refusal_flag:
        refusal_flag = detect_refusal(stripped)

    if refusal_flag:
        # Per §6: refusals scored as correct rejection (noise trial, noise response)
        # The *correctness* label for the trial depends on whether the question
        # is signal or noise. For our purposes, the scoring pipeline returns
        # correct=False (model did not produce the answer) and the SDT pipeline
        # handles the signal/noise assignment.
        return {
            "correct": False,
            "preamble_flag": preamble_flag,
            "refusal_flag": True,
            "match_type": "refusal",
            "best_similarity": 0.0,
            "matched_alias": None,
            "stripped_text": stripped,
        }

    # Normalise the generated answer
    norm_answer = normalise_answer(stripped)

    # Step 3: Exact match against aliases
    best_similarity = 0.0
    matched_alias = None

    for alias in aliases:
        norm_alias = normalise_answer(alias)
        if norm_answer == norm_alias:
            return {
                "correct": True,
                "preamble_flag": preamble_flag,
                "refusal_flag": False,
                "match_type": "exact",
                "best_similarity": 1.0,
                "matched_alias": alias,
                "stripped_text": stripped,
            }

    # Step 4: String similarity fallback
    for alias in aliases:
        norm_alias = normalise_answer(alias)
        sim = SequenceMatcher(None, norm_answer, norm_alias).ratio()
        if sim > best_similarity:
            best_similarity = sim
            matched_alias = alias

    if best_similarity >= similarity_threshold:
        return {
            "correct": True,
            "preamble_flag": preamble_flag,
            "refusal_flag": False,
            "match_type": "similarity",
            "best_similarity": best_similarity,
            "matched_alias": matched_alias,
            "stripped_text": stripped,
        }

    return {
        "correct": False,
        "preamble_flag": preamble_flag,
        "refusal_flag": False,
        "match_type": "incorrect",
        "best_similarity": best_similarity,
        "matched_alias": matched_alias,
        "stripped_text": stripped,
    }


def score_answer_robustness(
    generated_text: str,
    aliases: list[str],
    thresholds: list[float] = [0.80, 0.85, 0.90],
) -> dict:
    """Score at multiple thresholds for robustness check per §A.7.

    Returns dict mapping threshold -> score_result.
    """
    return {t: score_answer(generated_text, aliases, t) for t in thresholds}
