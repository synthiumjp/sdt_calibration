"""
inference_engine.py — Shared inference infrastructure for SDT Calibration Project 4.1

Provides model loading, prompt formatting, generation with logit extraction,
force-decode NLP computation, and first-token logit extraction.

Designed for reuse by the Weber's Law companion project (4.2).

Pre-registration references:
  - Prompt templates: Appendix A §A.2–A.5
  - Generation parameters: Appendix A §A.6
  - Seed strategy: Appendix A §A.6
  - Signals to record: Appendix A §A.8
"""

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# JSON helper — numpy types are not serializable by default
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Constants (locked per pre-registration)
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "Answer the following question with a short factual answer. "
    "Respond with only the answer, nothing else."
)

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
TEMP_INDEX = {0.1: 0, 0.3: 1, 0.5: 2, 0.7: 3, 1.0: 4, 1.5: 5, 2.0: 6}

MODEL_CONFIGS = {
    "llama3_instruct": {
        "path": r"models\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
        "model_type": "llama3_instruct",
    },
    "mistral_instruct": {
        "path": r"models\Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "stop_tokens": ["</s>"],
        "model_type": "mistral_instruct",
    },
    "llama3_base": {
        "path": r"models\Meta-Llama-3-8B.Q5_K_M.gguf",
        "stop_tokens": ["\n", "<|end_of_text|>"],
        "model_type": "llama3_base",
    },
}

# Top-K logits to store per first-token vector (§A.8(b): top-100)
TOP_K_LOGITS = 100


# ---------------------------------------------------------------------------
# Prompt formatting (Appendix A §A.3–A.5, exact templates)
# ---------------------------------------------------------------------------

def format_prompt_paradigm_a(model_type: str, question: str) -> str:
    """Format Paradigm A (generation) prompt per §A.3.

    Note: llama-cpp-python adds <|begin_of_text|> automatically for Llama-3.
    """
    if model_type == "llama3_instruct":
        # §A.3.1 — Llama-3 Instruct chat template
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_MSG}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Q: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "mistral_instruct":
        # §A.3.2 — Mistral v0.3 instruct template
        return f"[INST] {SYSTEM_MSG}\n\nQ: {question} [/INST]"
    elif model_type == "llama3_base":
        # §A.3.3 — Base model raw completion
        return f"{SYSTEM_MSG}\n\nQ: {question}\nA:"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def format_prompt_paradigm_b(model_type: str, question: str,
                              options: dict) -> str:
    """Format Paradigm B (4AFC) prompt per §A.5.

    options: dict with keys 'A', 'B', 'C', 'D' mapping to option text.
    """
    option_block = (
        f"A) {options['A']}\n"
        f"B) {options['B']}\n"
        f"C) {options['C']}\n"
        f"D) {options['D']}"
    )
    if model_type == "llama3_instruct":
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_MSG}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Q: {question}\n{option_block}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "mistral_instruct":
        return f"[INST] {SYSTEM_MSG}\n\nQ: {question}\n{option_block} [/INST]"
    elif model_type == "llama3_base":
        # §A.5 / continuation prompt §A.3.3 variant for 4AFC
        return (
            f"{SYSTEM_MSG}\n\n"
            f"Q: {question}\n{option_block}\nAnswer:"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Data classes for trial results
# ---------------------------------------------------------------------------

@dataclass
class ParadigmAResult:
    """Result for one Paradigm A trial (§A.8)."""
    question_id: str
    dataset: str
    model: str
    temperature: float
    # (a) Generated answer text (full, before stripping)
    generated_text: str
    # (b) First-token logit vector: top-100 logits + indices
    first_token_top_logits: list  # [(index, logit), ...]
    # (c) NLP of complete answer
    nlp: float
    # (d) Softmax probability of generated answer at temperature
    answer_softmax_prob: float
    # (e)–(g) filled by scoring pipeline
    correct: Optional[bool] = None
    preamble_flag: bool = False
    refusal_flag: bool = False
    # Auxiliary
    num_tokens: int = 0
    raw_sequence_logprob: float = 0.0  # unnormalized sum of log-probs
    generation_time_s: float = 0.0
    seed: int = 0


@dataclass
class ParadigmBResult:
    """Result for one Paradigm B trial (§A.8)."""
    question_id: str
    model: str
    # (a) Log-probs of {A, B, C, D} at first position
    label_logprobs: dict  # {'A': float, 'B': float, 'C': float, 'D': float}
    # (b) Model's choice
    model_choice: str
    # (c) Binary correctness
    correct: bool
    # (d) Position of correct answer
    correct_position: str
    # Auxiliary
    label_probs_sum: float = 0.0  # sum of softmax probs for compliance check


@dataclass
class AnalysisAResult:
    """Result for one Analysis A trial (§A.8)."""
    question_id: str
    dataset: str
    model: str
    # (a) NLP for force-decoded correct answer (max across aliases)
    nlp_correct: float
    # (b) NLP for force-decoded incorrect generation
    nlp_incorrect: Optional[float]  # None if T=1.0 answer was correct
    # (c) Which alias achieved max NLP
    best_alias: str
    # Auxiliary
    all_alias_nlps: dict = field(default_factory=dict)  # alias -> NLP


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class SDTInferenceEngine:
    """Wraps llama-cpp-python for SDT data collection.

    Handles model loading, generation with logit extraction,
    force-decoding, and first-token logit extraction.
    """

    def __init__(self, model_key: str, base_dir: str = r"C:\sdt_calibration"):
        from llama_cpp import Llama

        config = MODEL_CONFIGS[model_key]
        model_path = str(Path(base_dir) / config["path"])
        self.model_key = model_key
        self.model_type = config["model_type"]
        self.stop_tokens = config["stop_tokens"]

        print(f"Loading {model_key} from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=512,
            logits_all=True,
            verbose=False,
        )
        print(f"  Model loaded. Vocab size: {self.llm.n_vocab()}")

    def generate_paradigm_a(
        self,
        question: str,
        temperature: float,
        trial_index: int,
    ) -> dict:
        """Generate response and extract all Paradigm A signals.

        Returns dict with all §A.8 fields (except correctness, which
        comes from the scoring pipeline).
        """
        prompt = format_prompt_paradigm_a(self.model_type, question)
        temp_idx = TEMP_INDEX[temperature]
        seed = trial_index * 1000 + temp_idx

        t0 = time.perf_counter()
        result = self.llm(
            prompt,
            max_tokens=64,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.0,
            stop=self.stop_tokens,
            seed=seed,
            logprobs=TOP_K_LOGITS,  # request top-100 log-probs per token
        )
        gen_time = time.perf_counter() - t0

        # Extract generated text
        generated_text = result["choices"][0]["text"]

        # Extract token-level log-probabilities from the response
        logprobs_data = result["choices"][0].get("logprobs")

        nlp = 0.0
        raw_seq_logprob = 0.0
        num_tokens = 0
        answer_softmax_prob = 0.0
        log_answer_prob = float("-inf")
        first_token_top_logits = []

        if logprobs_data and logprobs_data.get("token_logprobs"):
            token_lps = logprobs_data["token_logprobs"]
            # Filter None values (first token may be None in some versions)
            valid_lps = [lp for lp in token_lps if lp is not None]
            num_tokens = len(valid_lps)

            if num_tokens > 0:
                raw_seq_logprob = sum(valid_lps)
                # (c) NLP = (1/L) * sum(log p(t_i | t_{<i}))
                nlp = raw_seq_logprob / num_tokens
                # (d) Softmax probability of the generated answer at T.
                # §A.8(d) / §B.4: this is the sequence probability
                # P(answer) = Π p(t_i | t_{<i}).
                # For multi-token answers this underflows to 0.0, so we
                # store the log-probability and clamp the exp() version.
                # The analysis pipeline uses log_answer_prob for ECE
                # when the linear-scale value is degenerate (see §B.4
                # methodological note in analysis_pipeline.py).
                log_answer_prob = raw_seq_logprob
                # Clamp to avoid underflow: exp(-745) ≈ 5e-324 (float64 min)
                answer_softmax_prob = math.exp(max(raw_seq_logprob, -700))

            # (b) First-token top logits
            # Use the top_logprobs field for the first generated token
            if logprobs_data.get("top_logprobs") and logprobs_data["top_logprobs"]:
                first_top = logprobs_data["top_logprobs"][0]
                if first_top:
                    # Convert to list of (token_str, logprob) sorted by logprob
                    sorted_logprobs = sorted(
                        first_top.items(), key=lambda x: x[1], reverse=True
                    )[:TOP_K_LOGITS]
                    first_token_top_logits = [
                        {"token": tok, "logprob": lp}
                        for tok, lp in sorted_logprobs
                    ]

        # Fall back to raw logit extraction only if API returned nothing.
        # The API may return fewer than TOP_K_LOGITS entries — that's fine.
        # The pre-reg specifies top-100 for storage efficiency, not as a
        # hard requirement. We record however many we get.
        if len(first_token_top_logits) == 0:
            first_token_top_logits = self._extract_first_token_logits_raw(prompt)

        return {
            "generated_text": generated_text,
            "first_token_top_logits": first_token_top_logits,
            "nlp": nlp,
            "answer_softmax_prob": answer_softmax_prob,
            "log_answer_prob": log_answer_prob,
            "num_tokens": num_tokens,
            "raw_sequence_logprob": raw_seq_logprob,
            "generation_time_s": gen_time,
            "seed": seed,
        }

    def _extract_first_token_logits_raw(self, prompt: str) -> list:
        """Extract top-100 logits at the first generated token position
        by evaluating the prompt and reading raw logits.

        This is the fallback if the logprobs API doesn't return enough
        top_logprobs entries.
        """
        try:
            # Tokenize the prompt
            prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"))
            # Evaluate prompt to fill KV cache and get logits
            self.llm.reset()
            self.llm.eval(prompt_tokens)
            # Get logits at the position after the last prompt token
            # This is the first generation position
            scores = self.llm.scores  # shape: (n_evaluated, n_vocab)
            if scores is not None and len(scores) > 0:
                last_logits = np.array(scores[-1])  # logits at generation position
                top_indices = np.argpartition(last_logits, -TOP_K_LOGITS)[-TOP_K_LOGITS:]
                top_indices = top_indices[np.argsort(last_logits[top_indices])[::-1]]
                return [
                    {"token_id": int(idx), "logit": float(last_logits[idx])}
                    for idx in top_indices
                ]
        except Exception as e:
            print(f"  Warning: raw logit extraction failed: {e}")
        return []

    def extract_4afc_logits(self, question: str, options: dict) -> dict:
        """Extract 4AFC choice by generating 1 token at T=1.0.

        Per §A.5: the model's choice is the label with the highest
        log-probability (argmax).

        Technical note: llama-cpp-python 0.3.16 (Vulkan) does not expose
        per-position logit vectors through the Python API. The model's
        choice is obtained by generating one token at T=1.0. The full
        four-way log-probability distribution is not available; proportion
        correct is used for d'_4AFC as pre-registered (§5.3.2 Step 3).
        Compliance is assessed as: did the model generate one of {A,B,C,D}.
        """
        prompt = format_prompt_paradigm_b(self.model_type, question, options)

        result = self.llm(
            prompt,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.0,
            logprobs=True,
            stop=self.stop_tokens,
        )

        generated = result["choices"][0]["text"].strip()

        # Extract the label from the generated text
        model_choice = None
        for label in ["A", "B", "C", "D"]:
            if generated.startswith(label):
                model_choice = label
                break

        # Format compliance: did the model produce a valid label?
        is_compliant = model_choice is not None

        # Get the logprob of the generated token (single value)
        logprobs_data = result["choices"][0].get("logprobs", {})
        token_lps = logprobs_data.get("token_logprobs", [])
        generated_logprob = float(token_lps[0]) if token_lps and token_lps[0] is not None else None

        return {
            "model_choice": model_choice,
            "generated_text": generated,
            "generated_logprob": generated_logprob,
            "is_compliant": is_compliant,
        }

    def force_decode_nlp(self, question: str, answer_text: str) -> float:
        """Force-decode an answer string and compute its NLP.

        Per §A.4: force-decode at T=1.0 (NLP is temperature-invariant
        because we use logits before temperature scaling).

        NLP = (1/L) * sum(log p(t_i | t_{<i}))
        """
        prompt = format_prompt_paradigm_a(self.model_type, question)

        # Tokenize prompt + answer
        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"))
        answer_tokens = self.llm.tokenize(answer_text.encode("utf-8"), add_bos=False)

        if len(answer_tokens) == 0:
            return float("-inf")

        # Evaluate full sequence (prompt + answer)
        full_tokens = prompt_tokens + answer_tokens
        self.llm.reset()
        self.llm.eval(full_tokens)

        # Get logits for all positions
        scores = self.llm.scores
        if scores is None or len(scores) == 0:
            return float("-inf")

        # For each answer token, get the log-prob from the position before it
        # Position of first answer token in the full sequence = len(prompt_tokens)
        # The logit at position i predicts token at position i+1
        # So logit at position (prompt_len - 1) predicts first answer token
        prompt_len = len(prompt_tokens)
        answer_len = len(answer_tokens)

        log_probs_sum = 0.0
        for i in range(answer_len):
            logit_pos = prompt_len - 1 + i  # position whose logits predict answer_tokens[i]
            if logit_pos >= len(scores):
                break
            logits = np.array(scores[logit_pos], dtype=np.float64)
            # Log-softmax (T=1.0, temperature-invariant)
            logits_shifted = logits - np.max(logits)
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted)))
            log_prob = logits_shifted[answer_tokens[i]] - log_sum_exp
            log_probs_sum += float(log_prob)

        nlp = log_probs_sum / answer_len
        return nlp

    def unload(self):
        """Free model from GPU memory."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
            # Force garbage collection to release VRAM
            import gc
            gc.collect()
