"""
Smoke test v2 for Mistral-7B-Instruct-v0.3 and Llama-3-8B-base.
Phase 0 verification: logit extraction, answer quality, throughput.

FIX from v1: stop tokens now properly passed to llm() call.

Usage:
    python smoke_test_v2.py
"""

import time
import numpy as np
from pathlib import Path

MODEL_DIR = Path(r"C:\sdt_calibration\models")

MODELS = {
    "mistral-7b-instruct": {
        "path": MODEL_DIR / "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "type": "instruct",
        "stop": ["</s>"],
    },
    "llama-3-8b-base": {
        "path": MODEL_DIR / "Meta-Llama-3-8B.Q5_K_M.gguf",
        "type": "base",
        "stop": ["\n", "\n\n"],
    },
}

SYSTEM_MSG = "Answer the following question with a short factual answer. Respond with only the answer, nothing else."

TEST_QUESTIONS = [
    ("What is the capital of France?", ["Paris"]),
    ("Who wrote Romeo and Juliet?", ["William Shakespeare", "Shakespeare"]),
    ("What planet is closest to the Sun?", ["Mercury"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("In what year did World War II end?", ["1945"]),
    ("What is the largest ocean on Earth?", ["Pacific", "Pacific Ocean"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "Da Vinci", "Leonardo"]),
    ("What is the square root of 144?", ["12"]),
    ("What gas do plants absorb from the atmosphere?", ["Carbon dioxide", "CO2"]),
    ("What is the capital of Japan?", ["Tokyo"]),
]


def format_prompt_mistral(question: str) -> str:
    return f"[INST] {SYSTEM_MSG}\n\nQ: {question} [/INST]"


def format_prompt_llama3_base(question: str) -> str:
    return f"{SYSTEM_MSG}\n\nQ: {question}\nA:"


def check_answer(generated: str, accepted: list[str]) -> bool:
    gen_clean = generated.strip().lower().rstrip(".")
    for prefix in ["the answer is", "sure!", "sure,", "i think", "i believe", "it's", "it is"]:
        if gen_clean.startswith(prefix):
            gen_clean = gen_clean[len(prefix):].strip().lstrip(":").strip()
    for ans in accepted:
        if ans.lower() == gen_clean or ans.lower() in gen_clean or gen_clean in ans.lower():
            return True
    return False


def test_model(model_name: str, model_config: dict) -> dict:
    from llama_cpp import Llama

    model_path = model_config["path"]
    model_type = model_config["type"]
    stop_tokens = model_config["stop"]

    print(f"\n{'='*70}")
    print(f"TESTING: {model_name}")
    print(f"Stop tokens: {stop_tokens}")
    print(f"{'='*70}")

    if not model_path.exists():
        print(f"  ERROR: Model file not found at {model_path}")
        return {"status": "FAIL", "reason": "file_not_found"}

    file_size_gb = model_path.stat().st_size / (1024**3)
    print(f"  File size: {file_size_gb:.2f} GB")

    print(f"\n  Loading model...")
    t0 = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=512,
        logits_all=True,
        verbose=False,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if model_type == "instruct":
        format_fn = format_prompt_mistral
    else:
        format_fn = format_prompt_llama3_base

    results = []
    timings = []

    print(f"\n  Running {len(TEST_QUESTIONS)} test questions at T=1.0...\n")

    for i, (question, accepted) in enumerate(TEST_QUESTIONS):
        prompt = format_fn(question)

        t0 = time.time()
        output = llm(
            prompt,
            max_tokens=64,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.0,
            seed=i * 1000 + 4,
            logprobs=True,
            stop=stop_tokens,  # <-- THE FIX
        )
        elapsed = time.time() - t0
        timings.append(elapsed)

        gen_text = output["choices"][0]["text"].strip()
        correct = check_answer(gen_text, accepted)
        finish = output["choices"][0].get("finish_reason", "unknown")

        logprobs_data = output["choices"][0].get("logprobs")
        has_logprobs = logprobs_data is not None and logprobs_data.get("token_logprobs") is not None
        if has_logprobs:
            token_logprobs = logprobs_data["token_logprobs"]
            n_tokens = len(token_logprobs)
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            nlp = sum(valid_logprobs) / len(valid_logprobs) if valid_logprobs else None
        else:
            n_tokens = 0
            nlp = None

        status_str = "CORRECT" if correct else "WRONG"
        logprob_str = f"NLP={nlp:.3f}" if nlp is not None else "NO_LOGPROBS"
        print(f"  Q{i+1}: [{status_str}] {gen_text[:50]:<50} "
              f"({elapsed:.2f}s, {n_tokens}tok, {finish}, {logprob_str})")

        results.append({
            "question": question,
            "generated": gen_text,
            "correct": correct,
            "has_logprobs": has_logprobs,
            "nlp": nlp,
            "n_tokens": n_tokens,
            "finish_reason": finish,
            "time_s": elapsed,
        })

    # Summary
    n_correct = sum(1 for r in results if r.get("correct", False))
    n_logprobs = sum(1 for r in results if r.get("has_logprobs", False))
    avg_time = np.mean(timings) if timings else 0
    max_token_hits = sum(1 for r in results if r.get("finish_reason") == "length")

    print(f"\n  {'─'*50}")
    print(f"  SUMMARY: {model_name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy:           {n_correct}/{len(TEST_QUESTIONS)}")
    print(f"  Logprob extraction: {n_logprobs}/{len(TEST_QUESTIONS)}")
    print(f"  Avg time/question:  {avg_time:.3f}s")
    print(f"  Max-token hits:     {max_token_hits}/{len(TEST_QUESTIONS)} (should be 0 for short answers)")
    print(f"  Est. Paradigm A:    {avg_time * 8000 * 7 / 3600:.1f} hours")

    # Logit vector check
    print(f"\n  First-token logit vector check...")
    test_prompt = format_fn("What is the capital of France?")
    try:
        tokens = llm.tokenize(test_prompt.encode())
        llm.reset()
        llm.eval(tokens)
        logits = llm._scores
        if logits is not None and len(logits) > 0:
            last_logits = logits[-1]
            top_5_idx = np.argsort(last_logits)[-5:][::-1]
            print(f"  Vocab size: {len(last_logits)}")
            for idx in top_5_idx:
                tok_str = llm.detokenize([idx]).decode("utf-8", errors="replace")
                print(f"    [{idx:>6}] {last_logits[idx]:>8.3f}  '{tok_str}'")
            logit_ok = True
        else:
            logit_ok = False
            print(f"  WARNING: No logit scores available")
    except Exception as e:
        logit_ok = False
        print(f"  ERROR: {e}")

    # 4AFC check for base model
    if model_type == "base":
        print(f"\n  4AFC format compliance check...")
        afc_prompt = (
            f"{SYSTEM_MSG}\n\n"
            f"Q: What is the capital of France?\n"
            f"A) Paris\nB) London\nC) Berlin\nD) Madrid\n"
            f"Answer:"
        )
        try:
            tokens_4afc = llm.tokenize(afc_prompt.encode())
            llm.reset()
            llm.eval(tokens_4afc)
            logits_4afc = llm._scores[-1]

            label_logits = {}
            for label in ["A", "B", "C", "D"]:
                tids = llm.tokenize(f" {label}".encode(), add_bos=False)
                if tids:
                    label_logits[label] = float(logits_4afc[tids[0]])

            vals = np.array(list(label_logits.values()))
            exp_v = np.exp(vals - np.max(vals))
            probs = exp_v / exp_v.sum()
            print(f"  Relative probs: { {k: f'{p:.3f}' for k, p in zip(label_logits.keys(), probs)} }")

            all_exp = np.exp(logits_4afc.astype(np.float64) - np.max(logits_4afc))
            all_probs = all_exp / all_exp.sum()
            abs_mass = sum(
                float(all_probs[llm.tokenize(f" {l}".encode(), add_bos=False)[0]])
                for l in ["A", "B", "C", "D"]
            )
            print(f"  Absolute mass on {{A,B,C,D}}: {abs_mass:.4f}")
        except Exception as e:
            print(f"  4AFC error: {e}")

    accuracy_threshold = 5 if model_type == "base" else 7
    overall = "PASS" if (n_correct >= accuracy_threshold and n_logprobs == len(TEST_QUESTIONS) and logit_ok) else "FAIL"

    print(f"\n  ╔{'═'*48}╗")
    print(f"  ║  {model_name:^44}  ║")
    print(f"  ║  Overall: {overall:^38}  ║")
    print(f"  ╚{'═'*48}╝")

    del llm
    return {
        "status": overall,
        "accuracy": f"{n_correct}/{len(TEST_QUESTIONS)}",
        "logprobs_ok": n_logprobs == len(TEST_QUESTIONS),
        "logit_vector_ok": logit_ok,
        "avg_time_per_q": round(avg_time, 3),
        "max_token_hits": max_token_hits,
        "est_paradigm_a_hours": round(avg_time * 8000 * 7 / 3600, 1),
    }


def main():
    print("SDT Calibration — Phase 0 Smoke Tests v2")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_results = {}
    for name, config in MODELS.items():
        all_results[name] = test_model(name, config)

    print(f"\n\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Status':<8} {'Acc':<8} {'LP':<5} {'MaxTok':<8} {'Time/Q':<9} {'PA hrs'}")
    print(f"{'─'*25} {'─'*8} {'─'*8} {'─'*5} {'─'*8} {'─'*9} {'─'*8}")
    for name, r in all_results.items():
        print(f"{name:<25} {r['status']:<8} {r['accuracy']:<8} "
              f"{'OK' if r['logprobs_ok'] else 'FAIL':<5} "
              f"{r['max_token_hits']:<8} {r['avg_time_per_q']:<9} {r['est_paradigm_a_hours']}")

    print(f"\n  Ref: Llama-3-8B-Instruct = 0.29s/q, ~9h Paradigm A")
    print(f"\n  Key: MaxTok = trials hitting max_tokens (want 0)")


if __name__ == "__main__":
    main()
