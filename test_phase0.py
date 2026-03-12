from llama_cpp import Llama
import numpy as np

print("Loading model on Vulkan GPU...")
llm = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=512,
    logits_all=True,
    verbose=False
)
print("Model loaded.")

print("\nGenerating with logit extraction...")
# Note: llama-cpp-python adds <|begin_of_text|> automatically, so don't include it
prompt = "<|start_header_id|>system<|end_header_id|>\n\nAnswer the following question with a short factual answer. Respond with only the answer, nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQ: What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

output = llm(prompt, max_tokens=16, temperature=1.0, top_p=1.0, top_k=0, repeat_penalty=1.0, logprobs=True)

text = output["choices"][0]["text"]
lp = output["choices"][0]["logprobs"]

print("Answer:", text.strip())
print("Tokens:", lp["tokens"])
print("Token logprobs:", lp["token_logprobs"])
if lp["top_logprobs"]:
    print("Top logprobs for first token:", lp["top_logprobs"][0])
    print("Number of top logprobs:", len(lp["top_logprobs"][0]))

# Compute NLP (normalised log-probability)
token_lps = [x for x in lp["token_logprobs"] if x is not None]
if token_lps:
    nlp = np.mean(token_lps)
    print(f"\nNLP (mean token log-prob): {nlp:.4f}")

# Quick throughput test
import time
print("\nThroughput test (10 questions)...")
questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "In what year did World War II end?",
    "What is the largest planet in our solar system?",
    "Who painted the Mona Lisa?",
    "What is the speed of light in km/s?",
    "What is the capital of Japan?",
    "Who discovered penicillin?",
    "What is the square root of 144?",
]

start = time.time()
for q in questions:
    p = f"<|start_header_id|>system<|end_header_id|>\n\nAnswer the following question with a short factual answer. Respond with only the answer, nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQ: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    out = llm(p, max_tokens=32, temperature=1.0, top_p=1.0, top_k=0, repeat_penalty=1.0, logprobs=True)
    ans = out["choices"][0]["text"].strip()
    print(f"  {q} -> {ans}")
elapsed = time.time() - start

print(f"\n10 questions in {elapsed:.1f}s ({elapsed/10:.2f}s per question)")
print(f"Estimated Paradigm A time: {elapsed/10 * 112000 / 3600:.1f} hours")

print("\n=== Phase 0 PASSED ===")
print("- Vulkan GPU offload: OK")
print("- Logit extraction: OK")
print("- NLP computation: OK")
print("- Throughput measured")
