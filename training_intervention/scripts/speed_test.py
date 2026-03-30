"""Quick speed test for inference with logprobs."""
from llama_cpp import Llama
import time

llm = Llama(
    model_path=r"C:\sdt_calibration\models\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    n_ctx=512, n_gpu_layers=-1, logits_all=True, verbose=False
)

prompt = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "What is the capital of France?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

t0 = time.time()
for i in range(10):
    r = llm.create_completion(
        prompt=prompt, max_tokens=32, temperature=1.0,
        logprobs=1, stop=["<|eot_id|>"]
    )
elapsed = time.time() - t0

print(f"10 questions in {elapsed:.1f}s = {10/elapsed:.2f} q/s")

answer = r["choices"][0]["text"].strip()
print(f"Answer: {answer}")

lps = r["choices"][0]["logprobs"]["token_logprobs"]
print(f"Logprobs: {lps}")

valid = [lp for lp in lps if lp is not None]
nlp = sum(valid) / len(valid) if valid else float("nan")
print(f"NLP: {nlp:.4f}")
print(f"Tokens: {len(valid)}")
