"""Check tokenisation of DPO pairs — diagnose NaN loss."""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

with open("data/triviaqa/dpo_pairs_science_pilot.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

print(f"Total pairs: {len(pairs)}")

problems = 0
for i, p in enumerate(pairs[:20]):
    msgs_c = [{"role": "user", "content": p["prompt"]},
              {"role": "assistant", "content": p["chosen"]}]
    msgs_r = [{"role": "user", "content": p["prompt"]},
              {"role": "assistant", "content": p["rejected"]}]
    prompt_msgs = [{"role": "user", "content": p["prompt"]}]
    
    chosen_text = tok.apply_chat_template(msgs_c, tokenize=False, add_generation_prompt=False)
    rejected_text = tok.apply_chat_template(msgs_r, tokenize=False, add_generation_prompt=False)
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    
    chosen_enc = tok(chosen_text, max_length=256, truncation=True, return_tensors="pt")
    rejected_enc = tok(rejected_text, max_length=256, truncation=True, return_tensors="pt")
    prompt_enc = tok(prompt_text, max_length=256, truncation=True, return_tensors="pt")
    
    c_len = chosen_enc.input_ids.shape[1]
    r_len = rejected_enc.input_ids.shape[1]
    p_len = prompt_enc.input_ids.shape[1]
    c_resp = c_len - p_len
    r_resp = r_len - p_len
    
    flag = ""
    if c_resp <= 0 or r_resp <= 0:
        flag = " *** NO RESPONSE TOKENS ***"
        problems += 1
    
    if i < 5 or flag:
        print(f"\nPair {i}:{flag}")
        print(f"  Prompt tokens: {p_len}")
        print(f"  Chosen total: {c_len}, response: {c_resp}")
        print(f"  Rejected total: {r_len}, response: {r_resp}")
        print(f"  Chosen: {p['chosen'][:80]}")
        print(f"  Rejected: {p['rejected'][:80]}")

print(f"\nProblems (no response tokens): {problems}/20")
