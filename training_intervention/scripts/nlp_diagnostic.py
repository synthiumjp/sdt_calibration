import json, numpy as np, os

models = ['baseline', 'dpo_conditional', 'dpo_agnostic',
          'sft_conditional', 'sft_agnostic', 'catto_conditional']

print(f"{'Model':<25} {'Correct':>9} {'Incorrect':>10} {'Gap':>8} {'Acc':>6}")
print("-" * 62)

for name in models:
    fpath = f"results/pilot/pilot_trials_{name}.json"
    if not os.path.exists(fpath):
        print(f"{name:<25} MISSING")
        continue
    trials = json.load(open(fpath))
    c = [t["nlp"] for t in trials if t["correct"]]
    ic = [t["nlp"] for t in trials if not t["correct"]]
    acc = len(c) / len(trials)
    print(f"{name:<25} {np.mean(c):>9.4f} {np.mean(ic):>10.4f} {np.mean(c)-np.mean(ic):>8.4f} {acc:>6.3f}")
