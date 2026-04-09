import pandas as pd

df = pd.read_csv('results/pilot/baseline_llamacpp_science.csv')
print(f"C1 Science: N={len(df)}, acc={df.is_correct.mean():.3f}, NLP std={df.nlp.std():.4f}")
c = df[df.is_correct == True]['nlp']
ic = df[df.is_correct == False]['nlp']
print(f"  Correct mean={c.mean():.4f}, std={c.std():.4f}")
print(f"  Incorrect mean={ic.mean():.4f}, std={ic.std():.4f}")
print(f"  Gap={c.mean()-ic.mean():.4f}")

df2 = pd.read_csv('C:/sdt_calibration/m1_type2/m1_trial_data.csv')
# Check what models are in M1 data
print(f"\nM1 models: {df2['model'].unique() if 'model' in df2.columns else 'no model column'}")
sci = df2[df2.domain == "Science & Technology"]
# Filter to Llama only if model column exists
if 'model' in df2.columns:
    llama = sci[sci.model.str.contains('llama', case=False, na=False)]
    if len(llama) > 0:
        sci = llama
        print(f"Filtered to Llama: N={len(sci)}")
print(f"\nM1 Science: N={len(sci)}, acc={sci.correct.mean():.3f}, NLP std={sci.nlp.std():.4f}")
c2 = sci[sci.correct == True]['nlp']
ic2 = sci[sci.correct == False]['nlp']
print(f"  Correct mean={c2.mean():.4f}, std={c2.std():.4f}")
print(f"  Incorrect mean={ic2.mean():.4f}, std={ic2.std():.4f}")
print(f"  Gap={c2.mean()-ic2.mean():.4f}")
