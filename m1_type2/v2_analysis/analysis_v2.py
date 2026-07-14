"""
M1 v2 replacement analysis — model-free metacognition.

Primary measure: normalised metacognitive information (Dayan 2023 meta-I family)
    meta_I2r = I(accuracy ; confidence_bin) / H(accuracy)      [bits/bit]
  - model-free, no Type-1 d' required, valid for any task
  - permutation null (plug-in MI is upward biased at small n)
  - bootstrap 95% percentile CI (trial-level resampling)

Supporting: AUROC2 (Type-2 ROC area) — model-free discrimination, with bootstrap CI.

Outputs: results_v2/*.csv and a printed summary.
"""
import numpy as np, pandas as pd, os
from scipy import stats

SEED = 42
rng = np.random.default_rng(SEED)
N_RATINGS = 4
NBINS = 2 * N_RATINGS
N_BOOT = 2000
N_PERM = 2000

os.makedirs("results_v2", exist_ok=True)
df = pd.read_csv("sdt_calibration-main/m1_type2/m1_trial_data.csv")
df["correct"] = df["correct"].astype(int)

MODELS = ["llama3_instruct","mistral_instruct","llama3_base","gemma2_instruct"]
LABELS = {"llama3_instruct":"Llama-3-Instruct","mistral_instruct":"Mistral-Instruct",
          "llama3_base":"Llama-3-Base","gemma2_instruct":"Gemma-2-Instruct"}
DOMS = ["History & Politics","Arts & Literature","Geography","Science & Technology"]

# ---------- core measures ----------
def bin_edges(nlp):
    q = np.linspace(0,1,NBINS+1)[1:-1]
    return np.quantile(nlp, q)

def ratings(nlp, e):
    return np.digitize(nlp, e) + 1

def _mi_bits(acc, conf):
    n = len(acc)
    if n == 0: return 0.0
    mi = 0.0
    pa1 = acc.mean(); pa0 = 1-pa1
    for a,pa in ((0,pa0),(1,pa1)):
        if pa == 0: continue
        mask_a = (acc==a)
        for k in np.unique(conf):
            pk = (conf==k).mean()
            pak = (mask_a & (conf==k)).mean()
            if pak > 0:
                mi += pak*np.log2(pak/(pa*pk))
    return mi

def _Hacc(acc):
    p = acc.mean()
    if p<=0 or p>=1: return 0.0
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))

def auroc2(acc, conf):
    """Type-2 AUC: confidence discriminating correct vs incorrect (rank-based)."""
    order = np.argsort(conf, kind="mergesort")
    c = acc[order]
    # Mann-Whitney U over confidence ranks
    ranks = stats.rankdata(conf)
    n1 = acc.sum(); n0 = len(acc)-n1
    if n1==0 or n0==0: return np.nan
    R1 = ranks[acc==1].sum()
    U1 = R1 - n1*(n1+1)/2
    return U1/(n1*n0)

def meta_i2r_point(acc, conf):
    H = _Hacc(acc)
    if H==0: return np.nan
    return _mi_bits(acc,conf)/H

def analyse_cell(sub, e, n_boot=N_BOOT, n_perm=N_PERM):
    acc = sub.correct.values
    conf = ratings(sub.nlp.values, e)
    H = _Hacc(acc)
    if H==0 or len(np.unique(acc))<2:
        return None
    mi2r = _mi_bits(acc,conf)/H
    a2 = auroc2(acc,conf)
    # permutation null
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _mi_bits(acc, rng.permutation(conf))/H
    p_perm = (1+(null>=mi2r).sum())/(n_perm+1)
    mi2r_bc = mi2r - null.mean()
    # bootstrap CI (trial-level) on bias-corrected meta-I2r and AUROC2
    n = len(acc)
    bmi = np.empty(n_boot); ba = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0,n,n)
        a_b, c_b = acc[idx], conf[idx]
        Hb = _Hacc(a_b)
        bmi[b] = (_mi_bits(a_b,c_b)/Hb - null.mean()) if Hb>0 else np.nan
        ba[b]  = auroc2(a_b,c_b)
    mi_lo,mi_hi = np.nanpercentile(bmi,[2.5,97.5])
    a_lo,a_hi   = np.nanpercentile(ba,[2.5,97.5])
    return dict(acc=acc.mean(), n=n,
                meta_i2r=mi2r_bc, mi_lo=mi_lo, mi_hi=mi_hi, p_perm=p_perm,
                auroc2=a2, a_lo=a_lo, a_hi=a_hi, null_mean=null.mean())

# ============ 1. AGGREGATE  T=1.0 both datasets ============
rows=[]
for ds in ["triviaqa","nq"]:
    for m in MODELS:
        sub = df[(df.model==m)&(df.dataset==ds)&(df.temperature==1.0)]
        e = bin_edges(sub.nlp.values)
        r = analyse_cell(sub,e)
        if r: rows.append(dict(dataset=ds,model=LABELS[m],**r))
agg = pd.DataFrame(rows)
agg.to_csv("results_v2/aggregate.csv",index=False)
print("="*78); print("AGGREGATE  T=1.0"); print("="*78)
for ds in ["triviaqa","nq"]:
    print(f"\n--- {ds} ---")
    ss = agg[agg.dataset==ds].sort_values("meta_i2r",ascending=False)
    print(f"{'Model':<20}{'acc':>6}{'meta-I2r':>10}{'95% CI':>18}{'AUROC2':>9}{'p_perm':>8}")
    for _,x in ss.iterrows():
        print(f"{x['model']:<20}{x['acc']:>6.3f}{x['meta_i2r']:>10.4f}"
              f"   [{x['mi_lo']:.3f},{x['mi_hi']:.3f}]{x['auroc2']:>9.3f}{x['p_perm']:>8.3f}")

# ============ 2. DOMAIN  T=1.0 TriviaQA ============
drows=[]
for m in MODELS:
    e = bin_edges(df[(df.model==m)&(df.dataset=="triviaqa")&(df.temperature==1.0)].nlp.values)
    for dom in DOMS:
        sub = df[(df.model==m)&(df.dataset=="triviaqa")&(df.temperature==1.0)&(df.domain==dom)]
        r = analyse_cell(sub,e,n_boot=1000,n_perm=1000)
        if r: drows.append(dict(model=LABELS[m],domain=dom,**r))
dom = pd.DataFrame(drows)
dom.to_csv("results_v2/domain.csv",index=False)
print("\n"+"="*78); print("DOMAIN meta-I2r  T=1.0 TriviaQA (bootstrap CI)"); print("="*78)
piv = dom.pivot(index="domain",columns="model",values="meta_i2r").reindex(DOMS)
print(piv.round(3).to_string())

# ============ 3. TEMPERATURE  TriviaQA ============
trows=[]
for m in MODELS:
    for t in [0.3,0.5,0.7,1.0]:
        sub = df[(df.model==m)&(df.dataset=="triviaqa")&(df.temperature==t)]
        e = bin_edges(sub.nlp.values)
        r = analyse_cell(sub,e,n_boot=800,n_perm=800)
        if r: trows.append(dict(model=LABELS[m],T=t,**r))
temp = pd.DataFrame(trows)
temp.to_csv("results_v2/temperature.csv",index=False)
print("\n"+"="*78); print("TEMPERATURE dissociation TriviaQA"); print("="*78)
print(f"{'Model':<20}{'mI range':>10}{'rho(mI,T)':>11}{'rho(acc,T)':>12}")
for m in MODELS:
    s = temp[temp.model==LABELS[m]].sort_values("T")
    rng_mi = s.meta_i2r.max()-s.meta_i2r.min()
    rho_mi,_ = stats.spearmanr(s['T'],s.meta_i2r)
    rho_ac,_ = stats.spearmanr(s['T'],s.acc)
    print(f"{LABELS[m]:<20}{rng_mi:>10.3f}{rho_mi:>+11.2f}{rho_ac:>+12.2f}")

print("\nSaved: results_v2/aggregate.csv, domain.csv, temperature.csv")
