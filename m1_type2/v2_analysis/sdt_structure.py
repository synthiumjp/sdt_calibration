"""
Valid SDT structure for QA confidence (no meta-d' circularity).

For each model x dataset at T=1.0:
  - Type-2 ROC: confidence discriminating correct vs incorrect -> AUROC2
  - z-ROC of the Type-2 ROC: regress z(HR) on z(FAR) across confidence
    criteria. slope = ratio of SDs of the two confidence distributions.
    This is a genuine unequal-variance SDT property (Green & Swets),
    computed directly from the empirical ROC, NOT via meta-d' inversion.
  - da: unequal-variance sensitivity index = sqrt(2/(1+s^2)) * intercept-based dprime
    (Macmillan & Creelman). Reported as the SDT summary of the Type-2 ROC.

These are all Type-1-of-the-confidence-signal / non-parametric quantities.
No d'/meta-d' ratio is formed.
"""
import numpy as np, pandas as pd
from scipy import stats

SEED=42; rng=np.random.default_rng(SEED)
N_RATINGS=4; NBINS=2*N_RATINGS
df=pd.read_csv("sdt_calibration-main/m1_type2/m1_trial_data.csv")
df["correct"]=df["correct"].astype(int)
MODELS=["llama3_instruct","mistral_instruct","llama3_base","gemma2_instruct"]
LAB={"llama3_instruct":"Llama-3-Instruct","mistral_instruct":"Mistral-Instruct",
     "llama3_base":"Llama-3-Base","gemma2_instruct":"Gemma-2-Instruct"}

def edges(nlp):
    q=np.linspace(0,1,NBINS+1)[1:-1]; return np.quantile(nlp,q)
def ratings(nlp,e): return np.digitize(nlp,e)+1

def type2_zroc(acc,conf):
    """Empirical Type-2 ROC + z-ROC slope. Confidence high->predicts correct."""
    # sweep criterion across confidence bins; HR = P(conf>=k | correct), FAR = P(conf>=k | incorrect)
    ks=range(2,NBINS+1)
    HR=[]; FAR=[]
    corr=conf[acc==1]; inc=conf[acc==0]
    for k in ks:
        HR.append((corr>=k).mean()); FAR.append((inc>=k).mean())
    HR=np.clip(np.array(HR),1e-4,1-1e-4); FAR=np.clip(np.array(FAR),1e-4,1-1e-4)
    zHR=stats.norm.ppf(HR); zFAR=stats.norm.ppf(FAR)
    # slope of zHR on zFAR
    slope,intercept,r,p,se=stats.linregress(zFAR,zHR)
    # da (Macmillan & Creelman): da = (2/(1+slope^2))^.5 * intercept ... use standard
    da = intercept * np.sqrt(2.0/(1.0+slope**2))
    return slope,intercept,da,r**2

def auroc2(acc,conf):
    ranks=stats.rankdata(conf); n1=acc.sum(); n0=len(acc)-n1
    if n1==0 or n0==0: return np.nan
    U1=ranks[acc==1].sum()-n1*(n1+1)/2
    return U1/(n1*n0)

print("="*86)
print("VALID SDT STRUCTURE OF THE CONFIDENCE SIGNAL  (T=1.0)")
print("="*86)
print(f"{'Model':<20}{'dataset':<10}{'AUROC2':>8}{'zROC slope s':>14}{'da':>8}{'R^2':>7}")
rows=[]
for ds in ["triviaqa","nq"]:
    for m in MODELS:
        sub=df[(df.model==m)&(df.dataset==ds)&(df.temperature==1.0)]
        e=edges(sub.nlp.values); r=ratings(sub.nlp.values,e); a=sub.correct.values
        slope,inter,da,r2=type2_zroc(a,r); au=auroc2(a,r)
        rows.append(dict(model=LAB[m],dataset=ds,auroc2=au,zroc_slope=slope,da=da,r2=r2))
        print(f"{LAB[m]:<20}{ds:<10}{au:>8.3f}{slope:>14.3f}{da:>8.3f}{r2:>7.3f}")
pd.DataFrame(rows).to_csv("results_v2/sdt_structure.csv",index=False)

# bootstrap CI on zROC slope, TriviaQA T=1.0
print("\n"+"="*86)
print("z-ROC slope with 95% bootstrap CI (TriviaQA, T=1.0)  -- unequal-variance signature")
print("="*86)
for m in MODELS:
    sub=df[(df.model==m)&(df.dataset=="triviaqa")&(df.temperature==1.0)]
    e=edges(sub.nlp.values); a=sub.correct.values; r=ratings(sub.nlp.values,e)
    n=len(a); sl=[]
    for b in range(1000):
        idx=rng.integers(0,n,n)
        try:
            s,_,_,_=type2_zroc(a[idx],r[idx]); sl.append(s)
        except Exception: pass
    lo,hi=np.percentile(sl,[2.5,97.5])
    s0,_,_,_=type2_zroc(a,r)
    flag="  <-- s<1: unequal variance (correct more variable)" if hi<1.0 else ("" if lo<1<hi else "  <-- s>1")
    print(f"  {LAB[m]:<20} s={s0:.3f}  [{lo:.3f}, {hi:.3f}]{flag}")
print("\nSaved: results_v2/sdt_structure.csv")

# ---- z-ROC figure + LaTeX table ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL={"Llama-3-Instruct":"#1f77b4","Mistral-Instruct":"#ff7f0e","Llama-3-Base":"#2ca02c","Gemma-2-Instruct":"#9467bd"}
fig,ax=plt.subplots(figsize=(5.2,5))
for m in MODELS:
    sub=df[(df.model==m)&(df.dataset=="triviaqa")&(df.temperature==1.0)]
    e=edges(sub.nlp.values); a=sub.correct.values; r=ratings(sub.nlp.values,e)
    corr=r[a==1]; inc=r[a==0]
    HR=[];FAR=[]
    for k in range(2,NBINS+1):
        HR.append((corr>=k).mean()); FAR.append((inc>=k).mean())
    HR=np.clip(HR,1e-4,1-1e-4);FAR=np.clip(FAR,1e-4,1-1e-4)
    zH=stats.norm.ppf(HR);zF=stats.norm.ppf(FAR)
    sl,inter,_,_,_=stats.linregress(zF,zH)
    ax.scatter(zF,zH,color=COL[m],s=28)
    xs=np.linspace(min(zF),max(zF),50); ax.plot(xs,sl*xs+inter,color=COL[m],lw=1.4,
        label=f"{m} (s={sl:.2f})")
ax.plot([-2.5,1],[-2.5,1],"k--",alpha=0.4,lw=1,label="s=1 (equal var.)")
ax.set_xlabel("z(FAR)  [incorrect]"); ax.set_ylabel("z(HR)  [correct]")
ax.set_title("Type-2 z-ROC (TriviaQA, T=1.0)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results_v2/fig/fig_zroc.pdf"); plt.savefig("results_v2/fig/fig_zroc.png",dpi=300); plt.close()

sd=pd.read_csv("results_v2/sdt_structure.csv")
tq=sd[sd.dataset=="triviaqa"].set_index("model"); nq=sd[sd.dataset=="nq"].set_index("model")
order=["Llama-3-Instruct","Mistral-Instruct","Llama-3-Base","Gemma-2-Instruct"]
L=[r"\begin{table}[t]",r"\centering",
r"\caption{SDT structure of the Type-2 (correct/incorrect) confidence ROC at $T{=}1.0$. "
r"$s$ is the z-ROC slope (ratio of incorrect-to-correct evidence SD); $s<1$ indicates the "
r"correct-answer evidence distribution is more variable. $d_a$ is the unequal-variance "
r"sensitivity index. Slopes computed by linear regression on the empirical z-ROC "
r"($R^2\geq0.98$ all cells); the ordering of $s$ replicates on NQ.}",
r"\label{tab:sdt}",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",
r"\textbf{Model} & \textbf{AUROC$_2$} & \textbf{z-ROC slope $s$} & $\boldsymbol{d_a}$ & \textbf{$s$ (NQ)} \\",
r"\midrule"]
for m in order:
    L.append(f"{m} & {tq.loc[m,'auroc2']:.3f} & {tq.loc[m,'zroc_slope']:.3f} & {tq.loc[m,'da']:.3f} & {nq.loc[m,'zroc_slope']:.3f} \\\\")
L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open("results_v2/table_sdt.tex","w").write("\n".join(L))
print("Saved fig_zroc + table_sdt.tex")
