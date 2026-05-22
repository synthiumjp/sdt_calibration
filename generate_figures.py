"""
generate_figures.py — SDT EMNLP paper, 5 models (incl. 70B scale probe)

Generates all 8 figures as PDF+PNG from analysis pipeline outputs.
70B appears only in T=1.0 TriviaQA figures (Fig 4 H2 scatter, Fig 5 H5 domains).

Usage:
    cd C:\\sdt_calibration
    python generate_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, spearmanr

# ============================================================
# Config
# ============================================================
RESULTS_DIR = os.path.join("results", "analysis")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FULL = os.path.join(RESULTS_DIR, "full_results.json")
ROC = os.path.join(RESULTS_DIR, "roc_data.json")
BOOT = os.path.join(RESULTS_DIR, "bootstrap_results.json")

# 4 main models (full temperature sweep)
MODELS_4 = ["llama3_instruct", "mistral_instruct", "llama3_base", "gemma2_instruct"]
# 5 models (includes 70B at T=1.0 TQA only)
MODELS_5 = MODELS_4 + ["llama31_70b_instruct"]

LABELS = {
    "llama3_instruct": "Llama-3-8B-Instruct",
    "mistral_instruct": "Mistral-7B-Instruct",
    "llama3_base": "Llama-3-8B-Base",
    "gemma2_instruct": "Gemma-2-9B-Instruct",
    "llama31_70b_instruct": "Llama-3.1-70B-Instruct",
}
LABELS_SHORT = {
    "llama3_instruct": "Llama-I",
    "mistral_instruct": "Mistral-I",
    "llama3_base": "Llama-B",
    "gemma2_instruct": "Gemma-I",
    "llama31_70b_instruct": "70B-I",
}
COLORS = {
    "llama3_instruct": "#1f77b4",
    "mistral_instruct": "#ff7f0e",
    "llama3_base": "#2ca02c",
    "gemma2_instruct": "#9B59B6",
    "llama31_70b_instruct": "#E74C3C",
}
TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
TEMP_STRS = [str(t) for t in TEMPS]
TEMP_COLORS = plt.cm.viridis(np.linspace(0, 0.95, len(TEMPS)))

DOMAINS = ["Arts & Literature", "Geography", "History & Politics",
           "Science & Technology", "Unclassified"]
DOMAIN_SHORT = ["Arts &\nLit", "Geo", "Hist &\nPol", "Sci &\nTech", "Unclass."]


def load_data():
    with open(FULL, encoding="utf-8") as f:
        full = json.load(f)
    roc = None
    if os.path.exists(ROC):
        with open(ROC, encoding="utf-8") as f:
            roc = json.load(f)
    boot = None
    if os.path.exists(BOOT):
        with open(BOOT, encoding="utf-8") as f:
            boot = json.load(f)
    return full, roc, boot


def save(fig, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(FIG_DIR, name + ext), dpi=200, bbox_inches="tight")
    print(f"  [saved] {name}")
    plt.close(fig)


# ============================================================
# Fig 1: ROC curves by temperature (4 models, TriviaQA)
# ============================================================
def fig1_roc(roc):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("ROC Curves by Temperature (TriviaQA)", fontsize=14, y=0.98)
    for ax, model in zip(axes.flat, MODELS_4):
        if model not in roc or "triviaqa" not in roc[model]:
            ax.set_title(LABELS[model])
            continue
        for i, t in enumerate(TEMP_STRS):
            if t not in roc[model]["triviaqa"]:
                continue
            d = roc[model]["triviaqa"][t]
            fa = [0.0] + d["fa_rates"] + [1.0]
            hr = [0.0] + d["hit_rates"] + [1.0]
            ax.plot(fa, hr, color=TEMP_COLORS[i], linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("False Alarm Rate")
        ax.set_ylabel("Hit Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    # Legend
    handles = [Line2D([0], [0], color=TEMP_COLORS[i], lw=2, label=f"T={t}")
               for i, t in enumerate(TEMPS)]
    axes[1, 1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig1_roc_curves")


# ============================================================
# Fig 2: z-ROC plots (4 models, TriviaQA, all temps overlaid)
# ============================================================
def fig2_zroc(full, roc):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("z-ROC Plots (TriviaQA, all temperatures)", fontsize=14, y=0.98)
    for ax, model in zip(axes.flat, MODELS_4):
        slope_t10 = full["paradigm_a"][model]["triviaqa"]["1.0"]["z_roc"]["slope"]
        intercept = full["paradigm_a"][model]["triviaqa"]["1.0"]["z_roc"]["intercept"]
        if model in roc and "triviaqa" in roc[model]:
            for i, t in enumerate(TEMP_STRS):
                if t not in roc[model]["triviaqa"]:
                    continue
                d = roc[model]["triviaqa"][t]
                hr = np.array(d["hit_rates"])
                fa = np.array(d["fa_rates"])
                mask = (hr > 0.001) & (hr < 0.999) & (fa > 0.001) & (fa < 0.999)
                if mask.sum() < 2:
                    continue
                z_hr = norm.ppf(hr[mask])
                z_fa = norm.ppf(fa[mask])
                ax.scatter(z_fa, z_hr, c=[TEMP_COLORS[i]], s=20, alpha=0.7, zorder=3)
        # Regression line at T=1.0
        x = np.linspace(-3, 2, 100)
        ax.plot(x, slope_t10 * x + intercept, "k-", linewidth=2, zorder=2)
        ax.plot(x, x, "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(-3, 2)
        ax.set_ylim(-3, 2)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("z(False Alarm Rate)")
        ax.set_ylabel("z(Hit Rate)")
        ax.text(-2.8, 1.5, f"slope={slope_t10:.2f}", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig2_zroc_plots")


# ============================================================
# Fig 3: H1 temperature effects (AUC, d_a, c vs T) — 4 models
# ============================================================
def fig3_h1(full, boot):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("H1: Temperature Effects on SDT Parameters", fontsize=14, y=1.02)
    metrics = [
        ("AUC", "auc", "AUC vs Temperature (TQA)"),
        ("d_a", "d_a", "$d_a$ vs Temperature (TQA)"),
        ("c", "c", "Criterion vs Temperature (TQA)"),
    ]
    for ax, (key, json_key, title) in zip(axes, metrics):
        for model in MODELS_4:
            vals = []
            ci_lo, ci_hi = [], []
            for t in TEMP_STRS:
                if t not in full["paradigm_a"].get(model, {}).get("triviaqa", {}):
                    continue
                cell = full["paradigm_a"][model]["triviaqa"][t]
                if json_key == "d_a":
                    vals.append(cell["uv"]["d_a"])
                elif json_key == "c":
                    vals.append(cell["c"])
                else:
                    vals.append(cell[json_key])
                # Bootstrap CIs
                if boot and model in boot and "triviaqa" in boot[model] and t in boot[model]["triviaqa"]:
                    bc = boot[model]["triviaqa"][t]
                    ci_key = f"{json_key}_ci"
                    if ci_key in bc:
                        ci_lo.append(bc[ci_key][0])
                        ci_hi.append(bc[ci_key][1])
                    else:
                        ci_lo.append(vals[-1])
                        ci_hi.append(vals[-1])
                else:
                    ci_lo.append(vals[-1])
                    ci_hi.append(vals[-1])
            t_vals = TEMPS[:len(vals)]
            ax.plot(t_vals, vals, "o-", color=COLORS[model], label=LABELS[model], markersize=5)
            ax.fill_between(t_vals, ci_lo, ci_hi, color=COLORS[model], alpha=0.15)
        ax.set_xlabel("Temperature")
        ax.set_title(title, fontsize=11)
        if key == "AUC":
            ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "fig3_h1_temperature")


# ============================================================
# Fig 4: H2 SDT operating points with ECE (5 models, T=1.0)
# ============================================================
def fig4_h2(full):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("H2: SDT Operating Points with ECE", fontsize=13)
    datasets = {"triviaqa": "o", "nq": "s"}
    all_ece = []
    for model in MODELS_5:
        for ds, marker in datasets.items():
            if ds not in full["paradigm_a"].get(model, {}):
                continue
            if "1.0" not in full["paradigm_a"][model][ds]:
                continue
            cell = full["paradigm_a"][model][ds]["1.0"]
            da = cell["uv"]["d_a"]
            c = cell["c"]
            ece = cell["ece"]["ece"] if isinstance(cell["ece"], dict) else cell["ece"]
            all_ece.append(ece)
            ax.scatter(da, c, c=ece, cmap="RdYlGn_r", vmin=0, vmax=0.5,
                       s=200, marker=marker, edgecolors=COLORS[model],
                       linewidths=2.5, zorder=5)
            ds_label = "TQA" if ds == "triviaqa" else "NQ"
            short = LABELS_SHORT[model]
            ax.annotate(f"{short}\n{ds_label}", (da, c),
                        textcoords="offset points", xytext=(12, 5), fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 0.5))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="ECE")
    ax.set_xlabel("$d_a$ (Sensitivity)", fontsize=12)
    ax.set_ylabel("$c$ (Criterion)", fontsize=12)
    # Legend for shapes and edge colors
    from matplotlib.patches import Patch
    handles = []
    for model in MODELS_5:
        handles.append(Line2D([0], [0], marker="o", color="w", markeredgecolor=COLORS[model],
                              markeredgewidth=2, markersize=10, label=LABELS[model]))
    handles.append(Line2D([0], [0], marker="o", color="grey", markersize=8, linestyle="None", label="TriviaQA"))
    handles.append(Line2D([0], [0], marker="s", color="grey", markersize=8, linestyle="None", label="NQ"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig4_h2_scatter")


# ============================================================
# Fig 5: H5 domain-specific sensitivity (5 models, T=1.0 TQA)
# ============================================================
def fig5_h5(full):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("H5: Domain-Specific Sensitivity (TriviaQA, T=1.0)", fontsize=13)
    domain_data = full["domain_level"]["paradigm_a"]
    n_domains = len(DOMAINS)
    n_models = len(MODELS_5)
    bar_width = 0.15
    x = np.arange(n_domains)
    for i, model in enumerate(MODELS_5):
        if model not in domain_data:
            continue
        vals = []
        for dom in DOMAINS:
            if dom in domain_data[model]:
                vals.append(domain_data[model][dom]["d_a"])
            else:
                vals.append(0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, color=COLORS[model], label=LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_SHORT, fontsize=10)
    ax.set_ylabel("$d_a$", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim(0)
    fig.tight_layout()
    save(fig, "fig5_h5_domains")


# ============================================================
# Fig 6: E5 z-ROC slope × temperature (4 models, TQA + NQ)
# ============================================================
def fig6_e5(full):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ds, ds_title in zip(axes, ["triviaqa", "nq"], ["TriviaQA", "NQ"]):
        ax.set_title(f"z-ROC Slope × Temperature ({ds_title})", fontsize=11)
        for model in MODELS_4:
            slopes = []
            t_vals = []
            for t in TEMP_STRS:
                if t not in full["paradigm_a"].get(model, {}).get(ds, {}):
                    continue
                s = full["paradigm_a"][model][ds][t]["z_roc"]["slope"]
                slopes.append(s)
                t_vals.append(float(t))
            if len(slopes) < 3:
                continue
            rho, p = spearmanr(t_vals, slopes)
            star = "*" if p < 0.01 else ""
            short = LABELS[model].split("-")[0] + "-"
            ax.plot(t_vals, slopes, "o-", color=COLORS[model], markersize=5,
                    label=f"{short} ($\\rho$={rho:.2f}{star})")
        ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("z-ROC Slope")
        ax.set_ylim(0.3, 1.1)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "fig6_e5_zroc_slope")


# ============================================================
# Fig 7: H3 Bland-Altman (3 models with 4AFC data)
# ============================================================
def fig7_h3(full):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("H3: Bland-Altman — Paradigm A vs B", fontsize=13)
    h3_models = ["llama3_instruct", "mistral_instruct", "llama3_base"]
    domain_a = full["domain_level"]["paradigm_a"]
    domain_b = full["domain_level"].get("paradigm_b", {})
    for model in h3_models:
        if model not in domain_a or model not in domain_b:
            continue
        for dom in DOMAINS:
            if dom not in domain_a[model] or dom not in domain_b[model]:
                continue
            da_a = domain_a[model][dom]["d_a"]
            # 4AFC d' is stored differently
            pb = domain_b[model][dom]
            d_4afc = pb.get("d_prime_4afc", pb.get("d_prime", pb.get("d_4afc", None)))
            if d_4afc is None:
                continue
            mean_d = (da_a + d_4afc) / 2
            diff = da_a - d_4afc
            ax.scatter(mean_d, diff, s=120, color=COLORS[model], alpha=0.8, zorder=3)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean $d$ (average of $d_a$ and $d'_{4AFC}$)", fontsize=11)
    ax.set_ylabel("Difference ($d_a - d'_{4AFC}$)", fontsize=11)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[m],
                      markersize=10, label=LABELS[m]) for m in h3_models]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    save(fig, "fig7_h3_bland_altman")


# ============================================================
# Fig 8: NQ replication (AUC + d_a vs T, 4 models)
# ============================================================
def fig8_nq(full, boot):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NQ Replication", fontsize=14, y=1.02)
    for ax, (key, json_key, title) in zip(axes, [
        ("AUC", "auc", "AUC vs Temperature (NQ)"),
        ("d_a", "d_a", "$d_a$ vs Temperature (NQ)"),
    ]):
        for model in MODELS_4:
            vals = []
            for t in TEMP_STRS:
                if t not in full["paradigm_a"].get(model, {}).get("nq", {}):
                    continue
                cell = full["paradigm_a"][model]["nq"][t]
                if json_key == "d_a":
                    vals.append(cell["uv"]["d_a"])
                else:
                    vals.append(cell[json_key])
            t_vals = TEMPS[:len(vals)]
            ax.plot(t_vals, vals, "o-", color=COLORS[model], label=LABELS[model], markersize=5)
        ax.set_xlabel("Temperature")
        ax.set_title(title, fontsize=11)
        if key == "AUC":
            ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "fig8_nq_replication")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    full, roc, boot = load_data()
    print(f"  Models in full_results: {list(full['paradigm_a'].keys())}")

    print("\nGenerating figures...")
    if roc:
        fig1_roc(roc)
    else:
        print("  [skip] fig1 — no roc_data.json")

    fig2_zroc(full, roc)
    fig3_h1(full, boot)
    fig4_h2(full)
    fig5_h5(full)
    fig6_e5(full)

    try:
        fig7_h3(full)
    except Exception as e:
        print(f"  [skip] fig7 — {e}")

    fig8_nq(full, boot)
    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
