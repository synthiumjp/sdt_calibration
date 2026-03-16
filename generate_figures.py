"""
generate_figures.py — Publication figures for SDT Calibration paper.

Run from C:\sdt_calibration:
    pip install matplotlib seaborn
    python generate_figures.py

Reads from results/analysis/ and outputs to results/analysis/figures/
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, linregress
from pathlib import Path

BASE_DIR = Path(r"C:\sdt_calibration")
OUT_DIR = BASE_DIR / "results" / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_COLORS = {
    "llama3_instruct": "#2176AE",
    "mistral_instruct": "#E85D04",
    "llama3_base": "#57A773",
}
MODEL_LABELS = {
    "llama3_instruct": "Llama-3-8B-Instruct",
    "mistral_instruct": "Mistral-7B-Instruct",
    "llama3_base": "Llama-3-8B-Base",
}
TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
TEMP_CMAP = plt.cm.viridis(np.linspace(0.1, 0.95, len(TEMPS)))

def load(name):
    with open(BASE_DIR / "results" / "analysis" / name, "r", encoding="utf-8") as f:
        return json.load(f)

def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} done")

def main():
    full = load("full_results.json")
    roc_data = load("roc_data.json")
    bootstrap = load("bootstrap_results.json")
    secondary = load("secondary_analyses.json")
    pa = full["paradigm_a"]

    # Fig 1: ROC curves
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, model in zip(axes, ["llama3_instruct", "mistral_instruct", "llama3_base"]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
        for i, temp in enumerate(TEMPS):
            r = roc_data.get(model, {}).get("triviaqa", {}).get(str(temp), {})
            if not r: continue
            hr = np.array(r["hit_rates"])
            fa = np.array(r["fa_rates"])
            fa_p = np.concatenate([[0], fa, [1]])
            hr_p = np.concatenate([[0], hr, [1]])
            s = np.argsort(fa_p)
            ax.plot(fa_p[s], hr_p[s], color=TEMP_CMAP[i], lw=1.2, alpha=0.85, label=f"T={temp}")
        ax.set_xlabel("False Alarm Rate"); ax.set_ylabel("Hit Rate")
        ax.set_title(MODEL_LABELS[model]); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
    axes[2].legend(loc="lower right", framealpha=0.9, fontsize=7)
    fig.suptitle("ROC Curves by Temperature (TriviaQA)", fontsize=11, y=1.02)
    plt.tight_layout(); save(fig, "fig1_roc_curves")

    # Fig 2: z-ROC plots
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, model in zip(axes, ["llama3_instruct", "mistral_instruct", "llama3_base"]):
        for i, temp in enumerate(TEMPS):
            r = roc_data.get(model, {}).get("triviaqa", {}).get(str(temp), {})
            if not r: continue
            hr = np.clip(np.array(r["hit_rates"]), 0.001, 0.999)
            fa = np.clip(np.array(r["fa_rates"]), 0.001, 0.999)
            z_hr, z_fa = norm.ppf(hr), norm.ppf(fa)
            ax.scatter(z_fa, z_hr, color=TEMP_CMAP[i], s=8, alpha=0.6, zorder=2)
            if abs(temp - 1.0) < 0.01:
                mask = np.isfinite(z_fa) & np.isfinite(z_hr)
                if np.sum(mask) > 2:
                    slope, intercept, _, _, _ = linregress(z_fa[mask], z_hr[mask])
                    x_fit = np.linspace(z_fa[mask].min(), z_fa[mask].max(), 50)
                    ax.plot(x_fit, slope * x_fit + intercept, "k-", lw=1.5, zorder=3)
                    ax.text(0.05, 0.95, f"slope={slope:.2f}", transform=ax.transAxes, fontsize=8, va="top", fontweight="bold")
        ax.plot([-3, 3], [-3, 3], "k--", lw=0.5, alpha=0.3)
        ax.set_xlabel("z(False Alarm Rate)"); ax.set_ylabel("z(Hit Rate)")
        ax.set_title(MODEL_LABELS[model]); ax.set_xlim(-3, 2); ax.set_ylim(-3, 2)
        ax.set_aspect("equal")
    fig.suptitle("z-ROC Plots (TriviaQA, all temperatures)", fontsize=11, y=1.02)
    plt.tight_layout(); save(fig, "fig2_zroc_plots")

    # Fig 3: H1 temperature effects
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for panel, (ax, key, ylabel, title) in enumerate(zip(axes,
        [lambda m,t: pa[m]["triviaqa"][t]["auc"], lambda m,t: pa[m]["triviaqa"][t]["uv"]["d_a"], lambda m,t: pa[m]["triviaqa"][t]["c"]],
        ["AUC", "$d_a$", "Criterion $c$"],
        ["AUC vs Temperature (TQA)", "$d_a$ vs Temperature (TQA)", "Criterion vs Temperature (TQA)"])):
        for model in ["llama3_instruct", "mistral_instruct", "llama3_base"]:
            vals = [key(model, str(t)) for t in TEMPS]
            ax.plot(TEMPS, vals, "o-", color=MODEL_COLORS[model], lw=1.5, ms=4, label=MODEL_LABELS[model])
            if panel < 2:  # bootstrap CIs for AUC and d_a
                ci_key = "auc_ci" if panel == 0 else "d_a_ci"
                lo = [bootstrap.get(model, {}).get("triviaqa", {}).get(str(t), {}).get(ci_key, [vals[i], vals[i]])[0] for i, t in enumerate(TEMPS)]
                hi = [bootstrap.get(model, {}).get("triviaqa", {}).get(str(t), {}).get(ci_key, [vals[i], vals[i]])[1] for i, t in enumerate(TEMPS)]
                ax.fill_between(TEMPS, lo, hi, color=MODEL_COLORS[model], alpha=0.15)
        ax.set_xlabel("Temperature"); ax.set_ylabel(ylabel); ax.set_title(title)
        if panel == 0: ax.legend(fontsize=6.5)
    fig.suptitle("H1: Temperature Effects on SDT Parameters", fontsize=11, y=1.02)
    plt.tight_layout(); save(fig, "fig3_h1_temperature")

    # Fig 4: H2 scatter
    h2 = full["h2"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for p in h2["points"]:
        model, dataset = p["model"], p["dataset"]
        marker = "o" if dataset == "triviaqa" else "s"
        sc = ax.scatter(p["d_a"], p["c"], c=p["ece"], cmap="RdYlGn_r", vmin=0, vmax=0.5,
                        s=80, marker=marker, edgecolors=MODEL_COLORS.get(model, "gray"), linewidths=1.5, zorder=3)
        label = f"{MODEL_LABELS.get(model, model)[:6]}\n{'TQA' if dataset == 'triviaqa' else 'NQ'}"
        ax.annotate(label, (p["d_a"], p["c"]), fontsize=6, xytext=(0.05, 0.08), textcoords="offset fontsize")
    plt.colorbar(sc, ax=ax, label="ECE", shrink=0.8)
    ax.set_xlabel("$d_a$ (Sensitivity)"); ax.set_ylabel("$c$ (Criterion)")
    ax.set_title("H2: SDT Operating Points with ECE")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", ms=8, label="TriviaQA"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", ms=8, label="NQ"),
    ] + [Line2D([0], [0], marker="o", color="w", markeredgecolor=MODEL_COLORS[m], markerfacecolor="white", ms=8, markeredgewidth=1.5, label=MODEL_LABELS[m][:12]) for m in MODEL_COLORS]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6.5)
    plt.tight_layout(); save(fig, "fig4_h2_scatter")

    # Fig 5: H5 domains
    h5 = secondary["h5"]
    domains = ["Arts & Literature", "Geography", "History & Politics", "Science & Technology", "Unclassified"]
    domain_short = ["Arts &\nLit", "Geo", "Hist &\nPol", "Sci &\nTech", "Unclass."]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(domains)); width = 0.25
    for i, model in enumerate(["llama3_instruct", "mistral_instruct", "llama3_base"]):
        da_vals = [h5[model]["domains"].get(d, 0) for d in domains]
        ax.bar(x + i * width, da_vals, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width); ax.set_xticklabels(domain_short, fontsize=7.5)
    ax.set_ylabel("$d_a$"); ax.set_title("H5: Domain-Specific Sensitivity (TriviaQA, T=1.0)")
    ax.legend(fontsize=7); ax.set_ylim(0)
    plt.tight_layout(); save(fig, "fig5_h5_domains")

    # Fig 6: E5 z-ROC slope x T
    e5 = secondary["e5"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, dataset, ds_label in zip(axes, ["triviaqa", "nq"], ["TriviaQA", "NQ"]):
        for model in ["llama3_instruct", "mistral_instruct", "llama3_base"]:
            e = e5.get(model, {}).get(dataset, {})
            if not e: continue
            temps = e.get("temps", TEMPS); slopes = e.get("slopes", [])
            rho = e.get("spearman_rho_slope_T", 0); p = e.get("spearman_p_slope_T", 1)
            sig = "*" if p < 0.05 else ""
            ax.plot(temps, slopes, "o-", color=MODEL_COLORS[model], lw=1.5, ms=4,
                    label=f"{MODEL_LABELS[model][:8]} (\u03c1={rho:.2f}{sig})")
        ax.axhline(y=1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Temperature"); ax.set_ylabel("z-ROC Slope")
        ax.set_title(f"z-ROC Slope \u00d7 Temperature ({ds_label})"); ax.legend(fontsize=6.5); ax.set_ylim(0.3, 1.1)
    plt.tight_layout(); save(fig, "fig6_e5_zroc_slope")

    # Fig 7: H3 Bland-Altman
    pb = full["paradigm_b"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for model in ["llama3_instruct", "mistral_instruct", "llama3_base"]:
        h5_domains = h5[model]["domains"]
        pb_domains = pb[model]["domain_results"]
        for domain in domains:
            if domain in h5_domains and domain in pb_domains:
                da_yesno = h5_domains[domain]; d_4afc = pb_domains[domain]["d_prime_4afc"]
                mean_d = (da_yesno + d_4afc) / 2; diff_d = da_yesno - d_4afc
                ax.scatter(mean_d, diff_d, color=MODEL_COLORS[model], s=50, alpha=0.8, zorder=3)
    ax.axhline(y=0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Mean $d$ (average of $d_a$ and $d\'_{4AFC}$)")
    ax.set_ylabel("Difference ($d_a$ \u2212 $d\'_{4AFC}$)")
    ax.set_title("H3: Bland-Altman \u2014 Paradigm A vs B")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m], ms=8, label=MODEL_LABELS[m]) for m in MODEL_COLORS], fontsize=7)
    plt.tight_layout(); save(fig, "fig7_h3_bland_altman")

    # Fig 8: NQ replication
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, key, ylabel in zip(axes, [lambda m,t: pa[m]["nq"][t]["auc"], lambda m,t: pa[m]["nq"][t]["uv"]["d_a"]], ["AUC", "$d_a$"]):
        for model in ["llama3_instruct", "mistral_instruct", "llama3_base"]:
            vals = [key(model, str(t)) for t in TEMPS]
            ax.plot(TEMPS, vals, "o-", color=MODEL_COLORS[model], lw=1.5, ms=4, label=MODEL_LABELS[model])
        ax.set_xlabel("Temperature"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Temperature (NQ)"); ax.legend(fontsize=6.5)
    fig.suptitle("NQ Replication", fontsize=11, y=1.02)
    plt.tight_layout(); save(fig, "fig8_nq_replication")

    print("\nAll 8 figures generated.")

if __name__ == "__main__":
    main()
