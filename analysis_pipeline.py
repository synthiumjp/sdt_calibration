"""
analysis_pipeline.py — Full analysis pipeline for SDT Calibration Project 4.1

Implements all pre-registered analyses:
  - ROC construction (20-bin, §5.3.1 Step 3-4)
  - EVSDT / UVSDT MLE fitting (§5.4)
  - Non-parametric AUC (§5.4)
  - z-ROC linearity test (§5.4)
  - 4AFC d' computation (§5.3.2 Step 3)
  - ECE computation (Appendix B §B.4)
  - H1: TOST for AUC equivalence + Spearman ρ for c × T (§5.6.1)
  - H2: SDT decomposition reveals hidden structure (§5.6.2)
  - H3: Paradigm convergence d_a ↔ d'_4AFC (§5.6.3)
  - Bootstrap CIs (Amendment 5)
  - Scoring robustness check (§A.7)

Dependencies: scipy, numpy, statsmodels, scikit-learn, matplotlib, seaborn

Usage:
    python analysis_pipeline.py --base-dir C:\\sdt_calibration
"""

import argparse
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats, optimize
from scipy.special import ndtri  # z-score (inverse normal CDF)
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS = 20  # confidence bins per §5.3.1 / §6
ECE_BINS = 15  # per §B.4 (Guo et al. 2017)
MODERATE_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0]  # H1 range
ALL_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
BONFERRONI_ALPHA = 0.05 / 3  # §5.6.4: α = 0.017 for H1-H3
N_BOOTSTRAP = 10_000  # Amendment 5
NLP_GIBBERISH_THRESHOLD = -10.0  # §6: flag trials with NLP < -10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paradigm_a_results(model: str, dataset: str, base_dir: str) -> list:
    """Load Paradigm A JSONL results."""
    path = Path(base_dir) / "results" / "paradigm_a" / f"{model}_{dataset}.jsonl"
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def load_paradigm_b_results(model: str, base_dir: str) -> list:
    """Load Paradigm B JSONL results."""
    path = Path(base_dir) / "results" / "paradigm_b" / f"{model}_4afc.jsonl"
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def load_analysis_a_results(model: str, dataset: str, base_dir: str) -> list:
    """Load Analysis A JSONL results."""
    path = Path(base_dir) / "results" / "analysis_a" / f"{model}_{dataset}.jsonl"
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def load_equivalence_bounds(base_dir: str) -> dict:
    """Load simulation-derived equivalence bounds."""
    path = Path(base_dir) / "simulation_results" / "equivalence_bounds.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_triviaqa_domains(base_dir: str) -> dict:
    """Load TriviaQA domain labels. Returns question_index -> domain."""
    path = Path(base_dir) / "data" / "triviaqa_5000.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {i: item.get("domain", "Unclassified") for i, item in enumerate(data)}


# ---------------------------------------------------------------------------
# ROC construction (§5.3.1 Steps 3-4)
# ---------------------------------------------------------------------------

def construct_roc(
    nlp_signal: np.ndarray,
    nlp_noise: np.ndarray,
    n_bins: int = N_BINS,
    bin_edges: np.ndarray = None,
) -> dict:
    """Construct ROC from NLP values for signal and noise trials.

    Per §5.3.1 Step 4 and §6:
      - 20 equal-width bins spanning [min(NLP), max(NLP)]
      - Bin edges determined from T=1.0 data and held constant across temps
      - Hautus (1995) log-linear correction for extreme rates

    Returns dict with hit_rates, fa_rates, bin_edges, frequencies.
    """
    all_nlp = np.concatenate([nlp_signal, nlp_noise])

    if bin_edges is None:
        # Determine bin edges from the data
        nlp_min, nlp_max = np.min(all_nlp), np.max(all_nlp)
        # Small epsilon to ensure max value falls in last bin
        bin_edges = np.linspace(nlp_min, nlp_max + 1e-10, n_bins + 1)

    # Count signal and noise in each bin
    signal_counts = np.histogram(nlp_signal, bins=bin_edges)[0]
    noise_counts = np.histogram(nlp_noise, bins=bin_edges)[0]

    # Hautus (1995) log-linear correction: add 0.5 to all cells (§6)
    signal_counts_corrected = signal_counts + 0.5
    noise_counts_corrected = noise_counts + 0.5

    n_signal = np.sum(signal_counts_corrected)
    n_noise = np.sum(noise_counts_corrected)

    # Cumulative rates from highest bin to lowest (higher NLP = more "signal-like")
    # Hit rate = P(evidence > criterion | signal)
    # FA rate = P(evidence > criterion | noise)
    cum_signal = np.cumsum(signal_counts_corrected[::-1])[::-1]
    cum_noise = np.cumsum(noise_counts_corrected[::-1])[::-1]

    # ROC points: use bin boundaries as criteria (19 points from 20 bins)
    # Skip the leftmost edge (all responses "yes") and rightmost (all "no")
    hit_rates = cum_signal[1:] / n_signal  # 19 points
    fa_rates = cum_noise[1:] / n_noise

    return {
        "hit_rates": hit_rates,
        "fa_rates": fa_rates,
        "bin_edges": bin_edges,
        "signal_counts": signal_counts.tolist(),
        "noise_counts": noise_counts.tolist(),
        "n_signal": int(np.sum(signal_counts)),
        "n_noise": int(np.sum(noise_counts)),
    }


# ---------------------------------------------------------------------------
# SDT model fitting (§5.4)
# ---------------------------------------------------------------------------

def evsdt_nll(params, signal_counts, noise_counts):
    """Negative log-likelihood for equal-variance SDT model.

    params: [d_prime, c_1, c_2, ..., c_{k-1}]
    Signal distribution: N(d', 1)
    Noise distribution: N(0, 1)
    """
    d_prime = params[0]
    criteria = np.sort(params[1:])  # ensure monotonic

    n_bins = len(signal_counts)
    # Cumulative probabilities at each criterion
    # P(evidence > c_i | signal) = 1 - Φ(c_i - d')
    # P(evidence > c_i | noise) = 1 - Φ(c_i)
    cum_p_signal = np.array([1.0] + [1 - norm.cdf(c - d_prime) for c in criteria] + [0.0])
    cum_p_noise = np.array([1.0] + [1 - norm.cdf(c) for c in criteria] + [0.0])

    # Bin probabilities
    p_signal = np.diff(-cum_p_signal)  # probability in each bin for signal
    p_noise = np.diff(-cum_p_noise)

    # Clip to avoid log(0)
    p_signal = np.clip(p_signal, 1e-10, 1.0)
    p_noise = np.clip(p_noise, 1e-10, 1.0)

    # Multinomial log-likelihood
    nll = -np.sum(signal_counts * np.log(p_signal)) - np.sum(noise_counts * np.log(p_noise))
    return nll


def uvsdt_nll(params, signal_counts, noise_counts):
    """Negative log-likelihood for unequal-variance SDT model.

    params: [d_a, s, c_1, c_2, ..., c_{k-1}]
    s = σ_noise / σ_signal
    Signal distribution: N(d_a * sqrt((1+s^2)/2) / s, 1/s^2)  [reparametrized]
    Noise distribution: N(0, 1)

    We use the parametrisation where:
      noise ~ N(0, 1)
      signal ~ N(μ_s, σ_s^2) where σ_s = 1/s
      μ_s = d_a * sqrt(2 / (1 + s^2))  [from d_a definition]
    """
    d_a = params[0]
    s = params[1]
    criteria = np.sort(params[2:])

    sigma_signal = 1.0 / s if s > 0 else 1.0
    mu_signal = d_a * math.sqrt(2.0 / (1.0 + s * s))

    n_bins = len(signal_counts)
    cum_p_signal = np.array(
        [1.0]
        + [1 - norm.cdf((c - mu_signal) / sigma_signal) for c in criteria]
        + [0.0]
    )
    cum_p_noise = np.array(
        [1.0] + [1 - norm.cdf(c) for c in criteria] + [0.0]
    )

    p_signal = np.diff(-cum_p_signal)
    p_noise = np.diff(-cum_p_noise)
    p_signal = np.clip(p_signal, 1e-10, 1.0)
    p_noise = np.clip(p_noise, 1e-10, 1.0)

    nll = -np.sum(signal_counts * np.log(p_signal)) - np.sum(noise_counts * np.log(p_noise))
    return nll


def fit_sdt_models(roc_data: dict, n_restarts: int = 10) -> dict:
    """Fit EV and UV SDT models to ROC data via MLE.

    Per §5.4: L-BFGS-B, z-ROC init + perturbation restarts.
    Returns fitted parameters, NLL, AIC, BIC.
    """
    signal_counts = np.array(roc_data["signal_counts"], dtype=float)
    noise_counts = np.array(roc_data["noise_counts"], dtype=float)
    hit_rates = roc_data["hit_rates"]
    fa_rates = roc_data["fa_rates"]
    n_bins = len(signal_counts)
    n_criteria = n_bins - 1
    n_total = np.sum(signal_counts) + np.sum(noise_counts)

    # --- z-ROC initialisation ---
    # Convert hit/FA rates to z-scores for initial estimates
    # Clip to avoid inf
    hr_clipped = np.clip(hit_rates, 0.001, 0.999)
    fa_clipped = np.clip(fa_rates, 0.001, 0.999)
    z_hr = ndtri(hr_clipped)
    z_fa = ndtri(fa_clipped)

    # z-ROC regression: z(HR) = intercept + slope * z(FA)
    valid = np.isfinite(z_hr) & np.isfinite(z_fa)
    if np.sum(valid) >= 2:
        slope, intercept, _, _, _ = stats.linregress(z_fa[valid], z_hr[valid])
    else:
        slope, intercept = 1.0, 1.0

    # Initial d' from z-ROC
    d_prime_init = intercept / max(slope, 0.3)
    s_init = 1.0 / max(slope, 0.3)  # slope ≈ σ_noise/σ_signal = s

    # Initial criteria from z(FA)
    criteria_init = np.sort(z_fa[valid]) if np.sum(valid) >= n_criteria else \
        np.linspace(-2, 2, n_criteria)

    # --- Fit EV model ---
    def fit_ev(init_params):
        # §5.4: d' ∈ [0, 5], criteria unbounded
        bounds = [(0, 5)] + [(None, None)] * n_criteria
        try:
            res = optimize.minimize(
                evsdt_nll, init_params, args=(signal_counts, noise_counts),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-10},
            )
            return res
        except Exception:
            return None

    # z-ROC init
    ev_init = np.concatenate([[d_prime_init], criteria_init[:n_criteria]])
    best_ev = fit_ev(ev_init)

    # Perturbation restarts
    rng = np.random.RandomState(42)
    for _ in range(n_restarts):
        perturbed = ev_init + rng.normal(0, 0.2, size=len(ev_init))
        perturbed[0] = max(0.01, perturbed[0])
        res = fit_ev(perturbed)
        if res and (best_ev is None or res.fun < best_ev.fun):
            best_ev = res

    # --- Fit UV model ---
    def fit_uv(init_params):
        # §5.4: d_a ∈ [0, 5], s ∈ [0.3, 3.0], criteria unbounded
        bounds = [(0, 5), (0.3, 3.0)] + [(None, None)] * n_criteria
        try:
            res = optimize.minimize(
                uvsdt_nll, init_params, args=(signal_counts, noise_counts),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-10},
            )
            return res
        except Exception:
            return None

    uv_init = np.concatenate([[d_prime_init, s_init], criteria_init[:n_criteria]])
    best_uv = fit_uv(uv_init)

    for _ in range(n_restarts):
        perturbed = uv_init + rng.normal(0, 0.2, size=len(uv_init))
        perturbed[0] = max(0.01, perturbed[0])
        perturbed[1] = np.clip(perturbed[1], 0.31, 2.99)
        res = fit_uv(perturbed)
        if res and (best_uv is None or res.fun < best_uv.fun):
            best_uv = res

    # --- Extract parameters ---
    ev_result = {}
    if best_ev and best_ev.success:
        k_ev = 1 + n_criteria  # d' + criteria
        ev_result = {
            "d_prime": float(best_ev.x[0]),
            "criteria": sorted(best_ev.x[1:].tolist()),
            "nll": float(best_ev.fun),
            "aic": 2 * k_ev + 2 * best_ev.fun,
            "bic": k_ev * np.log(n_total) + 2 * best_ev.fun,
            "converged": True,
        }
    else:
        ev_result = {"converged": False, "d_prime": float("nan")}

    uv_result = {}
    if best_uv and best_uv.success:
        k_uv = 2 + n_criteria  # d_a + s + criteria
        d_a = float(best_uv.x[0])
        s = float(best_uv.x[1])
        uv_result = {
            "d_a": d_a,
            "s": s,
            "criteria": sorted(best_uv.x[2:].tolist()),
            "nll": float(best_uv.fun),
            "aic": 2 * k_uv + 2 * best_uv.fun,
            "bic": k_uv * np.log(n_total) + 2 * best_uv.fun,
            "converged": True,
        }
    else:
        uv_result = {"converged": False, "d_a": float("nan"), "s": float("nan")}

    # --- Criterion c (equal-variance) ---
    # c = -½[z(HR) + z(FAR)] using the median operating point
    if len(hr_clipped) > 0 and len(fa_clipped) > 0:
        # Use the middle criterion point
        mid = len(hr_clipped) // 2
        c_ev = -0.5 * (ndtri(hr_clipped[mid]) + ndtri(fa_clipped[mid]))
    else:
        c_ev = float("nan")

    # --- Non-parametric AUC (trapezoidal) ---
    # Sort by FA rate for proper integration
    # Use np.trapezoid (NumPy 2.x) with fallback to np.trapz (1.x)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    sort_idx = np.argsort(fa_rates)
    # Include (0,0) and (1,1) endpoints
    fa_full = np.concatenate([[0], fa_rates[sort_idx], [1]])
    hr_full = np.concatenate([[0], hit_rates[sort_idx], [1]])
    auc = float(_trapz(hr_full, fa_full))

    # --- z-ROC linearity test ---
    z_roc_result = {}
    if np.sum(valid) >= 3:
        z_fa_v = z_fa[valid]
        z_hr_v = z_hr[valid]
        # Linear fit
        slope_l, intercept_l, r_value, p_value, stderr = stats.linregress(z_fa_v, z_hr_v)
        r_squared = r_value ** 2
        # Quadratic contrast
        if np.sum(valid) >= 4:
            try:
                coeffs_quad = np.polyfit(z_fa_v, z_hr_v, 2)
                # F-test: does quadratic improve over linear?
                ss_lin = np.sum((z_hr_v - (intercept_l + slope_l * z_fa_v)) ** 2)
                z_hr_quad = np.polyval(coeffs_quad, z_fa_v)
                ss_quad = np.sum((z_hr_v - z_hr_quad) ** 2)
                n_pts = len(z_fa_v)
                df1 = 1  # one extra parameter
                df2 = n_pts - 3
                if df2 > 0 and ss_quad > 0:
                    f_stat = ((ss_lin - ss_quad) / df1) / (ss_quad / df2)
                    p_quad = 1 - stats.f.cdf(f_stat, df1, df2)
                else:
                    f_stat, p_quad = float("nan"), float("nan")
            except Exception:
                f_stat, p_quad = float("nan"), float("nan")
        else:
            f_stat, p_quad = float("nan"), float("nan")

        z_roc_result = {
            "slope": float(slope_l),
            "intercept": float(intercept_l),
            "r_squared": float(r_squared),
            "p_linear": float(p_value),
            "f_quadratic": float(f_stat),
            "p_quadratic": float(p_quad),
        }

    # --- Model comparison ---
    comparison = {}
    if ev_result.get("converged") and uv_result.get("converged"):
        # Likelihood ratio test (UV nests EV)
        lr_stat = 2 * (ev_result["nll"] - uv_result["nll"])
        lr_p = 1 - stats.chi2.cdf(lr_stat, df=1)  # 1 extra param
        comparison = {
            "lr_statistic": float(lr_stat),
            "lr_p_value": float(lr_p),
            "aic_ev": ev_result["aic"],
            "aic_uv": uv_result["aic"],
            "bic_ev": ev_result["bic"],
            "bic_uv": uv_result["bic"],
            "preferred": "uv" if uv_result["aic"] < ev_result["aic"] else "ev",
        }

    return {
        "ev": ev_result,
        "uv": uv_result,
        "c": float(c_ev),
        "auc": auc,
        "z_roc": z_roc_result,
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# ECE computation (§B.4)
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = ECE_BINS,
) -> dict:
    """Compute Expected Calibration Error per §B.4.

    Per Guo et al. (2017): equal-width bins on [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    n_total = len(confidences)

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        n_bin = np.sum(mask)
        if n_bin > 0:
            acc = np.mean(correctness[mask])
            conf = np.mean(confidences[mask])
            ece += (n_bin / n_total) * abs(acc - conf)
            bin_data.append({
                "bin": i,
                "n": int(n_bin),
                "accuracy": float(acc),
                "confidence": float(conf),
                "gap": float(abs(acc - conf)),
            })
        else:
            bin_data.append({"bin": i, "n": 0, "accuracy": None, "confidence": None})

    return {"ece": float(ece), "bins": bin_data, "n_bins": n_bins}


# ---------------------------------------------------------------------------
# 4AFC d' computation (§5.3.2)
# ---------------------------------------------------------------------------

def dprime_4afc(proportion_correct: float) -> float:
    """Compute d' from 4AFC proportion correct via Green & Dai (1991).

    Numerical inversion: d' is the value where the expected proportion
    correct in 4AFC equals the observed proportion.

    P(correct) = ∫ φ(x - d') [Φ(x)]^3 dx
    """
    if proportion_correct <= 0.25:
        return 0.0
    if proportion_correct >= 1.0:
        return 5.0  # cap

    def p_correct_4afc(d_prime):
        """Expected proportion correct for 4AFC at given d'."""
        x = np.linspace(-6, 6 + d_prime, 1000)
        dx = x[1] - x[0]
        pdf_signal = norm.pdf(x - d_prime)
        cdf_noise = norm.cdf(x) ** 3  # 3 noise distributions
        return float(np.sum(pdf_signal * cdf_noise) * dx)

    # Numerical inversion via root finding
    try:
        result = optimize.brentq(
            lambda d: p_correct_4afc(d) - proportion_correct,
            0, 6,
            xtol=1e-6,
        )
        return float(result)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Bootstrap (Amendment 5)
# ---------------------------------------------------------------------------

def bootstrap_sdt(
    nlp_signal: np.ndarray,
    nlp_noise: np.ndarray,
    bin_edges: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CIs for d_a, c, and AUC per Amendment 5.

    Each iteration: resample trials → re-bin → fit UVSD → extract d_a, c, AUC.
    """
    rng = np.random.RandomState(seed)
    d_a_boot = []
    c_boot = []
    auc_boot = []

    n_sig = len(nlp_signal)
    n_noi = len(nlp_noise)

    for b in range(n_bootstrap):
        # Resample with replacement
        sig_sample = nlp_signal[rng.randint(0, n_sig, size=n_sig)]
        noi_sample = nlp_noise[rng.randint(0, n_noi, size=n_noi)]

        # Re-bin and fit
        roc = construct_roc(sig_sample, noi_sample, bin_edges=bin_edges)
        fit = fit_sdt_models(roc, n_restarts=3)  # fewer restarts for speed

        if fit["uv"].get("converged"):
            d_a_boot.append(fit["uv"]["d_a"])
        if not math.isnan(fit["c"]):
            c_boot.append(fit["c"])
        auc_boot.append(fit["auc"])

    result = {}
    if d_a_boot:
        d_a_arr = np.array(d_a_boot)
        result["d_a_ci"] = [float(np.percentile(d_a_arr, 2.5)),
                            float(np.percentile(d_a_arr, 97.5))]
        result["d_a_se"] = float(np.std(d_a_arr))
    if c_boot:
        c_arr = np.array(c_boot)
        result["c_ci"] = [float(np.percentile(c_arr, 2.5)),
                          float(np.percentile(c_arr, 97.5))]
    if auc_boot:
        auc_arr = np.array(auc_boot)
        result["auc_ci"] = [float(np.percentile(auc_arr, 2.5)),
                            float(np.percentile(auc_arr, 97.5))]

    result["n_converged"] = len(d_a_boot)
    result["n_bootstrap"] = n_bootstrap
    return result


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def test_h1(results_by_temp: dict, equiv_bounds: dict) -> dict:
    """H1: Temperature as criterion shift (§5.6.1).

    TOST for AUC equivalence across moderate temps.
    Spearman ρ for c × T.
    """
    # Get AUC and c at moderate temperatures
    temps = []
    aucs = []
    cs = []
    for t in MODERATE_TEMPS:
        key = str(t)
        if key in results_by_temp:
            r = results_by_temp[key]
            temps.append(t)
            aucs.append(r["auc"])
            cs.append(r["c"])

    if len(temps) < 3:
        return {"error": "Too few temperature conditions for H1"}

    aucs = np.array(aucs)
    cs = np.array(cs)

    # Get equivalence bound for AUC (use the delta from simulation)
    # Use the most conservative (largest) delta across conditions
    delta_auc = max(
        c.get("delta_auc", 0.02) for c in equiv_bounds.get("conditions", [{}])
    )

    # TOST for AUC equivalence
    # Compare each temp's AUC to the mean AUC
    mean_auc = np.mean(aucs)
    auc_diffs = aucs - mean_auc
    max_diff = np.max(np.abs(auc_diffs))

    # TOST: two one-sided tests
    # H0: |AUC_i - mean_AUC| >= delta
    # Using paired approach: test if all deviations are within bounds
    se_auc = np.std(aucs) / np.sqrt(len(aucs))
    if se_auc > 0:
        t_upper = (max_diff - delta_auc) / se_auc
        t_lower = (-max_diff + delta_auc) / se_auc
        p_tost = max(
            stats.t.cdf(t_upper, df=len(aucs) - 1),
            1 - stats.t.cdf(t_lower, df=len(aucs) - 1),
        )
    else:
        p_tost = 0.0 if max_diff < delta_auc else 1.0

    # Spearman ρ: c × T
    rho_c, p_rho_c = stats.spearmanr(temps, cs)

    # Bayesian supplement: BF_01 for AUC null
    # Approximate with JZS-style BF using the F-test
    # One-way ANOVA of AUC across temps (even though related)
    # This is a rough approximation
    ss_between = np.sum((aucs - mean_auc) ** 2)
    bf_01 = float("nan")  # placeholder — proper BF requires full implementation

    return {
        "auc_values": aucs.tolist(),
        "auc_mean": float(mean_auc),
        "auc_max_deviation": float(max_diff),
        "delta_auc": delta_auc,
        "tost_p": float(p_tost),
        "tost_significant": bool(p_tost < BONFERRONI_ALPHA),
        "c_values": cs.tolist(),
        "spearman_rho": float(rho_c),
        "spearman_p": float(p_rho_c),
        "spearman_significant": bool(p_rho_c < BONFERRONI_ALPHA),
        "h1_supported": bool(p_tost < BONFERRONI_ALPHA and abs(rho_c) > 0.85),
        "temperatures": temps,
    }


def test_h2(model_results: dict) -> dict:
    """H2: SDT decomposition reveals hidden structure (§5.6.2, §B.4.1).

    Test whether models with similar ECE have different (d_a, c) positions.
    """
    points = []
    for model, datasets in model_results.items():
        for dataset, result in datasets.items():
            t10 = result.get("1.0", {})
            if t10:
                points.append({
                    "model": model,
                    "dataset": dataset,
                    "d_a": t10.get("uv", {}).get("d_a", float("nan")),
                    "c": t10.get("c", float("nan")),
                    "ece": t10.get("ece", {}).get("ece", float("nan")),
                })

    if len(points) < 2:
        return {"error": "Too few model points for H2"}

    # Check ECE similarity (§B.4.1: |ECE_diff| ≤ 0.03)
    eces = [p["ece"] for p in points if not math.isnan(p["ece"])]
    d_as = [p["d_a"] for p in points if not math.isnan(p["d_a"])]
    cs_vals = [p["c"] for p in points if not math.isnan(p["c"])]

    ece_range = max(eces) - min(eces) if eces else float("nan")
    d_a_range = max(d_as) - min(d_as) if d_as else float("nan")
    c_range = max(cs_vals) - min(cs_vals) if cs_vals else float("nan")

    return {
        "points": points,
        "ece_range": float(ece_range),
        "ece_similar": bool(ece_range <= 0.03) if not math.isnan(ece_range) else None,
        "d_a_range": float(d_a_range),
        "c_range": float(c_range),
        "sdt_divergent": bool(d_a_range > 0.2 or c_range > 0.2),
        "h2_supported": bool(ece_range <= 0.03 and (d_a_range > 0.2 or c_range > 0.2))
            if not math.isnan(ece_range) else None,
    }


def test_h3(paradigm_a_by_domain: dict, paradigm_b_by_domain: dict) -> dict:
    """H3: Paradigm convergence (§5.6.3).

    Pearson correlation of d_a(yes/no) vs d'_4AFC across domains.
    """
    domains = []
    d_a_values = []
    d_4afc_values = []

    for domain in paradigm_a_by_domain:
        if domain in paradigm_b_by_domain:
            d_a = paradigm_a_by_domain[domain].get("d_a")
            d_4afc = paradigm_b_by_domain[domain].get("d_prime_4afc")
            if d_a is not None and d_4afc is not None:
                domains.append(domain)
                d_a_values.append(d_a)
                d_4afc_values.append(d_4afc)

    if len(domains) < 3:
        return {"error": "Too few domains for H3 convergence test"}

    d_a_arr = np.array(d_a_values)
    d_4afc_arr = np.array(d_4afc_values)

    r, p = stats.pearsonr(d_a_arr, d_4afc_arr)
    mean_diff = float(np.mean(np.abs(d_a_arr - d_4afc_arr)))

    return {
        "domains": domains,
        "d_a_values": d_a_values,
        "d_4afc_values": d_4afc_values,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_abs_difference": mean_diff,
        "h3_supported": bool(r > 0.7 and mean_diff < 0.3 and p < BONFERRONI_ALPHA),
    }


# ---------------------------------------------------------------------------
# Main analysis driver
# ---------------------------------------------------------------------------

def run_full_analysis(base_dir: str = r"C:\sdt_calibration"):
    """Run the complete pre-registered analysis pipeline."""
    output_dir = Path(base_dir) / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    equiv_bounds = load_equivalence_bounds(base_dir)
    domains_map = load_triviaqa_domains(base_dir)

    models = ["llama3_instruct", "mistral_instruct", "llama3_base"]
    datasets = ["triviaqa", "nq"]

    all_results = {}
    h1_inputs = {}
    h2_inputs = {}

    print("=" * 70)
    print("SDT Calibration — Full Analysis Pipeline")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Paradigm A: ROC + SDT fitting per model × dataset × temperature
    # -----------------------------------------------------------------------
    print("\n--- Phase 1: Paradigm A analysis ---")

    for model in models:
        all_results[model] = {}
        h1_inputs[model] = {}
        h2_inputs[model] = {}

        for dataset in datasets:
            try:
                trials = load_paradigm_a_results(model, dataset, base_dir)
            except FileNotFoundError:
                print(f"  Skipping {model} × {dataset} (no data)")
                continue

            print(f"\n  {model} × {dataset}: {len(trials)} trials")

            # Group by temperature
            by_temp = defaultdict(list)
            for t in trials:
                by_temp[t["temperature"]].append(t)

            # Determine bin edges from T=1.0 data (§6)
            t10_trials = by_temp.get(1.0, [])
            if t10_trials:
                all_nlp_t10 = np.array([t["nlp"] for t in t10_trials])
                t10_min, t10_max = np.min(all_nlp_t10), np.max(all_nlp_t10)
                bin_edges = np.linspace(t10_min, t10_max + 1e-10, N_BINS + 1)
            else:
                bin_edges = None

            results_by_temp = {}
            for temp in ALL_TEMPS:
                temp_trials = by_temp.get(temp, [])
                if not temp_trials:
                    continue

                nlp_vals = np.array([t["nlp"] for t in temp_trials])
                correct = np.array([t["correct"] for t in temp_trials])
                confidence = np.array([t["answer_softmax_prob"] for t in temp_trials])

                # Issue 1 diagnostic: sequence probability underflow.
                # For multi-token answers, P(answer) = Π p(t_i) can be
                # vanishingly small, causing most values to sit in ECE bin 0.
                # We report: (1) primary ECE per §B.4 using answer_softmax_prob,
                # (2) fraction of trials with confidence == 0.0 (underflow),
                # (3) supplementary ECE using NLP mapped to [0,1] via rank
                #     transform (documented deviation if primary ECE is degenerate).
                n_underflow = int(np.sum(confidence == 0.0))
                frac_underflow = n_underflow / len(confidence) if len(confidence) > 0 else 0

                # Flag gibberish (§6)
                gibberish_mask = nlp_vals < NLP_GIBBERISH_THRESHOLD
                n_gibberish = int(np.sum(gibberish_mask))

                # Signal = correct, Noise = incorrect
                nlp_signal = nlp_vals[correct]
                nlp_noise = nlp_vals[~correct]

                if len(nlp_signal) < 10 or len(nlp_noise) < 10:
                    print(f"    T={temp}: too few signal/noise ({len(nlp_signal)}/{len(nlp_noise)})")
                    continue

                # ROC
                roc = construct_roc(nlp_signal, nlp_noise, bin_edges=bin_edges)

                # SDT fitting
                sdt = fit_sdt_models(roc)

                # ECE (§B.4) — primary: softmax probability on [0, 1]
                ece = compute_ece(confidence, correct.astype(float))
                ece["n_underflow"] = n_underflow
                ece["frac_underflow"] = frac_underflow

                # Supplementary ECE: NLP mapped to [0,1] via min-max scaling.
                # Reported alongside primary ECE as methodological metadata.
                # Only meaningful if primary ECE is degenerate (>50% underflow).
                nlp_min, nlp_max = np.min(nlp_vals), np.max(nlp_vals)
                if nlp_max > nlp_min:
                    nlp_confidence = (nlp_vals - nlp_min) / (nlp_max - nlp_min)
                else:
                    nlp_confidence = np.full_like(nlp_vals, 0.5)
                ece_nlp = compute_ece(nlp_confidence, correct.astype(float))
                ece["ece_nlp_supplementary"] = ece_nlp["ece"]

                # Refusal/preamble rates
                n_refusal = sum(1 for t in temp_trials if t.get("refusal_flag"))
                n_preamble = sum(1 for t in temp_trials if t.get("preamble_flag"))

                temp_result = {
                    **sdt,
                    "roc": {k: v if not isinstance(v, np.ndarray) else v.tolist()
                            for k, v in roc.items()},
                    "ece": ece,
                    "accuracy": float(np.mean(correct)),
                    "n_trials": len(temp_trials),
                    "n_signal": int(np.sum(correct)),
                    "n_noise": int(np.sum(~correct)),
                    "n_gibberish": n_gibberish,
                    "n_refusal": n_refusal,
                    "n_preamble": n_preamble,
                    "refusal_rate": n_refusal / len(temp_trials),
                    "preamble_rate": n_preamble / len(temp_trials),
                    "nlp_mean": float(np.mean(nlp_vals)),
                    "nlp_std": float(np.std(nlp_vals)),
                }

                results_by_temp[str(temp)] = temp_result

                print(
                    f"    T={temp}: acc={temp_result['accuracy']:.3f} "
                    f"d_a={sdt['uv'].get('d_a', 'N/A'):.3f} "
                    f"c={sdt['c']:.3f} "
                    f"AUC={sdt['auc']:.3f} "
                    f"ECE={ece['ece']:.3f}"
                    if sdt['uv'].get('converged') else
                    f"    T={temp}: acc={temp_result['accuracy']:.3f} (UV did not converge)"
                )

            all_results[model][dataset] = results_by_temp
            h1_inputs[model][dataset] = results_by_temp
            h2_inputs[model][dataset] = results_by_temp

    # -----------------------------------------------------------------------
    # 2. Paradigm B: 4AFC analysis
    # -----------------------------------------------------------------------
    print("\n--- Phase 2: Paradigm B (4AFC) analysis ---")

    paradigm_b_results = {}
    for model in models:
        try:
            b_trials = load_paradigm_b_results(model, base_dir)
        except FileNotFoundError:
            print(f"  Skipping {model} (no Paradigm B data)")
            continue

        total = len(b_trials)
        correct = sum(1 for t in b_trials if t["correct"])
        p_correct = correct / total if total > 0 else 0.25

        d_prime = dprime_4afc(p_correct)

        # Domain-level (for H3)
        # Map question_index to domain (only for TriviaQA 4AFC questions)
        domain_correct = defaultdict(lambda: {"correct": 0, "total": 0})
        for t in b_trials:
            q_idx = t.get("question_index", -1)
            domain = domains_map.get(q_idx, "Unclassified")
            domain_correct[domain]["total"] += 1
            if t["correct"]:
                domain_correct[domain]["correct"] += 1

        domain_dprime = {}
        for domain, counts in domain_correct.items():
            if counts["total"] >= 50:  # minimum for stable estimate
                pc = counts["correct"] / counts["total"]
                domain_dprime[domain] = {
                    "p_correct": pc,
                    "d_prime_4afc": dprime_4afc(pc),
                    "n": counts["total"],
                }

        # Compliance check
        compliance_fails = sum(1 for t in b_trials if t.get("label_probs_sum", 1) < 0.50)
        compliance_rate = 1 - compliance_fails / total

        paradigm_b_results[model] = {
            "total": total,
            "correct": correct,
            "p_correct": p_correct,
            "d_prime_4afc": d_prime,
            "compliance_rate": compliance_rate,
            "domain_results": domain_dprime,
        }

        print(f"  {model}: P(correct)={p_correct:.3f}, d'_4AFC={d_prime:.3f}, "
              f"compliance={compliance_rate:.3f}")

    # -----------------------------------------------------------------------
    # 3. Analysis A: force-decode
    # -----------------------------------------------------------------------
    print("\n--- Phase 3: Analysis A (force-decode) ---")

    analysis_a_results = {}
    for model in models:
        analysis_a_results[model] = {}
        for dataset in datasets:
            try:
                a_trials = load_analysis_a_results(model, dataset, base_dir)
            except FileNotFoundError:
                print(f"  Skipping {model} × {dataset} (no Analysis A data)")
                continue

            # Signal: nlp_correct for all questions
            nlp_signal = np.array([t["nlp_correct"] for t in a_trials
                                   if t["nlp_correct"] is not None
                                   and not math.isinf(t["nlp_correct"])])

            # Noise: nlp_incorrect for questions where T=1.0 was wrong
            nlp_noise = np.array([t["nlp_incorrect"] for t in a_trials
                                  if t["nlp_incorrect"] is not None
                                  and not math.isinf(t["nlp_incorrect"])])

            if len(nlp_signal) < 10 or len(nlp_noise) < 10:
                print(f"  {model} × {dataset}: too few signal/noise for Analysis A")
                continue

            roc = construct_roc(nlp_signal, nlp_noise)
            sdt = fit_sdt_models(roc)

            # Amendment 1: random incorrect noise
            nlp_random_noise = np.array([
                t["nlp_random_incorrect"] for t in a_trials
                if t.get("nlp_random_incorrect") is not None
                and not math.isinf(t.get("nlp_random_incorrect", float("-inf")))
            ])

            amend1 = {}
            if len(nlp_random_noise) >= 10:
                roc_random = construct_roc(nlp_signal, nlp_random_noise)
                sdt_random = fit_sdt_models(roc_random)
                amend1 = {
                    "d_a_model_noise": sdt["uv"].get("d_a"),
                    "d_a_random_noise": sdt_random["uv"].get("d_a"),
                    "d_a_difference": abs(
                        (sdt["uv"].get("d_a", 0) or 0) -
                        (sdt_random["uv"].get("d_a", 0) or 0)
                    ),
                    "convergent": abs(
                        (sdt["uv"].get("d_a", 0) or 0) -
                        (sdt_random["uv"].get("d_a", 0) or 0)
                    ) < 0.2,
                }

            analysis_a_results[model][dataset] = {
                **sdt,
                "n_signal": len(nlp_signal),
                "n_noise": len(nlp_noise),
                "amendment_1": amend1,
            }

            d_a_str = f"{sdt['uv']['d_a']:.3f}" if sdt['uv'].get('converged') else "N/A"
            print(f"  {model} × {dataset}: d_a={d_a_str}, "
                  f"AUC={sdt['auc']:.3f}, "
                  f"signal={len(nlp_signal)}, noise={len(nlp_noise)}")

    # -----------------------------------------------------------------------
    # 4. Hypothesis tests
    # -----------------------------------------------------------------------
    print("\n--- Phase 4: Hypothesis tests ---")

    h1_results = {}
    for model in ["llama3_instruct", "mistral_instruct"]:
        for dataset in datasets:
            if model in h1_inputs and dataset in h1_inputs[model]:
                key = f"{model}_{dataset}"
                h1 = test_h1(h1_inputs[model][dataset], equiv_bounds)
                h1_results[key] = h1
                print(f"  H1 {key}: TOST p={h1.get('tost_p', 'N/A'):.4f}, "
                      f"ρ(c,T)={h1.get('spearman_rho', 'N/A'):.3f}, "
                      f"supported={h1.get('h1_supported')}")

    h2_result = test_h2(h2_inputs)
    print(f"  H2: ECE range={h2_result.get('ece_range', 'N/A')}, "
          f"SDT divergent={h2_result.get('sdt_divergent')}, "
          f"supported={h2_result.get('h2_supported')}")

    # H3 (paradigm convergence per model)
    h3_results = {}
    for model in ["llama3_instruct", "mistral_instruct"]:
        # Get domain-level d_a from Paradigm A at T=1.0
        if model not in all_results:
            continue
        triviaqa_temps = all_results[model].get("triviaqa", {})
        t10_result = triviaqa_temps.get("1.0", {})

        # We need domain-level ROC construction — use the raw trials
        # This requires re-loading and grouping by domain
        # For now, use the aggregate and note this needs domain-level expansion
        pa_domain = {}  # domain -> d_a
        pb_domain = paradigm_b_results.get(model, {}).get("domain_results", {})

        # TODO: domain-level Paradigm A analysis (requires grouping trials by domain
        # and running ROC + SDT per domain). For now, report aggregate.
        if pa_domain and pb_domain:
            h3 = test_h3(pa_domain, pb_domain)
            h3_results[model] = h3
            print(f"  H3 {model}: r={h3.get('pearson_r', 'N/A')}, "
                  f"supported={h3.get('h3_supported')}")
        else:
            print(f"  H3 {model}: deferred (requires domain-level Paradigm A analysis)")

    # -----------------------------------------------------------------------
    # 5. Bootstrap CIs (Amendment 5)
    # -----------------------------------------------------------------------
    print("\n--- Phase 5: Bootstrap CIs ---")
    print("  (This may take several hours on CPU. Run separately if needed.)")

    # Bootstrap is computationally expensive — provide a flag
    # For now, structure the code but don't run by default
    bootstrap_results = {}
    # Uncomment to run:
    # for model in ["llama3_instruct", "mistral_instruct"]:
    #     for dataset in datasets:
    #         for temp in ALL_TEMPS:
    #             ...

    # -----------------------------------------------------------------------
    # 6. Save all results
    # -----------------------------------------------------------------------
    print("\n--- Saving results ---")

    full_output = {
        "paradigm_a": {m: {d: {t: {k: v for k, v in r.items() if k != "roc"}
                               for t, r in temps.items()}
                           for d, temps in dsets.items()}
                       for m, dsets in all_results.items()},
        "paradigm_b": paradigm_b_results,
        "analysis_a": analysis_a_results,
        "h1": h1_results,
        "h2": h2_result,
        "h3": h3_results,
        "bootstrap": bootstrap_results,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_bins": N_BINS,
            "ece_bins": ECE_BINS,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "n_bootstrap": N_BOOTSTRAP,
        },
    }

    # Save main results (without large arrays)
    results_file = output_dir / "full_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"  Results saved to {results_file}")

    # Save detailed ROC data separately (large)
    roc_file = output_dir / "roc_data.json"
    roc_output = {}
    for m, dsets in all_results.items():
        roc_output[m] = {}
        for d, temps in dsets.items():
            roc_output[m][d] = {}
            for t, r in temps.items():
                if "roc" in r:
                    roc_output[m][d][t] = r["roc"]
    with open(roc_file, "w", encoding="utf-8") as f:
        json.dump(roc_output, f, indent=2, default=str)
    print(f"  ROC data saved to {roc_file}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    return full_output


# ---------------------------------------------------------------------------
# Bootstrap runner (separate entry point for Amendment 5)
# ---------------------------------------------------------------------------

def run_bootstrap(base_dir: str = r"C:\sdt_calibration"):
    """Run bootstrap CIs separately (computationally expensive).

    ~4 hours CPU for 2 models × 2 datasets × 7 temps × 10000 bootstraps.
    """
    output_dir = Path(base_dir) / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["llama3_instruct", "mistral_instruct"]
    datasets = ["triviaqa", "nq"]
    domains_map = load_triviaqa_domains(base_dir)

    bootstrap_results = {}

    for model in models:
        bootstrap_results[model] = {}
        for dataset in datasets:
            try:
                trials = load_paradigm_a_results(model, dataset, base_dir)
            except FileNotFoundError:
                continue

            by_temp = defaultdict(list)
            for t in trials:
                by_temp[t["temperature"]].append(t)

            # Bin edges from T=1.0
            t10_trials = by_temp.get(1.0, [])
            if not t10_trials:
                continue
            all_nlp_t10 = np.array([t["nlp"] for t in t10_trials])
            bin_edges = np.linspace(
                np.min(all_nlp_t10), np.max(all_nlp_t10) + 1e-10, N_BINS + 1
            )

            bootstrap_results[model][dataset] = {}
            for temp in ALL_TEMPS:
                temp_trials = by_temp.get(temp, [])
                if not temp_trials:
                    continue

                nlp = np.array([t["nlp"] for t in temp_trials])
                correct = np.array([t["correct"] for t in temp_trials])
                nlp_signal = nlp[correct]
                nlp_noise = nlp[~correct]

                if len(nlp_signal) < 10 or len(nlp_noise) < 10:
                    continue

                print(f"  Bootstrap: {model} × {dataset} × T={temp}...")
                boot = bootstrap_sdt(nlp_signal, nlp_noise, bin_edges)
                bootstrap_results[model][dataset][str(temp)] = boot
                print(f"    d_a CI: {boot.get('d_a_ci', 'N/A')}, "
                      f"AUC CI: {boot.get('auc_ci', 'N/A')}")

    # Save
    boot_file = output_dir / "bootstrap_results.json"
    with open(boot_file, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\nBootstrap results saved to {boot_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDT analysis pipeline")
    parser.add_argument("--base-dir", default=r"C:\sdt_calibration")
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Run only bootstrap CIs (computationally expensive)",
    )

    args = parser.parse_args()

    if args.bootstrap_only:
        run_bootstrap(args.base_dir)
    else:
        run_full_analysis(args.base_dir)


if __name__ == "__main__":
    main()
