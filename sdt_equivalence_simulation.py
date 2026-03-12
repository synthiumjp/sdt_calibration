"""
SDT for LLM Calibration — Phase 1: Monte Carlo Equivalence Bound Simulation
=============================================================================

Implements Section 5.5 of the pre-registration plan (v2.1).
Produces equivalence bound δ for the TOST test of H1.

Pipeline per iteration:
  1. Generate N synthetic signal/noise values from UVSD distributions
  2. Bin into 20 equal-width bins, construct ROC
  3. Fit UVSD model by MLE (L-BFGS-B, z-ROC-seeded + 10 perturbation restarts)
  4. Extract fitted d_a and AUC

Repeat 10,000 times → sampling distribution → δ = 2 × SD(fitted d_a).

Grid: d_a ∈ {0.5, 1.0, 1.5, 2.0, 2.5}, s ∈ {0.70, 0.80, 0.90, 1.00}

Usage:
  python sdt_equivalence_simulation.py
  python sdt_equivalence_simulation.py --iterations 10000 --restarts 10 --workers 12

Author: | March 2026
"""

import numpy as np
from scipy import optimize, stats
import json, os, time, argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count


def generate_uvsd_data(d_a, s, n_signal, n_noise, rng):
    sigma_signal = 1.0 / s
    mu_signal = d_a / (s * np.sqrt(2.0 / (1.0 + s**2)))
    noise = rng.normal(0.0, 1.0, n_noise)
    signal = rng.normal(mu_signal, sigma_signal, n_signal)
    return signal, noise


def bin_evidence(signal_ev, noise_ev, n_bins=20):
    all_ev = np.concatenate([signal_ev, noise_ev])
    lo, hi = np.min(all_ev), np.max(all_ev)
    pad = (hi - lo) * 0.001
    edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
    sc, _ = np.histogram(signal_ev, bins=edges)
    nc, _ = np.histogram(noise_ev, bins=edges)
    sc = sc.astype(float) + 0.5  # Hautus (1995)
    nc = nc.astype(float) + 0.5
    cum_s = np.cumsum(sc[::-1])[::-1]
    cum_n = np.cumsum(nc[::-1])[::-1]
    hr = cum_s[1:] / np.sum(sc)
    fa = cum_n[1:] / np.sum(nc)
    return hr, fa, edges, sc, nc


def compute_auc(hr, fa):
    fa_full = np.concatenate([[0.0], fa, [1.0]])
    hr_full = np.concatenate([[0.0], hr, [1.0]])
    idx = np.argsort(fa_full)
    try:
        return np.trapezoid(hr_full[idx], fa_full[idx])
    except AttributeError:
        return np.trapz(hr_full[idx], fa_full[idx])


def uvsd_neg_loglik(params, sc, nc):
    k = len(sc)
    d_a, s = params[0], params[1]
    crit = params[2:]
    sig_s = 1.0 / s
    mu_s = d_a / (s * np.sqrt(2.0 / (1.0 + s**2)))
    n_cum = stats.norm.cdf(crit)
    s_cum = stats.norm.cdf((crit - mu_s) / sig_s)
    eps = 1e-10
    np_ = np.clip(np.concatenate([[n_cum[0]], np.diff(n_cum), [1.0 - n_cum[-1]]]), eps, 1 - eps)
    sp_ = np.clip(np.concatenate([[s_cum[0]], np.diff(s_cum), [1.0 - s_cum[-1]]]), eps, 1 - eps)
    return -(np.sum(sc * np.log(sp_)) + np.sum(nc * np.log(np_)))


def zroc_init(hr, fa):
    z_hr = stats.norm.ppf(np.clip(hr, 0.001, 0.999))
    z_fa = stats.norm.ppf(np.clip(fa, 0.001, 0.999))
    slope, intercept, _, _, _ = stats.linregress(z_fa, z_hr)
    s0 = np.clip(slope, 0.3, 3.0)
    d0 = np.clip(np.sqrt(2.0 / (1.0 + s0**2)) * intercept, 0.01, 5.0)
    return d0, s0


def fit_uvsd(sc, nc, hr, fa, n_restarts=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    k = len(sc)
    n_c = k - 1
    d0, s0 = zroc_init(hr, fa)
    sig_s = 1.0 / s0
    mu_s = d0 / (s0 * np.sqrt(2.0 / (1.0 + s0**2)))
    c_base = np.linspace(min(-2.5, -0.5), max(mu_s + 2.5 * sig_s, 3.5), n_c)
    bounds = [(0.01, 5.0), (0.3, 3.0)] + [(-6.0, 10.0)] * n_c
    opts = {'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6}
    best_nll, best_x, best_conv = np.inf, None, False
    for r in range(n_restarts + 1):
        if r == 0:
            x0 = np.concatenate([[d0, s0], c_base])
        else:
            x0 = np.concatenate([
                [np.clip(d0 + rng.normal(0, 0.3), 0.01, 5.0),
                 np.clip(s0 + rng.normal(0, 0.1), 0.3, 3.0)],
                np.sort(c_base + rng.normal(0, 0.3, n_c))
            ])
        try:
            res = optimize.minimize(uvsd_neg_loglik, x0, args=(sc, nc),
                                     method='L-BFGS-B', bounds=bounds, options=opts)
            if res.fun < best_nll:
                best_nll, best_x, best_conv = res.fun, res.x, res.success
        except:
            continue
    if best_x is None:
        return np.nan, np.nan, np.inf, False
    return best_x[0], best_x[1], best_nll, best_conv


def _single_iter(args):
    d_a_true, s_true, n_sig, n_noi, n_bins, n_restarts, seed = args
    rng = np.random.default_rng(seed)
    sig, noi = generate_uvsd_data(d_a_true, s_true, n_sig, n_noi, rng)
    hr, fa, edges, sc, nc = bin_evidence(sig, noi, n_bins)
    auc = compute_auc(hr, fa)
    d_a_f, s_f, nll, conv = fit_uvsd(sc, nc, hr, fa, n_restarts, rng)
    return d_a_f, s_f, auc, conv


def run_condition(d_a_true, s_true, n_sig=2500, n_noi=2500, n_bins=20,
                  n_iter=10000, n_restarts=10, base_seed=42, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    args = [(d_a_true, s_true, n_sig, n_noi, n_bins, n_restarts,
             base_seed + i) for i in range(n_iter)]
    t0 = time.time()
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_single_iter, args)
    else:
        results = [_single_iter(a) for a in args]
    elapsed = time.time() - t0
    d_a = np.array([r[0] for r in results])
    s_arr = np.array([r[1] for r in results])
    auc = np.array([r[2] for r in results])
    conv = np.array([r[3] for r in results])
    v = ~np.isnan(d_a)
    dv, sv, av = d_a[v], s_arr[v], auc[v]
    sd_d, sd_a = np.std(dv, ddof=1), np.std(av, ddof=1)
    return {
        'params': {'d_a_true': d_a_true, 's_true': s_true,
                    'n_sig': n_sig, 'n_noi': n_noi, 'n_bins': n_bins,
                    'n_iter': n_iter, 'n_restarts': n_restarts},
        'dist': {
            'd_a_mean': float(np.mean(dv)), 'd_a_sd': float(sd_d),
            'd_a_median': float(np.median(dv)),
            'd_a_2.5': float(np.percentile(dv, 2.5)),
            'd_a_97.5': float(np.percentile(dv, 97.5)),
            'd_a_bias': float(np.mean(dv) - d_a_true),
            's_mean': float(np.mean(sv)), 's_sd': float(np.std(sv, ddof=1)),
            's_bias': float(np.mean(sv) - s_true),
            'auc_mean': float(np.mean(av)), 'auc_sd': float(sd_a),
            'auc_2.5': float(np.percentile(av, 2.5)),
            'auc_97.5': float(np.percentile(av, 97.5)),
        },
        'bounds': {'d_a_delta': float(2 * sd_d), 'auc_delta': float(2 * sd_a)},
        'diag': {'conv_rate': float(np.mean(conv)), 'n_valid': int(sum(v)),
                  'n_failed': int(sum(~v)), 'time_s': elapsed,
                  'time_per_iter': elapsed / n_iter}
    }, dv, av


def main():
    ap = argparse.ArgumentParser(description="SDT Equivalence Bound Simulation")
    ap.add_argument('--iterations', type=int, default=10000)
    ap.add_argument('--restarts', type=int, default=10)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--output-dir', type=str, default='simulation_results')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    d_a_grid = [0.5, 1.0, 1.5, 2.0, 2.5]
    s_grid = [0.70, 0.80, 0.90, 1.00]
    nw = a.workers or max(1, cpu_count() - 1)

    os.makedirs(a.output_dir, exist_ok=True)

    print("=" * 78)
    print("SDT Equivalence Bound Simulation (Pre-registration Phase 1)")
    print(f"Grid: {len(d_a_grid)}×{len(s_grid)} = {len(d_a_grid)*len(s_grid)} conditions")
    print(f"Iterations: {a.iterations} | Restarts: {a.restarts} (z-ROC seeded) | "
          f"Workers: {nw}")
    print(f"Trials: 5000 (2500 sig + 2500 noi) | Bins: 20 | Seed: {a.seed}")
    print("=" * 78)

    all_res = {}
    idx = 0
    for d_a in d_a_grid:
        for s in s_grid:
            idx += 1
            key = f"d_a={d_a}_s={s}"
            print(f"\n[{idx}/{len(d_a_grid)*len(s_grid)}] {key}")
            res, dv, av = run_condition(
                d_a, s, 2500, 2500, 20, a.iterations, a.restarts,
                a.seed + idx * 100000, nw)
            all_res[key] = res
            np.savez_compressed(os.path.join(a.output_dir, f"samples_{key}.npz"),
                                 d_a=dv, auc=av)
            d = res['dist']; b = res['bounds']; g = res['diag']
            print(f"  d_a={d['d_a_mean']:.3f}±{d['d_a_sd']:.4f} "
                  f"(bias={d['d_a_bias']:+.4f}) δ={b['d_a_delta']:.4f}")
            print(f"  s={d['s_mean']:.3f}±{d['s_sd']:.4f} "
                  f"(bias={d['s_bias']:+.4f})")
            print(f"  AUC={d['auc_mean']:.4f}±{d['auc_sd']:.4f} "
                  f"δ={b['auc_delta']:.4f}")
            print(f"  Conv={g['conv_rate']:.3f} Time={g['time_s']:.0f}s")

    out = {'metadata': {'timestamp': datetime.now().isoformat(),
                          'args': vars(a), 'n_workers': nw,
                          'method': 'L-BFGS-B + z-ROC init'},
           'conditions': all_res}
    fp = os.path.join(a.output_dir, "equivalence_bounds.json")
    with open(fp, 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'d_a':>5} {'s':>5} {'mean':>7} {'bias':>7} {'SD':>7} "
          f"{'delta':>7} {'AUC':>7} {'dAUC':>7} {'Conv':>6}")
    print("-" * 78)
    for d_a in d_a_grid:
        for s in s_grid:
            r = all_res[f"d_a={d_a}_s={s}"]
            d = r['dist']; b = r['bounds']; g = r['diag']
            print(f"{d_a:>5.1f} {s:>5.2f} {d['d_a_mean']:>7.3f} "
                  f"{d['d_a_bias']:>+7.4f} {d['d_a_sd']:>7.4f} "
                  f"{b['d_a_delta']:>7.4f} {d['auc_mean']:>7.4f} "
                  f"{b['auc_delta']:>7.4f} {g['conv_rate']*100:>5.1f}%")
    print(f"\nSaved: {fp}")


if __name__ == "__main__":
    main()
