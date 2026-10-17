"""Statistical utilities: paired bootstrap, one-SE K selection, fusion search.

All randomness is seeded; results are deterministic on disk.
"""
from __future__ import annotations
import itertools
import numpy as np

from .metrics_lib import evaluate, softmax

METRIC_KEYS = ["acc", "bal_acc", "f1_macro", "f1_weighted"]


def _metric(logp, y, num_classes, key):
    return evaluate(logp, y, num_classes)[key]


def paired_bootstrap(logp_a, logp_b, y, num_classes, keys=METRIC_KEYS,
                     B=500, seed=12345):
    """Stratified-free iid bootstrap paired on the same resample.

    Returns per key: point estimates for A/B, delta, 95% percentile CI of
    delta, two-sided bootstrap p (fraction of resamples with sign opposite
    to observed delta, incl. tie correction).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    res = {}
    for key in keys:
        base_a = _metric(logp_a, y, num_classes, key)
        base_b = _metric(logp_b, y, num_classes, key)
        d = np.empty(B)
        for b in range(B):
            ii = idx[b]
            d[b] = _metric(logp_a[ii], y[ii], num_classes, key) - \
                   _metric(logp_b[ii], y[ii], num_classes, key)
        obs = base_a - base_b
        lo, hi = np.percentile(d, [2.5, 97.5])
        if obs >= 0:
            pval = (np.sum(d <= 0) + 1.0) / (B + 1.0)
        else:
            pval = (np.sum(d >= 0) + 1.0) / (B + 1.0)
        res[key] = {"a": base_a, "b": base_b, "delta": obs,
                    "ci95": (float(lo), float(hi)), "p": float(min(pval, 1.0))}
    return res


def mc_aggregate(pass_logp, K=None, t=1.0):
    """Average probabilities across the first K FPS passes -> log-space sum."""
    p = pass_logp[:K].astype(np.float64)
    p = softmax(p, t)
    return np.log(p.mean(0) + 1e-300)


def one_se_k(pass_logp, y, num_classes, key="f1_macro",
             ks=(1, 2, 3, 5, 10, 25), B=300, seed=67890):
    """One-standard-error selection of K.

    For each K, the mean metric of B bootstrap "MC ensembles" (each pass-set
    resampled with replacement from the 25 available passes, preserving
    nested ordering) and its SE are estimated; the smallest K whose mean is
    within 1 SE of the best mean is selected.
    """
    rng = np.random.default_rng(seed)
    kmax = pass_logp.shape[0]
    rows = []
    means, ses = {}, {}
    for K in ks:
        vals = np.empty(B)
        for b in range(B):
            take = np.sort(rng.choice(kmax, size=K, replace=True))
            agg = mc_aggregate(pass_logp[take], K)
            vals[b] = _metric(agg, y, num_classes, key)
        means[K], ses[K] = vals.mean(), vals.std(ddof=1)
        rows.append({"K": K, "mean": vals.mean(), "se": vals.std(ddof=1)})
    best_k = max(ks, key=lambda k: means[k])
    thr = means[best_k] - ses[best_k]
    chosen = min(k for k in ks if means[k] >= thr)
    return {"chosen_K": chosen, "best_K": best_k,
            "threshold": thr, "table": rows}


def fusion_search(val_logps, y_val, num_classes, weights_choices=(0, 1, 2),
                  temps=None, key="f1_macro", verbose_top=None):
    """Enumerate integer weight vectors x temperature grid for M components.

    val_logps: list/tuple of (N,C) arrays. Temperature applies to every
    component (shared T, matching paper protocol). Vectorised via GEMM on
    log-softmax inputs: combined logp = sum_j w_j * logsoftmax(lp_j/T).

    Returns (best_cfg, best_score, df_of_all_configs).
    """
    import pandas as pd
    if temps is None:
        temps = np.round(np.arange(0.5, 2.01, 0.1), 2)
    M = len(val_logps)
    lps = [a.astype(np.float64) for a in val_logps]
    weight_vecs = [w for w in itertools.product(weights_choices, repeat=M)
                   if sum(w) > 0]
    records = []
    for t in temps:
        scaled = []
        for lp in lps:
            z = lp / t
            z -= z.max(1, keepdims=True)
            scaled.append(z - np.log(np.exp(z).sum(1, keepdims=True)))
        S = np.stack(scaled, 0)  # (M,N,C)
        for w in weight_vecs:
            comb = np.tensordot(np.array(w, np.float64), S, axes=(0, 0))
            score = _metric(comb, y_val, num_classes, key)
            records.append({"weights": w, "T": float(t), key: score})
    df = pd.DataFrame(records).sort_values(key, ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    return ({"weights": tuple(int(x) for x in best["weights"]),
             "T": float(best["T"])},
            float(best[key]), df)


def apply_fusion(test_logps, cfg):
    """Combine (N,C) components with selected weights/temperature."""
    w, t = cfg["weights"], cfg["T"]
    out = np.zeros_like(test_logps[0], dtype=np.float64)
    for wj, lp in zip(w, test_logps):
        if wj == 0:
            continue
        z = lp.astype(np.float64) / t
        z -= z.max(1, keepdims=True)
        out += wj * (z - np.log(np.exp(z).sum(1, keepdims=True)))
    return out
