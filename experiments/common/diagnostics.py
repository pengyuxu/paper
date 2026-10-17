"""Calibration, uncertainty and error-overlap diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics_lib import softmax


def ece(probs, y, n_bins=15):
    """Expected calibration error, equal-width confidence bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs, sizes = [], [], []
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1]) if i > 0 else \
            (conf >= bins[i]) & (conf <= bins[i + 1])
        n = m.sum()
        if n == 0:
            accs.append(np.nan); confs.append(np.nan); sizes.append(0); continue
        a, c = correct[m].mean(), conf[m].mean()
        e += n / len(y) * abs(a - c)
        accs.append(a); confs.append(c); sizes.append(int(n))
    return {"ece": float(e), "bin_acc": accs, "bin_conf": confs, "bin_n": sizes}


def aurc(probs, y):
    """Area under risk-coverage curve (lower better)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    order = np.argsort(-conf)
    correct = correct[order]
    n = len(y)
    cum_risk = 1.0 - np.cumsum(correct) / (np.arange(n) + 1)
    return {"aurc": float(np.mean(cum_risk)),
            "coverage": (np.arange(n) + 1) / n, "risk": cum_risk}


def bald_mi(pass_probs, eps=1e-12):
    """BALD mutual-information per sample from K MC passes:
    H(mean p) - mean_k H(p_k)."""
    p = np.clip(pass_probs.astype(np.float64), eps, 1.0)
    h_mean = -np.sum(p.mean(0) * np.log(p.mean(0)), axis=1)
    h_each = -np.sum(p * np.log(p), axis=2).mean(0)
    return h_mean - h_each


def flip_rate(logp_single, logp_ensemble):
    a = np.asarray(logp_single).argmax(1)
    b = np.asarray(logp_ensemble).argmax(1)
    return float((a != b).mean())


def logit_gap(logp):
    s = np.sort(np.asarray(logp, np.float64), axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def error_overlap_runs(logps_list, y):
    """Mean pairwise phi-coefficient between binary error vectors of runs."""
    E = [(lp.argmax(1) != y).astype(np.float64) for lp in logps_list]
    R = len(E)
    if R < 2:
        return {"mean_phi": np.nan, "pairwise": []}
    pairs = []
    for i in range(R):
        for j in range(i + 1, R):
            a, b = E[i], E[j]
            pa, pb = a.mean(), b.mean()
            nom = (a * b).mean() - pa * pb
            den = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
            pairs.append(float(nom / den) if den > 0 else np.nan)
    return {"mean_phi": float(np.nanmean(pairs)), "pairwise": pairs}


def class_size_groups(logp, y, num_classes, qs=(0.25, 0.75)):
    """Metrics grouped by train-set size quartiles (sizes passed via groups)."""
    raise NotImplementedError  # wrapper implemented in table pipeline with sizes


def size_grouped_metrics(logp, y, num_classes, train_sizes):
    """Split classes into small/medium/large by support-in-training tertiles,
    report macro-F1 per group."""
    from sklearn.metrics import f1_score
    pred = np.asarray(logp).argmax(1)
    present = np.unique(y)
    f1 = f1_score(y, pred, labels=present, average=None, zero_division=0)
    sizes = np.array([train_sizes.get(int(c), 0) for c in present])
    q1, q2 = np.quantile(sizes, [1 / 3, 2 / 3])
    out = {}
    for name, m in (("small", sizes <= q1),
                    ("medium", (sizes > q1) & (sizes <= q2)),
                    ("large", sizes > q2)):
        out[name] = {"n_classes": int(m.sum()),
                     "macro_f1": float(np.nanmean(f1[m])) if m.any() else np.nan}
    return out


def all_diagnostics(pass_logp, y, single_logp=None):
    """pass_logp: (K,N,C) raw logits/logp from MC passes."""
    probs_k = softmax(pass_logp)  # (K,N,C)
    ens_logp = np.log(probs_k.mean(0) + 1e-300)
    probs = softmax(ens_logp)
    out = {}
    out["ece15"] = ece(probs, y)["ece"]
    out["aurc"] = aurc(probs, y)["aurc"]
    out["bald_mean"] = float(bald_mi(probs_k).mean())
    if single_logp is not None:
        out["flip_rate"] = flip_rate(single_logp, ens_logp)
    out["gap_mean_single"] = float(np.nanmean(logit_gap(single_logp))) if single_logp is not None else None
    out["gap_mean_ens"] = float(np.nanmean(logit_gap(ens_logp)))
    return out
