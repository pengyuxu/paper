"""C1b: tuned 3DZD nearest-classifier variant (additional ensemble member).

Official c1 is 1-NN raw Euclidean + [0.8,1.2] volume filter (reproduced
exactly). Here we tune ONLY on v3 val (gallery=first 7428, queries=last 1816),
then freeze and apply to test (gallery=all 9244). Produces per-class soft
scores for fusion: results_final/c1b_arrays.npz
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
NTR = 7428
NC = 97


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def vol_mask(Vq, Vg, lo, hi):
    ratio = Vq[:, None] / np.maximum(Vg[None, :], 1e-9)
    return (ratio < lo) | (ratio > hi)


def knn_soft(dist, yg, yg_all, k, tau, vol=None, prior_mode="none"):
    """Return per-class score from k nearest neighbours.

    tau None -> uniform votes; 'inv' -> inverse distance; else softmax(-d/tau)
    over the k neighbours accumulated per class.
    """
    D = dist.copy()
    if vol is not None:
        D[vol] = np.inf
    kk = min(k, D.shape[1])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    rows = np.arange(D.shape[0])[:, None]
    dk = D[rows, idx]
    valid = np.isfinite(dk)
    yk = yg[idx]
    if tau is None:
        w = valid.astype(np.float64)
    elif tau == "inv":
        w = valid / (dk + 1e-6)
        w[~valid] = 0.0
    else:
        d = np.where(valid, dk, np.nan)
        md = np.nanmin(d, axis=1, keepdims=True)
        w = np.where(valid, np.exp(-(dk - md) / tau), 0.0)
    score = np.zeros((D.shape[0], NC))
    np.add.at(score, (np.repeat(rows, kk), yk.reshape(-1)), w.reshape(-1))
    if prior_mode == "add":  # tie-break only; add tiny train prior
        prior = np.bincount(yg_all, minlength=NC) / len(yg_all)
        score += 1e-6 * prior[None, :]
    return score


def preprocess(Xtr, Xq, Xte, mode):
    if mode == "none":
        return Xtr, Xq, Xte
    if mode == "zscore":
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr).astype(np.float32), sc.transform(Xq).astype(np.float32), \
            sc.transform(Xte).astype(np.float32)
    if mode == "l2norm":
        def f(X):
            n = np.linalg.norm(X, axis=1, keepdims=True)
            return (X / np.maximum(n, 1e-9)).astype(np.float32)
        return f(Xtr), f(Xq), f(Xte)
    if mode == "log1p":
        def f(X):
            return np.sign(X) * np.log1p(np.abs(X)).astype(np.float32)
        return f(Xtr), f(Xq), f(Xte)
    raise ValueError(mode)


def main():
    z = np.load(BASE / "results_final/c1_arrays.npz")
    Xtr, Vtr, tr_y = z["Xtr"], z["Vtr"], z["tr_y"]
    Xte, Vte = z["Xte"], z["Vte"]
    yval = tr_y[NTR:]
    Xg0, Vg0, yg0 = Xtr[:NTR], Vtr[:NTR], tr_y[:NTR]
    Xq0, Vq0 = Xtr[NTR:], Vtr[NTR:]

    Vv = vol_mask(Vq0, Vg0, 0.8, 1.2)
    Vt = vol_mask(Vte, Vtr, 0.8, 1.2)

    grid = []
    best = None
    for prep in ("none", "zscore", "l2norm", "log1p"):
        Xg, Xq, _ = preprocess(Xg0, Xq0, Xte, prep)
        Dv = cdist(Xq, Xg).astype(np.float64)
        for vf in (None, Vv):
            for k in (1, 3, 5, 9, 15):
                for tau in (None, "inv", 0.05, 0.1, 0.2, 0.5):
                    sc = knn_soft(Dv, yg0, yg0, k, tau, vf,
                                 prior_mode="add" if k > 1 else "none")
                    p = sc.argmax(1)
                    a, b, m = met(yval, p)
                    grid.append((m, a, b, prep, "vol" if vf is not None else "novol", k, str(tau)))
                    if best is None or m > best[0]:
                        best = (m, a, b, prep, vf is not None, k, tau)

    grid.sort(reverse=True)
    print("top 12 by val macroF1:")
    for g in grid[:12]:
        print("  macF1=%.4f acc=%.4f bal=%.4f prep=%s vol=%s k=%d tau=%s" % g)
    m, a, b, prep, usevol, k, tau = best
    print("FROZEN: prep=%s vol=%s k=%d tau=%s val(macF1=%.4f acc=%.4f bal=%.4f)"
          % (prep, usevol, k, tau, m, a, b))

    Xg, Xq, Xtp = preprocess(Xtr, Xq0, Xte, prep)
    Dte = cdist(Xtp, Xg).astype(np.float64)
    s_te = knn_soft(Dte, tr_y, tr_y, k, tau, Vt if usevol else None,
                   prior_mode="add" if k > 1 else "none")
    p_te = s_te.argmax(1)
    # val soft with frozen cfg for fusion validation
    Xg2, Xq2, _ = preprocess(Xg0, Xq0, Xte, prep)
    Dv2 = cdist(Xq2, Xg2).astype(np.float64)
    s_va = knn_soft(Dv2, yg0, yg0, k, tau, Vv if usevol else None,
                   prior_mode="add" if k > 1 else "none")

    gt = pd.read_csv(BASE / "csv/test_set_ground_truth.csv")
    yte = gt.class_id.values
    print("TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, p_te))

    np.savez_compressed(BASE / "results_final/c1b_arrays.npz",
                        val_soft=(s_va / np.maximum(s_va.sum(1, keepdims=True), 1e-12)).astype(np.float32),
                        val_pred=s_va.argmax(1), val_y=yval,
                        test_soft=(s_te / np.maximum(s_te.sum(1, keepdims=True), 1e-12)).astype(np.float32),
                        test_pred=p_te, test_y=yte)
    print("saved results_final/c1b_arrays.npz")


if __name__ == "__main__":
    main()
