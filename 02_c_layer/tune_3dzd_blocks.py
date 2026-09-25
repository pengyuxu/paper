"""C1c: block-weighted 3DZD distance (whole / +surface / -surface).

Each 121-d block distance is z-scored (finite entries only; volume-filtered
entries set to a large finite sentinel), then combined with tunable weights.
Selection on v3 val (7428/1816); frozen, then applied to test (gallery 9244).
k-NN soft scores saved for fusion: results_final/c1c_arrays.npz
"""
import itertools
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
NTR = 7428
NC = 97
BLOCKS = ((0, 121), (121, 242), (242, 363))
BIG = 1e6


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def block_dists(Xq, Xg, Vq, Vg, lo=0.8, hi=1.2):
    out = []
    ratio = Vq[:, None] / np.maximum(Vg[None, :], 1e-9)
    vm = (ratio < lo) | (ratio > hi)
    for a, b in BLOCKS:
        D = cdist(Xq[:, a:b], Xg[:, a:b]).astype(np.float64)
        fin = np.isfinite(D) & ~vm
        mu, sd = D[fin].mean(), D[fin].std() + 1e-12
        Z = (D - mu) / sd
        Z[~fin] = BIG
        out.append(Z)
    return out


def knn_soft(Z, yg, k, tau):
    D = Z
    kk = min(k, D.shape[1])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    rows = np.arange(D.shape[0])[:, None]
    dk = D[rows, idx]
    if tau is None:
        w = (dk < BIG / 2).astype(np.float64)
    elif tau == "inv":
        w = 1.0 / (dk + 1.0)
    else:
        md = dk.min(1, keepdims=True)
        w = np.exp(-(dk - md) / tau)
    sc = np.zeros((D.shape[0], NC))
    np.add.at(sc, (np.repeat(np.arange(D.shape[0]), kk), yg[idx].reshape(-1)), w.reshape(-1))
    return sc


def main():
    z = np.load(BASE / "results_final/c1_arrays.npz")
    Xtr, Vtr, y = z["Xtr"], z["Vtr"], z["tr_y"]
    Xte, Vte = z["Xte"], z["Vte"]
    yv = y[NTR:]

    Bv = block_dists(Xtr[NTR:], Xtr[:NTR], Vtr[NTR:], Vtr[:NTR])
    Bt = block_dists(Xte, Xtr, Vte, Vtr)

    results = []
    soft_cache_val = {}
    for wp, wn in itertools.product(np.arange(0.25, 2.01, 0.25), repeat=2):
        Dv = Bv[0] + wp * Bv[1] + wn * Bv[2]
        for k in (1, 3, 5):
            for tau in (None, "inv", 0.1, 0.3):
                sc = knn_soft(Dv, y[:NTR], k, tau)
                p = sc.argmax(1)
                results.append((f1_score(yv, p, average="macro", zero_division=0),
                                accuracy_score(yv, p), wp, wn, k, str(tau)))
                soft_cache_val[(wp, wn, k, str(tau))] = sc
    results.sort(reverse=True)
    print("top 8:")
    for r in results[:8]:
        print("  macF1=%.4f acc=%.4f wp=%.2f wn=%.2f k=%d tau=%s" % r)
    _, _, wp, wn, k, tau_s = results[0]
    tau = None if tau_s == "None" else ("inv" if tau_s == "inv" else float(tau_s))
    print("FROZEN wp=%.2f wn=%.2f k=%d tau=%s" % (wp, wn, k, tau_s))
    sv = soft_cache_val[(wp, wn, k, tau_s)]
    Dt = Bt[0] + wp * Bt[1] + wn * Bt[2]
    st = knn_soft(Dt, y, k, tau)
    gt = pd.read_csv(BASE / "csv/test_set_ground_truth.csv")
    yte = gt.class_id.values
    print("VAL acc=%.4f bal=%.4f macF1=%.4f" % met(yv, sv.argmax(1)))
    print("TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, st.argmax(1)))
    np.savez_compressed(BASE / "results_final/c1c_arrays.npz",
                        val_soft=(sv / np.maximum(sv.sum(1, keepdims=True), 1e-12)).astype(np.float32),
                        val_pred=sv.argmax(1), val_y=yv,
                        test_soft=(st / np.maximum(st.sum(1, keepdims=True), 1e-12)).astype(np.float32),
                        test_pred=st.argmax(1), test_y=yte)
    print("saved results_final/c1c_arrays.npz")


if __name__ == "__main__":
    main()
