"""Second pre-registered experiment: BOTH deep signals are MC-FPS averaged
(reproducible; single clean draws are stochastic due to random FPS init).

v3 MC: one-SE nested K (already selected K=3 by B=300 bootstrap).
v5 s2_swa MC: one-SE nested K (B=300, separate seed); also report K=25 full
average as the maximally-stable fixed rule.

Blend grid (16): w3,w5 in {0,1,2} not both 0; tau2 in {1,2}; c1:2 tau=.5.
Selection val macF1; paired bootstrap vs frozen production val predictions;
single test report for the winner only. No CSV is written automatically.
"""
import sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "clayer"))
R = BASE / "results_final"; C = R / "cache"
from fuse_c import c1_class_scores, softmax_with_taus  # noqa: E402

tr = pd.read_csv(BASE / "csv/train_set_2.csv")
ytr_all = tr.class_id.values
yval = ytr_all[7428:]
yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values
nval = len(yval)


def macf1(y, p):
    tcb = np.bincount(y, minlength=97); pres = tcb > 0
    pcb = np.bincount(p, minlength=97)
    tp = np.bincount(y[p == y], minlength=97)
    den = 2 * tp + (pcb - tp) + (tcb - tp)
    f1v = np.divide(2 * tp, den, out=np.zeros(97, float), where=den > 0)
    uni = pres | (pcb > 0)
    return (f1v * uni).sum() / uni.sum()


def probs(lp):
    z = lp - lp.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nested_k(arr, seed):
    rng = np.random.default_rng(seed); B = 300; Ks = list(range(1, 26))
    bm = np.zeros(25); bs = np.zeros(25)
    for _ in range(B):
        ix = rng.integers(0, nval, nval); yb = yval[ix]
        for K in Ks:
            x = macf1(yb, probs(arr[:K, ix].mean(0)).argmax(1))
            bm[K - 1] += x; bs[K - 1] += x * x
    bm /= B; se = np.sqrt(np.maximum(bs / B - bm ** 2, 0))
    ib = bm.argmax(); thr = bm[ib] - se[ib]
    ksel = min(K for K in Ks if bm[K - 1] >= thr)
    return ksel, Ks[ib], bm[ib], se[ib]


a3 = np.load(C / "mcfps_val_logp.npy").astype(np.float64)
b3 = np.load(C / "mcfps_test_logp.npy").astype(np.float64)
a5 = np.load(C / "mcfps_v5s2_val_logp.npy").astype(np.float64)
b5 = np.load(C / "mcfps_v5s2_test_logp.npy").astype(np.float64)
k3, b3k, m3, s3 = nested_k(a3, 2024)
k5, b5k, m5, s5 = nested_k(a5, 7)
print("one-SE K: v3 Ksel=%d (best %d %.4f+-%.4f) | v5 Ksel=%d (best %d %.4f+-%.4f)"
      % (k3, b3k, m3, s3, k5, b5k, m5, s5))

c1 = np.load(R / "c1_arrays.npz")
c1v = c1_class_scores(ytr_all[:7428], c1["val_dist"], [0.5])[0.5]
c1t = c1_class_scores(ytr_all, c1["test_dist"], [0.5])[0.5]
c2 = np.load(R / "c2_arrays.npz")
froz = np.load(R / "full_fusion_arrays.npz")["val_pred_blend"]
froz_t = pd.read_csv(R / "test_full_fusion.csv").predicted_label.values

# two fixed MC depths: one-SE and full K=25
for tag, K3, K5 in (("oneSE", k3, k5), ("full25", 25, 25)):
    P3v, P3t = probs(a3[:K3].mean(0)), probs(b3[:K3].mean(0))
    P5v, P5t = probs(a5[:K5].mean(0)), probs(b5[:K5].mean(0))
    rows = []
    for w3, w5, t2 in product(range(3), range(3), (1.0, 2.0)):
        if w3 == 0 and w5 == 0:
            continue
        qv = softmax_with_taus(c2["val_margin"], [t2])[t2]
        qt = softmax_with_taus(c2["test_margin"], [t2])[t2]
        sv = w3 * P3v + w5 * P5v + 2 * c1v + 2 * qv
        st = w3 * P3t + w5 * P5t + 2 * c1t + 2 * qt
        rows.append((w3, w5, t2, sv.argmax(1), st.argmax(1),
                     macf1(yval, sv.argmax(1))))
    rows.sort(key=lambda r: (-r[5], -(r[0] + r[1]), -r[2]))
    print("\n==== %s (K3=%d K5=%d) ====" % (tag, K3, K5))
    for r in rows[:6]:
        w3, w5, t2, pv, pt, f = r
        print("w3=%d w5=%d t2=%.0f val acc=%.4f bal=%.4f mac=%.4f"
              % (w3, w5, t2, accuracy_score(yval, pv),
                 balanced_accuracy_score(yval, pv), f))
    w = rows[0]
    rng = np.random.default_rng(99); d = np.zeros(500)
    for _ in range(500):
        ix = rng.integers(0, nval, nval)
        d[_] = macf1(yval[ix], w[3][ix]) - macf1(yval[ix], froz[ix])
    print("winner w3=%d w5=%d t2=%.0f val mac=%.4f | bootstrap vs frozen: "
          "mean=%+.4f P=%.3f CI=[%+.4f,%+.4f]"
          % (w[0], w[1], w[2], w[5], d.mean(), (d > 0).mean(),
             np.percentile(d, 2.5), np.percentile(d, 97.5)))
    p = w[4]
    print("TEST winner acc=%.4f bal=%.4f mac=%.4f | frozen prod mac=%.4f"
          % (accuracy_score(yte, p), balanced_accuracy_score(yte, p),
             f1_score(yte, p, average="macro", zero_division=0),
             f1_score(yte, froz_t, average="macro", zero_division=0)))

# single deep-model MC references (test once, already-touched GT diagnostic)
for nm, b in (("v3 MC K25", b3[:25]), ("v5 MC K25", b5[:25])):
    p = probs(b.mean(0)).argmax(1)
    print("single %s TEST acc=%.4f bal=%.4f mac=%.4f"
          % (nm, accuracy_score(yte, p), balanced_accuracy_score(yte, p),
             f1_score(yte, p, average="macro", zero_division=0)))
