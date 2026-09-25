"""MC-FPS averaged deep signal: nested K selection (one-SE, val only), then a
small pre-registered blend grid (16 candidates) with c1:2 tau=.5 fixed:
  w_mc,w_v5 in {0,1,2} not both 0 ; tau2 in {1,2}
where w_mc = v3 mean-logp over K FPS draws, w_v5 = v5 s2_swa single draw.

References on val: frozen production full-fusion predictions (full_fusion_arrays)
and the reproduced structure. Test touched once for the winner. Write CSV only
with strong val evidence (P>=.8, CI lower>=-.002 vs frozen-production val).
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


mcv = np.load(C / "mcfps_val_logp.npy").astype(np.float64)   # (25,N,97) logp
mct = np.load(C / "mcfps_test_logp.npy").astype(np.float64)

# ---- nested K one-SE selection on val (B=300 bootstrap over val rows) ----
Ks = list(range(1, 26))
single = {}
for K in Ks:
    single[K] = probs(mcv[:K].mean(0))
rng = np.random.default_rng(2024)
B = 300
bmean = np.zeros(len(Ks)); bsq = np.zeros(len(Ks))
for b in range(B):
    ix = rng.integers(0, nval, nval); yb = yval[ix]
    for i, K in enumerate(Ks):
        x = macf1(yb, probs(mcv[:K, ix].mean(0)).argmax(1))
        bmean[i] += x; bsq[i] += x * x
bmean /= B; bse = np.sqrt(np.maximum(bsq / B - bmean ** 2, 0))
ib = int(np.argmax(bmean)); thr = bmean[ib] - bse[ib]
Ksel = min(K for i, K in enumerate(Ks) if bmean[i] >= thr)
print("== nested K (one-SE): Ksel=%d, best K=%d best mean=%.4f +-%.4f =="
      % (Ksel, Ks[ib], bmean[ib], bse[ib]))
for i, K in enumerate(Ks):
    p = single[K].argmax(1)
    print("K=%2d val acc=%.4f bal=%.4f mac=%.4f"
          % (K, accuracy_score(yval, p), balanced_accuracy_score(yval, p),
             f1_score(yval, p, average="macro", zero_division=0)))

Pmcv = probs(mcv[:Ksel].mean(0)); Pmct = probs(mct[:Ksel].mean(0))

v5v = probs(np.load(C / "v5_s2_swa_val.npz")["logp"].astype(np.float64))
v5t = probs(np.load(C / "v5_s2_swa_test.npz")["logp"].astype(np.float64))

c1 = np.load(R / "c1_arrays.npz")
c1v = c1_class_scores(ytr_all[:7428], c1["val_dist"], [0.5])[0.5]
c1t = c1_class_scores(ytr_all, c1["test_dist"], [0.5])[0.5]
c2 = np.load(R / "c2_arrays.npz")

rows = []
for wm, w5, t2 in product(range(3), range(3), (1.0, 2.0)):
    if wm == 0 and w5 == 0:
        continue
    c2v = softmax_with_taus(c2["val_margin"], [t2])[t2]
    c2t = softmax_with_taus(c2["test_margin"], [t2])[t2]
    sv = wm * Pmcv + w5 * v5v + 2 * c1v + 2 * c2v
    st = wm * Pmct + w5 * v5t + 2 * c1t + 2 * c2t
    rows.append((wm, w5, t2, sv.argmax(1), st.argmax(1)))

froz_prod = np.load(R / "full_fusion_arrays.npz")["val_pred_blend"]
rows.sort(key=lambda r: (-macf1(yval, r[3]), -(r[0] + r[1]), -r[2]))
print("\n%-5s %-5s %-5s %7s %7s %7s" % ("w_mc", "w_v5", "tau2", "val_acc", "val_bal", "macF1"))
for r in rows:
    wm, w5, t2, pv, pt = r
    print("%-5d %-5d %-5.1f %7.4f %7.4f %7.4f" % (
        wm, w5, t2, accuracy_score(yval, pv), balanced_accuracy_score(yval, pv),
        macf1(yval, pv)))
win = rows[0]
print("\nwinner wm=%d w5=%d tau2=%.1f" % (win[0], win[1], win[2]))

rng = np.random.default_rng(99); d = np.zeros(500)
for b in range(500):
    ix = rng.integers(0, nval, nval)
    d[b] = macf1(yval[ix], win[3][ix]) - macf1(yval[ix], froz_prod[ix])
print("paired bootstrap vs frozen production: mean=%+.4f P=%.3f CI95=[%+.4f,%+.4f]"
      % (d.mean(), (d > 0).mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)))

p = win[4]
print("\nTEST winner acc=%.4f bal=%.4f mac=%.4f" % (
    accuracy_score(yte, p), balanced_accuracy_score(yte, p),
    f1_score(yte, p, average="macro", zero_division=0)))
print("TEST frozen production acc=%.4f (92.00 macF1)"
      % accuracy_score(yte, pd.read_csv(R / "test_full_fusion.csv").predicted_label.values))

ok = (d > 0).mean() >= 0.8 and np.percentile(d, 2.5) >= -0.002
if ok and macf1(yval, win[3]) > macf1(yval, froz_prod):
    ids = pd.read_csv(BASE / "new_data/test_set_2.csv").anonymised_protein_id
    pd.DataFrame({"anonymised_protein_id": ids, "predicted_label": p}).to_csv(
        R / "test_mcfps_fusion.csv", index=False)
    np.savez_compressed(R / "mcfps_fusion_arrays.npz",
                        val_pred=win[3], test_pred=p, K=Ksel)
    print("\nwrote results_final/test_mcfps_fusion.csv  K=%d" % Ksel)
else:
    print("\nval evidence insufficient; keeping test_full_fusion.csv")
