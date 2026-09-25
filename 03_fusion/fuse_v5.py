"""Pre-registered small fusion grid adding v5 (s2_swa) to the frozen one-SE
blend structure (c1:2 tau=.5, c2:2).

Candidates (16): w3,w5 in {0,1,2} not both 0; tau2 in {1,2}.
Frozen reference: w3=1,w5=0,tau2=2 -> VAL 90.67 / TEST 92.00.
Selection: val macF1 argmax, ties -> lower complexity; paired bootstrap vs
frozen; test reported once for the winner. Write test_v5_fusion.csv only if
val evidence (P>=0.8 and CI lower>=-0.002) supports replacement.
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
    f1 = np.divide(2 * tp, den, out=np.zeros(97, float), where=den > 0)
    uni = pres | (pcb > 0)
    return (f1 * uni).sum() / uni.sum()


def probs(lp):
    z = lp - lp.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


v3v = probs(np.load(C / "val_clean.npz")["logp"].astype(np.float64))
v3t = probs(np.load(C / "test_clean.npz")["logp"].astype(np.float64))
v5v = probs(np.load(C / "v5_s2_swa_val.npz")["logp"].astype(np.float64))
v5t = probs(np.load(C / "v5_s2_swa_test.npz")["logp"].astype(np.float64))

c1 = np.load(R / "c1_arrays.npz")
c1v = c1_class_scores(ytr_all[:7428], c1["val_dist"], [0.5])[0.5]
c1t = c1_class_scores(ytr_all, c1["test_dist"], [0.5])[0.5]
c2 = np.load(R / "c2_arrays.npz")

rows = []
for w3, w5, t2 in product(range(3), range(3), (1.0, 2.0)):
    if w3 == 0 and w5 == 0:
        continue
    c2v = softmax_with_taus(c2["val_margin"], [t2])[t2]
    c2t = softmax_with_taus(c2["test_margin"], [t2])[t2]
    sv = w3 * v3v + w5 * v5v + 2 * c1v + 2 * c2v
    st = w3 * v3t + w5 * v5t + 2 * c1t + 2 * c2t
    rows.append((w3, w5, t2, sv.argmax(1), st.argmax(1),
                 macf1(yval, sv.argmax(1))))

rows.sort(key=lambda r: (-r[5], -(r[0] + r[1]), -r[2]))
print("%-6s %-6s %-5s %7s %7s %7s" % ("w_v3", "w_v5", "tau2", "val_acc", "val_bal", "macF1"))
froz = None
for r in rows:
    w3, w5, t2, pv, pt, f = r
    print("%-6d %-6d %-5.1f %7.4f %7.4f %7.4f" % (
        w3, w5, t2, accuracy_score(yval, pv), balanced_accuracy_score(yval, pv), f))
    if w3 == 1 and w5 == 0 and t2 == 2.0:
        froz = r

win = rows[0]
print("\nwinner: w3=%d w5=%d tau2=%.1f val mac=%.4f" % (win[0], win[1], win[2], win[5]))
print("frozen: w3=1 w5=0 tau2=2.0 val mac=%.4f" % froz[5])

rng = np.random.default_rng(42); B = 500
d = np.zeros(B)
for b in range(B):
    ix = rng.integers(0, nval, nval)
    d[b] = macf1(yval[ix], win[3][ix]) - macf1(yval[ix], froz[3][ix])
print("paired bootstrap vs frozen: mean=%+.4f P(better)=%.3f CI95=[%+.4f,%+.4f]"
      % (d.mean(), (d > 0).mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)))

p = win[4]
print("\nTEST winner acc=%.4f bal=%.4f mac=%.4f" % (
    accuracy_score(yte, p), balanced_accuracy_score(yte, p),
    f1_score(yte, p, average="macro", zero_division=0)))
print("TEST frozen acc=%.4f bal=%.4f mac=%.4f" % (
    accuracy_score(yte, froz[4]), balanced_accuracy_score(yte, froz[4]),
    f1_score(yte, froz[4], average="macro", zero_division=0)))

ok = (d > 0).mean() >= 0.8 and np.percentile(d, 2.5) >= -0.002
if ok and win[5] > froz[5]:
    ids = pd.read_csv(BASE / "new_data/test_set_2.csv")["anonymised_protein_id"]
    pd.DataFrame({"anonymised_protein_id": ids, "predicted_label": p}).to_csv(
        R / "test_v5_fusion.csv", index=False)
    print("\nwrote results_final/test_v5_fusion.csv")
else:
    print("\nval evidence insufficient; keeping frozen test_full_fusion.csv")
