"""C1 evaluation: official 3D-SURFER 3DZD 1-NN protocol.
- val: gallery = first 7428 (v3 split), queries = last 1816
- test: gallery = all 9244 (exact official protocol), volume-ratio filter [0.8,1.2]
Also checks agreement with the predictions shipped in the official repo.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
D = BASE / "results_final/3dzd"


def load(keys):
    X, V, miss = [], [], []
    for k in keys:
        f = D / (str(k) + ".npz")
        if not f.exists():
            miss.append(k); X.append(np.zeros(363, np.float32)); V.append(0.0); continue
        z = np.load(f)
        X.append(z["inv"]); V.append(float(z["volume"]))
    return np.asarray(X, np.float32), np.asarray(V, np.float64), miss


def nn_predict(Xq, Vq, Xg, Vg, yg, vol_lo=0.8, vol_hi=1.2):
    dist = cdist(Xq, Xg)
    ratio = Vq[:, None] / np.maximum(Vg[None, :], 1e-9)
    dist[(ratio < vol_lo) | (ratio > vol_hi)] = np.inf
    idx = np.argmin(dist, axis=1)
    return yg[idx], idx, dist


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0),
            f1_score(y, p, average="micro", zero_division=0))


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    tr_keys = tr.protein_id.values; tr_y = tr.class_id.values
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_raw = te.anonymised_protein_id.astype(str).values
    te_keys = np.array([k.replace(".vtk", "") for k in te_raw])  # npz keys lack .vtk

    Xtr, Vtr, m1 = load(tr_keys)
    Xte, Vte, m2 = load(te_keys)
    print("missing descriptors: train=%d test=%d" % (len(m1), len(m2)))
    print("zero-pos-surface train=%d test=%d" %
          ((np.all(Xtr[:, 121:242] == 0, axis=1)).sum(),
           (np.all(Xte[:, 121:242] == 0, axis=1)).sum()))
    print("zero-neg-surface train=%d test=%d" %
          ((np.all(Xtr[:, 242:] == 0, axis=1)).sum(),
           (np.all(Xte[:, 242:] == 0, axis=1)).sum()))

    # val (gallery = train split)
    pv, iv, Dv = nn_predict(Xtr[7428:], Vtr[7428:], Xtr[:7428], Vtr[:7428], tr_y[:7428])
    print("VAL  acc=%.4f bal=%.4f macF1=%.4f" % met(tr_y[7428:], pv)[:3])

    # test with full gallery (official)
    pt, it, Dm = nn_predict(Xte, Vte, Xtr, Vtr, tr_y)
    gt = pd.read_csv(BASE / "csv/test_set_ground_truth.csv")
    yte = gt.class_id.values
    print("TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt)[:3])

    # agreement with official shipped predictions
    shipped = pd.read_csv(BASE / "external/SHREC2025/test_predictions.csv")
    sh_keys = shipped.query_key.astype(str).values
    sh_pred = shipped.predicted_label.values.astype(int)
    order = {k: i for i, k in enumerate(te_keys)}
    pair = [(order[k], p) for k, p in zip(sh_keys, sh_pred) if k in order]
    oi, op = zip(*pair)
    agree = (pt[list(oi)] == np.array(op)).mean()
    print("agreement with official shipped test_predictions: %.4f (n=%d)" % (agree, len(oi)))
    print("official shipped metrics: acc=%.4f bal=%.4f macF1=%.4f" % met(yte[list(oi)], np.array(op))[:3])

    # rare-class recall
    vc = pd.Series(tr_y).value_counts()
    rare = [c for c in vc.index if vc[c] <= 10]
    m = np.isin(yte, rare)
    print("rare(n<=10) test recall ours=%.3f official=%.3f" %
          ((pt[m] == yte[m]).mean(), (np.array(op)[np.isin(yte[list(oi)], rare)] ==
                                      yte[list(oi)][np.isin(yte[list(oi)], rare)]).mean()))

    out = pd.DataFrame({"anonymised_protein_id": te_raw, "predicted_label": pt})
    out.to_csv(BASE / "results_final/test_c1_3dzd_nn.csv", index=False)
    np.savez_compressed(BASE / "results_final/c1_arrays.npz",
                        Xtr=Xtr, Vtr=Vtr, tr_y=tr_y, Xte=Xte, Vte=Vte,
                        val_pred=pv, val_nn_idx=iv, val_dist=Dv.astype(np.float32),
                        test_pred=pt, test_nn_idx=it, test_dist=Dm.astype(np.float32))
    print("saved results_final/test_c1_3dzd_nn.csv")


if __name__ == "__main__":
    main()
