"""C2b: Tatsuma FPFH features + ExtraTrees (the alternative learner in the
official cross_val.py). Different error pattern from the RBF SVC, so used as
an additional fusion member. L2 Normalizer, class_weight balanced.
Val model fit on first 7428; test model refit on all 9244.
"""
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
D = BASE / "results_final/fpfh"


def load(keys):
    X, miss = [], []
    for k in keys:
        p = D / (str(k) + ".dat")
        if not p.exists():
            miss.append(k); X.append(np.zeros(612, np.float32)); continue
        v = np.fromfile(p, dtype=np.float32)
        X.append(v if len(v) == 612 else np.zeros(612, np.float32))
    return np.asarray(X, np.float32), miss


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def to97(clf, proba, n):
    out = np.full((n, 97), 1.0 / 97, np.float32)
    out[:, clf.classes_] = proba
    out /= out.sum(1, keepdims=True)
    return out


def make():
    return Pipeline([("normalizer", Normalizer()),
                     ("et", ExtraTreesClassifier(
                         n_estimators=500, max_features=None,
                         class_weight="balanced", random_state=42, n_jobs=-1))])


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    keys, ytr = tr.protein_id.values, tr.class_id.values
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_keys = te.anonymised_protein_id.astype(str).values
    Xtr, m1 = load(keys)
    Xte, m2 = load([k.replace(".vtk", "") for k in te_keys])
    print("missing: train=%d test=%d %s" % (len(m1), len(m2), Xtr.shape), flush=True)
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values

    t = time.time()
    clf = make(); clf.fit(Xtr[:7428], ytr[:7428])
    pv = clf.predict(Xtr[7428:]); sv = to97(clf, clf.predict_proba(Xtr[7428:]).astype(np.float32), 1816)
    print("val fit %.1fmin VAL acc=%.4f bal=%.4f macF1=%.4f"
          % ((time.time() - t) / 60, *met(ytr[7428:], pv)), flush=True)

    t = time.time()
    clf = make(); clf.fit(Xtr, ytr)
    pt = clf.predict(Xte); st = to97(clf, clf.predict_proba(Xte).astype(np.float32), Xte.shape[0])
    print("full fit %.1fmin TEST acc=%.4f bal=%.4f macF1=%.4f"
          % ((time.time() - t) / 60, *met(yte, pt)), flush=True)

    np.savez_compressed(BASE / "results_final/c2b_arrays.npz",
                        val_soft=sv, val_pred=pv, val_y=ytr[7428:],
                        test_soft=st, test_pred=pt, test_y=yte)
    print("saved results_final/c2b_arrays.npz", flush=True)


if __name__ == "__main__":
    main()
