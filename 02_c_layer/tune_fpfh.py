"""C2/C2b val hyperparameter tuning for FPFH 612-d descriptors.
Selects on v3 val split (last 1816 train proteins) by macro-F1; freezes;
then refits on all 9244 and reports test once.

Winners overwrite c2_arrays.npz / c2b_arrays.npz (same schemas as the
official-parameter evaluations), so fuse_c.py picks them up unchanged."""
import time, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
D = BASE / "results_final/fpfh"


def load(keys):
    X = np.empty((len(keys), 612), np.float32)
    for i, k in enumerate(keys):
        v = np.fromfile(D / (str(k) + ".dat"), dtype=np.float32)
        if len(v) != 612 or not np.isfinite(v).all():
            v = np.zeros(612, np.float32)
        X[i] = v
    return X


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def ovr_margin(clf, X, n=97):
    dec = clf.decision_function(X)
    m = np.zeros((X.shape[0], n), np.float32)
    m[:, clf.classes_] = dec.astype(np.float32)
    return m


def to97(proba, classes, n=97):
    m = np.full((proba.shape[0], n), 1.0 / n, np.float32)
    m[:, classes] = proba
    m /= m.sum(1, keepdims=True)
    return m.astype(np.float32)


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    keys, ytr = tr.protein_id.values, tr.class_id.values
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_keys = te.anonymised_protein_id.astype(str).values
    Xtr = load(keys)
    Xte = load([k.replace(".vtk", "") for k in te_keys])
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values
    idx_fit, idx_val = np.arange(7428), np.arange(7428, 9244)

    # ---------------- SVM grid ----------------
    grid = list(itertools.product([4, 6, 8, 12, 16], [8, 16, 32, 64]))
    best = None
    print("== SVM gamma x C grid (%d fits) ==" % len(grid), flush=True)
    for gi, (gamma, C) in enumerate(grid):
        clf = Pipeline([("n", Normalizer()),
                        ("s", SVC(class_weight="balanced", gamma=gamma, C=C,
                                  random_state=42, cache_size=1500))])
        t = time.time(); clf.fit(Xtr[idx_fit], ytr[idx_fit])
        pv = clf.predict(Xtr[idx_val])
        a, b, f = met(ytr[idx_val], pv)
        tag = "OFFICIAL" if (gamma, C) == (8, 32) else ""
        print("  gamma=%2d C=%2d  val acc=%.4f bal=%.4f macF1=%.4f (%.1fmin) %s"
              % (gamma, C, a, b, f, (time.time() - t) / 60, tag), flush=True)
        key = (f, a)
        if best is None or key > best[0]:
            best = (key, gamma, C)
    _, bg, bC = best
    print("FROZEN SVM gamma=%g C=%g -> refit all 9244" % (bg, bC), flush=True)
    clf = Pipeline([("n", Normalizer()),
                    ("s", SVC(class_weight="balanced", gamma=bg, C=bC,
                              random_state=42, cache_size=4000))])
    t = time.time(); clf.fit(Xtr, ytr)
    clf_v = Pipeline([("n", Normalizer()),
                      ("s", SVC(class_weight="balanced", gamma=bg, C=bC,
                                random_state=42, cache_size=1500))]).fit(
        Xtr[idx_fit], ytr[idx_fit])
    pv = clf_v.predict(Xtr[idx_val]); pt = clf.predict(Xte)
    print("TEST acc=%.4f bal=%.4f macF1=%.4f (%.1fmin)"
          % (*met(yte, pt), (time.time() - t) / 60), flush=True)
    np.savez_compressed(BASE / "results_final/c2_arrays.npz",
                        val_pred=pv, val_margin=ovr_margin(clf_v, Xtr[idx_val]),
                        val_y=ytr[idx_val],
                        test_pred=pt, test_margin=ovr_margin(clf, Xte),
                        test_y=yte,
                        test_margin_fit7428=ovr_margin(clf_v, Xte))
    pd.DataFrame({"anonymised_protein_id": te_keys, "predicted_label": pt}
                 ).to_csv(BASE / "results_final/test_c2_fpfh_svm.csv", index=False)

    # ---------------- ExtraTrees grid ----------------
    egrid = list(itertools.product([None, 0.5, "sqrt"], [1, 2]))
    beste = None
    print("== ExtraTrees max_features x min_leaf grid ==", flush=True)
    for mf, leaf in egrid:
        et = ExtraTreesClassifier(n_estimators=500, max_features=mf,
                                  min_samples_leaf=leaf, class_weight="balanced",
                                  n_jobs=-1, random_state=42)
        t = time.time(); et.fit(Xtr[idx_fit], ytr[idx_fit])
        pv = et.predict(Xtr[idx_val])
        a, b, f = met(ytr[idx_val], pv)
        tag = "OFFICIAL" if (mf is None and leaf == 1) else ""
        print("  mf=%-4s leaf=%d val acc=%.4f bal=%.4f macF1=%.4f (%.1fmin) %s"
              % (str(mf), leaf, a, b, f, (time.time() - t) / 60, tag), flush=True)
        key = (f, a)
        if beste is None or key > beste[0]:
            beste = (key, mf, leaf)
    _, mf, leaf = beste
    print("FROZEN ET mf=%s leaf=%d -> refit all 9244" % (mf, leaf), flush=True)
    et_v = ExtraTreesClassifier(n_estimators=500, max_features=mf,
                                min_samples_leaf=leaf, class_weight="balanced",
                                n_jobs=-1, random_state=42).fit(Xtr[idx_fit], ytr[idx_fit])
    et = ExtraTreesClassifier(n_estimators=500, max_features=mf,
                              min_samples_leaf=leaf, class_weight="balanced",
                              n_jobs=-1, random_state=42).fit(Xtr, ytr)
    pv, pt = et_v.predict(Xtr[idx_val]), et.predict(Xte)
    print("ET TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt), flush=True)
    np.savez_compressed(BASE / "results_final/c2b_arrays.npz",
                        val_soft=to97(et_v.predict_proba(Xtr[idx_val]), et_v.classes_),
                        val_pred=pv, val_y=ytr[idx_val],
                        test_soft=to97(et.predict_proba(Xte), et.classes_),
                        test_pred=pt, test_y=yte)
    print("saved c2_arrays.npz / c2b_arrays.npz", flush=True)


if __name__ == "__main__":
    main()
