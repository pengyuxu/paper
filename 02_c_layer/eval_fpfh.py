"""C2 evaluation: Tatsuma FPFH+potential stats (612-d) -> L2 normalize ->
RBF SVC(C=32, gamma=8, class_weight=balanced), exact params from ml/main.py.
Val: fit on first 7428. Test: refit on all 9244 (official protocol)."""
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
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


def ovo_margins(clf, X, n_classes=97):
    """SVC default decision_function_shape='ovr' -> (n, n_classes_seen).
    Align columns to fixed 0..96 label grid (missing classes -> 0)."""
    dec = clf.decision_function(X)
    m = np.zeros((X.shape[0], n_classes), np.float32)
    m[:, clf.classes_] = dec.astype(np.float32)
    return m


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    keys, ytr = tr.protein_id.values, tr.class_id.values
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_keys = te.anonymised_protein_id.astype(str).values
    te_npz_keys = np.array([k.replace(".vtk", "") for k in te_keys])
    Xtr, m1 = load(keys); Xte, m2 = load(te_npz_keys)
    print("missing: train=%d test=%d" % (len(m1), len(m2)), "Xtr", Xtr.shape)

    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values
    for name, idx_tr, idx_va in [("val-fit7428", np.arange(7428), np.arange(7428, 9244))]:
        clf = Pipeline([("normalizer", Normalizer()),
                        ("svc", SVC(class_weight="balanced", gamma=8, C=32,
                                    random_state=42, cache_size=2000))])
        t = time.time(); clf.fit(Xtr[idx_tr], ytr[idx_tr])
        pv = clf.predict(Xtr[idx_va])
        val_margin = ovo_margins(clf, Xtr[idx_va])
        test_margin_7428 = ovo_margins(clf, Xte)
        print("%s fit %.1fmin VAL acc=%.4f bal=%.4f macF1=%.4f"
              % (name, (time.time() - t) / 60, *met(ytr[idx_va], pv)))

    print("refitting on full 9244 ...")
    clf = Pipeline([("normalizer", Normalizer()),
                    ("svc", SVC(class_weight="balanced", gamma=8, C=32,
                                random_state=42, cache_size=4000))])
    t = time.time(); clf.fit(Xtr, ytr)
    pt = clf.predict(Xte)
    test_margin = ovo_margins(clf, Xte)
    print("full fit %.1fmin TEST acc=%.4f bal=%.4f macF1=%.4f nSV=%d"
          % ((time.time() - t) / 60, *met(yte, pt), int(clf.named_steps["svc"].n_support_.sum())))
    pd.DataFrame({"anonymised_protein_id": te_keys, "predicted_label": pt}
                 ).to_csv(BASE / "results_final/test_c2_fpfh_svm.csv", index=False)
    np.savez_compressed(BASE / "results_final/c2_arrays.npz",
                        Xtr=Xtr, tr_y=ytr, Xte=Xte,
                        val_pred=pv, val_margin=val_margin,
                        test_pred=pt, test_margin=test_margin,
                        test_margin_fit7428=test_margin_7428)
    print("saved results_final/test_c2_fpfh_svm.csv")


if __name__ == "__main__":
    main()
