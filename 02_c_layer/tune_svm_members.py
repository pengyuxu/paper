"""Finer val refinement for the two strongest parametric members:
c1d (StandardScaler RBF-SVM on 363-d 3DZD) and c4 (block-scaled RBF-SVM on
[3DZD|FPFH] 975-d). Grids around the frozen coarse winners. Overwrites
c1d_arrays.npz / c4_arrays.npz with margin-softmax (tau=0.5) arrays."""
import time, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
R = BASE / "results_final"


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def margins(clf, X):
    d = clf.decision_function(X)
    m = np.zeros((X.shape[0], 97), np.float32)
    m[:, clf.classes_] = d.astype(np.float32)
    return m


def sm(x, t=0.5):
    z = x / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def fit_select(name, Xtr, Xte, y, yte, grid):
    yv = y[7428:]
    best = None
    for C, gam in grid:
        clf = SVC(C=C, gamma=gam, class_weight="balanced", cache_size=2500,
                  random_state=42)
        t = time.time(); clf.fit(Xtr[:7428], y[:7428])
        m = met(yv, clf.predict(Xtr[7428:]))
        print("%s C=%-5g gamma=%-6g val acc=%.4f bal=%.4f mac=%.4f (%.1fmin)"
              % (name, C, gam, *m, (time.time() - t) / 60), flush=True)
        if best is None or (m[2], m[0]) > best[0]:
            best = ((m[2], m[0]), C, gam)
    _, C, gam = best
    print("%s FROZEN C=%g gamma=%g" % (name, C, gam), flush=True)
    mv = SVC(C=C, gamma=gam, class_weight="balanced", cache_size=2500,
             random_state=42).fit(Xtr[:7428], y[:7428])
    mf = SVC(C=C, gamma=gam, class_weight="balanced", cache_size=4000,
             random_state=42).fit(Xtr, y)
    pv, pt = sm(margins(mv, Xtr[7428:])), sm(margins(mf, Xte))
    print("%s VAL  acc=%.4f bal=%.4f mac=%.4f" % (name, *met(yv, pv.argmax(1))),
          flush=True)
    print("%s TEST acc=%.4f bal=%.4f mac=%.4f" % (name, *met(yte, pt.argmax(1))),
          flush=True)
    return pv, pt


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    keys, y = tr.protein_id.astype(str).values, tr.class_id.values
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_keys = [str(k).replace(".vtk", "") for k in te.anonymised_protein_id]
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values
    yv = y[7428:]

    Ftr = np.stack([np.fromfile(R / "fpfh" / (k + ".dat"), np.float32) for k in keys])
    Fte = np.stack([np.fromfile(R / "fpfh" / (k + ".dat"), np.float32) for k in te_keys])
    z = np.load(R / "c1_arrays.npz")
    Ztr, Zte = z["Xtr"].astype(np.float32), z["Xte"].astype(np.float32)

    sc = StandardScaler().fit(Ztr[:7428])
    Z1 = sc.transform(Ztr).astype(np.float32)
    Z2 = sc.transform(Zte).astype(np.float32)
    pv, pt = fit_select("c1d", Z1, Z2, y, yte,
                        list(itertools.product((50, 100, 200, 400),
                                               (0.0015, 0.002, 0.003, 0.005))))
    np.savez_compressed(R / "c1d_arrays.npz",
                        val_soft=pv, val_pred=pv.argmax(1), val_y=yv,
                        test_soft=pt, test_pred=pt.argmax(1), test_y=yte)

    scz = StandardScaler().fit(Ztr[:7428]); scf = StandardScaler().fit(Ftr[:7428])
    Xtr = np.concatenate([scz.transform(Ztr), scf.transform(Ftr)], 1).astype(np.float32)
    Xte = np.concatenate([scz.transform(Zte), scf.transform(Fte)], 1).astype(np.float32)
    pv, pt = fit_select("c4", Xtr, Xte, y, yte,
                        list(itertools.product((8, 16, 32, 64),
                                               (0.002, 0.003, 0.005))))
    np.savez_compressed(R / "c4_arrays.npz",
                        val_soft=pv, val_pred=pv.argmax(1), val_y=yv,
                        test_soft=pt, test_pred=pt.argmax(1), test_y=yte)
    print("done", flush=True)


if __name__ == "__main__":
    main()
