"""c4: RBF-SVM on concatenated descriptors [3DZD 363 | FPFH 612], each block
standardized with fit-split (7428) statistics. Selected on v3 val by
macro-F1; test model refit on all 9244. Soft scores = softmax(OVR margin
/ 0.5) (Platt calibration hurts on this imbalanced task).
Saves results_final/c4_arrays.npz."""
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


def softmax(x, t):
    z = x / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    keys, y = tr.protein_id.astype(str).values, tr.class_id.values
    yv = y[7428:]
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_keys = [str(k).replace(".vtk", "") for k in te.anonymised_protein_id]
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values

    Ftr = np.stack([np.fromfile(R / "fpfh" / (k + ".dat"), np.float32) for k in keys])
    Fte = np.stack([np.fromfile(R / "fpfh" / (k + ".dat"), np.float32) for k in te_keys])
    z = np.load(R / "c1_arrays.npz")
    Ztr, Zte = z["Xtr"].astype(np.float32), z["Xte"].astype(np.float32)

    scz = StandardScaler().fit(Ztr[:7428]); scf = StandardScaler().fit(Ftr[:7428])
    Xtr = np.concatenate([scz.transform(Ztr), scf.transform(Ftr)], 1).astype(np.float32)
    Xte = np.concatenate([scz.transform(Zte), scf.transform(Fte)], 1).astype(np.float32)

    best = None
    for C, gam in itertools.product((16, 32, 64, 100), ("scale", 0.001, 0.003)):
        clf = SVC(C=C, gamma=gam, class_weight="balanced", cache_size=2500,
                  random_state=42)
        t = time.time(); clf.fit(Xtr[:7428], y[:7428])
        m = met(yv, clf.predict(Xtr[7428:]))
        print("C=%-4g gamma=%-6s val acc %.4f bal %.4f mac %.4f (%.1fmin)"
              % (C, str(gam), *m, (time.time() - t) / 60), flush=True)
        if best is None or (m[2], m[0]) > best[0]:
            best = ((m[2], m[0]), C, gam)
    _, C, gam = best
    print("frozen C=%g gamma=%s" % (C, gam), flush=True)

    mv = SVC(C=C, gamma=gam, class_weight="balanced", cache_size=2500,
             random_state=42).fit(Xtr[:7428], y[:7428])
    mf = SVC(C=C, gamma=gam, class_weight="balanced", cache_size=4000,
             random_state=42).fit(Xtr, y)
    pv, pt = softmax(margins(mv, Xtr[7428:]), 0.5), softmax(margins(mf, Xte), 0.5)
    print("VAL  acc=%.4f bal=%.4f macF1=%.4f" % met(yv, pv.argmax(1)), flush=True)
    print("TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt.argmax(1)), flush=True)
    np.savez_compressed(R / "c4_arrays.npz",
                        val_soft=pv, val_pred=pv.argmax(1), val_y=yv,
                        test_soft=pt, test_pred=pt.argmax(1), test_y=yte)
    print("saved c4_arrays.npz", flush=True)


if __name__ == "__main__":
    main()
