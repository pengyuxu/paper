"""Extra cheap C-layer members with inductive biases not already covered:

c1d: RBF-SVM directly on raw 3DZD descriptors (363-d) -- parametric, whereas
     c1/c1b/c1c are all nearest-neighbour distance families.
c2c: FPFH RBF-SVM with Platt probabilities (frozen gamma=12, C=64 from
     tune_fpfh.py) instead of temperature-scaled margins.
c2d: cosine k-NN class vote on the L2-normalized FPFH 612-d descriptors.

Everything selected on v3 val (last 1816); test models refit on all 9244.
Saves c1d_arrays.npz / c2c_arrays.npz / c2d_arrays.npz, each with
val_soft, val_pred, val_y, test_soft, test_pred, test_y (97-column grid)."""
import time, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
R = BASE / "results_final"


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def to97(P, classes, n=97):
    m = np.full((P.shape[0], n), 1.0 / n, np.float32)
    m[:, classes] = P
    m /= m.sum(1, keepdims=True)
    return m.astype(np.float32)


def knn_soft(Xfit, yfit, Xq, k):
    """Cosine k-NN with class-weighted vote (inputs must be L2-normalized);
    ties resolved by nearest neighbour among tied classes (distance min)."""
    sim = Xq @ Xfit.T
    idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
    n = Xq.shape[0]
    soft = np.full((n, 97), 1e-9, np.float64)
    w = np.take_along_axis(sim, idx, 1)
    w = np.clip(w, 0, None) + 1e-6
    lab = yfit[idx]
    for i in range(n):
        np.add.at(soft[i], lab[i], w[i])
    soft /= soft.sum(1, keepdims=True)
    return soft.astype(np.float32)


def main():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    ytr_all = tr.class_id.values
    yfit, yval = ytr_all[:7428], ytr_all[7428:9244]
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values

    # ---------------- c2c / c2d: FPFH features ----------------
    F = np.fromfile  # noqa
    keys = tr.protein_id.astype(str).values
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_keys = te.anonymised_protein_id.astype(str).values

    def load_fpfh(ks):
        X = np.empty((len(ks), 612), np.float32)
        for i, k in enumerate(ks):
            v = np.fromfile(R / "fpfh" / (str(k).replace(".vtk", "") + ".dat"),
                            dtype=np.float32)
            X[i] = v if len(v) == 612 else 0
        return X

    Xtr = load_fpfh(keys); Xte = load_fpfh(te_keys)
    nz = Normalizer()
    Xtr_n = nz.fit_transform(Xtr).astype(np.float32)
    Xte_n = nz.transform(Xte).astype(np.float32)

    # ---- c2c: SVM with Platt probabilities (frozen gamma=12 C=64) ----
    print("== c2c FPFH-SVM probability ==", flush=True)
    t = time.time()
    sv = Pipeline([("n", Normalizer()),
                   ("s", SVC(C=64, gamma=12, class_weight="balanced",
                             probability=True, cache_size=2000,
                             random_state=42))])
    sv.fit(Xtr[:7428], yfit)
    pv = to97(sv.predict_proba(Xtr[7428:9244]), sv.classes_)
    print("  val  acc=%.4f bal=%.4f macF1=%.4f (%.1fmin)"
          % (*met(yval, pv.argmax(1)), (time.time() - t) / 60), flush=True)
    sf = Pipeline([("n", Normalizer()),
                   ("s", SVC(C=64, gamma=12, class_weight="balanced",
                             probability=True, cache_size=4000,
                             random_state=42))]).fit(Xtr, ytr_all)
    pt = to97(sf.predict_proba(Xte), sf.classes_)
    print("  TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt.argmax(1)), flush=True)
    np.savez_compressed(R / "c2c_arrays.npz",
                        val_soft=pv, val_pred=pv.argmax(1), val_y=yval,
                        test_soft=pt, test_pred=pt.argmax(1), test_y=yte)

    # ---- c2d: cosine k-NN, k chosen on val ----
    print("== c2d FPFH cosine kNN ==", flush=True)
    bestk, bestkey = None, None
    for k in (1, 3, 5, 9, 15, 25):
        ps = knn_soft(Xtr_n[:7428], yfit, Xtr_n[7428:9244], k)
        m = met(yval, ps.argmax(1))
        print("  k=%2d val acc=%.4f bal=%.4f macF1=%.4f" % (k, *m), flush=True)
        if bestkey is None or (m[2], m[0]) > bestkey:
            bestkey, bestk = (m[2], m[0]), k
    print("  frozen k=%d" % bestk, flush=True)
    pv = knn_soft(Xtr_n[:7428], yfit, Xtr_n[7428:9244], bestk)
    pt = knn_soft(Xtr_n, ytr_all, Xte_n, bestk)
    print("  TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt.argmax(1)), flush=True)
    np.savez_compressed(R / "c2d_arrays.npz",
                        val_soft=pv, val_pred=pv.argmax(1), val_y=yval,
                        test_soft=pt, test_pred=pt.argmax(1), test_y=yte)

    # ---------------- c1d: RBF-SVM on raw 3DZD 363-d ----------------
    print("== c1d 3DZD-SVM ==", flush=True)
    z = np.load(R / "c1_arrays.npz")
    Ztr, Zte = z["Xtr"].astype(np.float32), z["Xte"].astype(np.float32)
    assert Ztr.shape == (9244, 363) and Zte.shape[0] == len(yte)
    grid = list(itertools.product(["raw", "std"], ["scale", 0.003, 0.01, 0.03],
                                  [10.0, 100.0]))
    best = None
    for prep, gamma, C in grid:
        if prep == "std":
            pipe = Pipeline([("sc", StandardScaler()),
                             ("s", SVC(C=C, gamma=gamma, class_weight="balanced",
                                       cache_size=1500, random_state=42))])
        else:
            pipe = Pipeline([("n", Normalizer()),
                             ("s", SVC(C=C, gamma=gamma, class_weight="balanced",
                                       cache_size=1500, random_state=42))])
        t = time.time(); pipe.fit(Ztr[:7428], yfit)
        m = met(yval, pipe.predict(Ztr[7428:9244]))
        print("  %s gamma=%-6s C=%-4g acc=%.4f bal=%.4f macF1=%.4f (%.1fmin)"
              % (prep, str(gamma), C, *m, (time.time() - t) / 60), flush=True)
        if best is None or (m[2], m[0]) > best[0]:
            best = ((m[2], m[0]), prep, gamma, C)
    _, prep, gamma, C = best
    print("  frozen %s gamma=%s C=%g" % (prep, gamma, C), flush=True)

    def mkpipe(prob):
        pre = StandardScaler() if prep == "std" else Normalizer()
        return Pipeline([("p", pre),
                         ("s", SVC(C=C, gamma=gamma, class_weight="balanced",
                                   probability=prob, cache_size=3000,
                                   random_state=42))])

    pv_m = mkpipe(True).fit(Ztr[:7428], yfit)
    pf = mkpipe(True).fit(Ztr, ytr_all)
    pv = to97(pv_m.predict_proba(Ztr[7428:9244]), pv_m.classes_)
    pt = to97(pf.predict_proba(Zte), pf.classes_)
    print("  VAL  acc=%.4f bal=%.4f macF1=%.4f" % met(yval, pv.argmax(1)), flush=True)
    print("  TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt.argmax(1)), flush=True)
    np.savez_compressed(R / "c1d_arrays.npz",
                        val_soft=pv, val_pred=pv.argmax(1), val_y=yval,
                        test_soft=pt, test_pred=pt.argmax(1), test_y=yte)
    print("done", flush=True)


if __name__ == "__main__":
    main()
