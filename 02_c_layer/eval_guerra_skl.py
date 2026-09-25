"""C3b: classical classifiers on Guerra topological descriptors (171-d
cleaned canonical view). Complements the official SimpleNN (c3) with a
different inductive bias. Select on v3 val (last 1816) by macro-F1;
freeze; refit transforms+model on all 9244 train; report test once.
Saves results_final/c3b_arrays.npz with val/test soft probs (97 grid)."""
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import importlib.util
spec = importlib.util.spec_from_file_location(
    "eval_guerra", Path(__file__).resolve().parent / "eval_guerra.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

BASE = g.BASE


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def to97(proba, classes, n=97):
    m = np.full((proba.shape[0], n), 1.0 / n, np.float32)
    m[:, classes] = proba
    m /= m.sum(1, keepdims=True)
    return m.astype(np.float32)


def candidates():
    return {
        "logreg_C0.1": lambda: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
        "logreg_C1":   lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "logreg_C10":  lambda: LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced"),
        "rbf_C32_gscale": lambda: SVC(C=32, gamma="scale", class_weight="balanced",
                                      probability=True, cache_size=2000),
        "rbf_C64_gscale": lambda: SVC(C=64, gamma="scale", class_weight="balanced",
                                      probability=True, cache_size=2000),
        "et500": lambda: ExtraTreesClassifier(n_estimators=500, max_features=None,
                                              min_samples_leaf=1, class_weight="balanced",
                                              n_jobs=-1, random_state=42),
        "et500_mf0.5": lambda: ExtraTreesClassifier(n_estimators=500, max_features=0.5,
                                                    class_weight="balanced", n_jobs=-1,
                                                    random_state=42),
    }


def main():
    A0, S0, ytr_all, miss = g.load_descs()
    assert not miss, miss
    Av, Sv = g.clean(A0, S0, 7428)
    Xv_all = np.concatenate([Sv, Av], 1).astype(np.float32)
    Xfit, yfit = Xv_all[:7428], ytr_all[:7428]
    Xval, yval = Xv_all[7428:9244], ytr_all[7428:]

    print("== val candidate comparison ==", flush=True)
    results = {}
    best = None
    for name, ctor in candidates().items():
        m = ctor(); t = time.time(); m.fit(Xfit, yfit)
        pv = m.predict(Xval); a, b, f = met(yval, pv)
        results[name] = m
        print("  %-16s acc=%.4f bal=%.4f macF1=%.4f (%.1fmin)"
              % (name, a, b, f, (time.time() - t) / 60), flush=True)
        if best is None or (f, a) > best[0]:
            best = ((f, a), name)
    name = best[1]
    print("FROZEN:", name, flush=True)

    At, St = g.clean(A0, S0, 9244)
    Xall = np.concatenate([St, At], 1).astype(np.float32)
    m_full = dict(candidates())[name]().fit(Xall[:9244], ytr_all[:9244])
    m_val = dict(candidates())[name]().fit(Xfit, yfit)

    def soft(mdl, X):
        pr = mdl.predict_proba(X)
        return to97(pr, mdl.classes_)

    psv, pst = soft(m_val, Xval), soft(m_full, Xall[9244:])
    pv, pt = psv.argmax(1), pst.argmax(1)
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values
    print("VAL acc=%.4f bal=%.4f macF1=%.4f" % met(yval, pv), flush=True)
    print("TEST acc=%.4f bal=%.4f macF1=%.4f" % met(yte, pt), flush=True)
    np.savez_compressed(BASE / "results_final/c3b_arrays.npz",
                        val_soft=psv, val_pred=pv, val_y=yval,
                        test_soft=pst, test_pred=pt, test_y=yte)
    print("saved results_final/c3b_arrays.npz", flush=True)


if __name__ == "__main__":
    main()
