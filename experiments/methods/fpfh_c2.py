"""C2 / C2b / C2d FPFH classifiers on both dataset versions (paper §4.6).

Descriptors (experiments/results/desc/fpfh_{train,test}.npz) are version
independent; the version only fixes train/val label rows and the test id/gt
pairing.

  c2  : L2-normalised 612-d -> RBF-SVM (C=64, gamma=12); decision margins
        saved for the late-fusion panel.
  c2b : ExtraTrees (500 trees, max_features=None, class_weight balanced).
  c2d : cosine 1-NN on the L2-normalised descriptor.

Outputs per version: experiments/results/<ver>/descriptors/
  fpfh_c2_metrics.json, fpfh_c2_<split>.npz (ids, scores, labels) for val/test,
  and matching c2b/c2d npz (scores: margins / proba / -cosine distance).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.common import paths as vp
from experiments.common import metrics_lib

DESC = vp.EXP_DIR / "results" / "desc"


def l2norm(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def load_arrays(version):
    cfg = vp.version_cfg(version)
    tr = np.load(DESC / "fpfh_train.npz", allow_pickle=False)
    te = np.load(DESC / "fpfh_test.npz", allow_pickle=False)
    tdf = pd.read_csv(vp.csv_dir(version) / "train_set_2.csv")
    id2lab = dict(zip(tdf.protein_id.astype(str), tdf.class_id.astype(int)))
    tr_ids = tr["ids"].astype(str)
    ytr = np.array([id2lab[i] for i in tr_ids], np.int64)
    n = vp.TRAIN_ROWS
    Xtr, Xva, ytr_, yva = tr["X"][:n], tr["X"][n:], ytr[:n], ytr[n:]

    gtdf = pd.read_csv(vp.csv_dir(version) / cfg["gt_file"])
    g = gtdf.set_index(cfg["gt_anon_col"])[cfg["gt_label_col"]].to_dict()
    gmap = {str(k)[:-4] if str(k).endswith(".vtk") else str(k): int(v)
            for k, v in g.items()}
    test_df = pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"])
    keep = [str(x)[:-4] if str(x).endswith(".vtk") else str(x)
            for x in test_df[cfg["test_id_col"]]]
    pos = {i: k for k, i in enumerate(te["ids"].astype(str))}
    idx = np.array([pos[i] for i in keep], np.int64)
    Xte = te["X"][idx]
    yte = np.array([gmap[i] for i in keep], np.int64)
    ids_te = np.array(keep)
    ids_va = tr_ids[n:]
    return ((Xtr, ytr_), (Xva, yva, ids_va), (Xte, yte, ids_te))


def metrics_row(scores, y, nc):
    return {k: float(metrics_lib.evaluate(scores, y, nc)[k])
            for k in metrics_lib.METRIC_ORDER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    nc = vp.version_cfg(args.version)["num_classes"]
    outdir = vp.RESULTS_DIR / args.version / "descriptors"
    outdir.mkdir(parents=True, exist_ok=True)

    (Xtr, ytr), (Xva, yva, idva), (Xte, yte, idte) = load_arrays(args.version)
    print("shapes", Xtr.shape, Xva.shape, Xte.shape, "classes", nc, flush=True)

    Ztr, Zva, Zte = l2norm(Xtr), l2norm(Xva), l2norm(Xte)
    summary = {}

    def dump(tag, split, ids, scores, labels):
        np.savez_compressed(outdir / ("fpfh_%s_%s.npz" % (tag, split)),
                            ids=ids.astype(str),
                            scores=scores.astype(np.float32), labels=labels)

    # ---- c2: RBF-SVM C=64 gamma=12 on L2-normalised descriptors -----------
    from sklearn.svm import SVC
    clf = SVC(C=64.0, kernel="rbf", gamma=12.0,
              decision_function_shape="ovr", cache_size=2000)
    clf.fit(Ztr, ytr)
    sva, ste = clf.decision_function(Zva), clf.decision_function(Zte)
    summary["c2_val"] = metrics_row(sva, yva, nc)
    summary["c2_test"] = metrics_row(ste, yte, nc)
    dump("c2", "val", idva, sva, yva)
    dump("c2", "test", idte, ste, yte)

    # ---- c2b: ExtraTrees ---------------------------------------------------
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=500, max_features=None,
                              class_weight="balanced", n_jobs=-1,
                              random_state=17)
    et.fit(Xtr, ytr)
    pva, pte = et.predict_proba(Xva), et.predict_proba(Xte)
    summary["c2b_val"] = metrics_row(np.log(pva + 1e-12), yva, nc)
    summary["c2b_test"] = metrics_row(np.log(pte + 1e-12), yte, nc)
    dump("c2b", "val", idva, pva, yva)
    dump("c2b", "test", idte, pte, yte)

    # ---- c2d: cosine 1-NN --------------------------------------------------
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(Ztr)
    dva = knn.kneighbors(Zva, return_distance=True)[0].ravel()
    dte = knn.kneighbors(Zte, return_distance=True)[0].ravel()
    pva = ytr[knn.kneighbors(Zva, return_distance=False).ravel()]
    pte = ytr[knn.kneighbors(Zte, return_distance=False).ravel()]
    onehot = np.full((len(pva), nc), -1.0, np.float32)
    onehot[np.arange(len(pva)), pva] = 1.0 - dva
    onehote = np.full((len(pte), nc), -1.0, np.float32)
    onehote[np.arange(len(pte)), pte] = 1.0 - dte
    summary["c2d_val"] = metrics_row(onehot, yva, nc)
    summary["c2d_test"] = metrics_row(onehote, yte, nc)
    dump("c2d", "val", idva, onehot, yva)
    dump("c2d", "test", idte, onehote, yte)

    with open(outdir / "fpfh_c2_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    for tag in ("c2", "c2b", "c2d"):
        for sp in ("val", "test"):
            m = summary["%s_%s" % (tag, sp)]
            print("%-4s %-4s acc=%.4f f1m=%.4f Pm=%.4f Rm=%.4f f1w=%.4f"
                  % (tag, sp, m["acc"], m["f1_macro"], m["prec_macro"],
                     m["rec_macro"], m["f1_weighted"]), flush=True)


if __name__ == "__main__":
    main()
