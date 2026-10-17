"""C1 / C1b / C1c / C1d 3DZD classifiers on both versions (paper §4.6).

Descriptors in experiments/results/desc/dzd_{train,test}.npz are 364-vectors
(363 invariants + positive-shell volume fraction f+).

  c1  : Euclidean 1-NN on the 363 invariants with the f+ volume-ratio gate
        V_q/V_g in [0.8,1.2]; gallery members outside the gate get infinite
        distance, replaced per query by that query's worst FINITE class
        distance. Fusion score: per-class nearest (gated) distance.
  c1b : inverse-distance k-NN with k chosen on validation (k in 1..25).
  c1c : per-block (121-coordinate) z-scoring fitted on the 7,428 train rows;
        equal block-weighted Euclidean, 1-NN, per-class min distance.
  c1d : StandardScaler (all 363) + RBF-SVM C=50, gamma=0.003; margins saved.

Outputs: experiments/results/<ver>/descriptors/dzd_*_{val,test}.npz and
dzd_c1_metrics.json.
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


def load_arrays(version):
    cfg = vp.version_cfg(version)
    tr = np.load(DESC / "dzd_train.npz", allow_pickle=False)
    te = np.load(DESC / "dzd_test.npz", allow_pickle=False)
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
    keep = [str(x)[:-4] if str(x).endswith(".vtk") else str(x)
            for x in pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"]
                                 )[cfg["test_id_col"]]]
    pos = {i: k for k, i in enumerate(te["ids"].astype(str))}
    Xte = te["X"][np.array([pos[i] for i in keep], np.int64)]
    yte = np.array([gmap[i] for i in keep], np.int64)
    return ((Xtr, ytr_), (Xva, yva, tr_ids[n:]),
            (Xte, yte, np.array(keep)))


def metrics_row(scores, y, nc):
    return {k: float(metrics_lib.evaluate(scores, y, nc)[k])
            for k in metrics_lib.METRIC_ORDER}


def gated_class_distance(Zq, fq, Zg, fg, yg, nc, lo=0.8, hi=1.2):
    """Per-class nearest distance with the volume-ratio gate.

    Zq (Q,363); Zg (G,363); fq/fg volume fractions; yg gallery labels.
    Infinite (gated-out) class entries are filled with the query's worst
    finite class distance.
    """
    # pairwise euclidean distances in small query blocks (memory bound)
    Dout = np.full((len(Zq), nc), np.inf, np.float32)
    for s in range(0, len(Zq), 64):
        d = np.linalg.norm(Zq[s:s + 64, None, :] - Zg[None, :, :], axis=2)
        ratio = fq[s:s + 64, None] / np.maximum(fg[None, :], 1e-9)
        ok = (ratio >= lo) & (ratio <= hi)
        d = np.where(ok, d, np.inf)
        for c in range(nc):
            mm = yg == c
            if mm.any():
                Dout[s:s + 64, c] = np.min(d[:, mm], axis=1)
    finite = np.isfinite(Dout)
    worst = np.where(finite, Dout, -np.inf).max(1, keepdims=True)
    Dout = np.where(finite, Dout, worst)
    return Dout  # negative distance used as score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    nc = vp.version_cfg(args.version)["num_classes"]
    outdir = vp.RESULTS_DIR / args.version / "descriptors"
    outdir.mkdir(parents=True, exist_ok=True)

    (Xtr, ytr), (Xva, yva, idva), (Xte, yte, idte) = load_arrays(args.version)
    Ztr, Zva, Zte = Xtr[:, :363], Xva[:, :363], Xte[:, :363]
    ftr, fva, fte = Xtr[:, 363], Xva[:, 363], Xte[:, 363]
    print("shapes", Ztr.shape, Zva.shape, Zte.shape, "classes", nc, flush=True)
    summary = {}

    def dump(tag, split, ids, scores, labels):
        np.savez_compressed(outdir / ("dzd_%s_%s.npz" % (tag, split)),
                            ids=ids.astype(str),
                            scores=scores.astype(np.float32), labels=labels)

    # ---- c1: gated 1-NN ----------------------------------------------------
    dva = gated_class_distance(Zva, fva, Ztr, ftr, ytr, nc)
    dte = gated_class_distance(Zte, fte, Ztr, ftr, ytr, nc)
    summary["c1_val"] = metrics_row(-dva, yva, nc)
    summary["c1_test"] = metrics_row(-dte, yte, nc)
    dump("c1", "val", idva, -dva, yva)
    dump("c1", "test", idte, -dte, yte)

    # ---- c1b: inverse-distance kNN, k tuned on validation -----------------
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=25).fit(Ztr)
    d25v, i25v = knn.kneighbors(Zva)
    d25t, i25t = knn.kneighbors(Zte)

    def invknn_scores(d, idx, k):
        w = 1.0 / np.maximum(d[:, :k], 1e-6)
        lab = ytr[idx[:, :k]]
        S = np.zeros((len(d), nc), np.float32)
        for j in range(k):
            S[np.arange(len(d)), lab[:, j]] += w[:, j]
        return S

    best_k, best_f = 1, -1
    for k in range(1, 26):
        f1 = metrics_row(invknn_scores(d25v, i25v, k), yva, nc)["f1_macro"]
        if f1 > best_f:
            best_f, best_k = f1, k
    sva = invknn_scores(d25v, i25v, best_k)
    ste = invknn_scores(d25t, i25t, best_k)
    summary["c1b_val"] = metrics_row(sva, yva, nc)
    summary["c1b_test"] = metrics_row(ste, yte, nc)
    summary["c1b_k"] = best_k
    dump("c1b", "val", idva, sva, yva)
    dump("c1b", "test", idte, ste, yte)

    # ---- c1c: block-zscored equal-block weighted 1-NN ----------------------
    def block_z(Z):
        out = np.empty_like(Z)
        for b in range(3):
            z = Z[:, b * 121:(b + 1) * 121]
            mu, sd = mus[b], sds[b]
            out[:, b * 121:(b + 1) * 121] = (z - mu) / sd
        return out / np.sqrt(121.0)   # equalise block contribution

    mus, sds = [], []
    for b in range(3):
        z = Ztr[:, b * 121:(b + 1) * 121]
        mu, sd = z.mean(0), z.std(0)
        mus.append(mu); sds.append(np.maximum(sd, 1e-8))
    Btr, Bva, Bte = block_z(Ztr), block_z(Zva), block_z(Zte)
    dva = gated_class_distance(Bva, fva, Btr, ftr, ytr, nc)
    dte = gated_class_distance(Bte, fte, Btr, ftr, ytr, nc)
    summary["c1c_val"] = metrics_row(-dva, yva, nc)
    summary["c1c_test"] = metrics_row(-dte, yte, nc)
    dump("c1c", "val", idva, -dva, yva)
    dump("c1c", "test", idte, -dte, yte)

    # ---- c1d: StandardScaler + RBF-SVM C=50 gamma=0.003 --------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(Ztr)
    clf = SVC(C=50.0, kernel="rbf", gamma=0.003,
              decision_function_shape="ovr", cache_size=2000)
    clf.fit(sc.transform(Ztr), ytr)
    sva = clf.decision_function(sc.transform(Zva))
    ste = clf.decision_function(sc.transform(Zte))
    summary["c1d_val"] = metrics_row(sva, yva, nc)
    summary["c1d_test"] = metrics_row(ste, yte, nc)
    dump("c1d", "val", idva, sva, yva)
    dump("c1d", "test", idte, ste, yte)

    with open(outdir / "dzd_c1_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    for tag in ("c1", "c1b", "c1c", "c1d"):
        for sp in ("val", "test"):
            m = summary["%s_%s" % (tag, sp)]
            print("%-4s %-4s acc=%.4f f1m=%.4f Pm=%.4f Rm=%.4f f1w=%.4f"
                  % (tag, sp, m["acc"], m["f1_macro"], m["prec_macro"],
                     m["rec_macro"], m["f1_weighted"]), flush=True)
    print("c1b tuned k =", best_k, flush=True)


if __name__ == "__main__":
    main()
