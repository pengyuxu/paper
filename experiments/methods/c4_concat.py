"""C4 concatenated descriptor classifier (paper §4.6).

[3DZD(363) | FPFH(612)] = 975 coordinates, standardised per block using
statistics of the 7,428 training rows, RBF-SVM frozen at C=16, gamma=0.003.
No Guerra block.  Decision margins are cached for the fusion panel.

Outputs: experiments/results/<ver>/descriptors/c4_{val,test}.npz + json.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.common import paths as vp
from experiments.common import metrics_lib
from experiments.methods.dzd_c1 import load_arrays as load_dzd
from experiments.methods.fpfh_c2 import load_arrays as load_fpfh

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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

    (Dtr, ytr), (Dva, yva, idva), (Dte, yte, idte) = load_dzd(args.version)
    (Ftr, _), (Fva, _, _), (Fte, _, _) = load_fpfh(args.version)
    Ztr = np.concatenate([Dtr[:, :363], Ftr], 1)
    Zva = np.concatenate([Dva[:, :363], Fva], 1)
    Zte = np.concatenate([Dte[:, :363], Fte], 1)
    print("c4 shapes", Ztr.shape, Zva.shape, Zte.shape, nc, flush=True)

    Ztr_s = np.empty_like(Ztr)
    Zva_s = np.empty_like(Zva)
    Zte_s = np.empty_like(Zte)
    for b, e in ((0, 363), (363, 975)):
        sc = StandardScaler().fit(Ztr[:, b:e])
        Ztr_s[:, b:e] = sc.transform(Ztr[:, b:e])
        Zva_s[:, b:e] = sc.transform(Zva[:, b:e])
        Zte_s[:, b:e] = sc.transform(Zte[:, b:e])

    clf = SVC(C=16.0, kernel="rbf", gamma=0.003,
              decision_function_shape="ovr", cache_size=3000)
    clf.fit(Ztr_s, ytr)
    sva, ste = clf.decision_function(Zva_s), clf.decision_function(Zte_s)
    summary = {"c4_val": metrics_row(sva, yva, nc),
               "c4_test": metrics_row(ste, yte, nc)}
    np.savez_compressed(outdir / "c4_val.npz", ids=idva.astype(str),
                        scores=sva.astype(np.float32), labels=yva)
    np.savez_compressed(outdir / "c4_test.npz", ids=idte.astype(str),
                        scores=ste.astype(np.float32), labels=yte)
    with open(outdir / "c4_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    for sp in ("val", "test"):
        m = summary["c4_%s" % sp]
        print("c4 %-4s acc=%.4f f1m=%.4f Pm=%.4f Rm=%.4f f1w=%.4f"
              % (sp, m["acc"], m["f1_macro"], m["prec_macro"],
                 m["rec_macro"], m["f1_weighted"]), flush=True)


if __name__ == "__main__":
    main()
