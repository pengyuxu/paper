"""Re-rate SHREC2025 historical track submissions on the corrected dataset.

Historical submissions live in results/<author>/submission_*.csv with
columns protein_id,class_id on the PAPER track (97 classes, 2311 test items,
protein_id = anonymised id matching dataset_paper_version/test_set.csv with
'.vtk' stripped).

The 2025-10-28 correction (paths.VERSIONS['updated']):
  * test proteins 840/1277/1311 removed -> 2308 items;
  * old class 16 split into 16/97/98; all other class ids unchanged
    (verified by joining the two train_set_2.csv files: the only changed
    (paper,updated) label pairs are (16,97) and (16,98)).

Historical predictors never output 97/98, so their predictions can be scored
against the updated ground truth directly: an old-16 prediction is correct
only when the corrected label remains 16, and necessarily wrong for items
relabelled 97/98. No prediction remapping is applied (and none is possible

from a 97-class submission).

Outputs:
  experiments/results/updated/history/history_rerating.csv  (one row/method)
  experiments/results/paper/history/history_papergrid.csv   (97-grid reference)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import paths as vp
from experiments.common.metrics_lib import evaluate

SUBMISSIONS = {
    "tatsuma": "results/tatsuma/submission_tatsuma.csv",
    "barisin_geom": "results/barisin_geom/submission_barisin_geom.csv",
    "barisin_geom_prop": "results/barisin_geom_prop/submission_barisin_geom_prop.csv",
    "guerra_v1": "results/guerra_v1/submission_guerra_v1.csv",
    "guerra_v2": "results/guerra_v2/submission_guerra_v2.csv",
    "guerra_v3": "results/guerra_v3/submission_guerra_v3.csv",
    "guerra_v4": "results/guerra_v4/submission_guerra_v4.csv",
    "he_v1": "results/he_v1/submission_he_v1.csv",
    "he_v2": "results/he_v2/submission_he_v2.csv",
    "kagaya_3D-Surfer": "results/kagaya_3D-Surfer/submission_kagaya_3D-Surfer.csv",
    "li_RISurProNet": "results/li_RISurProNet/submission_li_RISurProNet.csv",
    "peng": "results/peng/submission_peng.csv",
    "tehrani_v1": "results/tehrani_v1/submission_tehrani_v1.csv",
    "tehrani_v2": "results/tehrani_v2/submission_tehrani_v2.csv",
    "yang_ProtoCluster": "results/yang_ProtoCluster/submission_yang_ProtoCluster.csv",
}

METRICS = ["acc", "bal_acc", "prec_macro", "rec_macro", "f1_macro",
           "prec_weighted", "rec_weighted", "f1_weighted"]


def load_gt(version):
    cfg = vp.version_cfg(version)
    ids = pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"])[
        cfg["test_id_col"]].astype(str)
    ids = ids.str.replace(r"\.vtk$", "", regex=True).values
    gt = pd.read_csv(vp.csv_dir(version) / cfg["gt_file"])
    g = gt.set_index(cfg["gt_anon_col"])[cfg["gt_label_col"]].to_dict()
    g = {str(k).replace(".vtk", ""): int(v) for k, v in g.items()}
    return ids, np.array([g[i] for i in ids], np.int64)


def score(csv_path, ids, y, nc):
    df = pd.read_csv(csv_path)
    pred = dict(zip(df["protein_id"].astype(str), df["class_id"].astype(int)))
    p = np.array([pred[i] for i in ids], np.int64)
    onehot = np.zeros((len(p), nc), np.float32)
    onehot[np.arange(len(p)), p] = 1.0
    m = evaluate(onehot, y, nc)
    return p, {k: float(m[k]) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["updated", "paper"])
    args = ap.parse_args()

    summary = {}
    for ver in args.versions:
        nc = vp.version_cfg(ver)["num_classes"]
        ids, y = load_gt(ver)
        rows = []
        detail = {}
        for name, rel in SUBMISSIONS.items():
            p = vp.REPO_ROOT / rel
            if not p.exists():
                continue
            pred, m = score(p, ids, y, nc)
            m["method"] = name
            rows.append(m)
            detail[name] = {"pred": pred.tolist(), "metrics": m}
        df = pd.DataFrame(rows)[["method"] + METRICS].sort_values(
            "f1_macro", ascending=False)
        outdir = vp.RESULTS_DIR / ver / "history"
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / ("history_%sgrid.csv" % ver), index=False)
        summary[ver] = rows
        print("== %s (%d classes, n=%d)" % (ver, nc, len(ids)))
        print(df.to_string(index=False, float_format=lambda x: "%.4f" % x))
        with open(outdir / ("history_%s_predictions.json" % ver), "w") as f:
            json.dump({"ids": ids.tolist(), "labels": y.tolist(),
                       "methods": detail}, f)
    print("HISTORY_RERATING_DONE")


if __name__ == "__main__":
    main()
