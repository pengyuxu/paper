"""Archive and score the Barisin updated-profile reruns (Task 16).

Reads the two inference caches produced by barisin_run.py --mode infer:
  work/updated/log/classification_shrec2025/barisin_run{1,2}/preds_test.npz
(ids/probs/labels, updated test split n=2308, 99-class grid), re-evaluates
them through metrics_lib so the numbers are one-command reproducible, copies
the caches into results/updated/barisin/ and writes
results/updated/barisin/barisin_updated_metrics.json (consumed by
make_table10.py).

Run1 = riconv2_cls_v2_largest (6ch geometry);
Run2 = riconv2_cls_v2_features2_largest (8ch geometry + potentials).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import json
import shutil

import numpy as np

from experiments.common import paths as vp
from experiments.common.metrics_lib import METRIC_ORDER, evaluate

RUNS = (1, 2)


def main():
    nc = vp.version_cfg("updated")["num_classes"]
    outdir = vp.RESULTS_DIR / "updated" / "barisin"
    outdir.mkdir(parents=True, exist_ok=True)
    doc = {"version": "updated", "split": "test", "num_classes": nc}

    for run in RUNS:
        exp = vp.log_dir("updated") / ("barisin_run%d" % run)
        src = exp / "preds_test.npz"
        d = np.load(src, allow_pickle=True)
        probs, y = d["probs"].astype(np.float64), d["labels"]
        assert len(y) == 2308, "expected updated test n=2308, got %d" % len(y)
        m = evaluate(probs, y, nc)

        # best val accuracy / epoch span from the training metrics csv
        vm = np.genfromtxt(exp / "logs" / "val_metrics.csv", delimiter=",",
                           names=True)
        best_val = float(np.nanmax(vm["val_acc"]))
        n_ep = int(vm.shape[0])

        dst = outdir / ("barisin_run%d_preds_test.npz" % run)
        shutil.copy2(src, dst)
        doc["run%d" % run] = {
            "model": ("riconv2_cls_v2_largest" if run == 1
                      else "riconv2_cls_v2_features2_largest"),
            "n": int(len(y)), "epochs_trained": n_ep,
            "best_val_acc": best_val,
            "cache": dst.name,
            **{k: float(m[k]) for k in METRIC_ORDER}}
        print("run%d n=%d ep=%d best_val=%.4f test acc=%.4f bal=%.4f "
              "f1m=%.4f f1w=%.4f" % (
                  run, len(y), n_ep, best_val, m["acc"], m["bal_acc"],
                  m["f1_macro"], m["f1_weighted"]))

    with open(outdir / "barisin_updated_metrics.json", "w") as f:
        json.dump(doc, f, indent=2)
    print("saved", outdir / "barisin_updated_metrics.json")


if __name__ == "__main__":
    main()
