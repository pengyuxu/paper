"""Render Supplementary Table S7 (training ablation, dual-version) per Task 14 / TR-14.1.

Upper block "standalone": each run's deep cache MC-FPS K=25 aggregated in
probability space (stats_lib.mc_aggregate) and evaluated in isolation.
Lower block "frozen_fusion": the same cache substituted into the frozen
Sec-4.11 fusion (members deep + c1 + c2, weights 2/2/2,
tau = (1.0, 0.5[dzd], 1.0[fpfh])).

Caches cited per row; ablation runs use best_model K25 dumps (no swa dump
was produced for them -- disclosed in the md footer).

Outputs: experiments/results/<ver>/tables/table9_ablation.{csv,md}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse

import numpy as np
import pandas as pd

from experiments.common import paths as vp
from experiments.common.metrics_lib import evaluate
from experiments.methods.fusion_panel import (align, load_deep, load_member,
                                              logsoftmax_np)

RUNS = ["v3_noSynth_s2", "abl_focalg2_s2", "abl_noCutMix_s2",
        "abl_cbbeta_s2"]
METKEYS = ["acc", "bal_acc", "f1_macro", "prec_macro", "rec_macro",
           "f1_weighted"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    ap.add_argument("--ckpt", default="best_model")
    args = ap.parse_args()

    nc = vp.version_cfg(args.version)["num_classes"]
    deepdir = vp.RESULTS_DIR / args.version / "deep"
    rows = []

    c1 = {sp: load_member(args.version, "dzd_c1", sp) for sp in ("val", "test")}
    c2 = {sp: load_member(args.version, "fpfh_c2", sp)
          for sp in ("val", "test")}

    for run in RUNS:
        stem = "%s__%s__seeded42_K25" % (run, args.ckpt)
        tp = deepdir / ("%s_test.npz" % stem)
        if not tp.exists():
            rows.append({"block": "standalone", "run": run, "cache": stem,
                         "note": "N/A: cache missing"})
            continue
        deep = {"test": load_deep(str(tp)),
                "val": load_deep(str(tp).replace("_test.npz", "_val.npz"))}

        # upper block: standalone MC25
        for split in ("val", "test"):
            _ids, S, y = deep[split]
            m = evaluate(S, y, nc)
            rows.append({"block": "standalone", "run": run, "split": split,
                         "cache": stem, **{k: float(m[k]) for k in METKEYS}})

        # lower block: frozen fusion deep+c1+c2, w=(2,2,2), tau=(1,.5,1)
        recs = {"deep": deep, "c1": c1, "c2": c2}
        keys, taus, w = ["deep", "c1", "c2"], [1.0, 0.5, 1.0], [2.0] * 3
        for split in ("val", "test"):
            _ids, Ss, y = align(recs, keys, split)
            L = sum(w[m] * logsoftmax_np(Ss[m], taus[m]) for m in range(3))
            m = evaluate(L, y, nc)
            rows.append({"block": "frozen_fusion", "run": run, "split": split,
                         "cache": stem + " + dzd_c1 + fpfh_c2",
                         **{k: float(m[k]) for k in METKEYS}})

    df = pd.DataFrame(rows)
    outdir = vp.RESULTS_DIR / args.version / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "table9_ablation.csv", index=False)

    L = ["| Block | Run | Split | Acc | Bal | MacF1 | PrecM | RecM | F1w |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "note" in r:
            L.append("| %s | %s | - | %s |" % (r["block"], r["run"],
                                                r["note"]))
            continue
        L.append("| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | "
                 "%.4f |" % (r["block"], r["run"], r["split"], r["acc"],
                             r["bal_acc"], r["f1_macro"], r["prec_macro"],
                             r["rec_macro"], r["f1_weighted"]))
    L.append("")
    L.append("caches: deep/<run>__%s__seeded42_K25_{val,test}.npz (MC-FPS "
             "K=25, probability-space aggregation); descriptors "
             "dzd_c1/fpfh_c2; frozen fusion w=(2,2,2) tau=(1.0,0.5,1.0). "
             "NOTE: ablation runs use best_model checkpoints (no swa K25 "
             "dump produced)." % args.ckpt)
    with open(outdir / "table9_ablation.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
