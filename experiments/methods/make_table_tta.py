"""Render the Sec. 5.6 rotational-TTA table from tta25 caches.

Reads experiments/results/<ver>/deep/<run>__<ckpt>__tta25_{val,test}.npz
(scores (25,N,C) raw logits; view 0 = identity) and reports identity-only
vs TTA25 mean-logit (same aggregation as tta_eval.py) on both splits.

Outputs: experiments/results/<ver>/tables/table_tta.{csv,md}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse

import numpy as np
import pandas as pd

from experiments.common import paths as vp
from experiments.common.metrics_lib import evaluate

METKEYS = ["acc", "bal_acc", "f1_macro", "prec_macro", "rec_macro",
           "f1_weighted"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    ap.add_argument("--run", default="v3_noSynth_s2")
    ap.add_argument("--ckpt", default="best_model")
    args = ap.parse_args()

    nc = vp.version_cfg(args.version)["num_classes"]
    tag = "%s__%s__tta25" % (args.run, args.ckpt)
    rows = []
    for split in ("val", "test"):
        p = vp.RESULTS_DIR / args.version / "deep" / (
            "%s_%s.npz" % (tag, split))
        d = np.load(p, allow_pickle=True)
        S = d["scores"].astype(np.float64)  # (25,N,C)
        y = d["labels"]
        for method, L in (("identity", S[0]), ("tta25_meanlogit", S.mean(0))):
            m = evaluate(L, y, nc)
            rows.append({"split": split, "method": method,
                         "cache": p.name,
                         **{k: float(m[k]) for k in METKEYS}})

    df = pd.DataFrame(rows)
    outdir = vp.RESULTS_DIR / args.version / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "table_tta.csv", index=False)

    L = ["| Split | Method | Acc | Bal | MacF1 | PrecM | RecM | F1w |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            r["split"], r["method"], r["acc"], r["bal_acc"], r["f1_macro"],
            r["prec_macro"], r["rec_macro"], r["f1_weighted"]))
    L.append("")
    L.append("cache: %s_{val,test}.npz (25 views = identity + 24 cube "
             "rotations, mean-logit, FPS pass 0 fixed)" % tag)
    with open(outdir / "table_tta.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
