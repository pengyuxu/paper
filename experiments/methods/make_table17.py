"""Render Task 17 supplemental analyses (paper Sec. 5.13/5.14, dual version).

Outputs per version under experiments/results/<ver>/tables/:
  table17_groupsize.csv/.md - test accuracy by training-set-size quartile
      groups (rare <=10, small 11-50, medium 51-200, large >200; sizes counted
      over ALL rows of train_set_2.csv, i.e. train+val - this reproduces the
      paper's 25/23/42/7 class counts) for three conditions:
      v3 single FPS draw (first42_K1), v3 MC-FPS K=25, Final frozen panel
      (v5-swa K25 + dzd_c1 + fpfh_c2, w=(2,2,2), tau=(1,.5,1)).
  perclass_final.csv       - per-class precision/recall/f1/support (Final)
  confusion_final.csv      - full C x C test confusion matrix (Final)
  confusion_final_pairs.md - weakest classes + top directed confusion pairs

All numbers recomputed from npz caches; nothing copied from logs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse

import numpy as np
import pandas as pd

from experiments.common import paths as vp
from experiments.common.metrics_lib import evaluate, per_class_table
from experiments.methods.fusion_panel import (align, load_deep, load_member,
                                              logsoftmax_np)

GROUPS = [("rare", 0, 10), ("small", 11, 50), ("medium", 51, 200),
          ("large", 201, 10 ** 9)]
CONDITIONS = ["v3_single", "v3_K25", "final"]


def group_of(size):
    for gi, (_n, lo, hi) in enumerate(GROUPS):
        if lo <= size <= hi:
            return gi
    raise ValueError(size)


def group_bal_acc(pred, y, sizes, gi):
    """Mean per-class recall over present test classes inside the group."""
    nc = len(sizes)
    recs = []
    for c in range(nc):
        if group_of(int(sizes[c])) != gi:
            continue
        m = y == c
        if m.sum() == 0:
            continue
        recs.append(float((pred[m] == c).sum()) / float(m.sum()))
    return float(np.mean(recs)) if recs else float("nan"), len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    ver = args.version
    nc = vp.version_cfg(ver)["num_classes"]
    deepdir = vp.RESULTS_DIR / ver / "deep"
    tdir = vp.RESULTS_DIR / ver / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # class sizes over the full train_set_2.csv (train+val), paper definition
    tcsv = pd.read_csv(vp.csv_dir(ver) / "train_set_2.csv")
    sizes = np.bincount(tcsv["class_id"].values, minlength=nc)

    # three test conditions
    _i, L_single, y = load_deep(str(
        deepdir / "v3_noSynth_s2__best_model__first42_K1_test.npz"))
    _i, L_k25, y = load_deep(str(
        deepdir / "v3_noSynth_s2__best_model__seeded42_K25_test.npz"))
    recs = {"v5": {"test": load_deep(str(
                deepdir / "v5_s2__swa_model__seeded42_K25_test.npz"))},
            "c1": {"test": load_member(ver, "dzd_c1", "test")},
            "c2": {"test": load_member(ver, "fpfh_c2", "test")}}
    _i, Ss, y = align(recs, ["v5", "c1", "c2"], "test")
    L_final = sum(2.0 * logsoftmax_np(Ss[m], t) for m, t in
                  enumerate([1.0, 0.5, 1.0]))
    conds = {"v3_single": L_single, "v3_K25": L_k25, "final": L_final}

    # sanity: overall metrics must agree with existing artifacts
    ov = {k: evaluate(L, y, nc)["acc"] for k, L in conds.items()}
    print("overall acc: " + ", ".join(
        "%s=%.4f" % (k, v) for k, v in ov.items()), flush=True)

    gmap = np.array([group_of(int(sizes[c])) for c in y])

    # ---- table17 groupsize ------------------------------------------------
    rows = []
    for gi, (gname, lo, hi) in enumerate(GROUPS):
        m = gmap == gi
        cls_in = [c for c in range(nc)
                  if lo <= int(sizes[c]) <= hi]
        for cname, L in conds.items():
            pred = L.argmax(1)
            bal, n_present = group_bal_acc(pred, y, sizes, gi)
            rows.append({"group": gname, "condition": cname,
                         "n_classes": len(cls_in),
                         "n_test_classes_present": n_present,
                         "n_test": int(m.sum()),
                         "acc": float((pred[m] == y[m]).mean()) if m.sum()
                         else float("nan"),
                         "bal_acc": bal})
    gdf = pd.DataFrame(rows)
    gdf.to_csv(tdir / "table17_groupsize.csv", index=False)

    L = ["| Group (train-set size) | Condition | #classes | #test obj | "
         "Acc | Group BalAcc |",
         "|---|---|---|---|---|---|"]
    for (gname, _, _), gi in [(z, i) for i, z in enumerate(GROUPS)]:
        sub = gdf[gdf.group == gname]
        for _, r in sub.iterrows():
            L.append("| %s | %s | %d | %d | %.4f | %.4f |" % (
                gname, r.condition, r.n_classes, r.n_test, r.acc,
                r.bal_acc))
    L.append("")
    L.append("groups by train_set_2.csv class sizes (all %d rows): "
             "rare<=10, small 11-50, medium 51-200, large >200. "
             "Conditions: v3 first42_K1 single draw, v3 seeded42 K=25 "
             "(probability-space aggregate), Final frozen panel "
             "v5_s2_swa K25 + dzd_c1 + fpfh_c2 w=(2,2,2) tau=(1,.5,1). "
             "Group BalAcc = mean per-class recall over test-present classes "
             "in the group." % len(tcsv))
    (tdir / "table17_groupsize.md").write_text("\n".join(L) + "\n")

    # ---- per-class table + confusion (Final panel) ------------------------
    pc = per_class_table(L_final, y, nc)
    pc.insert(1, "train_size", sizes[pc.class_id.values])
    pc.insert(2, "group", [GROUPS[group_of(int(s))][0]
                           for s in pc["train_size"]])
    pc = pc.sort_values("recall")
    pc.to_csv(tdir / "perclass_final.csv", index=False)

    pred = L_final.argmax(1)
    cm = np.zeros((nc, nc), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    pd.DataFrame(cm).to_csv(tdir / "confusion_final.csv", index=False)

    off = [(cm[a, b], a, b) for a in range(nc) for b in range(nc)
           if a != b and cm[a, b] > 0]
    off.sort(reverse=True)
    weak = pc[pc.n > 0].head(10)
    P = ["# Final-panel per-class / confusion analysis (test split)", "",
         "## Weakest classes by recall",
         "| class | group | n | recall | precision | f1 |",
         "|---|---|---|---|---|---|"]
    for _, r in weak.iterrows():
        P.append("| %d | %s | %d | %.3f | %.3f | %.3f |" % (
            r.class_id, r.group, r.n, r.recall, r.precision, r.f1))
    P += ["", "## Top directed confusion pairs (true -> pred : count)",
          "| true -> pred | count | reverse count |",
          "|---|---|---|"]
    for cnt, a, b in off[:15]:
        P.append("| %d -> %d | %d | %d |" % (a, b, cnt, int(cm[b, a])))
    P.append("")
    P.append("sources: perclass_final.csv, confusion_final.csv (%dx%d, "
             "rows=true cols=pred); Final panel predictions from "
             "v5_s2__swa_model__seeded42_K25 + dzd_c1 + fpfh_c2, "
             "w=(2,2,2) tau=(1,.5,1)." % (nc, nc))
    (tdir / "confusion_final_pairs.md").write_text("\n".join(P) + "\n")

    print(gdf.pivot_table(index="group", columns="condition",
                          values="acc").loc[[g for g, _, _ in GROUPS]]
          .to_string())
    print("saved table17_groupsize, perclass_final, confusion_final to", tdir)


if __name__ == "__main__":
    main()
