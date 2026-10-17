"""Generate dual-version Table 6 (classical layer-C members) from cached
metrics JSONs: dzd_c1_metrics.json, fpfh_c2_metrics.json,
guerra_{trainfit,fullrefit}_metrics.json, c4_metrics.json.

Outputs: experiments/results/<version>/tables/table6_classical.{csv,md}
Every number traces to a member cache + JSON listed in the provenance column.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse
import json

import pandas as pd

from experiments.common import paths as vp

# (row key, metrics json, description, metric prefix inside json)
ORDER = [
    ("c1", "dzd_c1_metrics.json", "363 1-NN+gate", "c1"),
    ("c1b", "dzd_c1_metrics.json", "363 inv-dist kNN", "c1b"),
    ("c1c", "dzd_c1_metrics.json", "363 block-z", "c1c"),
    ("c1d", "dzd_c1_metrics.json", "363 RBF-SVM", "c1d"),
    ("c2", "fpfh_c2_metrics.json", "612 RBF-SVM", "c2"),
    ("c2b", "fpfh_c2_metrics.json", "612 ExtraTrees", "c2b"),
    ("c2d", "fpfh_c2_metrics.json", "612 cosine kNN", "c2d"),
    ("c3", "guerra_trainfit_metrics.json", "171 SimpleNN 8-view (trainfit)",
     "c3"),
    ("c3_fullrefit", "guerra_fullrefit_metrics.json",
     "171 SimpleNN 8-view (9244 refit, submission)", "c3"),
    ("c3b_fullrefit", "guerra_fullrefit_metrics.json",
     "171 RBF-SVM (9244 refit, submission)", "c3b"),
    ("c3b", "guerra_trainfit_metrics.json", "171 RBF-SVM", "c3b"),
    ("c4", "c4_metrics.json", "975 RBF-SVM", "c4"),
]
COLS = [("acc", "Acc"), ("bal_acc", "Bal"), ("f1_macro", "Mac")]

# paper Table 6 reference (revised manuscript)
PAPER_REF = {
    "c1": (.9025, .8519, .8226, .9104, .8921, .8873),
    "c1b": (.9020, .8581, .8478, .9056, .8636, .8639),
    "c1c": (.9058, .8859, .8694, .9138, .8995, .8806),
    "c1d": (.9174, .8752, .8593, .9216, .8953, .8983),
    "c2": (.9036, .8559, .8446, .9061, .8600, .8423),
    "c2b": (.8992, .7978, .8095, .9022, .8018, .8086),
    "c2d": (.8756, .8039, .7945, .8927, .8387, .8334),
    "c3": (.8144, .6075, .5924, .8311, .7077, .7122),
    "c3b": (.8128, .5872, .5880, .8208, .6486, .6663),
    "c4": (.9229, .8448, .8467, .9324, .8633, .8822),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    ddir = vp.RESULTS_DIR / args.version / "descriptors"
    outdir = vp.RESULTS_DIR / args.version / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    rows = []
    for key, jf, desc, pref in ORDER:
        p = ddir / jf
        if not p.exists():
            continue
        if jf not in loaded:
            loaded[jf] = json.load(open(p))
        d = loaded[jf]
        fullfit = key.endswith("fullrefit")
        if "%s_test" % pref not in d or (not fullfit
                                         and "%s_val" % pref not in d):
            continue
        row = {"member": key, "descriptor": desc}
        for sp in ("val", "test"):
            for mk, lab in COLS:
                e = d.get("%s_%s" % (pref, sp))
                row["%s_%s" % (sp, mk)] = float(e[mk]) if e else None
        row["source"] = jf
        if args.version == "paper" and key in PAPER_REF:
            ref = PAPER_REF[key]
            row["paper_ref_val_mac"] = ref[2]
            row["paper_ref_test_mac"] = ref[5]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "table6_classical.csv", index=False)

    head = "| Member | Val Acc | Val Bal | Val Mac | Test Acc | Test Bal | Test Mac |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    def f(v):
        return "—" if v is None else "%.4f" % v
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["member"], f(r["val_acc"]), f(r["val_bal_acc"]),
            f(r["val_f1_macro"]), f(r["test_acc"]), f(r["test_bal_acc"]),
            f(r["test_f1_macro"])))
    if args.version == "paper":
        lines.append("")
        lines.append("Paper-reference test macro-F1: " + ", ".join(
            "%s=%.4f" % (k, v[5]) for k, v in PAPER_REF.items()))
    with open(outdir / "table6_classical.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(df[["member", "val_f1_macro", "test_f1_macro"]].to_string(index=False))
    print("saved", outdir / "table6_classical.csv")


if __name__ == "__main__":
    main()
