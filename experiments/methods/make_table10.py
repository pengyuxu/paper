"""Render dual-version Table 8 (cross-method comparison).

Sources, all optional (N/A rows skipped / placeholders):
  results/<version>/history/history_<version>grid.csv  (15 legacy submissions)
  results/updated/barisin/barisin_updated_metrics.json  (new updated runs)
  results/<version>/fusion/fusion_summary.json          (our p11 / final)

Outputs: results/<version>/tables/table10_crossmethod.{csv,md}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse
import json

import pandas as pd

from experiments.common import paths as vp

# display ordering: known methods first, ours last
ORDER = ["kagaya_3D-Surfer", "tatsuma", "barisin_geom_prop", "barisin_geom",
         "guerra_v3", "guerra_v2", "guerra_v1", "barisin_run1_updated",
         "barisin_run2_updated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    base = vp.RESULTS_DIR / args.version
    rows = []

    hp = base / "history" / ("history_%sgrid.csv" % args.version)
    if hp.exists():
        h = pd.read_csv(hp)
        for _, r in h.iterrows():
            rows.append({"method": r["method"], "source": "legacy submission",
                         "acc": r["acc"], "bal": r["bal_acc"],
                         "mac": r["f1_macro"], "f1w": r["f1_weighted"]})

    if args.version == "updated":
        bp = base / "barisin" / "barisin_updated_metrics.json"
        if bp.exists():
            bd = json.load(open(bp))
            for k in ("run1", "run2"):
                if k in bd:
                    e = bd[k]
                    rows.append({"method": "barisin_%s_updated" % k,
                                 "source": "new training (updated profile)",
                                 "acc": e["acc"], "bal": e["bal_acc"],
                                 "mac": e["f1_macro"],
                                 "f1w": e["f1_weighted"]})

    fp = base / "fusion" / "fusion_summary.json"
    if fp.exists():
        fs = json.load(open(fp))
        for key, label in (("p11_onse", "ours_P11 (1-SE)"),
                           ("final", "ours_Final")):
            if key in fs and "test" in fs[key]:
                e = fs[key]["test"]
                rows.append({"method": label, "source": "this work",
                             "acc": e["acc"], "bal": e["bal_acc"],
                             "mac": e["f1_macro"], "f1w": e["f1_weighted"]})

    def rank(r):
        name = r["method"]
        if name in ORDER:
            return (0, ORDER.index(name))
        if name.startswith("ours"):
            return (2, name)
        return (1, name)
    rows.sort(key=rank)
    df = pd.DataFrame(rows)
    tdir = base / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "table10_crossmethod.csv", index=False)

    L = ["| Method | Acc | Bal | Mac-F1 | Weighted-F1 | Source |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %s |" % (
            r["method"], r["acc"], r["bal"], r["mac"], r["f1w"], r["source"]))
    with open(tdir / "table10_crossmethod.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
