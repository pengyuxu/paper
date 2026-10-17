"""Render dual-version fusion Tables 7/8/11 (P7, P10, P11, final) from
fusion_summary.json produced by fusion_panel.py.

Outputs: experiments/results/<version>/tables/table78_fusion.{csv,md}
Missing panels are skipped (rerun after each scan to refresh).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse
import json

import pandas as pd

from experiments.common import paths as vp

PANELS = ["p7", "p10", "p11", "final"]
SEL = ["argmax", "onse"]
MEMBERS = {
    "p7": ["c1", "c1b", "c1c", "c1d", "c2", "c2b", "c2d"],
    "p10": ["c1", "c1b", "c1c", "c1d", "c2", "c2b", "c2d", "c3", "c3b", "c4"],
    "p11": ["c1", "c1b", "c1c", "c1d", "c2", "c2b", "c2d", "c3", "c3b",
            "c4", "v3deep"],
    "final": ["v5swa", "c1d", "c2"],
}
# paper revised manuscript references (val_mac, test_acc, test_mac)
PAPER_REF = {
    "p7": (0.8955, None, 0.8968),
    "p10": (0.9019, None, 0.8825),
    "p11": (0.9067, None, 0.9200),
    "final": (0.9165, 0.9479, 0.9148),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    fp = vp.RESULTS_DIR / args.version / "fusion" / "fusion_summary.json"
    d = json.load(open(fp))
    rows = []
    for p in PANELS:
        for s in SEL:
            key = "%s_%s" % (p, s)
            if key not in d:
                continue
            e = d[key]
            row = {"panel": p, "selection": s,
                   "weights": json.dumps(e["weights"]),
                   "tau_dzd": e.get("tau_dzd"),
                   "tau_fpfh": e.get("tau_fpfh"),
                   "val_acc": e["val"]["acc"],
                   "val_bal": e["val"]["bal_acc"],
                   "val_mac": e["val"]["f1_macro"],
                   "test_acc": e["test"]["acc"],
                   "test_bal": e["test"]["bal_acc"],
                   "test_mac": e["test"]["f1_macro"]}
            if "%s_se" % p in d:
                row["bootstrap_se"] = d["%s_se" % p]
            rows.append(row)
    df = pd.DataFrame(rows)
    outdir = vp.RESULTS_DIR / args.version / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "table78_fusion.csv", index=False)

    L = ["| Panel | Sel | Val Acc | Val Mac | Test Acc | Test Bal | Test Mac |"
         " w | tau(d,f) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s,%s |" % (
            r["panel"], r["selection"], r["val_acc"], r["val_mac"],
            r["test_acc"], r["test_bal"], r["test_mac"], r["weights"],
            r["tau_dzd"], r["tau_fpfh"]))
    if args.version == "paper":
        L.append("")
        L.append("Paper references (val mac / test mac): " + ", ".join(
            "%s %.4f/%.4f" % (k, v[0], v[2]) for k, v in PAPER_REF.items()))
    with open(outdir / "table78_fusion.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
