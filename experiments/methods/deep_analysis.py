"""Analyse deterministic deep dumps (deep_dump.py caches) and render
dual-version Tables 2/3/4 plus K-curve data.

Input cache naming:
  results/<version>/deep/<run>__<ckpt>__<fps><seed>_K<K>_<split>.npz
with npz keys ids, scores (K,N,C) f16 raw logits, labels.

Outputs under results/<version>/:
  deep/deep_analysis.json
  tables/table2_mc.csv, table3_diag.csv, table4_modes.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.common import paths as vp
from experiments.common import metrics_lib, diagnostics as diag, stats_lib

FNAME = re.compile(r"^(?P<run>.+?)__(?P<ckpt>[^_]+(?:_[^_]+)*?)__"
                   r"(?P<fps>first|seeded|gen)(?P<seed>\d+)_K(?P<K>\d+)_"
                   r"(?P<split>val|test)\.npz$")
KCURVE = [1, 2, 3, 5, 10, 25]


def load(ver, run, ckpt, fps, seed, K, split):
    fname = "%s__%s__%s%s_K%d_%s.npz" % (run, ckpt, fps, seed, K, split)
    p = vp.RESULTS_DIR / ver / "deep" / fname
    d = np.load(p)
    return d["scores"].astype(np.float32), d["labels"].astype(int)


def mac(logp, y, nc):
    return metrics_lib.evaluate(logp, y, nc)


def analyse_group(ver, run, ckpt, fps, seed, K, nc):
    out = {"run": run, "ckpt": ckpt, "fps": fps, "seed": seed, "K": K,
           "splits": {}}
    for split in ("val", "test"):
        try:
            S, y = load(ver, run, ckpt, fps, seed, K, split)
        except FileNotFoundError:
            continue
        kk = S.shape[0]
        per = [mac(S[k], y, nc) for k in range(kk)]
        macs = np.array([m["f1_macro"] for m in per])
        accs = np.array([m["acc"] for m in per])
        bals = np.array([m["bal_acc"] for m in per])
        curve = {}
        for kk2 in KCURVE:
            if kk2 > kk:
                continue
            e = mac(stats_lib.mc_aggregate(S, K=kk2), y, nc)
            curve[str(kk2)] = {"acc": e["acc"], "bal": e["bal_acc"],
                               "mac": e["f1_macro"]}
        dg = diag.all_diagnostics(S, y, single_logp=S[0])
        if kk > 1:
            pred = S.argmax(2)  # (K,N)
            tot = 0.0
            cnt = 0
            for a in range(kk):
                for b in range(a + 1, kk):
                    tot += float((pred[a] != pred[b]).mean())
                    cnt += 1
            dg["pairflip_mean"] = tot / cnt
        else:
            dg["pairflip_mean"] = None
        out["splits"][split] = {
            "n_pass": int(kk),
            "single_mean_mac": float(macs.mean()),
            "single_sd_mac": float(macs.std(ddof=1)) if kk > 1 else 0.0,
            "single_min_mac": float(macs.min()),
            "single_max_mac": float(macs.max()),
            "single_mean_acc": float(accs.mean()),
            "single_mean_bal": float(bals.mean()),
            "curve": curve,
            "diag": dg,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    args = ap.parse_args()
    nc = vp.version_cfg(args.version)["num_classes"]
    ddir = vp.RESULTS_DIR / args.version / "deep"
    groups = {}
    for p in glob.glob(str(ddir / "*.npz")):
        m = FNAME.match(os.path.basename(p))
        if not m:
            continue
        key = (m["run"], m["ckpt"], m["fps"], int(m["seed"]), int(m["K"]))
        groups.setdefault(key, set()).add(m["split"])

    analyses = []
    for key in sorted(groups):
        run, ckpt, fps, seed, K = key
        analyses.append(analyse_group(args.version, run, ckpt, fps, seed, K, nc))

    jpath = ddir / "deep_analysis.json"
    with open(jpath, "w") as f:
        json.dump(analyses, f, indent=1)

    tdir = vp.RESULTS_DIR / args.version / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # Table 2: MC performance
    t2 = []
    for a in analyses:
        for split, s in a["splits"].items():
            c = s["curve"]
            t2.append({"run": a["run"], "ckpt": a["ckpt"], "split": split,
                       "K": a["K"],
                       "single_mean": s["single_mean_mac"],
                       "single_sd": s["single_sd_mac"],
                       "single_min": s["single_min_mac"],
                       "single_max": s["single_max_mac"],
                       "mc3_mac": c.get("3", c.get("2", {})).get("mac"),
                       "mc25_mac": c.get("25", c.get("10", {})).get("mac"),
                       "mc25_acc": c.get("25", c.get("10", {})).get("acc"),
                       "mc25_bal": c.get("25", c.get("10", {})).get("bal")})
    pd.DataFrame(t2).to_csv(tdir / "table2_mc.csv", index=False)

    # Table 3: calibration / uncertainty
    t3 = []
    for a in analyses:
        for split, s in a["splits"].items():
            d = s["diag"]
            t3.append({"run": a["run"], "ckpt": a["ckpt"], "split": split,
                       "K": a["K"], "ece15": d["ece15"], "aurc": d["aurc"],
                       "bald": d["bald_mean"], "flip": d["flip_rate"],
                       "pairflip": d["pairflip_mean"],
                       "gap_single": d["gap_mean_single"],
                       "gap_ens": d["gap_mean_ens"]})
    pd.DataFrame(t3).to_csv(tdir / "table3_diag.csv", index=False)

    # Table 4: inference modes (D0=first-point deterministic,
    # Ds=fixed-seed CPU generator, random=mean of stochastic single passes,
    # K25=seeded hash ensemble)
    t4 = []
    byrun = defaultdict(dict)
    for a in analyses:
        for split, s in a["splits"].items():
            byrun[(a["run"], split)][a["fps"]] = (a, s)
    for (run, split), modes in sorted(byrun.items()):
        row = {"run": run, "split": split}
        if "seeded" in modes:
            a, s = modes["seeded"]
            row["ckpt_seeded"] = a["ckpt"]
            row["random_mean_mac"] = s["single_mean_mac"]
            row["random_mean_acc"] = s["single_mean_acc"]
            k25 = s["curve"].get("25", s["curve"].get("10"))
            if k25:
                row["k25_mac"] = k25["mac"]
                row["k25_acc"] = k25["acc"]
        if "first" in modes:
            a, s = modes["first"]
            row["ckpt_d0"] = a["ckpt"]
            c1 = s["curve"].get("1")
            if c1:
                row["d0_mac"] = c1["mac"]
                row["d0_acc"] = c1["acc"]
        if "gen" in modes:
            a, s = modes["gen"]
            row["ckpt_ds"] = a["ckpt"]
            c1 = s["curve"].get("1")
            if c1:
                row["ds_mac"] = c1["mac"]
                row["ds_acc"] = c1["acc"]
        t4.append(row)
    pd.DataFrame(t4).to_csv(tdir / "table4_modes.csv", index=False)
    print("groups analysed:", len(analyses), "->", tdir)


if __name__ == "__main__":
    main()
