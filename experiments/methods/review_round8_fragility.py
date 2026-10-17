#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1-CONS evidence: ordering fragility of the recomputed SHREC test table.

For the test split of each SHREC version, every cached single draw of our two
deep configurations is compared with every recomputed external system on
macro-F1, and each draw-level ordering is checked against the ordering implied
by the K = 25 MC-FPS mixture. Counts how many deep-vs-external pairs reverse
their sign for at least one draw, and where.

Writes experiments/results/review_round8_fragility.json.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
import numpy as np
import pandas as pd
from experiments.common.metrics_lib import evaluate
from experiments.common.stats_lib import mc_aggregate

NC = {"paper": 97, "updated": 99}
DEEP = {"v3": ("v3_noSynth_s2", "best_model"),
        "v5-SWA": ("v5_s2", "swa_model")}


def draws(ver, run, ckpt, nc):
    d = np.load("experiments/results/%s/deep/%s__%s__seeded42_K25_test.npz"
                % (ver, run, ckpt), allow_pickle=True)
    S = d["scores"].astype(np.float64); y = d["labels"].astype(np.int64)
    f1 = np.array([evaluate(S[k], y, nc)["f1_macro"] for k in range(S.shape[0])])
    mix = evaluate(mc_aggregate(S, S.shape[0]), y, nc)["f1_macro"]
    return f1, float(mix)


def main():
    out = {}
    for ver, nc in NC.items():
        ext = pd.read_csv("experiments/results/%s/history/history_%sgrid.csv"
                          % (ver, ver))
        ext = {r["method"]: float(r["f1_macro"]) for _, r in ext.iterrows()}
        rec = {"externals_f1": ext, "configs": {}, "reversals": [],
               "n_deep_vs_external_pairs": 0,
               "n_pairs_unstable": 0,
               "n_draw_level_flips": 0,
               "n_deep_vs_external_draw_level": 0}
        for lab, (run, ckpt) in DEEP.items():
            f1, mix = draws(ver, run, ckpt, nc)
            rec["configs"][lab] = {"mix_f1": mix,
                                   "draw_min": float(f1.min()),
                                   "draw_max": float(f1.max()),
                                   "n_draws_below_mix": int((f1 < mix).sum())[0]
                                   if False else int((f1 < mix).sum()),
                                   "n_draws_above_mix": int((f1 > mix).sum())}
            for name, e in ext.items():
                rec["n_deep_vs_external_pairs"] += 1
                rec["n_deep_vs_external_draw_level"] += len(f1)
                ref = np.sign(mix - e)
                flips = int(np.sum(np.sign(f1 - e) != ref))
                rec["n_draw_level_flips"] += flips
                if flips:
                    rec["n_pairs_unstable"] += 1
                    rec["reversals"].append({
                        "config": lab, "external": name, "external_f1": e,
                        "mix_f1": mix, "mix_side": "above" if ref > 0 else "below",
                        "n_draws_opposite": flips, "n_draws": len(f1)})
        out[ver] = rec
    with open("experiments/results/review_round8_fragility.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
