#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C4-BOOT evidence: the §5.2 paired-bootstrap contrast recomputed at B = 2,000.

Mirrors experiments/methods/review_analysis.py (the script that produced the
reported B = 500 values) exactly, including its single-rng nested resample
stream, so that the B = 500 run here reproduces review_analysis.json bit-for-bit
and only B changes between the two settings.

Contrast: MC-FPS K = 25 mixture (A) vs the mean over the 25 single draws (B),
macro-F1 on validation and test, paper and updated profiles.

Writes experiments/results/review_round8_boot.json.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
import numpy as np
from experiments.common.stats_lib import mc_aggregate

NC = {"paper": 97, "updated": 99}
RUN, CKPT = "v3_noSynth_s2", "best_model"
SEEDS = {"B=500": 500, "B=2000": 2000}
SEED = 12345


def load(ver, split):
    d = np.load("experiments/results/%s/deep/%s__%s__seeded42_K25_%s.npz"
                % (ver, RUN, CKPT, split), allow_pickle=True)
    return d["scores"].astype(np.float64), d["labels"].astype(np.int64)


def cm_to_metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    diag = np.einsum("...cc->...c", cm)
    colsum = cm.sum(-2); rowsum = cm.sum(-1); n = rowsum.sum(-1)
    acc = diag.sum(-1) / n
    rec = np.divide(diag, rowsum, out=np.zeros_like(diag), where=rowsum > 0)
    prec = np.divide(diag, colsum, out=np.zeros_like(diag), where=colsum > 0)
    present = rowsum > 0
    bal = np.where(present, rec, 0).sum(-1) / present.sum(-1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    return {"acc": acc, "bal": bal,
            "f1": np.where(present, f1, 0).sum(-1) / present.sum(-1)}


def boot_cm(pred, y, idx, C):
    B, n = idx.shape
    comb = (y * C + pred).astype(np.int64)
    v = (comb[idx] + np.arange(B)[:, None] * (C * C)).ravel()
    return np.bincount(v, minlength=B * C * C).reshape(B, C, C)


def point_cm(pred, y, C):
    return np.bincount((y * C + pred).astype(np.int64),
                       minlength=C * C).reshape(C, C)


def run(B):
    rng = np.random.default_rng(SEED)
    out = {}
    for ver in ("paper", "updated"):
        for split in ("val", "test"):
            scores, y = load(ver, split)
            C = NC[ver]; K, n = scores.shape[0], len(y)
            key = "%s/%s" % (ver, split)
            idx = rng.integers(0, n, size=(B, n))
            df1 = np.array([cm_to_metrics(point_cm(scores[k].argmax(1), y, C)[None])["f1"][0]
                            for k in range(K)])
            mix = mc_aggregate(scores, K)
            mixm = cm_to_metrics(point_cm(mix.argmax(1), y, C)[None])["f1"][0]
            bd = np.stack([cm_to_metrics(boot_cm(scores[k].argmax(1), y, idx, C))["f1"]
                           for k in range(K)], 0)
            mixb = cm_to_metrics(boot_cm(mix.argmax(1), y, idx, C))["f1"]
            d = mixb - bd.mean(0)
            obs = mixm - df1.mean()
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = ((np.sum(d <= 0) + 1.0) / (B + 1.0) if obs >= 0
                 else (np.sum(d >= 0) + 1.0) / (B + 1.0))
            out[key] = {"delta": float(obs), "ci95": [float(lo), float(hi)],
                        "p": float(min(p, 1.0))}
    return out


def main():
    res = {tag: run(BB) for tag, BB in SEEDS.items()}
    diff = max(max(abs(res["B=500"][k][m] - res["B=2000"][k][m])
                   for m in ("delta", "p")) for k in res["B=500"])
    ci = max(max(abs(res["B=500"][k]["ci95"][i] - res["B=2000"][k]["ci95"][i])
                 for i in (0, 1)) for k in res["B=500"])
    res["_max_abs_diff"] = max(diff, ci)
    with open("experiments/results/review_round8_boot.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
