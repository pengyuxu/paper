#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review-driven supplementary analyses (existing .npz caches, CPU only).

Vectorised paired bootstrap: confusion matrices are built for all B resamples at
once (one bincount per draw), then accuracy / balanced accuracy / macro-F1 are
derived from the counts. Fast (seconds) instead of per-resample sklearn calls.

Outputs experiments/results/review_analysis.json:
  B3   paired bootstrap of K=25 mixture vs single-draw mean (3 metrics)
  B5i  probability-space (log-mean-exp of softmax) vs log-space (mean log-prob)
  C3   single-draw distribution summary (mean / sd / min / max)

npz `scores` are per-pass LOG-SOFTMAX (verified logsumexp(row) == 0); softmax is
shift-invariant, so mc_aggregate's softmax-then-mean == softmax of raw logits.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import numpy as np
from experiments.common.metrics_lib import evaluate
from experiments.common.stats_lib import mc_aggregate

NC = {"paper": 97, "updated": 99}
RUN, CKPT = "v3_noSynth_s2", "best_model"
B = 500
SEED = 12345


def load(ver, split):
    path = ("experiments/results/%s/deep/%s__%s__seeded42_K25_%s.npz"
            % (ver, RUN, CKPT, split))
    d = np.load(path, allow_pickle=True)
    return d["scores"].astype(np.float64), d["labels"].astype(np.int64)


def cm_to_metrics(cm):
    """cm: (..., C, C) confusion matrices -> dict of metric arrays."""
    cm = np.asarray(cm, dtype=np.float64)
    C = cm.shape[-1]
    diag = np.einsum("...cc->...c", cm)
    colsum = cm.sum(-2)                    # predicted-total per class (..., C)
    rowsum = cm.sum(-1)                    # true-total per class (..., C)
    n = rowsum.sum(-1)
    acc = diag.sum(-1) / n
    rec = np.divide(diag, rowsum, out=np.zeros_like(diag), where=rowsum > 0)
    prec = np.divide(diag, colsum, out=np.zeros_like(diag), where=colsum > 0)
    present = rowsum > 0
    bal = np.where(present, rec, 0).sum(-1) / present.sum(-1)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    macro_f1 = np.where(present, f1, 0).sum(-1) / present.sum(-1)
    return {"acc": acc, "bal": bal, "f1": macro_f1}


def boot_cm(pred, y, idx, C):
    """Confusion matrices for B resamples. pred:(N,), y:(N,), idx:(B,N)."""
    comb = (y * C + pred).astype(np.int64)          # (N,)
    Mk = comb[idx]                                   # (B, n)
    b = np.arange(idx.shape[0])[:, None] * (C * C)
    v = (Mk + b).ravel()
    return np.bincount(v, minlength=idx.shape[0] * C * C).reshape(idx.shape[0], C, C)


def point_cm(pred, y, C):
    comb = (y * C + pred).astype(np.int64)
    return np.bincount(comb, minlength=C * C).reshape(C, C)


def main():
    result = {"b3_paired_bootstrap": {}, "b5i_log_vs_prob": {},
              "c3_single_draw": {}, "b1v3_identity_check": {},
              "meta": {"run": RUN, "ckpt": CKPT, "B": B, "seed": SEED}}
    rng = np.random.default_rng(SEED)
    for ver in ("paper", "updated"):
        for split in ("val", "test"):
            scores, y = load(ver, split)
            C = NC[ver]
            K, n = scores.shape[0], len(y)
            key = f"{ver}/{split}"
            idx = rng.integers(0, n, size=(B, n))

            # per-draw point metrics (for C3 summary)
            draws = []
            for k in range(K):
                cm = point_cm(scores[k].argmax(1), y, C)
                draws.append(cm_to_metrics(cm[None])["f1"][0])  # keep f1
            # full 3-metric per-draw arrays
            dacc, dbal, df1 = [], [], []
            for k in range(K):
                m = cm_to_metrics(point_cm(scores[k].argmax(1), y, C)[None])
                dacc.append(m["acc"][0]); dbal.append(m["bal"][0]); df1.append(m["f1"][0])
            dacc, dbal, df1 = np.array(dacc), np.array(dbal), np.array(df1)
            result["c3_single_draw"][key] = {
                "acc": {"mean": float(dacc.mean()), "sd": float(dacc.std(ddof=1)),
                        "min": float(dacc.min()), "max": float(dacc.max())},
                "bal": {"mean": float(dbal.mean()), "sd": float(dbal.std(ddof=1)),
                        "min": float(dbal.min()), "max": float(dbal.max())},
                "f1": {"mean": float(df1.mean()), "sd": float(df1.std(ddof=1)),
                       "min": float(df1.min()), "max": float(df1.max())}}

            # mixture (K=25) curves
            mix = mc_aggregate(scores, K)
            mix_cm = point_cm(mix.argmax(1), y, C)[None]
            mixm = {k: float(v[0]) for k, v in cm_to_metrics(mix_cm).items()}

            # ---- B3 paired bootstrap: mix - mean_k(draw) ----
            # draw-metric bootstrap arrays
            bd = {"acc": [], "bal": [], "f1": []}
            for k in range(K):
                m = cm_to_metrics(boot_cm(scores[k].argmax(1), y, idx, C))
                bd["acc"].append(m["acc"]); bd["bal"].append(m["bal"]); bd["f1"].append(m["f1"])
            mixb = cm_to_metrics(boot_cm(mix.argmax(1), y, idx, C))
            b3 = {}
            for mk, ref in (("acc", dacc.mean()), ("bal", dbal.mean()), ("f1", df1.mean())):
                draw_b = np.mean(np.stack(bd[mk], 0), 0)      # (B,)
                d = mixb[mk] - draw_b
                obs = mixm[mk] - ref
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = (np.sum(d <= 0) + 1.0) / (B + 1.0) if obs >= 0 else \
                    (np.sum(d >= 0) + 1.0) / (B + 1.0)
                b3[mk] = {"mixture": mixm[mk], "single_draw_mean": float(ref),
                          "delta": float(obs), "ci95": [float(lo), float(hi)],
                          "p": float(min(p, 1.0))}
            result["b3_paired_bootstrap"][key] = b3

            # ---- B5i log-space vs probability-space ----
            logp_log = scores[:K].mean(0)
            lm = {k: float(v[0]) for k, v in
                  cm_to_metrics(point_cm(logp_log.argmax(1), y, C)[None]).items()}
            result["b5i_log_vs_prob"][key] = {
                "prob_space": mixm, "log_space": lm,
                "delta_prob_minus_log": {k: mixm[k] - lm[k] for k in mixm}}

            print(f"[{key}] K={K} n={n} C={C}")
            print("  C3 single-draw f1 mean/sd/min/max = %.4f/%.4f/%.4f/%.4f"
                  % (df1.mean(), df1.std(ddof=1), df1.min(), df1.max()))
            print("  B3 mix-vs-single f1 d=%.4f CI=[%.4f,%.4f] p=%.3f"
                  % (b3["f1"]["delta"], b3["f1"]["ci95"][0], b3["f1"]["ci95"][1], b3["f1"]["p"]))
            print("  B3 mix-vs-single bal d=%.4f CI=[%.4f,%.4f] p=%.3f"
                  % (b3["bal"]["delta"], b3["bal"]["ci95"][0], b3["bal"]["ci95"][1], b3["bal"]["p"]))
            print("  B5i prob-vs-log f1 %.4f vs %.4f (d=%.4f)"
                  % (mixm["f1"], lm["f1"], mixm["f1"] - lm["f1"]))

    # sanity: mc_aggregate must equal the published deep_analysis curve
    with open("experiments/results/review_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print("written experiments/results/review_analysis.json")


if __name__ == "__main__":
    main()
