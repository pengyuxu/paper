#!/usr/bin/env python
"""Nested-K val macro-F1 curve with one-SE error bars for the MC-FPS signal.

For K=1..25 we average the first K clean forward passes, take the argmax
prediction and score val macro-F1. We bootstrap (B=300) over val samples to
get a standard error per K, and mark:
  - the K that maximises bootstrapped mean val macro-F1 (argmax)
  - the smallest K within one SE of the argmax (one-SE rule, the selection
    actually used for the production configuration)

Reads:
  - cache/mcfps_val_logp.npy  (K=25, N=1816, C=97)
  - 00_data/splits/train_set_2.csv  (class_id[7428:] = val labels)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJ / "cache" / "mcfps_val_logp.npy"
FALLBACK_CACHE = PROJ / "results_final" / "cache" / "mcfps_val_logp.npy"
DEFAULT_SPLITS = PROJ / "00_data" / "splits" / "train_set_2.csv"
OUT = PROJ / "figures" / "mcfps_k_curve.pdf"

VAL_OFFSET = 7428
B = 300
SEED = 7  # matches the v5 nested-K bootstrap seed in fuse_mcfps2.py


def load_cache(path):
    p = Path(path)
    if not p.exists() and str(p) == str(DEFAULT_CACHE) and FALLBACK_CACHE.exists():
        print("primary cache not found, falling back to %s" % FALLBACK_CACHE)
        p = FALLBACK_CACHE
    if not p.exists():
        raise SystemExit(
            "MC-FPS cache not found: %s\nRun Stage 1 first to produce it, e.g.:\n"
            "  python 01_a_layer/dump_mcfps.py --K 25" % p)
    arr = np.load(p)
    if arr.ndim != 3:
        raise SystemExit("expected (K,N,C) array, got shape %s" % (arr.shape,))
    return arr.astype(np.float64)


def probs(lp):
    z = lp - lp.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macf1(y, p, nclass=97):
    tcb = np.bincount(y, minlength=nclass)
    pres = tcb > 0
    pcb = np.bincount(p, minlength=nclass)
    tp = np.bincount(y[p == y], minlength=nclass)
    den = 2 * tp + (pcb - tp) + (tcb - tp)
    f1v = np.divide(2 * tp, den, out=np.zeros(nclass, float), where=den > 0)
    uni = pres | (pcb > 0)
    return (f1v * uni).sum() / uni.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--splits", default=str(DEFAULT_SPLITS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    logp = load_cache(args.cache)
    K, N, C = logp.shape
    yval = pd.read_csv(args.splits)["class_id"].values[VAL_OFFSET:].astype(np.int64)
    if len(yval) != N:
        raise SystemExit("val size mismatch: cache N=%d splits val=%d" % (N, len(yval)))

    rng = np.random.default_rng(args.seed)
    Ks = np.arange(1, K + 1)
    mean = np.zeros(K)
    se = np.zeros(K)
    for Kc in Ks:
        Pk = probs(logp[:Kc].mean(0))  # (N, C) averaged prob
        scores = np.empty(args.B)
        for b in range(args.B):
            ix = rng.integers(0, N, N)
            scores[b] = macf1(yval[ix], Pk[ix].argmax(1))
        mean[Kc - 1] = scores.mean()
        se[Kc - 1] = scores.std(ddof=1)

    ib = int(mean.argmax())
    thr = mean[ib] - se[ib]
    ksel = int(min(Kc for Kc in Ks if mean[Kc - 1] >= thr))
    print("argmax K=%d (macF1=%.4f +- %.4f)" % (ib + 1, mean[ib], se[ib]))
    print("one-SE selected K=%d" % ksel)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(Ks, mean, yerr=se, fmt="-o", color="#4c72b0", capsize=3,
                lw=1.5, ms=4, label="bootstrapped val macF1 (B=%d)" % args.B)
    ax.axvline(ib + 1, color="crimson", ls=":", lw=1.2, label="argmax K=%d" % (ib + 1))
    ax.axvline(ksel, color="green", ls="--", lw=1.4, label="one-SE K=%d" % ksel)
    ax.set_xlabel("K (number of MC-FPS passes averaged)")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Nested-K selection curve (one-SE rule)")
    ax.set_xticks(Ks[::2])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_P = Path(args.out)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_P)
    print("saved %s" % OUT_P)


if __name__ == "__main__":
    main()
