#!/usr/bin/env python
"""Characterise the Monte-Carlo FPS variance of the deep signal on val.

Reads cache/mcfps_val_logp.npy with shape (K=25, N=1816, C=97): K clean forward
passes (random FPS seed each pass). Produces figures/fps_variance.pdf with:
  left  - histogram of per-sample argmax agreement (modal-share over 25 passes)
  right - histogram of per-(sample,class) log-prob maxdiff across 25 passes

High agreement / small maxdiff => the K-pass mean is a stable, reproducible
deep signal (justifying MC-FPS averaging over a single stochastic pass).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJ / "cache" / "mcfps_val_logp.npy"
FALLBACK_CACHE = PROJ / "results_final" / "cache" / "mcfps_val_logp.npy"
OUT = PROJ / "figures" / "fps_variance.pdf"


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
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    logp = load_cache(args.cache)
    K, N, C = logp.shape
    print("loaded %s shape=(%d,%d,%d)" % (args.cache, K, N, C))

    # --- argmax agreement across the K passes ---
    argmaxes = logp.argmax(2)  # (K, N)
    counts = np.zeros((N, C), dtype=np.int64)
    np.add.at(counts, np.arange(N), argmaxes)
    mode_share = counts.max(1) / K  # 1.0 = all K passes agree
    exact_agree = float((mode_share == 1.0).mean())
    mean_share = float(mode_share.mean())

    # --- log-prob maxdiff across the K passes ---
    maxdiff = logp.max(0) - logp.min(0)  # (N, C)
    maxdiff_flat = maxdiff.ravel()
    # ignore exact zeros (classes that never move) for the histogram tail
    nz = maxdiff_flat[maxdiff_flat > 0]

    print("argmax exact-agreement (all %d pass): %.2f%%" % (K, exact_agree * 100))
    print("mean modal-share: %.4f" % mean_share)
    if nz.size:
        print("logp maxdiff (nonzero): median=%.4f p95=%.4f max=%.4f"
              % (np.median(nz), np.percentile(nz, 95), nz.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(mode_share, bins=bins, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Per-sample argmax agreement (modal share over %d passes)" % K)
    ax.set_ylabel("# val samples")
    ax.set_title("MC-FPS argmax consistency\nall-agree=%.1f%%  mean=%.3f"
                 % (exact_agree * 100, mean_share))
    ax.axvline(mean_share, color="crimson", ls="--", lw=1.2, label="mean")
    ax.legend()

    ax = axes[1]
    if nz.size:
        ax.hist(nz, bins=60, color="#55a868", edgecolor="white")
        ax.axvline(np.median(nz), color="crimson", ls="--", lw=1.2,
                   label="median=%.4f" % np.median(nz))
        ax.legend()
    ax.set_xlabel("Per-(sample,class) log-prob max-min across %d passes" % K)
    ax.set_ylabel("count")
    ax.set_title("MC-FPS log-prob variance")
    ax.set_yscale("log")

    fig.tight_layout()
    OUT_P = Path(args.out)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_P)
    print("saved %s" % OUT_P)


if __name__ == "__main__":
    main()
