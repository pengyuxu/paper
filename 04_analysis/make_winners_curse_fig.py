#!/usr/bin/env python
"""Reproducible illustration of the winners-curse mechanism.

There is no per-candidate results file for the model-selection grid (the
selection was a one-shot paired bootstrap), so this script simulates the
mechanism: a pool of m candidate models share a latent quality, and val / test
macF1 are independent noisy observations of that quality. Selecting the
val-argmax inflates the val estimate relative to the (re-)observed test score
-- the gap widens as the pool size m grows.

Three panels correspond to the candidate-pool sizes considered in the report:
m = 7, 10, 11. Each point is one (val, test) observation; red marks the
val-argmax winner. The dashed line is y = x.

Outputs figures/winners_curse.pdf. No GPU, no external data.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "figures" / "winners_curse.pdf"

POOL_SIZES = (7, 10, 11)
TRIALS = 500
MU = 0.880          # latent quality mean
SIGMA_Q = 0.020    # between-candidate quality spread
SIGMA_OBS = 0.012   # observation noise (val == test noise level)
SEED = 20250918     # project-local reproducible seed


def simulate(m, rng, trials=TRIALS):
    all_v, all_t = [], []
    win_v, win_t = [], []
    gap = []
    for _ in range(trials):
        theta = rng.normal(MU, SIGMA_Q, m)
        v = theta + rng.normal(0, SIGMA_OBS, m)
        t = theta + rng.normal(0, SIGMA_OBS, m)
        w = int(v.argmax())
        all_v.append(v); all_t.append(t)
        win_v.append(v[w]); win_t.append(t[w])
        gap.append(v[w] - t[w])
    return (np.concatenate(all_v), np.concatenate(all_t),
            np.array(win_v), np.array(win_t), np.array(gap))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)

    print("pool | trials | mean val-winner | mean test-winner | gap (winners curse)")
    for ax, m in zip(axes, POOL_SIZES):
        av, at, wv, wt, gap = simulate(m, rng, trials=args.trials)
        ax.scatter(av, at, s=8, alpha=0.18, color="#888888", label="candidates")
        ax.scatter(wv, wt, s=22, color="crimson", label="val-argmax winner",
                   zorder=3)
        lo = min(av.min(), at.min(), wv.min(), wt.min())
        hi = max(av.max(), at.max(), wv.max(), wt.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")
        ax.set_title("m = %d\nval-winner=%.4f  test-winner=%.4f  gap=%+.4f"
                     % (m, wv.mean(), wt.mean(), gap.mean()))
        ax.set_xlabel("val macF1")
        ax.grid(alpha=0.3)
        print("%4d | %6d |     %.4f      |     %.4f      | %+.4f"
              % (m, args.trials, wv.mean(), wt.mean(), gap.mean()))

    axes[0].set_ylabel("test macF1")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Winners curse: val-argmax selection inflates val vs test\n"
                 "(gap grows with candidate-pool size m)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT_P = Path(args.out)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_P)
    print("saved %s" % OUT_P)


if __name__ == "__main__":
    main()
