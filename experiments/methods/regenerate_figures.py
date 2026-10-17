"""Regenerate Figs. 1, 2 and S2 of the manuscript from rerun caches.

Every data point comes from experiments/results/{paper,updated}/ caches via the
same mc_aggregate + evaluate pipeline as make_manuscript_tables_dual.py —
no hand transcription, exactly matching the rendered tables.
(Fig. S1 framework: make_fig1_framework.py; Fig. 3 comparison: make_fig3_method_comparison.py)

Usage:  python3 experiments/methods/regenerate_figures.py
Output: figures/fig1_fps_nondeterminism.png
        figures/fig2_calibration_risk.png
        figures/figA2_confusion_final.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 400,
})
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix

from experiments.common.stats_lib import mc_aggregate
from experiments.common.metrics_lib import evaluate

RESULTS = "experiments/results"
FIG_OUT = "figures"
os.makedirs(FIG_OUT, exist_ok=True)
VERS = ("paper", "updated")
NC = {"paper": 97, "updated": 99}


# ─── helpers ──────────────────────────────────────────────────────────────────
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["scores"].astype(np.float64), d["labels"]  # scores: (K,N,C) log-softmax


def softmax_rows(z, t=1.0):
    # z: (N,C), returns (N,C) probabilities per row, temp-scaled
    zt = z / t
    ex = np.exp(zt - zt.max(1, keepdims=True))
    return ex / ex.sum(1, keepdims=True)


def reliability_bins(logp, y, n_bins=15):
    """15 equal-width ECE bins — returns (confidence_mid, accuracy, confidence, n, ece)."""
    p = np.exp(logp)
    conf = p.max(1)
    pred = p.argmax(1)
    acc = (pred == y).astype(float)
    # equal-width bins on confidence ∈ [0,1]
    edges = np.linspace(0, 1, n_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_n = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1]) if i < n_bins - 1 else \
               (conf >= edges[i]) & (conf <= edges[i + 1])
        if mask.sum() > 0:
            bin_conf[i] = conf[mask].mean()
            bin_acc[i] = acc[mask].mean()
            bin_n[i] = mask.sum()
    valid = ~np.isnan(bin_acc)
    ece = np.abs(bin_acc[valid] - bin_conf[valid]) @ bin_n[valid] / bin_n.sum()
    return mid, bin_conf, bin_acc, bin_n, ece


def risk_coverage(logp, y):
    """Risk-coverage curve: for each confidence threshold, compute coverage and error rate."""
    p = np.exp(logp)
    conf = p.max(1)
    pred = p.argmax(1)
    err = (pred != y).astype(float)
    order = np.argsort(conf)      # lowest-confidence first
    cum_err = err[order][::-1].cumsum()    # ascending coverage from most confident first
    cum_n = np.arange(1, len(y) + 1)
    coverage = cum_n / len(y)
    risk = cum_err / cum_n
    aurc = np.trapz(risk, coverage)          # same convention as Table 4 / diagnostics.aurc
    return coverage, risk, aurc


# ─── Fig 1: FPS nondeterminism ─────────────────────────────────────────────────
def make_fig1_fps():
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    fig.suptitle("Fig. 1 — FPS seed nondeterminism of the rebuilt RIConv++ v3 on validation (25 draws)",
                 fontsize=11, y=0.99)
    for vi, v in enumerate(VERS):
        scores, y = load_npz(f"{RESULTS}/{v}/deep/v3_noSynth_s2__best_model__seeded42_K25_val.npz")
        K, N, C = scores.shape
        pred_per_pass = scores.argmax(-1)          # (K, N)
        # per-sample range across K passes (max-min log-softmax over classes per sample)
        logp_range = scores.max(0) - scores.min(0)   # (N,C)
        per_sample_range_max = logp_range.max(-1)    # (N,)
        # pairwise log-softmax gaps over the exhaustive C(K,2)=300 pairs,
        # |delta_c| max over classes, pooled over all samples (matches the caption
        # and manuscript_stats.json, which the 108-row verification covers)
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]   # C(25,2) = 300
        gaps = np.concatenate([np.abs(scores[i] - scores[j]).max(-1)
                               for i, j in pairs])   # (300*N,)

        # (a) left: per-sample log-softmax range
        ax = axes[vi, 0]
        ax.hist(per_sample_range_max, bins=40, color="#c0392b", alpha=0.8, density=True)
        ax.axvline(np.median(per_sample_range_max), color="black", ls="--", lw=1,
                   label=f"median {np.median(per_sample_range_max):.2f}")
        ax.axvline(np.percentile(per_sample_range_max, 90), color="black", ls=":", lw=1,
                   label=f"p90 {np.percentile(per_sample_range_max, 90):.2f}")
        ax.set_title(f"({chr(97 + vi * 4)}) Log-softmax range  [{v}]", fontsize=8.5)
        ax.set_xlabel("max-min log-softmax across K=25 draws")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=7)

        # (b) right: all pairs pooled gap
        ax = axes[vi, 1]
        ax.hist(gaps, bins=60, color="#c0392b", alpha=0.7, density=True)
        ax.axvline(np.median(gaps), color="black", ls="--", lw=1,
                   label=f"median {np.median(gaps):.2f}")
        ax.axvline(np.percentile(gaps, 90), color="black", ls=":", lw=1,
                   label=f"p90 {np.percentile(gaps, 90):.2f}")
        ax.set_title(f"({chr(97 + vi * 4 + 1)}) 300 pairs, max|\u0394|  [{v}]", fontsize=8.5)
        ax.set_xlabel("per-sample log-softmax gap")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=7)

        # (c) val macro-F1 of 25 single draws vs MC-FPS K25
        # Need single draws → each pass's argmax → macro-F1
        mac_single = []
        for k in range(K):
            # eval per-pass: aggregate pass k alone (K=1), then evaluate
            agg = mc_aggregate(scores[k:k + 1], K=1)   # pass k only
            m = evaluate(agg, y, NC[v])
            mac_single.append(m["f1_macro"])
        agg_k25 = mc_aggregate(scores, K=25)
        m_k25 = evaluate(agg_k25, y, NC[v])
        mean_single = np.mean(mac_single)
        std_single = np.std(mac_single, ddof=1)

        ax = axes[vi, 2]
        ax.scatter(range(25), mac_single, color="#c0392b", s=20, label="single draws")
        ax.axhline(mean_single, color="#c0392b", ls="--", lw=1,
                   label=f"mean {mean_single:.4f} ± {std_single:.4f}")
        ax.axhline(m_k25["f1_macro"], color="#27ae60", ls="-", lw=1.5,
                   label=f"MC-FPS K25 {m_k25['f1_macro']:.4f}")
        ax.set_title(f"({chr(97 + vi * 4 + 2)}) Single draw vs K = 25  [{v}]", fontsize=8.5)
        ax.set_xlabel("draw index (seed order)")
        ax.set_ylabel("macro-F1")
        ax.legend(loc="lower right", fontsize=7)

        # (d) right: flip-rate — per-sample argmax unanimous?
        non_unanimous = (pred_per_pass.std(0) > 0)    # (N,) bool
        ax = axes[vi, 3]
        # Bar 0 (unanimous) vs 1 (non-unanimous)
        counts = np.bincount(non_unanimous.astype(int), minlength=2)
        bars = ax.bar([0, 1], counts, color=["#3498db", "#c0392b"], alpha=0.8)
        flip_rate = non_unanimous.mean()
        ax.set_title(f"({chr(97 + vi * 4 + 3)}) Argmax non-unanimity {flip_rate:.1%}", fontsize=8.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["unanimous", "non-unanimous"]); ax.set_xlabel("argmax agreement")
        ax.set_ylabel("# validation samples")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(c), ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{FIG_OUT}/fig1_fps_nondeterminism.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ─── Fig 2: Calibration + risk-coverage ────────────────────────────────────────
def make_fig2_cal():
    # Risk-coverage on the v3 test split (paper + updated). The single-draw
    # curve is the per-coverage mean over all 25 FPS draws, so its AURC matches
    # the single-draw mean reported in Table 4 (same convention as
    # diagnostics.aurc / make_manuscript_tables_dual.py).
    fig, r_axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Risk-coverage curves, v3 test split — MC-FPS K=25 lowers selective risk",
                 fontsize=10, y=1.02)
    for vi, v in enumerate(VERS):
        scores, y = load_npz(
            f"{RESULTS}/{v}/deep/v3_noSynth_s2__best_model__seeded42_K25_test.npz")
        K = scores.shape[0]
        # single draw: mean risk-coverage curve over all K draws
        cov_single = np.arange(1, len(y) + 1) / len(y)
        risk_draws = np.zeros((K, len(y)))
        for j in range(K):
            p = np.exp(scores[j])
            c = p.max(1); pr = p.argmax(1); err = (pr != y).astype(float)
            order = np.argsort(c)
            risk_draws[j] = err[order][::-1].cumsum() / np.arange(1, len(y) + 1)
        risk_s = risk_draws.mean(0)
        aurc_s = np.trapz(risk_s, cov_single)
        # K = 25 mixture
        logp_k25 = mc_aggregate(scores, K=25)
        cov_c, risk_c, aurc_c = risk_coverage(logp_k25, y)
        ax = r_axes[vi]
        ax.plot(cov_single, risk_s, color="#c0392b", lw=1.5,
                label=f"single draw AURC {aurc_s:.4f}")
        ax.plot(cov_c, risk_c, color="#27ae60", lw=1.5,
                label=f"K=25 mixture AURC {aurc_c:.4f}")
        ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate among kept samples)")
        ax.set_title(f"[{v}]")
        ax.legend(fontsize=8); ax.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{FIG_OUT}/fig2_calibration_risk.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close("all")
    print(f"wrote {out}")


# ─── Fig 4: Final panel confusion matrix ────────────────────────────────────────
def fused_logp_final(v, sp):
    """Reproduce Final panel fusion — use renderer's fused_logp directly."""
    from experiments.methods.make_manuscript_tables_dual import fused_logp
    return fused_logp(v, sp, "final")


def make_figA2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Fig. S2 \u2014 Test confusion matrix, Final frozen panel (log colour, off-diagonal mass)",
                 fontsize=11, y=1.03)
    for vi, v in enumerate(VERS):
        logp, y = fused_logp_final(v, "test")
        pred = np.exp(logp).argmax(1)
        nc = NC[v]
        cm = confusion_matrix(y, pred, labels=list(range(nc)), normalize="true")
        # mask the 9 absent classes (all zero columns) as NaN for visual clarity
        cm_masked = np.where(cm > 0, cm, np.nan)
        # log scale for off-diagonal visibility
        log_cm = np.log10(cm_masked + 1e-10)
        ax = axes[vi]
        im = ax.imshow(log_cm, cmap="YlOrRd", aspect="auto", vmin=-3, vmax=0)
        # highlight the 9 absent rows/cols
        absent = np.where(cm.sum(0) == 0)[0]
        for c in absent:
            ax.axhline(c, color="white", lw=0.5, alpha=0.5)
            ax.axvline(c, color="white", lw=0.5, alpha=0.5)
        ax.set_title(f"[{v}] Final test ({len(y)} meshes, {nc} classes)")
        ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
        ax.set_xlim(0, nc - 1); ax.set_ylim(nc - 1, 0)
        ax.set_xticks([]); ax.set_yticks([])   # too many classes for labels
        # colourbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log₁₀ row-normalised recall")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{FIG_OUT}/figA2_confusion_final.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig1_fps()
    make_fig2_cal()
    make_figA2()
    print("done. All three figures regenerated from rerun caches.")
