#!/usr/bin/env python3
"""Graphical abstract for the PR submission, rendered from the shipped caches.

Zero-transcription: flip rates (single draw vs K=25 mixture, the manuscript's
Table 3 definition), single-draw macro-F1 mean(sd), and the K=1/K=25 macro-F1
values come from deep_analysis.json; AURC reductions are recomputed from the
per-pass npz dumps with experiments/common/diagnostics.aurc; the example
sample is the paper-validation v3 sample with the most pass-to-pass label
variation. Outputs graphical_abstract.png (2656x1062 px, 200 dpi) and .pdf.

Output directory: <repo>/submission when it exists (the submission checkout),
otherwise <repo>/figures (standalone / reproduction-package checkout); can be
overridden with --outdir.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from experiments.common.metrics_lib import softmax  # noqa: E402
from experiments.common.diagnostics import aurc  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEEP = os.path.join(ROOT, "experiments/results/{v}/deep")

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument("--outdir", default=None,
                     help="output directory (default: submission/ if present, "
                          "otherwise figures/)")
_args = _parser.parse_args()


def resolve_outdir(cli_outdir=None):
    """Pick the output directory: --outdir > submission/ > figures/."""
    if cli_outdir:
        return os.path.abspath(cli_outdir)
    sub = os.path.join(ROOT, "submission")
    if os.path.isdir(sub):
        return sub
    return os.path.join(ROOT, "figures")


OUTDIR = resolve_outdir(_args.outdir)
os.makedirs(OUTDIR, exist_ok=True)
OUT_PNG = os.path.join(OUTDIR, "graphical_abstract.png")
OUT_PDF = os.path.join(OUTDIR, "graphical_abstract.pdf")

C_A, C_B, C_MIX = "#c0392b", "#2471a3", "#1e8449"


def analysis(ver):
    a = json.load(open(os.path.join(DEEP.format(v=ver), "deep_analysis.json")))
    return next(x for x in a if x["run"] == "v3_noSynth_s2"
                and x["ckpt"] == "best_model"
                and x["splits"]["val"]["n_pass"] == 25)


def aurc_reduction(ver, split):
    """1 - AURC(K=25 mixture) / mean_k AURC(single draw k), in percent."""
    d = np.load(os.path.join(DEEP.format(v=ver),
                             "v3_noSynth_s2__best_model__seeded42_K25_%s.npz" % split))
    s = d["scores"].astype(np.float64)
    y = d["labels"]
    singles = [aurc(softmax(s[k]), y)["aurc"] for k in range(s.shape[0])]
    mix_logp = np.log(softmax(s).mean(0) + 1e-300)
    mix = aurc(softmax(mix_logp), y)["aurc"]
    return (1.0 - mix / np.mean(singles)) * 100.0


# ---------- data ----------
e_p, e_u = analysis("paper"), analysis("updated")
k1_p = e_p["splits"]["val"]["single_mean_mac"]
k25_p = e_p["splits"]["val"]["curve"]["25"]["mac"]
k1_u = e_u["splits"]["val"]["single_mean_mac"]
k25_u = e_u["splits"]["val"]["curve"]["25"]["mac"]
sd_p = e_p["splits"]["val"]["single_sd_mac"]
flip = {v: {sp: e[f"splits"][sp]["diag"]["flip_rate"] * 100
            for sp in ("val", "test")}
        for v, e in (("paper", e_p), ("updated", e_u))}
ar_p, ar_u = aurc_reduction("paper", "val"), aurc_reduction("updated", "val")

# ordering fragility (review_round8_fragility.json): both unstable pairs on the
# paper version are against 3D-SURFER; 4 + 3 = 7 of 2 x 25 draw comparisons flip
_frag = json.load(open(os.path.join(
    ROOT, "experiments/results/review_round8_fragility.json")))
_fp = _frag["paper"]
assert all(r["external"] == "kagaya_3D-Surfer" for r in _fp["reversals"])
_fl = sum(r["n_draws_opposite"] for r in _fp["reversals"])
_nd = sum(r["n_draws"] for r in _fp["reversals"])
assert _fl == _fp["n_draw_level_flips"] == 7 and _nd == 50
frag_line = ("%d of 2 deep configs reverse vs 3D-SURFER: %d/%d draw reversals"
             % (_fp["n_pairs_unstable"], _fl, _nd))

# example flip sample: most pass-to-pass variation, binary prediction
d = np.load(os.path.join(DEEP.format(v="paper"),
                         "v3_noSynth_s2__best_model__seeded42_K25_val.npz"))
s_raw = d["scores"]
pred_p = s_raw.argmax(-1)
y_p = d["labels"]
flips = (pred_p != pred_p[0]).sum(0)
cand = [i for i in np.argsort(-flips) if len(np.unique(pred_p[:, i])) == 2]
si = int(cand[0])
probs = np.exp(s_raw[:, si].astype(np.float64))
probs /= probs.sum(1, keepdims=True)
true_c = int(y_p[si])
p_true = probs[:, true_c]
mix = probs.mean(0)

# one real protein cloud for illustration
cloud = np.load(os.path.join(ROOT, "work/clouds/txt8/0_pc.npy"))[:, :3]
cloud = cloud[::2]                                   # 4096 pts, lighter render

# ---------- figure ----------
fig = plt.figure(figsize=(13.28, 5.31), dpi=200)
gs = GridSpec(1, 3, width_ratios=[1.05, 1.0, 0.95], wspace=0.30,
              left=0.045, right=0.985, top=0.86, bottom=0.14)
fig.suptitle("Seed sensitivity of farthest-point sampling: one unseeded draw "
             "makes inference nondeterministic \u2014 averaging K = 25 passes removes it",
             fontsize=13.5, fontweight="bold", y=0.965)

# --- panel A: same input, different draws -> different labels ---
subA = gs[0, 0].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.62)
axA = fig.add_subplot(subA[0])
z = cloud[:, 2]
axA.scatter(cloud[:, 0], cloud[:, 1], c=z, cmap="viridis", s=1.2, alpha=0.75)
axA.set_xticks([]); axA.set_yticks([])
for sp in axA.spines.values():
    sp.set_visible(False)
axA.set_title("Same input, 25 FPS passes\n(unseeded initial centroid)",
              fontsize=11.5, fontweight="bold")
axA.set_xlabel("one protein surface point cloud", fontsize=9.5)

axA2 = fig.add_subplot(subA[1])
uniq = np.unique(pred_p[:, si])
cmap = {uniq[0]: C_A, uniq[1]: C_B}
axA2.scatter(np.arange(25), pred_p[:, si],
             c=[cmap[c] for c in pred_p[:, si]], s=42, marker="s", zorder=3)
axA2.set_yticks(list(uniq))
axA2.set_yticklabels([f"class {c}" for c in uniq], fontsize=8.5)
axA2.set_xticks([0, 6, 12, 18, 24])
axA2.set_xlabel("FPS pass index (one example sample)", fontsize=9)
axA2.grid(axis="y", ls=":", alpha=0.5)
axA2.set_title("label flips on %.1f%% / %.1f%% of val samples (single draw vs K = 25 mixture);\nsingle-draw macro-F1 varies by \u00b1%.3f (paper version)"
               % (flip['paper']['val'], flip['updated']['val'], sd_p),
               fontsize=8.5)

# --- panel B: probability-space averaging ---
axB = fig.add_subplot(gs[0, 1])
axB.plot(np.arange(1, 26), p_true, "o-", color=C_B, ms=5, lw=1.4,
         label="per-pass P(true class)")
axB.axhline(mix[true_c], color=C_MIX, lw=2.2,
            label=f"MC-FPS mixture = {mix[true_c]:.2f}")
axB.set_xlabel("FPS pass index", fontsize=10)
axB.set_ylabel("predicted probability of true class", fontsize=10)
axB.set_ylim(0, 1.02)
axB.set_title("MC-FPS: average the 25 probability vectors\n"
              "(no retraining, no code change)", fontsize=11.5,
              fontweight="bold")
axB.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
axB.grid(ls=":", alpha=0.5)
axB.text(0.03, 0.16, "argmax stabilizes; sampling variance \u2192 0",
         transform=axB.transAxes, fontsize=9.5, color=C_MIX, fontweight="bold")

# --- panel C: criterion + headline numbers ---
axC = fig.add_subplot(gs[0, 2])
axC.axis("off")
axC.set_title("Decision criterion and consequences", fontsize=11.5,
              fontweight="bold")
lines = [
    ("Fix the seed", C_A, "you control the code path (bit-exact reproducibility)"),
    ("Average over seeds", C_B, "released checkpoints or selective prediction"),
]
y0 = 0.97
for head, col, sub in lines:
    axC.add_patch(plt.Rectangle((0.02, y0 - 0.10), 0.045, 0.085, color=col,
                                transform=axC.transAxes, clip_on=False))
    axC.text(0.10, y0 - 0.038, head, transform=axC.transAxes, fontsize=11.5,
             fontweight="bold", color=col, va="center")
    axC.text(0.10, y0 - 0.095, sub, transform=axC.transAxes, fontsize=8.5,
             va="center", color="#333333")
    y0 -= 0.175
stats = [
    f"val macro-F1 (25-draw mean \u2192 K25): {k1_p:.3f}\u2192{k25_p:.3f} / "
    f"{k1_u:.3f}\u2192{k25_u:.3f}",
    "paired bootstrap vs 25-draw mean: P < .001 / .005 (val)",
    "paper-test macro gain within noise (P = .212) \u2014 null",
    f"AURC (risk\u2013coverage): \u2212{ar_p:.0f}% / \u2212{ar_u:.0f}% (val)",
    "reproduces on PointNet++ SSG/MSG (ModelNet40; earlier audit)",
    frag_line,
]
axC.text(0.02, y0 - 0.01, "Consequences", transform=axC.transAxes,
         fontsize=11, fontweight="bold")
yy = y0 - 0.08
for i, st in enumerate(stats):
    axC.text(0.02, yy, st, transform=axC.transAxes,
             fontsize=8.5 if i in (0, len(stats) - 1) else 9.0,
             fontweight="bold" if i == len(stats) - 1 else "normal",
             va="center", color="#7b241c" if i == len(stats) - 1 else "#222222")
    yy -= 0.070

fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print("flip single-vs-mixture val/test paper: %.1f%% / %.1f%%; updated: %.1f%% / %.1f%%"
      % (flip["paper"]["val"], flip["paper"]["test"],
         flip["updated"]["val"], flip["updated"]["test"]))
print("val macro-F1 single->K25: %.4f->%.4f / %.4f->%.4f" % (k1_p, k25_p, k1_u, k25_u))
print("AURC reduction val: %.1f%% / %.1f%% (paper / updated)" % (ar_p, ar_u))
print("example sample idx", si, "true class", true_c, "mixture P(true)=%.3f" % mix[true_c])
print("wrote", OUT_PNG)
print("wrote", OUT_PDF)
