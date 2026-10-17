"""Generate Fig. 3: strongest published baselines vs this work (test set).

All values are read from caches — no hand transcription:
  - experiments/results/{paper,updated}/tables/table10_crossmethod.csv (Block B/C)
  - experiments/results/{v}/deep/deep_analysis.json (v5 s2-SWA MC-FPS K25 test)
  - experiments/results/{v}/tables/table78_fusion.csv (P11 argmax, Final)

Usage:  python3 experiments/methods/make_fig3_method_comparison.py
Output: figures/fig3_method_comparison.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 400,
})

RESULTS = "experiments/results"
VERS = ("paper", "updated")
NC = {"paper": 97, "updated": 99}

def deep_entry(v, run, ckpt):
    d = json.load(open(f"{RESULTS}/{v}/deep/deep_analysis.json"))
    e = [x for x in d if x["run"] == run and x["K"] == 25 and x["ckpt"] == ckpt][0]
    c = e["splits"]["test"]["curve"]["25"]
    return c["acc"], c["mac"]

def t78_entry(v, panel):
    t = pd.read_csv(f"{RESULTS}/{v}/tables/table78_fusion.csv")
    r = t[t.panel == panel].iloc[0]
    return r["test_acc"], r["test_mac"]

# method rows: (display label, group, acc, mac)  group: legacy / retrained / ours
rows = {}
for v in VERS:
    t10 = pd.read_csv(f"{RESULTS}/{v}/tables/table10_crossmethod.csv")
    def get(name):
        r = t10[t10.method == name].iloc[0]
        return r["acc"], r["mac"]
    R = []
    R.append(("Tatsuma (FPFH-SVM)", "legacy", *get("tatsuma")))
    R.append(("3D-SURFER (Kagaya)", "legacy", *get("kagaya_3D-Surfer")))
    R.append(("Barisin v1 (geom)", "legacy", *get("barisin_geom")))
    R.append(("Barisin v2 (geom+prop)", "legacy", *get("barisin_geom_prop")))
    if v == "updated":
        R.append(("Barisin run2 (retrained)", "retrained", *get("barisin_run2_updated")))
    R.append(("Ours: Final (frozen)", "ours", *t78_entry(v, "final")))
    R.append(("Ours: P11 (val argmax)", "ours", *t78_entry(v, "p11")))
    R.append(("Ours: v5-SWA MC-FPS K25", "ours", *deep_entry(v, "v5_s2", "swa_model")))
    rows[v] = R

COL = {"legacy": "#8da0c9", "retrained": "#e8a33d", "ours": "#2e9e6b"}
COL_MAC = {"legacy": "#c3ccE4", "retrained": "#f3cf9a", "ours": "#a9dcc4"}
EDGE = {"legacy": "#5a6b99", "retrained": "#b37a1f", "ours": "#1d6e4b"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
fig.suptitle("Fig. 3 — Test-set comparison: strongest published baselines vs this work",
             fontsize=10, y=1.00)

for vi, v in enumerate(VERS):
    ax = axes[vi]
    R = rows[v]
    n = len(R)
    y = np.arange(n)
    h = 0.38
    acc = [r[2] for r in R]
    mac = [r[3] for r in R]
    grp = [r[1] for r in R]
    ax.barh(y + h/2, acc, height=h, color=[COL[g] for g in grp],
            edgecolor=[EDGE[g] for g in grp], lw=0.6, label="accuracy")
    ax.barh(y - h/2, mac, height=h, color=[COL_MAC[g] for g in grp],
            edgecolor=[EDGE[g] for g in grp], lw=0.6, hatch="//", label="macro-F1")
    for yi, (a, m) in enumerate(zip(acc, mac)):
        ax.text(a + .002, yi + h/2, f"{a:.3f}", va="center", fontsize=7)
        ax.text(max(m, 0.8005) + .002, yi - h/2, f"{m:.3f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in R])
    ax.set_xlim(0.80, 1.0)
    ax.set_xlabel("test score")
    nc, nt = (97, 2311) if v == "paper" else (99, 2308)
    ax.set_title(f"({'ab'[vi]}) {v} version ({nc} classes, n = {nt})")
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# single shared legend, placed outside below the panels (avoids covering bars)
handles = [
    plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.6, label="accuracy (solid)"),
    plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.6, hatch="//", label="macro-F1 (hatched)"),
    plt.Rectangle((0, 0), 1, 1, fc=COL["legacy"], ec=EDGE["legacy"], label="published baseline"),
    plt.Rectangle((0, 0), 1, 1, fc=COL["retrained"], ec=EDGE["retrained"], label="retrained baseline (updated only)"),
    plt.Rectangle((0, 0), 1, 1, fc=COL["ours"], ec=EDGE["ours"], label="this work"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.015))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
out = "figures/fig3_method_comparison.png"
plt.savefig(out, dpi=400, bbox_inches="tight")
print(f"wrote {out}")
