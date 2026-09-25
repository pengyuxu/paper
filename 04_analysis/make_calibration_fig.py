"""Fig. (calibration): reliability diagram and selective-prediction curves.

Panel (a): v3 validation reliability diagram, single draw vs K=25.
Panel (b): risk-coverage curves (AURC) for v3 validation and test,
           single draw vs K=25 averaging.
Reads results_final/calibration_summary.json and calibration_curves.npz
(produced by 01_a_layer/calibration_analysis.py). Writes
figures/calibration.{pdf,png}.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
summary = json.loads((BASE / "results_final/calibration_summary.json")
                     .read_text())
curves = np.load(BASE / "results_final/calibration_curves.npz")

fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0))

# ---- (a) reliability diagram: v3 validation ----
ax = axes[0]
bins = np.linspace(0, 1, 16)
w = 1 / 15
for rows, color, off, label in [
        (summary["v3_val"]["rel_single"], "#3B6FB6", -0.004, "single"),
        (summary["v3_val"]["rel_k25"], "#D9534F", 0.004, "K=25")]:
    for conf, acc, _, n in rows:
        ax.bar(conf + off, acc, width=0.82 * w, align="center",
               fc=color, ec=color, alpha=0.85 if label == "single" else 0.55)
ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1)
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
ax.set_title("(a) reliability (v3 val)", fontsize=9)
ax.text(0.53, 0.97, "single ECE .0164\nK=25 ECE .0152", fontsize=7.5,
        va="top", color="#333")

# ---- (b) risk-coverage ----
ax = axes[1]
styles = {"v3_val": ("#3B6FB6", "val"), "v3_test": ("#2E9E6B", "test")}
for key, (c, lab) in styles.items():
    cov = curves[key + "_cov"]
    ax.plot(cov, curves[key + "_single"], color=c, lw=1.8,
            label="%s single" % lab)
    ax.plot(cov, curves[key + "_k25"], color=c, lw=1.8, ls="--",
            label="%s K=25" % lab)
ax.set_xlabel("coverage"); ax.set_ylabel("selective risk")
ax.set_title("(b) risk-coverage (AURC)", fontsize=9)
ax.legend(fontsize=7.2, loc="upper left")
ax.set_xlim(0, 1)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(BASE / f"figures/calibration.{ext}", dpi=220,
                bbox_inches="tight")
print("saved", BASE / "figures/calibration.pdf")
