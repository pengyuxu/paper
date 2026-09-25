#!/usr/bin/env python
"""97x97 test confusion matrix of the frozen production fusion, log colour scale.

Reads:
  - expected_outputs/test_full_fusion.csv  (anonymised_protein_id, predicted_label)
  - 00_data/splits/test_set_ground_truth.csv (anonymised_protein_id, class_id)
Outputs figures/confusion_matrix.pdf.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_PRED = PROJ / "expected_outputs" / "test_full_fusion.csv"
DEFAULT_GT = PROJ / "00_data" / "splits" / "test_set_ground_truth.csv"
OUT = PROJ / "figures" / "confusion_matrix.pdf"
N_CLASS = 97


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", default=str(DEFAULT_PRED))
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred)
    gt_df = pd.read_csv(args.gt)
    merged = gt_df.merge(pred_df, on="anonymised_protein_id", how="inner")
    if len(merged) != len(gt_df):
        raise SystemExit("row mismatch after merge: gt=%d pred=%d merged=%d"
                         % (len(gt_df), len(pred_df), len(merged)))

    y_true = merged["class_id"].values
    y_pred = merged["predicted_label"].values
    labels = list(range(N_CLASS))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    diag = np.trace(cm)
    print("test samples=%d  correct=%d  acc=%.4f"
          % (len(y_true), diag, diag / len(y_true)))

    # Log colour scale: zeros shown as the lowest bin (1).
    disp = np.where(cm == 0, 1, cm).astype(float)
    vmax = max(disp.max(), 1.0)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(disp, cmap="viridis", aspect="equal",
                   norm=LogNorm(vmin=1, vmax=vmax),
                   interpolation="nearest")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Test confusion matrix (log scale) -- final fusion\n"
                 "acc=94.79  bal=91.48  macF1=92.25")
    ax.set_xticks(np.arange(0, N_CLASS, 10))
    ax.set_yticks(np.arange(0, N_CLASS, 10))
    ax.set_xticklabels(np.arange(0, N_CLASS, 10))
    ax.set_yticklabels(np.arange(0, N_CLASS, 10))
    fig.colorbar(im, ax=ax, label="# samples (log)", shrink=0.8)
    fig.tight_layout()
    OUT_P = Path(args.out)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_P)
    print("saved %s" % OUT_P)


if __name__ == "__main__":
    main()
