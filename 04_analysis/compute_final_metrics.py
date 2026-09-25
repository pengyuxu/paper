#!/usr/bin/env python
"""Compute final test metrics for the frozen production fusion and verify
them against the pre-registered target (acc=94.79, bal=91.48, macF1=92.25).

Reads:
  - expected_outputs/test_full_fusion.csv  (anonymised_protein_id, predicted_label)
  - 00_data/splits/test_set_ground_truth.csv (anonymised_protein_id, class_id)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_PRED = PROJ / "expected_outputs" / "test_full_fusion.csv"
DEFAULT_GT = PROJ / "00_data" / "splits" / "test_set_ground_truth.csv"

TARGET = (94.79, 91.48, 92.25)  # acc, bal, macF1 (percent)
TOL = 0.05  # percent points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", default=str(DEFAULT_PRED))
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred)
    gt_df = pd.read_csv(args.gt)

    # Align rows by protein id so ordering cannot silently drift.
    merged = gt_df.merge(pred_df, on="anonymised_protein_id", how="inner")
    if len(merged) != len(gt_df):
        raise SystemExit(
            "row mismatch after merge: gt=%d pred=%d merged=%d"
            % (len(gt_df), len(pred_df), len(merged)))

    y_true = merged["class_id"].values
    y_pred = merged["predicted_label"].values

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal = balanced_accuracy_score(y_true, y_pred) * 100.0
    mac = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    print("=== Final fusion test metrics ===")
    print("N samples        : %d" % len(y_true))
    print("accuracy         : %.2f" % acc)
    print("balanced acc     : %.2f" % bal)
    print("macro-F1         : %.2f" % mac)
    print("target (acc/bal/macF1): %.2f / %.2f / %.2f" % TARGET)

    ok = (abs(acc - TARGET[0]) <= TOL
          and abs(bal - TARGET[1]) <= TOL
          and abs(mac - TARGET[2]) <= TOL)
    if ok:
        print("PASS: metrics match the pre-registered target (tol=%.2f)." % TOL)
    else:
        diffs = (acc - TARGET[0], bal - TARGET[1], mac - TARGET[2])
        print("WARN: metrics deviate from target by (acc %+.2f, bal %+.2f, mac %+.2f)."
              % diffs)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
