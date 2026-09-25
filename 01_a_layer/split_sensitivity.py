"""Split-sensitivity check (paper Section III-D).

Applies the SAME frozen v3 single-draw checkpoint to a class-stratified
80/20 partition of the 9,244-row training list WITHOUT retraining.
Single-draw log-probabilities over all rows are obtained (after optional
stage-1 v3 dumps) in:
  results_final/cache/train_clean.npz  (rows 0..7427)
  results_final/cache/val_clean.npz    (rows 7428..9243)

Explains why the stratified "validation" macro-F1 (~0.97) is inflated:
about 80% of its rows are rows the checkpoint was trained on, where it
scores ~0.993 through memorization. On its never-seen rows it scores
~0.88, in line with the row-order split and with test.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "results_final" / "cache"
SEED = 42


def mac(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def main():
    y = pd.read_csv(BASE / "csv/train_set_2.csv").class_id.values
    lp = np.concatenate([
        np.load(CACHE / "train_clean.npz")["logp"].astype(np.float64),
        np.load(CACHE / "val_clean.npz")["logp"].astype(np.float64)])
    pred = lp.argmax(1)
    assert lp.shape == (9244, 97)

    print("row-order val   (n=1816) mac = %.4f" % mac(y[7428:], pred[7428:]))
    print("training rows   (n=7428) mac = %.4f (memorization)"
          % mac(y[:7428], pred[:7428]))

    _, vi = next(StratifiedShuffleSplit(n_splits=1, train_size=0.8,
                                        random_state=SEED)
                 .split(np.zeros(len(y)), y))
    vi = np.sort(vi)
    seen, unseen = vi[vi < 7428], vi[vi >= 7428]
    n_classes = (np.bincount(y[vi], minlength=97) > 0).sum()
    print("\nstratified split random_state=%d" % SEED)
    print("strat val       (n=%d, %d classes) mac = %.4f"
          % (len(vi), n_classes, mac(y[vi], pred[vi])))
    print("  seen-in-train (n=%d, %.0f%%)    mac = %.4f"
          % (len(seen), 100 * len(seen) / len(vi), mac(y[seen], pred[seen])))
    print("  never-seen    (n=%d)          mac = %.4f"
          % (len(unseen), mac(y[unseen], pred[unseen])))

    tr = np.bincount(y[:7428], minlength=97)
    print("\nrow-order training coverage: classes with <=2 train meshes: %s"
          % np.where(tr <= 2)[0].tolist())
    print("classes with one train mesh + one val member: %s"
          % np.where(tr == 1)[0].tolist())


if __name__ == "__main__":
    sys.exit(main())
