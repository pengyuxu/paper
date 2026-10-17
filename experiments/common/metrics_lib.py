"""Unified evaluation on the FIXED class grid 0..C-1.

Every metric reported in the paper goes through this module so that the paper
(97 classes) and updated (99 classes) versions are evaluated identically.
Classes absent from a split are still part of the grid; sklearn reports zero
for them (zero_division=0) and balanced accuracy is computed on classes
PRESENT in y_true (sklearn semantics, unchanged across versions).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics as skm

METRIC_ORDER = ["acc", "bal_acc", "prec_macro", "rec_macro", "f1_macro",
                "prec_weighted", "rec_weighted", "f1_weighted"]


def class_grid(num_classes: int) -> np.ndarray:
    return np.arange(num_classes)


def evaluate(logp: np.ndarray, y: np.ndarray, num_classes: int) -> dict:
    """logp: (N,C) log-probabilities (or scores); y: (N,) int labels."""
    logp = np.asarray(logp, np.float64)
    y = np.asarray(y, np.int64)
    pred = logp.argmax(1)
    labels = class_grid(num_classes)
    present = np.unique(y)
    out = {
        "acc": skm.accuracy_score(y, pred),
        "bal_acc": skm.balanced_accuracy_score(y, pred),
    }
    p, r, f, s = skm.precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    # macro / weighted only over classes present in y_true (matches paper tables);
    # absent classes are excluded from the reduction like sklearn default.
    pw, rw, fw, _ = skm.precision_recall_fscore_support(
        y, pred, labels=present, average="weighted", zero_division=0)
    out["prec_macro"] = p[present].mean()
    out["rec_macro"] = r[present].mean()
    out["f1_macro"] = f[present].mean()
    out["prec_weighted"] = pw
    out["rec_weighted"] = rw
    out["f1_weighted"] = fw
    return out


def per_class_table(logp: np.ndarray, y: np.ndarray, num_classes: int) -> pd.DataFrame:
    logp = np.asarray(logp, np.float64)
    y = np.asarray(y, np.int64)
    pred = logp.argmax(1)
    labels = class_grid(num_classes)
    p, r, f, support = skm.precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    cm = skm.confusion_matrix(y, pred, labels=labels)
    return pd.DataFrame({
        "class_id": labels, "n": support,
        "precision": p, "recall": r, "f1": f,
        "n_correct": np.diag(cm),
    })


def softmax(logp: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(logp, np.float64) / t
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def evaluate_test(logp: np.ndarray, ids: np.ndarray, gt: pd.DataFrame,
                  id_col: str, label_col: str, num_classes: int) -> dict:
    """Align dumped test predictions with ground truth by id.

    logp rows are ordered exactly like the test_list CSV; `ids` carries the
    anonymised numeric ids. GT is looked up by id to be robust to ordering.
    """
    gmap = gt.set_index(id_col)[label_col].to_dict()
    y = np.array([gmap[int(i)] for i in np.asarray(ids).ravel()], dtype=np.int64)
    assert len(y) == logp.shape[0]
    return evaluate(logp, y, num_classes)


def evaluate_subset(logp: np.ndarray, ids: np.ndarray, keep_ids: set,
                    gt: pd.DataFrame, id_col: str, label_col: str,
                    num_classes: int) -> dict:
    """Historical-method re-evaluation (updated N=2308 grid) on common ids."""
    ids = np.asarray(ids).ravel()
    m = np.array([int(i) in keep_ids for i in ids])
    gmap = gt.set_index(id_col)[label_col].to_dict()
    y = np.array([gmap[int(i)] for i in ids[m]], dtype=np.int64)
    return evaluate(np.asarray(logp)[m], y, num_classes)
