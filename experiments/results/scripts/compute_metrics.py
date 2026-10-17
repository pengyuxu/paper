import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def write_metrics_file(file_results, output_path):

    df_results = pd.read_csv(file_results)
    list_true = df_results["ground_truth_class"].to_list()
    list_pred = df_results["pred_class"].to_list()

    list_labels = list(range(0, 97))

    with open(output_path, "w") as f_metrics:

        # Global evaluation

        f_metrics.write("CLASSIFICATION REPORT\n")
        global_eval = classification_report(
            list_true, list_pred, zero_division=np.nan, labels=list_labels
        )
        f_metrics.write(global_eval + "\n")

        f_metrics.write("PRF1\n")
        prf1 = precision_recall_fscore_support(
            list_true,
            list_pred,
            average="weighted",
            zero_division=np.nan,
            labels=list_labels,
        )
        f_metrics.write(str(prf1) + "\n")

        # Accuracy and balanced accuracy scores

        f_metrics.write("ACCURACY SCORE\n")
        accuracy = accuracy_score(list_true, list_pred)
        f_metrics.write(str(accuracy) + "\n")

        f_metrics.write("BALANCED ACCURACY SCORE\n")
        balanced_accuracy = balanced_accuracy_score(list_true, list_pred)
        f_metrics.write(str(balanced_accuracy) + "\n")

        # F1 score

        f_metrics.write("F1 SCORE\n")
        f1 = f1_score(
            list_true,
            list_pred,
            average="weighted",
            zero_division=np.nan,
            labels=list_labels,
        )
        f_metrics.write(str(f1) + "\n")

        # Precision

        f_metrics.write("PRECISION\n")
        precision = precision_score(
            list_true,
            list_pred,
            average="weighted",
            zero_division=np.nan,
            labels=list_labels,
        )
        f_metrics.write(str(precision) + "\n")

        # Recall

        f_metrics.write("RECALL\n")
        recall = recall_score(
            list_true,
            list_pred,
            average="weighted",
            zero_division=np.nan,
            labels=list_labels,
        )
        f_metrics.write(str(recall) + "\n")
