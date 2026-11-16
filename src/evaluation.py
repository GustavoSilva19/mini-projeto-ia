
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    g_mean = np.sqrt(sensitivity * specificity)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "g_mean": g_mean,
        "roc_auc": auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    }

def summarize_results(results):
    

    metrics = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "g_mean",
        "roc_auc",
        "mse",
        "execution_time"
    ]

    summary = []

    grouped = results.groupby([
        "dataset",
        "algorithm",
        "smote"
    ])

    for (dataset, algorithm, smote), group in grouped:

        row = {
            "dataset": dataset,
            "algorithm": algorithm,
            "smote": smote
        }

        for metric in metrics:

            row[f"{metric}_mean"] = group[metric].mean()

            row[f"{metric}_std"] = group[metric].std()

        summary.append(row)

    return pd.DataFrame(summary)
