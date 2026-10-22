"""
evaluation.py — Lab 1: Language Identification
Custom evaluation metrics: Macro-F1, Accuracy, Confusion Matrix.
NO sklearn used anywhere.
"""

import numpy as np
from typing import List, Dict, Tuple


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the confusion matrix without sklearn.

    Returns
    -------
    matrix : ndarray, shape (n_classes, n_classes)
        matrix[i, j] = number of samples with true class i predicted as class j.
    classes : List[str]
        Sorted list of class names (row/column labels).
    """
    classes = sorted(set(y_true) | set(y_pred))
    idx     = {c: i for i, c in enumerate(classes)}
    n       = len(classes)

    matrix = np.zeros((n, n), dtype=np.int32)
    for true_lbl, pred_lbl in zip(y_true, y_pred):
        matrix[idx[true_lbl], idx[pred_lbl]] += 1

    return matrix, classes


# ── Per-class Precision, Recall, F1 ──────────────────────────────────────────

def classification_report(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1 score without sklearn.

    Returns
    -------
    report : dict mapping class_name → {precision, recall, f1, support}
    """
    matrix, classes = confusion_matrix(y_true, y_pred)
    report: Dict[str, Dict[str, float]] = {}

    for i, cls in enumerate(classes):
        TP = int(matrix[i, i])
        FP = int(matrix[:, i].sum()) - TP   # all predicted as i, minus true positives
        FN = int(matrix[i, :].sum()) - TP   # all actually i, minus true positives
        support = int(matrix[i, :].sum())

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        report[cls] = {
            "precision": round(precision, 6),
            "recall"   : round(recall,    6),
            "f1"       : round(f1,        6),
            "support"  : support,
        }

    return report


# ── Macro-F1 ──────────────────────────────────────────────────────────────────

def macro_f1_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute the Macro-averaged F1 score without sklearn.
    Macro average: compute F1 for each class independently, then take the mean.
    This treats all classes equally regardless of support.
    """
    report = classification_report(y_true, y_pred)
    f1_scores = [metrics["f1"] for metrics in report.values()]
    return float(np.mean(f1_scores))


# ── Accuracy ──────────────────────────────────────────────────────────────────

def accuracy_score(y_true: List[str], y_pred: List[str]) -> float:
    """Compute overall classification accuracy without sklearn."""
    if len(y_true) == 0:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


# ── Full Evaluation Report ────────────────────────────────────────────────────

def print_evaluation_report(
    y_true: List[str],
    y_pred: List[str],
    split_name: str = "Test",
) -> Tuple[float, float]:
    """
    Print a detailed evaluation report and return (macro_f1, accuracy).

    Prints:
      - Overall accuracy
      - Macro-F1
      - Per-class: precision, recall, F1, support
      - Confusion matrix (compact)
    """
    report  = classification_report(y_true, y_pred)
    mac_f1  = macro_f1_score(y_true, y_pred)
    acc     = accuracy_score(y_true, y_pred)
    matrix, classes = confusion_matrix(y_true, y_pred)

    bar = "=" * 70

    print(f"\n{bar}")
    print(f"  Evaluation Report  —  {split_name} Set")
    print(bar)
    print(f"  Samples  : {len(y_true)}")
    print(f"  Accuracy : {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")
    print(f"  Macro-F1 : {mac_f1:.4f}")
    print(f"{bar}")

    # Per-class table
    header = f"  {'Language':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}"
    print(header)
    print(f"  {'-'*62}")

    for cls in sorted(report.keys()):
        m = report[cls]
        print(
            f"  {cls:<22} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} "
            f"{m['support']:>8d}"
        )

    print(f"{bar}\n")

    return mac_f1, acc


def print_confusion_matrix(y_true: List[str], y_pred: List[str]) -> None:
    """Print a compact confusion matrix with class labels."""
    matrix, classes = confusion_matrix(y_true, y_pred)
    n = len(classes)

    # Abbreviated class names for compact display
    abbr = [c[:8] for c in classes]

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    # Header
    header = "         " + "".join(f"{a:>9}" for a in abbr)
    print(header)
    print("  " + "-" * (9 + 9 * n))
    for i, cls in enumerate(classes):
        row_str = f"  {abbr[i]:<7}" + "".join(f"{matrix[i, j]:>9}" for j in range(n))
        print(row_str)
    print()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_true = ["hin_Deva", "tam_Taml", "tel_Telu", "hin_Deva", "tam_Taml"]
    y_pred = ["hin_Deva", "tam_Taml", "hin_Deva", "hin_Deva", "tel_Telu"]

    print_evaluation_report(y_true, y_pred, split_name="Demo")
    print(f"Macro-F1 (standalone): {macro_f1_score(y_true, y_pred):.4f}")
    print(f"Accuracy (standalone): {accuracy_score(y_true, y_pred):.4f}")
