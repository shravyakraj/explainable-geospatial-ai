from pathlib import Path
"""Evaluation metrics and confidence interval utilities for deforestation classification."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_confusion_matrix(y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, int]:
    """Compute confusion matrix counts (TP, FP, TN, FN)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_binary, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(y_true)}


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)

    # 95% confidence z-score is ~1.95996
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 1.96
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4 * (n**2)))

    low = max(0.0, float(center - margin))
    high = min(1.0, float(center + margin))
    return (round(low, 4), round(high, 4))


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (PR-AUC) using trapezoidal rule."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.0

    thresholds = np.sort(np.unique(y_prob))[::-1]
    precisions = [1.0]
    recalls = [0.0]

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    precisions.append(np.sum(y_true == 1) / len(y_true))
    recalls.append(1.0)

    # Sort by recall for trapezoidal integration
    rec_sorted_idx = np.argsort(recalls)
    rec_sorted = np.array(recalls)[rec_sorted_idx]
    prec_sorted = np.array(precisions)[rec_sorted_idx]

    auc = np.trapz(prec_sorted, rec_sorted)
    return float(np.clip(auc, 0.0, 1.0))


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (ROC-AUC) using trapezoidal rule."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    thresholds = np.sort(np.unique(y_prob))[::-1]
    fprs = [0.0]
    tprs = [0.0]

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr = tp / n_pos
        fpr = fp / n_neg
        fprs.append(fpr)
        tprs.append(tpr)

    fprs.append(1.0)
    tprs.append(1.0)

    # Sort by FPR
    order = np.argsort(fprs)
    fpr_sorted = np.array(fprs)[order]
    tpr_sorted = np.array(tprs)[order]

    auc = np.trapz(tpr_sorted, fpr_sorted)
    return float(np.clip(auc, 0.0, 1.0))


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute comprehensive classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = compute_confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    total = len(y_true)

    # Counts and prevalence
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    pos_prevalence = n_pos / total if total > 0 else 0.0

    # Primary metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Deforestation recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Forest recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0

    # Negative class (Intact Forest) metrics
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_recall = specificity
    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0.0

    # Curve AUCs
    pr_auc = compute_pr_auc(y_true, y_prob)
    roc_auc = compute_roc_auc(y_true, y_prob)

    # 95% Wilson confidence intervals
    acc_ci = compute_wilson_ci(tp + tn, total)
    prec_ci = compute_wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    rec_ci = compute_wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    spec_ci = compute_wilson_ci(tn, tn + fp) if (tn + fp) > 0 else (0.0, 0.0)

    # Majority class baseline
    majority_class = 1 if n_pos >= n_neg else 0
    majority_acc = max(n_pos, n_neg) / total if total > 0 else 0.0

    return {
        "sample_count": total,
        "n_deforestation": n_pos,
        "n_intact_forest": n_neg,
        "prevalence_deforestation": round(pos_prevalence, 4),
        "majority_class": majority_class,
        "majority_baseline_accuracy": round(majority_acc, 4),
        "threshold": threshold,
        "confusion_matrix": cm,
        "accuracy": round(accuracy, 4),
        "accuracy_ci_95": acc_ci,
        "precision": round(precision, 4),
        "precision_ci_95": prec_ci,
        "recall": round(recall, 4),
        "recall_ci_95": rec_ci,
        "specificity": round(specificity, 4),
        "specificity_ci_95": spec_ci,
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "intact_forest_precision": round(neg_precision, 4),
        "intact_forest_recall": round(neg_recall, 4),
        "intact_forest_f1": round(neg_f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
    }


def plot_confusion_matrix_figure(cm: Dict[str, int], output_path: Path, title: str = "Test Set Confusion Matrix") -> None:
    """Plot and save confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    matrix = np.array([
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)

    classes = ["Intact Forest (0)", "Deforestation (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Label counts and percentages
    total = np.sum(matrix)
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = count / total * 100
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label (Threshold = 0.5)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot: %s", output_path)


def plot_roc_pr_curves_figure(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    """Plot ROC and PR curves side-by-side."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    thresholds = np.sort(np.unique(y_prob))[::-1]
    fprs, tprs = [0.0], [0.0]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        fprs.append(fp / n_neg if n_neg > 0 else 0)
        tprs.append(tp / n_pos if n_pos > 0 else 0)
    fprs.append(1.0)
    tprs.append(1.0)

    order = np.argsort(fprs)
    fpr_s = np.array(fprs)[order]
    tpr_s = np.array(tprs)[order]
    roc_auc = float(np.trapz(tpr_s, fpr_s))

    axes[0].plot(fpr_s, tpr_s, color="#1f77b4", lw=2.5, label=f"Siamese Model (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10)
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.02])
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    # PR curve
    precs, recs = [1.0], [0.0]
    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        fn = np.sum((y_true == 1) & (yp == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
    precs.append(n_pos / len(y_true))
    recs.append(1.0)

    rec_order = np.argsort(recs)
    rec_s = np.array(recs)[rec_order]
    prec_s = np.array(precs)[rec_order]
    pr_auc = float(np.trapz(prec_s, rec_s))
    baseline_prec = n_pos / len(y_true)

    axes[1].plot(rec_s, prec_s, color="#2ca02c", lw=2.5, label=f"Siamese Model (AUC = {pr_auc:.3f})")
    axes[1].axhline(y=baseline_prec, color="k", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({baseline_prec:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_xlim([-0.02, 1.02])
    axes[1].set_ylim([-0.02, 1.02])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC and PR curves: %s", output_path)
