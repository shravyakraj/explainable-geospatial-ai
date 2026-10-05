"""Training pipeline for bi-temporal Siamese deforestation classification."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import keras
from keras import callbacks, metrics, optimizers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from geospatial_ai.dataset import create_tf_dataset, get_split_records, load_numpy_arrays
from geospatial_ai.metrics import evaluate_predictions
from geospatial_ai.model import build_siamese_network, AbsoluteDifference, SignedDifference

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    import random
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Initialized deterministic seeds with seed=%d", seed)


def plot_learning_curves(history: Dict[str, List[float]], output_path: Path, title: str) -> None:
    """Plot and save loss, accuracy, and PR-AUC learning curves."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.suptitle(title, fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Accuracy
    acc_key = "binary_accuracy" if "binary_accuracy" in history else "accuracy"
    val_acc_key = f"val_{acc_key}"
    axes[1].plot(epochs, history[acc_key], label="Train Acc", color="#1f77b4", lw=2)
    if val_acc_key in history:
        axes[1].plot(epochs, history[val_acc_key], label="Val Acc", color="#ff7f0e", lw=2)
    axes[1].set_title("Binary Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # PR-AUC
    axes[2].plot(epochs, history["pr_auc"], label="Train PR-AUC", color="#1f77b4", lw=2)
    if "val_pr_auc" in history:
        axes[2].plot(epochs, history["val_pr_auc"], label="Val PR-AUC", color="#ff7f0e", lw=2)
    axes[2].set_title("Precision-Recall AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved learning curves plot to %s", output_path)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J index (sensitivity + specificity - 1) on validation data."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_j = -1.0
    best_th = 0.5

    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fn = np.sum((y_true == 1) & (yp == 0))
        tn = np.sum((y_true == 0) & (yp == 0))
        fp = np.sum((y_true == 0) & (yp == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j_index = tpr + tnr - 1.0

        if j_index > best_j:
            best_j = j_index
            best_th = float(th)

    return round(best_th, 4)


def train_siamese_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    val_t1: np.ndarray,
    val_t2: np.ndarray,
    val_y: np.ndarray,
    checkpoint_path: Path,
    log_dir: Path,
    epochs: int = 50,
    learning_rate: float = 2e-4,
    experiment_name: str = "siamese_baseline",
) -> Tuple[keras.Model, Dict, Dict]:
    """Train Siamese model with early stopping, checkpointing, and validation monitoring."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="binary_accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(curve="PR", name="pr_auc"),
            metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )

    csv_logger = callbacks.CSVLogger(str(log_dir / f"{experiment_name}_history.csv"))
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=16,
        restore_best_weights=True,
        verbose=1,
    )
    lr_scheduler_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    )

    logger.info("Starting training for experiment '%s' (%d epochs)...", experiment_name, epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb, csv_logger],
        verbose=1,
    )

    # Load best checkpoint weights for rigorous evaluation
    if checkpoint_path.exists():
        logger.info("Loading best saved checkpoint from %s", checkpoint_path)
        model = keras.models.load_model(str(checkpoint_path), custom_objects={"AbsoluteDifference": AbsoluteDifference, "SignedDifference": SignedDifference}, safe_mode=False)

    # Evaluate best model on validation split
    val_prob = model.predict({"input_t1": val_t1, "input_t2": val_t2}, batch_size=8, verbose=0)[:, 0]
    optimal_th = find_optimal_threshold(val_y, val_prob)

    val_metrics_default = evaluate_predictions(val_y, val_prob, threshold=0.5)
    val_metrics_opt = evaluate_predictions(val_y, val_prob, threshold=optimal_th)

    logger.info(
        "Validation Evaluation (thresh=0.5): Acc=%.4f (CI: %s), F1=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f",
        val_metrics_default["accuracy"],
        val_metrics_default["accuracy_ci_95"],
        val_metrics_default["f1_score"],
        val_metrics_default["pr_auc"],
        val_metrics_default["roc_auc"],
    )
    logger.info(
        "Validation Evaluation (opt thresh=%.4f): Acc=%.4f, F1=%.4f, Recall=%.4f, Spec=%.4f",
        optimal_th,
        val_metrics_opt["accuracy"],
        val_metrics_opt["f1_score"],
        val_metrics_opt["recall"],
        val_metrics_opt["specificity"],
    )

    # Plot curves
    plot_learning_curves(
        history.history,
        log_dir / f"{experiment_name}_curves.png",
        f"Learning Curves: {experiment_name}",
    )

    val_summary = {
        "default_threshold_0_5": val_metrics_default,
        "optimal_threshold": optimal_th,
        "optimal_threshold_metrics": val_metrics_opt,
        "pr_auc": val_metrics_default["pr_auc"],
        "roc_auc": val_metrics_default["roc_auc"],
        "val_loss": float(np.min(history.history["val_loss"])),
    }

    return model, history.history, val_summary
