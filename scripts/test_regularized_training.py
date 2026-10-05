"""Test regularized training strategies to prevent logit explosion and optimize validation ranking."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import keras
from keras import layers, models, ops, regularizers
import tensorflow as tf

from geospatial_ai.dataset import create_tf_dataset, get_split_records, load_numpy_arrays
from geospatial_ai.metrics import evaluate_predictions
from geospatial_ai.model import build_siamese_network
from geospatial_ai.train import set_seed, find_optimal_threshold

manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
train_records = get_split_records(manifest_path, "train", project_root=PROJECT_ROOT)
val_records = get_split_records(manifest_path, "val", project_root=PROJECT_ROOT)
val_t1, val_t2, val_y = load_numpy_arrays(val_records)

batch_size = 8
seed = 42
set_seed(seed)

train_ds = create_tf_dataset(train_records, batch_size=batch_size, is_training=True, augment=True, seed=seed)
val_ds = create_tf_dataset(val_records, batch_size=batch_size, is_training=False, augment=False, seed=seed)

# Candidate 1: Plain CNN with Directional Fusion + Label Smoothing (0.05) + L2(1e-4) + Cosine Decay
model = build_siamese_network(
    input_shape=(256, 256, 4),
    dense_units=64,
    dropout_rate=0.5,
    backbone_type="conv",
    pooling="gap",
    fusion_type="directional",
    l2_reg=1e-4,
    name="reg_directional_cnn",
)

epochs = 40
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-4,
    decay_steps=epochs * (len(train_records) // batch_size + 1),
    alpha=1e-5,
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Binary cross-entropy with label smoothing
loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.08)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        "binary_accuracy",
        keras.metrics.AUC(curve="PR", name="pr_auc"),
        keras.metrics.AUC(curve="ROC", name="roc_auc"),
    ],
)

# Custom callback to track composite validation score
class ValidationMonitor(keras.callbacks.Callback):
    def __init__(self, val_t1, val_t2, val_y):
        super().__init__()
        self.val_t1 = val_t1
        self.val_t2 = val_t2
        self.val_y = val_y
        self.best_score = -float("inf")
        self.best_weights = None
        self.best_epoch = 0
        self.best_metrics = None

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict({"input_t1": self.val_t1, "input_t2": self.val_t2}, verbose=0)[:, 0]
        th_opt = find_optimal_threshold(self.val_y, probs)
        res = evaluate_predictions(self.val_y, probs, threshold=th_opt)
        
        # Balanced score: PR-AUC + ROC-AUC + Balanced Accuracy
        score = res["pr_auc"] + res["roc_auc"] + res["balanced_accuracy"]
        if score > self.best_score:
            self.best_score = score
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.best_metrics = res
            self.best_opt_th = th_opt
            print(f" -> Epoch {epoch+1}: New best validation score {score:.4f} (PR-AUC={res['pr_auc']:.4f}, ROC-AUC={res['roc_auc']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, Acc={res['accuracy']:.4f}, Thresh={th_opt:.3f})")

monitor = ValidationMonitor(val_t1, val_t2, val_y)

model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[monitor], verbose=1)

model.set_weights(monitor.best_weights)
print("\n=== Best Validation Performance (Restored from Epoch %d) ===" % (monitor.best_epoch + 1))
print("Optimal Threshold:", monitor.best_opt_th)
print("Accuracy:", monitor.best_metrics["accuracy"])
print("Balanced Accuracy:", monitor.best_metrics["balanced_accuracy"])
print("Precision:", monitor.best_metrics["precision"])
print("Recall:", monitor.best_metrics["recall"])
print("Specificity:", monitor.best_metrics["specificity"])
print("F1:", monitor.best_metrics["f1_score"])
print("PR-AUC:", monitor.best_metrics["pr_auc"])
print("ROC-AUC:", monitor.best_metrics["roc_auc"])
print("Confusion Matrix:", monitor.best_metrics["confusion_matrix"])
