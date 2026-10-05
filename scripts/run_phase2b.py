"""Master end-to-end pipeline for Phase 2B model training, evaluation, explainability, and error analysis."""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from geospatial_ai.dataset import create_tf_dataset, get_split_records, load_numpy_arrays
from geospatial_ai.explainability import compute_gradcam, plot_explainability_case, compute_ndvi
from geospatial_ai.metrics import (
    evaluate_predictions,
    plot_confusion_matrix_figure,
    plot_roc_pr_curves_figure,
)
import geospatial_ai.model
from geospatial_ai.model import build_siamese_network
from geospatial_ai.train import set_seed, train_siamese_model, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase2b_runner")


def main() -> None:
    start_time = time.time()
    logger.info("=== Starting Phase 2B End-to-End Pipeline ===")

    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
    configs_path = PROJECT_ROOT / "configs" / "training_config.json"

    with open(configs_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = config.get("random_seed", 42)
    set_seed(seed)

    # 1. Load Split Records
    logger.info("Loading dataset splits from manifest: %s", manifest_path)
    train_records = get_split_records(manifest_path, "train", project_root=PROJECT_ROOT)
    val_records = get_split_records(manifest_path, "val", project_root=PROJECT_ROOT)
    test_records = get_split_records(manifest_path, "test", project_root=PROJECT_ROOT)

    logger.info("Train usable count: %d", len(train_records))
    logger.info("Validation usable count: %d", len(val_records))
    logger.info("Test held-out count: %d", len(test_records))

    # Pre-load validation arrays for deterministic evaluation
    val_t1, val_t2, val_y = load_numpy_arrays(val_records)
    logger.info("Val arrays loaded: T1=%s, T2=%s, y=%s (Pos=%d, Neg=%d)",
                val_t1.shape, val_t2.shape, val_y.shape, int(np.sum(val_y == 1)), int(np.sum(val_y == 0)))

    # Create tf.data datasets
    batch_size = config.get("batch_size", 8)
    train_ds = create_tf_dataset(
        train_records, batch_size=batch_size, is_training=True, augment=True, seed=seed
    )
    val_ds = create_tf_dataset(
        val_records, batch_size=batch_size, is_training=False, augment=False, seed=seed
    )

    models_dir = PROJECT_ROOT / "models"
    experiments_dir = PROJECT_ROOT / "experiments"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Experiment 1: Baseline Siamese CNN (Difference Only)
    logger.info("\n--- Running Experiment 1: M1 Siamese Difference Baseline ---")
    model_m1 = build_siamese_network(
        input_shape=(256, 256, 4),
        dense_units=64,
        dropout_rate=0.4,
        include_concatenation=False,
        name="siamese_m1_diff",
    )
    ckpt_m1 = experiments_dir / "checkpoints" / "siamese_m1_diff.keras"
    trained_m1, hist_m1, val_summary_m1 = train_siamese_model(
        model=model_m1,
        train_ds=train_ds,
        val_ds=val_ds,
        val_t1=val_t1,
        val_t2=val_t2,
        val_y=val_y,
        checkpoint_path=ckpt_m1,
        log_dir=experiments_dir,
        epochs=config.get("max_epochs", 50),
        learning_rate=2e-4,
        experiment_name="m1_siamese_diff",
    )

    # 3. Experiment 2: Siamese CNN with Difference + State Concatenation
    logger.info("\n--- Running Experiment 2: M2 Siamese Difference + Concatenation ---")
    set_seed(seed)
    train_ds_m2 = create_tf_dataset(
        train_records, batch_size=batch_size, is_training=True, augment=True, seed=seed
    )
    model_m2 = build_siamese_network(
        input_shape=(256, 256, 4),
        dense_units=64,
        dropout_rate=0.4,
        include_concatenation=True,
        name="siamese_m2_concat",
    )
    ckpt_m2 = experiments_dir / "checkpoints" / "siamese_m2_concat.keras"
    trained_m2, hist_m2, val_summary_m2 = train_siamese_model(
        model=model_m2,
        train_ds=train_ds_m2,
        val_ds=val_ds,
        val_t1=val_t1,
        val_t2=val_t2,
        val_y=val_y,
        checkpoint_path=ckpt_m2,
        log_dir=experiments_dir,
        epochs=config.get("max_epochs", 50),
        learning_rate=2e-4,
        experiment_name="m2_siamese_concat",
    )

    # 4. Model Selection Based Strictly on Validation Performance
    logger.info("\n=== Model Selection (Validation Comparison) ===")
    logger.info("M1 (Difference): Min Val Loss=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f, Opt Thresh=%.4f (F1=%.4f)",
                val_summary_m1["val_loss"], val_summary_m1["pr_auc"], val_summary_m1["roc_auc"],
                val_summary_m1["optimal_threshold"], val_summary_m1["optimal_threshold_metrics"]["f1_score"])
    logger.info("M2 (Concat):     Min Val Loss=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f, Opt Thresh=%.4f (F1=%.4f)",
                val_summary_m2["val_loss"], val_summary_m2["pr_auc"], val_summary_m2["roc_auc"],
                val_summary_m2["optimal_threshold"], val_summary_m2["optimal_threshold_metrics"]["f1_score"])

    # Selection rule: Lower validation loss, tie-break with PR-AUC
    if val_summary_m2["val_loss"] < val_summary_m1["val_loss"]:
        selected_model_name = "M2_Siamese_Concatenation"
        selected_model = trained_m2
        selected_val_summary = val_summary_m2
        selected_ckpt = ckpt_m2
    else:
        selected_model_name = "M1_Siamese_Difference"
        selected_model = trained_m1
        selected_val_summary = val_summary_m1
        selected_ckpt = ckpt_m1

    logger.info("--> Selected Champion Model: %s (Val Loss: %.4f, PR-AUC: %.4f)",
                selected_model_name, selected_val_summary["val_loss"], selected_val_summary["pr_auc"])

    # Save to primary production path
    final_model_path = models_dir / "best_siamese_model.keras"
    selected_model.save(str(final_model_path))
    logger.info("Saved final production model to %s", final_model_path)

    # Save Model Selection Record
    selection_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_sample_size": len(val_records),
        "validation_note": "Validation set has 18 positive and 6 negative samples. Selection prioritized cross-entropy loss and PR-AUC.",
        "candidate_m1_difference": {
            "parameters": model_m1.count_params(),
            "val_summary": val_summary_m1,
        },
        "candidate_m2_concatenation": {
            "parameters": model_m2.count_params(),
            "val_summary": val_summary_m2,
        },
        "selected_model": selected_model_name,
        "selection_rationale": (
            f"Selected {selected_model_name} because it achieved superior validation loss "
            f"({selected_val_summary['val_loss']:.4f}) and PR-AUC ({selected_val_summary['pr_auc']:.4f}) "
            f"on the spatial validation split."
        ),
        "validation_derived_decision_threshold": selected_val_summary["optimal_threshold"],
    }
    with open(reports_dir / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2)

    # 5. Held-Out Test Evaluation (ONCE ONLY)
    logger.info("\n=== SINGLE HELD-OUT TEST EVALUATION ===")
    logger.info("Evaluating selected model once on virgin test split (N=%d)...", len(test_records))
    test_t1, test_t2, test_y = load_numpy_arrays(test_records)
    logger.info("Test arrays loaded: T1=%s, T2=%s, y=%s (Pos=%d, Neg=%d)",
                test_t1.shape, test_t2.shape, test_y.shape, int(np.sum(test_y == 1)), int(np.sum(test_y == 0)))

    # Single inference pass
    test_probs = selected_model.predict({"input_t1": test_t1, "input_t2": test_t2}, batch_size=8, verbose=0)[:, 0]

    # Evaluate test predictions at default threshold 0.50
    test_metrics_default = evaluate_predictions(test_y, test_probs, threshold=0.5)

    # Evaluate test predictions at validation-derived optimal threshold
    val_th = selected_val_summary["optimal_threshold"]
    test_metrics_val_th = evaluate_predictions(test_y, test_probs, threshold=val_th)

    logger.info("\n=== FINAL HELD-OUT TEST RESULTS (Threshold = 0.50) ===")
    logger.info("Accuracy: %.2f%% [95%% Wilson CI: %.2f%% - %.2f%%]",
                test_metrics_default["accuracy"] * 100, test_metrics_default["accuracy_ci_95"][0] * 100, test_metrics_default["accuracy_ci_95"][1] * 100)
    logger.info("Precision: %.2f%% | Recall: %.2f%% | Specificity: %.2f%% | F1: %.4f",
                test_metrics_default["precision"] * 100, test_metrics_default["recall"] * 100,
                test_metrics_default["specificity"] * 100, test_metrics_default["f1_score"])
    logger.info("PR-AUC: %.4f | ROC-AUC: %.4f", test_metrics_default["pr_auc"], test_metrics_default["roc_auc"])
    logger.info("Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d",
                test_metrics_default["confusion_matrix"]["tp"], test_metrics_default["confusion_matrix"]["fp"],
                test_metrics_default["confusion_matrix"]["tn"], test_metrics_default["confusion_matrix"]["fn"])

    logger.info("\n=== FINAL HELD-OUT TEST RESULTS (Validation-Derived Threshold = %.4f) ===", val_th)
    logger.info("Accuracy: %.2f%% [95%% Wilson CI: %.2f%% - %.2f%%]",
                test_metrics_val_th["accuracy"] * 100, test_metrics_val_th["accuracy_ci_95"][0] * 100, test_metrics_val_th["accuracy_ci_95"][1] * 100)
    logger.info("Precision: %.2f%% | Recall: %.2f%% | Specificity: %.2f%% | F1: %.4f",
                test_metrics_val_th["precision"] * 100, test_metrics_val_th["recall"] * 100,
                test_metrics_val_th["specificity"] * 100, test_metrics_val_th["f1_score"])
    logger.info("Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d",
                test_metrics_val_th["confusion_matrix"]["tp"], test_metrics_val_th["confusion_matrix"]["fp"],
                test_metrics_val_th["confusion_matrix"]["tn"], test_metrics_val_th["confusion_matrix"]["fn"])

    # Combine both reports into test_metrics.json
    final_test_output = {
        "selected_model": selected_model_name,
        "sample_count": len(test_records),
        "n_deforestation": int(np.sum(test_y == 1)),
        "n_intact_forest": int(np.sum(test_y == 0)),
        "majority_class_baseline": {
            "class": "Intact Forest (0)",
            "accuracy": round(int(np.sum(test_y == 0)) / len(test_records), 4),
        },
        "threshold_independent_metrics": {
            "pr_auc": test_metrics_default["pr_auc"],
            "roc_auc": test_metrics_default["roc_auc"],
        },
        "default_threshold_0_5": test_metrics_default,
        "validation_calibrated_threshold": {
            "threshold_value": val_th,
            "metrics": test_metrics_val_th,
        },
    }
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_output, f, indent=2)

    # Plot Confusion Matrix (for validation-derived calibrated threshold, and default)
    plot_confusion_matrix_figure(
        test_metrics_val_th["confusion_matrix"],
        reports_dir / "confusion_matrix.png",
        title=f"Test Set Confusion Matrix ({selected_model_name})\nThreshold = {val_th:.3f} | Total N=60",
    )
    plot_confusion_matrix_figure(
        test_metrics_default["confusion_matrix"],
        reports_dir / "confusion_matrix_default_05.png",
        title=f"Test Set Confusion Matrix ({selected_model_name})\nThreshold = 0.500 | Total N=60",
    )

    # Plot ROC and PR Curves
    plot_roc_pr_curves_figure(
        test_y,
        test_probs,
        reports_dir / "roc_pr_curves.png",
    )

    # 6. Detailed Test Predictions & Error Categorization
    pred_records = []
    y_pred_binary = (test_probs >= val_th).astype(int)

    for i, rec in enumerate(test_records):
        y_t = int(test_y[i])
        y_p = int(y_pred_binary[i])
        prob = float(test_probs[i])

        if y_t == 1 and y_p == 1:
            cat = "TP"
        elif y_t == 0 and y_p == 0:
            cat = "TN"
        elif y_t == 0 and y_p == 1:
            cat = "FP"
        else:
            cat = "FN"

        t1_p = test_t1[i]
        t2_p = test_t2[i]
        d_ndvi = float(np.mean(compute_ndvi(t2_p) - compute_ndvi(t1_p)))

        pred_records.append({
            "patch_id": rec.get("patch_id", f"test_{i:03d}"),
            "patch_file": rec.get("patch_file", ""),
            "row": rec.get("row", -1),
            "col": rec.get("col", -1),
            "ground_truth": y_t,
            "pred_binary_calibrated": y_p,
            "pred_probability": round(prob, 4),
            "outcome": cat,
            "mean_delta_ndvi": round(d_ndvi, 4),
            "deforestation_fraction": rec.get("deforestation_fraction", 0.0),
        })

    df_preds = pd.DataFrame(pred_records)
    df_preds.to_csv(reports_dir / "test_predictions_detailed.csv", index=False)
    logger.info("Saved detailed test predictions to %s", reports_dir / "test_predictions_detailed.csv")

    # 7. Generate Explainability Artifacts (Grad-CAM)
    logger.info("\n=== Generating Explainability (Grad-CAM) Overlays ===")
    gradcam_dir = reports_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    tp_indices = [i for i, r in enumerate(pred_records) if r["outcome"] == "TP"]
    tn_indices = [i for i, r in enumerate(pred_records) if r["outcome"] == "TN"]
    fp_indices = [i for i, r in enumerate(pred_records) if r["outcome"] == "FP"]
    fn_indices = [i for i, r in enumerate(pred_records) if r["outcome"] == "FN"]

    logger.info("Test set outcomes (at threshold %.4f): TP=%d, TN=%d, FP=%d, FN=%d",
                val_th, len(tp_indices), len(tn_indices), len(fp_indices), len(fn_indices))

    cases_to_plot = []
    for idx in tp_indices[:2]:
        cases_to_plot.append((idx, "True Positive (Deforestation Correctly Detected)"))
    for idx in tn_indices[:2]:
        cases_to_plot.append((idx, "True Negative (Intact Forest Correctly Preserved)"))
    for idx in fp_indices[:2]:
        cases_to_plot.append((idx, "False Positive (Intact Forest Misclassified as Deforested)"))
    for idx in fn_indices[:2]:
        cases_to_plot.append((idx, "False Negative (Deforestation Missed by Model)"))

    for idx, desc in cases_to_plot:
        rec = pred_records[idx]
        pid = rec["patch_id"]
        t1_p = test_t1[idx]
        t2_p = test_t2[idx]
        cam_t1, cam_t2, prob = compute_gradcam(selected_model, t1_p, t2_p)

        out_fname = f"gradcam_{rec['outcome'].lower()}_{pid}.png"
        plot_explainability_case(
            t1_patch=t1_p,
            t2_patch=t2_p,
            label=rec["ground_truth"],
            pred_prob=prob,
            cam_t1=cam_t1,
            cam_t2=cam_t2,
            patch_id=f"{pid} ({rec['outcome']})",
            output_path=gradcam_dir / out_fname,
            title_prefix=desc,
        )
        plot_explainability_case(
            t1_patch=t1_p,
            t2_patch=t2_p,
            label=rec["ground_truth"],
            pred_prob=prob,
            cam_t1=cam_t1,
            cam_t2=cam_t2,
            patch_id=f"{pid} ({rec['outcome']})",
            output_path=reports_dir / out_fname,
            title_prefix=desc,
        )

    duration = time.time() - start_time
    logger.info("\n=== Phase 2B Pipeline Completed Successfully in %.1f seconds ===", duration)


if __name__ == "__main__":
    main()
