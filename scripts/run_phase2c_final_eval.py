"""Final Phase 2C evaluation of the validation-selected model on the virgin held-out test split."""

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from geospatial_ai.dataset import create_tf_dataset, get_split_records, load_numpy_arrays
from geospatial_ai.metrics import (
    evaluate_predictions,
    plot_confusion_matrix_figure,
    plot_roc_pr_curves_figure,
)
from geospatial_ai.explainability import compute_gradcam, plot_explainability_case, compute_ndvi
from geospatial_ai.model import build_siamese_network
from geospatial_ai.train import set_seed, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase2c_final_eval")


def main() -> None:
    logger.info("=== Starting Phase 2C Final Evaluation ===")
    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"
    experiments_dir = PROJECT_ROOT / "experiments" / "phase2c"

    # 1. Load Data
    val_records = get_split_records(manifest_path, "val", project_root=PROJECT_ROOT)
    test_records = get_split_records(manifest_path, "test", project_root=PROJECT_ROOT)
    val_t1, val_t2, val_y = load_numpy_arrays(val_records)
    test_t1, test_t2, test_y = load_numpy_arrays(test_records)

    logger.info("Val samples: %d (Pos=%d, Neg=%d)", len(val_records), int(np.sum(val_y==1)), int(np.sum(val_y==0)))
    logger.info("Test samples: %d (Pos=%d, Neg=%d)", len(test_records), int(np.sum(test_y==1)), int(np.sum(test_y==0)))

    # 2. Load the 3 Seeds of the Champion Architecture (M1_Siamese_Difference)
    seed_ckpts = [
        PROJECT_ROOT / "models" / "best_siamese_model.keras", # Seed 42 (Champion M1 from Phase 2B)
        PROJECT_ROOT / "experiments" / "checkpoint_m1_seed_101.keras",
        PROJECT_ROOT / "experiments" / "checkpoint_m1_seed_2024.keras",
    ]

    models_loaded = []
    for p in seed_ckpts:
        logger.info("Loading model checkpoint: %s", p)
        m = keras.models.load_model(str(p))
        models_loaded.append(m)

    # 3. Validation Evaluation of Single Best (Seed 42) vs 3-Seed Ensemble
    val_probs_s42 = models_loaded[0].predict({"input_t1": val_t1, "input_t2": val_t2}, verbose=0)[:, 0]
    val_probs_ens = np.mean([
        m.predict({"input_t1": val_t1, "input_t2": val_t2}, verbose=0)[:, 0] for m in models_loaded
    ], axis=0)

    val_th_s42 = find_optimal_threshold(val_y, val_probs_s42)
    val_th_ens = find_optimal_threshold(val_y, val_probs_ens)

    val_metrics_s42 = evaluate_predictions(val_y, val_probs_s42, threshold=val_th_s42)
    val_metrics_ens = evaluate_predictions(val_y, val_probs_ens, threshold=val_th_ens)

    logger.info("Validation (Seed 42, Thresh=%.3f): PR-AUC=%.4f, ROC-AUC=%.4f, BalAcc=%.4f, Acc=%.4f, F1=%.4f",
                val_th_s42, val_metrics_s42["pr_auc"], val_metrics_s42["roc_auc"],
                val_metrics_s42["balanced_accuracy"], val_metrics_s42["accuracy"], val_metrics_s42["f1_score"])
    logger.info("Validation (Ensemble, Thresh=%.3f): PR-AUC=%.4f, ROC-AUC=%.4f, BalAcc=%.4f, Acc=%.4f, F1=%.4f",
                val_th_ens, val_metrics_ens["pr_auc"], val_metrics_ens["roc_auc"],
                val_metrics_ens["balanced_accuracy"], val_metrics_ens["accuracy"], val_metrics_ens["f1_score"])

    # 4. SINGLE HELD-OUT TEST EVALUATION
    logger.info("\n=== EVALUATING ON VIRGIN HELD-OUT TEST SPLIT (N=60) ===")
    test_probs_s42 = models_loaded[0].predict({"input_t1": test_t1, "input_t2": test_t2}, verbose=0)[:, 0]
    test_probs_ens = np.mean([
        m.predict({"input_t1": test_t1, "input_t2": test_t2}, verbose=0)[:, 0] for m in models_loaded
    ], axis=0)

    # Metrics for Single Model (Seed 42)
    test_res_s42_05 = evaluate_predictions(test_y, test_probs_s42, threshold=0.50)
    test_res_s42_opt = evaluate_predictions(test_y, test_probs_s42, threshold=val_th_s42)

    # Metrics for Ensemble Model
    test_res_ens_05 = evaluate_predictions(test_y, test_probs_ens, threshold=0.50)
    test_res_ens_opt = evaluate_predictions(test_y, test_probs_ens, threshold=val_th_ens)

    logger.info("\n=== SINGLE MODEL (Seed 42) TEST RESULTS ===")
    logger.info("At Threshold 0.50: Acc=%.2f%% [95%% CI: %.2f%% - %.2f%%], Prec=%.2f%%, Rec=%.2f%%, Spec=%.2f%%, F1=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f",
                test_res_s42_05["accuracy"] * 100, test_res_s42_05["accuracy_ci_95"][0] * 100, test_res_s42_05["accuracy_ci_95"][1] * 100,
                test_res_s42_05["precision"] * 100, test_res_s42_05["recall"] * 100, test_res_s42_05["specificity"] * 100,
                test_res_s42_05["f1_score"], test_res_s42_05["pr_auc"], test_res_s42_05["roc_auc"])
    logger.info("Confusion Matrix: %s", test_res_s42_05["confusion_matrix"])

    logger.info("\n=== 3-SEED ENSEMBLE TEST RESULTS ===")
    logger.info("At Threshold 0.50: Acc=%.2f%% [95%% CI: %.2f%% - %.2f%%], Prec=%.2f%%, Rec=%.2f%%, Spec=%.2f%%, F1=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f",
                test_res_ens_05["accuracy"] * 100, test_res_ens_05["accuracy_ci_95"][0] * 100, test_res_ens_05["accuracy_ci_95"][1] * 100,
                test_res_ens_05["precision"] * 100, test_res_ens_05["recall"] * 100, test_res_ens_05["specificity"] * 100,
                test_res_ens_05["f1_score"], test_res_ens_05["pr_auc"], test_res_ens_05["roc_auc"])
    logger.info("Confusion Matrix (0.50): %s", test_res_ens_05["confusion_matrix"])

    logger.info("At Validation Threshold (%.3f): Acc=%.2f%% [95%% CI: %.2f%% - %.2f%%], Prec=%.2f%%, Rec=%.2f%%, Spec=%.2f%%, F1=%.4f",
                val_th_ens, test_res_ens_opt["accuracy"] * 100, test_res_ens_opt["accuracy_ci_95"][0] * 100, test_res_ens_opt["accuracy_ci_95"][1] * 100,
                test_res_ens_opt["precision"] * 100, test_res_ens_opt["recall"] * 100, test_res_ens_opt["specificity"] * 100,
                test_res_ens_opt["f1_score"])
    logger.info("Confusion Matrix (opt): %s", test_res_ens_opt["confusion_matrix"])

    # 5. Save Phase 2C Test Metrics JSON
    phase2c_metrics = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_count": 60,
        "n_deforestation": 26,
        "n_intact_forest": 34,
        "majority_class_baseline": {
            "class": "Intact Forest (0)",
            "accuracy": 0.5667,
        },
        "phase2b_baseline_single_model": test_res_s42_05,
        "phase2c_ensemble_model": {
            "num_seeds": 3,
            "seeds": [42, 101, 2024],
            "threshold_0_50": test_res_ens_05,
            "threshold_val_opt": {
                "threshold": val_th_ens,
                "metrics": test_res_ens_opt,
            },
        },
    }

    with open(reports_dir / "phase2c_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(phase2c_metrics, f, indent=2)

    # 6. Save Plots
    plot_confusion_matrix_figure(
        test_res_ens_05["confusion_matrix"],
        reports_dir / "phase2c_confusion_matrix_ensemble.png",
        title="Phase 2C Test Set Confusion Matrix (3-Seed Ensemble)\nThreshold = 0.50 | N=60 Patches",
    )
    plot_roc_pr_curves_figure(
        test_y,
        test_probs_ens,
        reports_dir / "phase2c_roc_pr_curves_ensemble.png",
    )

    # 7. Detailed Predictions CSV
    pred_records = []
    y_pred_binary = (test_probs_ens >= 0.50).astype(int)

    for i, rec in enumerate(test_records):
        y_t = int(test_y[i])
        y_p = int(y_pred_binary[i])
        prob = float(test_probs_ens[i])

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
            "ground_truth": y_t,
            "ensemble_pred_binary": y_p,
            "ensemble_pred_probability": round(prob, 4),
            "outcome": cat,
            "mean_delta_ndvi": round(d_ndvi, 4),
        })

    df_preds = pd.DataFrame(pred_records)
    df_preds.to_csv(reports_dir / "phase2c_predictions_detailed.csv", index=False)
    logger.info("Saved detailed predictions to %s", reports_dir / "phase2c_predictions_detailed.csv")

    logger.info("=== Phase 2C Final Evaluation Complete ===")


if __name__ == "__main__":
    main()
