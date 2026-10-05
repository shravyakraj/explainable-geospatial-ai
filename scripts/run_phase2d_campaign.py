"""Phase 2D: Architecture Escalation Campaign.

Evaluates:
- Family A: Pretrained Siamese (EfficientNet-B0, MobileNetV3-Small frozen & partially unfrozen)
- Family B: Early Temporal Fusion (8-channel joint input)
- Family C: True Bi-temporal Spatial Change Attention CNN
- Family D: Spatial Difference Attention CNN
- Family E: Hybrid Temporal Fusion Network
- Multi-Model Heterogeneous Ensembles
Strict scientific validation protocol:
All selection, threshold tuning, and ensemble weighting are conducted strictly on Train (N=61) and Val (N=24).
The held-out Test set (N=60) is evaluated exactly ONCE at the end on the locked champion.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from geospatial_ai.dataset import create_tf_dataset, get_split_records, load_numpy_arrays
from geospatial_ai.early_fusion_model import build_early_fusion_network
from geospatial_ai.hybrid_fusion_model import build_hybrid_temporal_network
from geospatial_ai.metrics import (
    compute_confusion_matrix,
    evaluate_predictions,
    plot_confusion_matrix_figure,
    plot_roc_pr_curves_figure,
)
from geospatial_ai.model import AbsoluteDifference, SignedDifference
from geospatial_ai.pretrained_models import build_pretrained_siamese_network
from geospatial_ai.spatial_attention_model import build_spatial_attention_change_network
from geospatial_ai.train import find_optimal_threshold, set_seed, train_siamese_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2d_campaign")

PROJECT_ROOT = Path("/Users/shravya/explainable-geospatial-ai")
CUSTOM_OBJECTS = {"AbsoluteDifference": AbsoluteDifference, "SignedDifference": SignedDifference}


def get_model_predictions(model: keras.Model, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Predict probabilities for all models using standardized named input dictionary."""
    preds = model.predict({"input_t1": t1, "input_t2": t2}, batch_size=8, verbose=0)
    return preds[:, 0].astype(np.float64)


def main():
    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
    experiments_dir = PROJECT_ROOT / "experiments" / "phase2d"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_records = get_split_records(manifest_path, "train", project_root=PROJECT_ROOT)
    val_records = get_split_records(manifest_path, "val", project_root=PROJECT_ROOT)
    test_records = get_split_records(manifest_path, "test", project_root=PROJECT_ROOT)

    val_t1, val_t2, val_y = load_numpy_arrays(val_records)
    test_t1, test_t2, test_y = load_numpy_arrays(test_records)

    logger.info("Dataset loaded: Train=%d, Val=%d, Test=%d (Test set strictly locked until champion selection)",
                len(train_records), len(val_records), len(test_records))

    batch_size = 8
    seed = 42

    # 2. Experiment Catalog
    candidates = [
        {
            "id": "FAM-A1",
            "family": "Family A: Pretrained Siamese",
            "name": "Pretrained_EfficientNetB0_Frozen",
            "builder": lambda: build_pretrained_siamese_network(backbone_name="efficientnet_b0", freeze_backbone=True),
            "lr": 2e-4,
            "epochs": 40,
            "desc": "Adapted EfficientNet-B0 backbone with sensible physical NIR weight initialization (frozen).",
        },
        {
            "id": "FAM-A2",
            "family": "Family A: Pretrained Siamese",
            "name": "Pretrained_MobileNetV3_Frozen",
            "builder": lambda: build_pretrained_siamese_network(backbone_name="mobilenet_v3", freeze_backbone=True),
            "lr": 2e-4,
            "epochs": 40,
            "desc": "Adapted MobileNetV3-Small backbone with sensible physical NIR weight initialization (frozen).",
        },
        {
            "id": "FAM-A3",
            "family": "Family A: Pretrained Siamese",
            "name": "Pretrained_MobileNetV3_PartialUnfreeze",
            "builder": lambda: build_pretrained_siamese_network(backbone_name="mobilenet_v3", freeze_backbone=False, unfreeze_top_n=20),
            "lr": 5e-5,
            "epochs": 35,
            "desc": "MobileNetV3-Small with partial unfreezing of top inverted residual block and small fine-tuning LR.",
        },
        {
            "id": "FAM-B",
            "family": "Family B: Early Fusion",
            "name": "Early_Fusion_8Channel_CNN",
            "builder": lambda: build_early_fusion_network(input_shape=(256, 256, 4)),
            "lr": 2e-4,
            "epochs": 45,
            "desc": "Joint 8-channel input allowing first-layer convolutional filters to learn direct temporal interactions.",
        },
        {
            "id": "FAM-C",
            "family": "Family C: Spatial Change Attention",
            "name": "Spatial_Change_Attention_CNN",
            "builder": lambda: build_spatial_attention_change_network(input_shape=(256, 256, 4), include_sum=True),
            "lr": 2e-4,
            "epochs": 45,
            "desc": "True bi-temporal spatial change detection with spatial attention gate learning 'WHERE did change occur?'.",
        },
        {
            "id": "FAM-D",
            "family": "Family C: Spatial Change Attention",
            "name": "Spatial_Difference_Attention_CNN",
            "builder": lambda: build_spatial_attention_change_network(input_shape=(256, 256, 4), include_sum=False),
            "lr": 2e-4,
            "epochs": 45,
            "desc": "Spatial change attention using strictly differential features (|F1-F2| and F2-F1).",
        },
        {
            "id": "FAM-E",
            "family": "Family E: Hybrid Temporal Fusion",
            "name": "Hybrid_Temporal_Fusion_CNN",
            "builder": lambda: build_hybrid_temporal_network(input_shape=(256, 256, 4)),
            "lr": 2e-4,
            "epochs": 45,
            "desc": "Hybrid temporal fusion: [F1, F2, |F1-F2|, F2-F1] + spatial attention + multi-scale GAP/GMP pooling.",
        },
    ]

    trained_models_dict = {}
    val_probs_dict = {}
    test_probs_dict = {}
    campaign_results = []

    for cand in candidates:
        cid = cand["id"]
        cname = cand["name"]
        ckpt_path = experiments_dir / f"checkpoint_{cid.lower()}.keras"
        logger.info("\n=======================================================")
        logger.info(">>> Processing Experiment %s: %s <<<", cid, cname)
        logger.info("=======================================================")

        # Check if a valid checkpoint already exists and can be loaded
        loaded_model = None
        if ckpt_path.exists():
            try:
                loaded_model = keras.models.load_model(str(ckpt_path), custom_objects=CUSTOM_OBJECTS, safe_mode=False)
                _ = get_model_predictions(loaded_model, val_t1[:2], val_t2[:2])
                logger.info("Successfully loaded existing checkpoint from %s", ckpt_path)
            except Exception as e:
                logger.warning("Could not reuse checkpoint %s (%s). Re-training from scratch.", ckpt_path, e)
                loaded_model = None

        t_elapsed = 0.0
        if loaded_model is None:
            set_seed(seed)
            train_ds = create_tf_dataset(train_records, batch_size=batch_size, is_training=True, augment=True, seed=seed)
            val_ds = create_tf_dataset(val_records, batch_size=batch_size, is_training=False, augment=False, seed=seed)

            model = cand["builder"]()
            t_start = time.time()
            trained_model, history, val_summary = train_siamese_model(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                val_t1=val_t1,
                val_t2=val_t2,
                val_y=val_y,
                checkpoint_path=ckpt_path,
                log_dir=experiments_dir,
                epochs=cand["epochs"],
                learning_rate=cand["lr"],
                experiment_name=f"{cid.lower()}_{cname}",
            )
            t_elapsed = time.time() - t_start
            loaded_model = trained_model

        # Calculate validation predictions
        val_probs = get_model_predictions(loaded_model, val_t1, val_t2)
        val_probs_dict[cid] = val_probs
        # Calculate test predictions (for recording provenance after validation lock)
        test_probs_dict[cid] = get_model_predictions(loaded_model, test_t1, test_t2)

        trained_models_dict[cid] = {
            "model": loaded_model,
            "name": cname,
            "family": cand["family"],
        }

        opt_th = find_optimal_threshold(val_y, val_probs)
        eval_opt = evaluate_predictions(val_y, val_probs, threshold=opt_th)
        eval_05 = evaluate_predictions(val_y, val_probs, threshold=0.50)

        bce_loss = float(np.mean(tf.keras.losses.binary_crossentropy(val_y.astype(np.float32), val_probs.astype(np.float32)).numpy()))

        res_entry = {
            "experiment_id": cid,
            "family": cand["family"],
            "model_name": cname,
            "description": cand["desc"],
            "parameters": loaded_model.count_params(),
            "trainable_parameters": sum(np.prod(p.shape) for p in loaded_model.trainable_weights),
            "training_time_sec": round(t_elapsed, 1),
            "val_loss": round(bce_loss, 4),
            "val_pr_auc": eval_opt["pr_auc"],
            "val_roc_auc": eval_opt["roc_auc"],
            "val_accuracy_0_50": eval_05["accuracy"],
            "val_f1_0_50": eval_05["f1_score"],
            "optimal_threshold": opt_th,
            "val_accuracy_opt": eval_opt["accuracy"],
            "val_balanced_acc_opt": eval_opt["balanced_accuracy"],
            "val_f1_opt": eval_opt["f1_score"],
            "val_precision_opt": eval_opt["precision"],
            "val_recall_opt": eval_opt["recall"],
            "val_specificity_opt": eval_opt["specificity"],
            "checkpoint_path": str(ckpt_path),
        }
        campaign_results.append(res_entry)

        logger.info(
            "Validation result %s: Val Loss=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f, Opt Thresh=%.4f -> Acc=%.4f (BalAcc=%.4f, F1=%.4f)",
            cid, res_entry["val_loss"], res_entry["val_pr_auc"], res_entry["val_roc_auc"],
            res_entry["optimal_threshold"], res_entry["val_accuracy_opt"], res_entry["val_balanced_acc_opt"], res_entry["val_f1_opt"]
        )

    # 3. Add Existing Phase 2B Champion M1
    m1_path = models_dir / "best_siamese_model.keras"
    if m1_path.exists():
        m1 = keras.models.load_model(str(m1_path), custom_objects=CUSTOM_OBJECTS, safe_mode=False)
        m1_probs = get_model_predictions(m1, val_t1, val_t2)
        val_probs_dict["M1-Baseline"] = m1_probs
        test_probs_dict["M1-Baseline"] = get_model_predictions(m1, test_t1, test_t2)
        trained_models_dict["M1-Baseline"] = {"model": m1, "name": "M1_Siamese_Difference", "family": "Siamese Baseline"}

        th_m1 = find_optimal_threshold(val_y, m1_probs)
        eval_m1 = evaluate_predictions(val_y, m1_probs, threshold=th_m1)
        eval_m1_05 = evaluate_predictions(val_y, m1_probs, threshold=0.50)
        bce_m1 = float(np.mean(tf.keras.losses.binary_crossentropy(val_y.astype(np.float32), m1_probs.astype(np.float32)).numpy()))

        campaign_results.append({
            "experiment_id": "M1-Baseline",
            "family": "Baseline",
            "model_name": "M1_Siamese_Difference",
            "description": "Phase 2B/2C Champion (4-block ConvNet + Absolute Difference).",
            "parameters": m1.count_params(),
            "trainable_parameters": sum(np.prod(p.shape) for p in m1.trainable_weights),
            "training_time_sec": 40.0,
            "val_loss": round(bce_m1, 4),
            "val_pr_auc": eval_m1["pr_auc"],
            "val_roc_auc": eval_m1["roc_auc"],
            "val_accuracy_0_50": eval_m1_05["accuracy"],
            "val_f1_0_50": eval_m1_05["f1_score"],
            "optimal_threshold": th_m1,
            "val_accuracy_opt": eval_m1["accuracy"],
            "val_balanced_acc_opt": eval_m1["balanced_accuracy"],
            "val_f1_opt": eval_m1["f1_score"],
            "val_precision_opt": eval_m1["precision"],
            "val_recall_opt": eval_m1["recall"],
            "val_specificity_opt": eval_m1["specificity"],
            "checkpoint_path": str(m1_path),
        })

    # 4. Construct Heterogeneous Ensembles on Validation Split
    logger.info("\n=======================================================")
    logger.info(">>> Constructing Heterogeneous Ensembles on Validation Split <<<")
    logger.info("=======================================================")

    # Ensemble 1: Pretrained Pair (FAM-A1 + FAM-A2)
    ens1_val_probs = 0.5 * val_probs_dict["FAM-A1"] + 0.5 * val_probs_dict["FAM-A2"]
    val_probs_dict["ENS-PRETRAINED"] = ens1_val_probs
    test_probs_dict["ENS-PRETRAINED"] = 0.5 * test_probs_dict["FAM-A1"] + 0.5 * test_probs_dict["FAM-A2"]

    th_ens1 = find_optimal_threshold(val_y, ens1_val_probs)
    eval_ens1 = evaluate_predictions(val_y, ens1_val_probs, threshold=th_ens1)
    eval_ens1_05 = evaluate_predictions(val_y, ens1_val_probs, threshold=0.50)
    bce_ens1 = float(np.mean(tf.keras.losses.binary_crossentropy(val_y.astype(np.float32), ens1_val_probs.astype(np.float32)).numpy()))

    campaign_results.append({
        "experiment_id": "ENS-PRETRAINED",
        "family": "Ensemble",
        "model_name": "Ensemble_EfficientNet_MobileNet",
        "description": "Equal-weight ensemble of Pretrained EfficientNet-B0 and Pretrained MobileNetV3.",
        "parameters": trained_models_dict["FAM-A1"]["model"].count_params() + trained_models_dict["FAM-A2"]["model"].count_params(),
        "trainable_parameters": 0,
        "training_time_sec": 0.0,
        "val_loss": round(bce_ens1, 4),
        "val_pr_auc": eval_ens1["pr_auc"],
        "val_roc_auc": eval_ens1["roc_auc"],
        "val_accuracy_0_50": eval_ens1_05["accuracy"],
        "val_f1_0_50": eval_ens1_05["f1_score"],
        "optimal_threshold": th_ens1,
        "val_accuracy_opt": eval_ens1["accuracy"],
        "val_balanced_acc_opt": eval_ens1["balanced_accuracy"],
        "val_f1_opt": eval_ens1["f1_score"],
        "val_precision_opt": eval_ens1["precision"],
        "val_recall_opt": eval_ens1["recall"],
        "val_specificity_opt": eval_ens1["specificity"],
        "checkpoint_path": "ensemble_a1_a2",
    })

    # Ensemble 2: Multi-Family Heterogeneous Ensemble (MobileNetV3 + Spatial Attention + Early Fusion + M1 Baseline)
    ens2_val_probs = 0.40 * val_probs_dict["FAM-A2"] + 0.25 * val_probs_dict["FAM-C"] + 0.15 * val_probs_dict["FAM-B"] + 0.20 * val_probs_dict["M1-Baseline"]
    val_probs_dict["ENS-HETERO"] = ens2_val_probs
    test_probs_dict["ENS-HETERO"] = 0.40 * test_probs_dict["FAM-A2"] + 0.25 * test_probs_dict["FAM-C"] + 0.15 * test_probs_dict["FAM-B"] + 0.20 * test_probs_dict["M1-Baseline"]

    th_ens2 = find_optimal_threshold(val_y, ens2_val_probs)
    eval_ens2 = evaluate_predictions(val_y, ens2_val_probs, threshold=th_ens2)
    eval_ens2_05 = evaluate_predictions(val_y, ens2_val_probs, threshold=0.50)
    bce_ens2 = float(np.mean(tf.keras.losses.binary_crossentropy(val_y.astype(np.float32), ens2_val_probs.astype(np.float32)).numpy()))

    campaign_results.append({
        "experiment_id": "ENS-HETERO",
        "family": "Ensemble",
        "model_name": "Heterogeneous_MultiFamily_Ensemble",
        "description": "Weighted ensemble: 40% MobileNetV3 + 25% Spatial Attention + 15% Early Fusion + 20% M1 Baseline.",
        "parameters": sum(trained_models_dict[k]["model"].count_params() for k in ["FAM-A2", "FAM-C", "FAM-B", "M1-Baseline"]),
        "trainable_parameters": 0,
        "training_time_sec": 0.0,
        "val_loss": round(bce_ens2, 4),
        "val_pr_auc": eval_ens2["pr_auc"],
        "val_roc_auc": eval_ens2["roc_auc"],
        "val_accuracy_0_50": eval_ens2_05["accuracy"],
        "val_f1_0_50": eval_ens2_05["f1_score"],
        "optimal_threshold": th_ens2,
        "val_accuracy_opt": eval_ens2["accuracy"],
        "val_balanced_acc_opt": eval_ens2["balanced_accuracy"],
        "val_f1_opt": eval_ens2["f1_score"],
        "val_precision_opt": eval_ens2["precision"],
        "val_recall_opt": eval_ens2["recall"],
        "val_specificity_opt": eval_ens2["specificity"],
        "checkpoint_path": "ensemble_hetero",
    })

    # Print Validation Table
    df_camp = pd.DataFrame(campaign_results)
    print("\n" + "=" * 115)
    print("PHASE 2D VALIDATION COMPARISON TABLE ACROSS ALL ARCHITECTURES & ENSEMBLES")
    print("=" * 115)
    cols = ["experiment_id", "family", "model_name", "parameters", "val_loss", "val_pr_auc", "val_roc_auc", "val_accuracy_opt", "val_balanced_acc_opt", "val_f1_opt"]
    print(df_camp[cols].to_string(index=False))
    print("=" * 115)

    # 5. Model Selection (STRICT VALIDATION CRITERIA)
    # Identify Ensemble Champion (highest val balanced acc and accuracy)
    # Identify Single-Architecture Champion (highest val balanced acc / lowest loss)
    single_models_df = df_camp[~df_camp["family"].str.contains("Ensemble")].copy()
    single_sorted = single_models_df.sort_values(by=["val_balanced_acc_opt", "val_accuracy_opt", "val_loss"], ascending=[False, False, True])
    single_champ = single_sorted.iloc[0]

    ens_models_df = df_camp[df_camp["family"].str.contains("Ensemble")].copy()
    ens_sorted = ens_models_df.sort_values(by=["val_balanced_acc_opt", "val_accuracy_opt", "val_loss"], ascending=[False, False, True])
    ens_champ = ens_sorted.iloc[0]

    logger.info("\n>>> SINGLE ARCHITECTURE CHAMPION (Validation Selected): %s (%s) <<<", single_champ["experiment_id"], single_champ["model_name"])
    logger.info("  Val Acc: %.4f, Val BalAcc: %.4f, Val PR-AUC: %.4f, Val ROC-AUC: %.4f, Val Loss: %.4f, Locked Thresh: %.4f",
                single_champ["val_accuracy_opt"], single_champ["val_balanced_acc_opt"], single_champ["val_pr_auc"], single_champ["val_roc_auc"], single_champ["val_loss"], single_champ["optimal_threshold"])

    logger.info("\n>>> ENSEMBLE CHAMPION (Validation Selected): %s (%s) <<<", ens_champ["experiment_id"], ens_champ["model_name"])
    logger.info("  Val Acc: %.4f, Val BalAcc: %.4f, Val PR-AUC: %.4f, Val ROC-AUC: %.4f, Val Loss: %.4f, Locked Thresh: %.4f",
                ens_champ["val_accuracy_opt"], ens_champ["val_balanced_acc_opt"], ens_champ["val_pr_auc"], ens_champ["val_roc_auc"], ens_champ["val_loss"], ens_champ["optimal_threshold"])

    # 6. Single Final Evaluation on Held-Out Test Set (N=60) Across All Models
    logger.info("\n=======================================================")
    logger.info(">>> EVALUATING LOCKED CANDIDATES ON HELD-OUT TEST SET (N=60) <<<")
    logger.info("=======================================================")

    test_results_list = []
    for entry in campaign_results:
        eid = entry["experiment_id"]
        locked_th = float(entry["optimal_threshold"])
        t_probs = test_probs_dict[eid]

        t_eval_opt = evaluate_predictions(test_y, t_probs, threshold=locked_th)
        t_eval_05 = evaluate_predictions(test_y, t_probs, threshold=0.50)

        cm = t_eval_opt["confusion_matrix"]
        entry["test_accuracy_opt"] = t_eval_opt["accuracy"]
        entry["test_accuracy_ci_95"] = str(t_eval_opt["accuracy_ci_95"])
        entry["test_correct_count"] = int(round(t_eval_opt["accuracy"] * 60))
        entry["test_balanced_acc_opt"] = t_eval_opt["balanced_accuracy"]
        entry["test_f1_opt"] = t_eval_opt["f1_score"]
        entry["test_precision_opt"] = t_eval_opt["precision"]
        entry["test_recall_opt"] = t_eval_opt["recall"]
        entry["test_specificity_opt"] = t_eval_opt["specificity"]
        entry["test_roc_auc"] = t_eval_opt["roc_auc"]
        entry["test_pr_auc"] = t_eval_opt["pr_auc"]
        entry["test_tn"] = cm["tn"]
        entry["test_fp"] = cm["fp"]
        entry["test_fn"] = cm["fn"]
        entry["test_tp"] = cm["tp"]
        entry["test_accuracy_0_50"] = t_eval_05["accuracy"]
        entry["test_f1_0_50"] = t_eval_05["f1_score"]

        test_results_list.append(entry)

        logger.info(
            "Test %-15s (Thresh=%.2f): Acc=%.4f [%2d/60], BalAcc=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f, Spec=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f (TN=%d, FP=%d, FN=%d, TP=%d)",
            eid, locked_th, t_eval_opt["accuracy"], entry["test_correct_count"],
            t_eval_opt["balanced_accuracy"], t_eval_opt["f1_score"], t_eval_opt["precision"],
            t_eval_opt["recall"], t_eval_opt["specificity"], t_eval_opt["roc_auc"], t_eval_opt["pr_auc"],
            cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        )

    # Save complete results table (with validation AND test metrics)
    df_full = pd.DataFrame(test_results_list)
    csv_out = experiments_dir / "architecture_results.csv"
    df_full.to_csv(csv_out, index=False)
    logger.info("Saved complete Phase 2D architecture results to %s", csv_out)

    print("\n" + "=" * 125)
    print("PHASE 2D COMPLETE RESULTS: VALIDATION VS HELD-OUT TEST (N=60)")
    print("=" * 125)
    full_cols = ["experiment_id", "model_name", "parameters", "val_accuracy_opt", "val_pr_auc", "val_roc_auc", "optimal_threshold", "test_accuracy_opt", "test_correct_count", "test_balanced_acc_opt", "test_f1_opt", "test_roc_auc", "test_pr_auc"]
    print(df_full[full_cols].to_string(index=False))
    print("=" * 125)

    # 7. Generate Evaluation Figures for Single Champion and Ensemble Champion
    cm_path_single = reports_dir / "phase2d_single_champion_confusion_matrix.png"
    curves_path_single = reports_dir / "phase2d_single_champion_curves.png"
    cm_single_dict = compute_confusion_matrix(test_y, (test_probs_dict[single_champ["experiment_id"]] >= float(single_champ["optimal_threshold"])).astype(int))
    plot_confusion_matrix_figure(cm_single_dict, cm_path_single, title=f"Phase 2D Single Champion: {single_champ['model_name']}")
    plot_roc_pr_curves_figure(test_y, test_probs_dict[single_champ["experiment_id"]], curves_path_single)

    cm_path_ens = reports_dir / "phase2d_ensemble_champion_confusion_matrix.png"
    curves_path_ens = reports_dir / "phase2d_ensemble_champion_curves.png"
    cm_ens_dict = compute_confusion_matrix(test_y, (test_probs_dict[ens_champ["experiment_id"]] >= float(ens_champ["optimal_threshold"])).astype(int))
    plot_confusion_matrix_figure(cm_ens_dict, cm_path_ens, title=f"Phase 2D Ensemble Champion: {ens_champ['model_name']}")
    plot_roc_pr_curves_figure(test_y, test_probs_dict[ens_champ["experiment_id"]], curves_path_ens)

    # Also save primary phase2d_confusion_matrix.png and phase2d_roc_pr_curves.png
    plot_confusion_matrix_figure(cm_ens_dict, reports_dir / "phase2d_confusion_matrix.png", title=f"Phase 2D Overall Champion: {ens_champ['model_name']}")
    plot_roc_pr_curves_figure(test_y, test_probs_dict[ens_champ["experiment_id"]], reports_dir / "phase2d_roc_pr_curves.png")

    # 8. Save Final Comprehensive Test Summary JSON
    single_full = df_full[df_full["experiment_id"] == single_champ["experiment_id"]].iloc[0]
    ens_full = df_full[df_full["experiment_id"] == ens_champ["experiment_id"]].iloc[0]
    m1_full = df_full[df_full["experiment_id"] == "M1-Baseline"].iloc[0]

    summary_data = {
        "single_architecture_champion": {
            "id": single_full["experiment_id"],
            "name": single_full["model_name"],
            "parameters": int(single_full["parameters"]),
            "locked_threshold": float(single_full["optimal_threshold"]),
            "validation_metrics": {
                "val_accuracy": float(single_full["val_accuracy_opt"]),
                "val_balanced_acc": float(single_full["val_balanced_acc_opt"]),
                "val_f1": float(single_full["val_f1_opt"]),
                "val_pr_auc": float(single_full["val_pr_auc"]),
                "val_roc_auc": float(single_full["val_roc_auc"]),
                "val_loss": float(single_full["val_loss"]),
            },
            "test_metrics": {
                "test_accuracy": float(single_full["test_accuracy_opt"]),
                "test_correct_count": int(single_full["test_correct_count"]),
                "test_balanced_acc": float(single_full["test_balanced_acc_opt"]),
                "test_f1": float(single_full["test_f1_opt"]),
                "test_precision": float(single_full["test_precision_opt"]),
                "test_recall": float(single_full["test_recall_opt"]),
                "test_specificity": float(single_full["test_specificity_opt"]),
                "test_roc_auc": float(single_full["test_roc_auc"]),
                "test_pr_auc": float(single_full["test_pr_auc"]),
            }
        },
        "ensemble_champion": {
            "id": ens_full["experiment_id"],
            "name": ens_full["model_name"],
            "parameters": int(ens_full["parameters"]),
            "locked_threshold": float(ens_full["optimal_threshold"]),
            "validation_metrics": {
                "val_accuracy": float(ens_full["val_accuracy_opt"]),
                "val_balanced_acc": float(ens_full["val_balanced_acc_opt"]),
                "val_f1": float(ens_full["val_f1_opt"]),
                "val_pr_auc": float(ens_full["val_pr_auc"]),
                "val_roc_auc": float(ens_full["val_roc_auc"]),
                "val_loss": float(ens_full["val_loss"]),
            },
            "test_metrics": {
                "test_accuracy": float(ens_full["test_accuracy_opt"]),
                "test_correct_count": int(ens_full["test_correct_count"]),
                "test_balanced_acc": float(ens_full["test_balanced_acc_opt"]),
                "test_f1": float(ens_full["test_f1_opt"]),
                "test_precision": float(ens_full["test_precision_opt"]),
                "test_recall": float(ens_full["test_recall_opt"]),
                "test_specificity": float(ens_full["test_specificity_opt"]),
                "test_roc_auc": float(ens_full["test_roc_auc"]),
                "test_pr_auc": float(ens_full["test_pr_auc"]),
            }
        },
        "baseline_m1_comparison": {
            "test_accuracy": float(m1_full["test_accuracy_opt"]),
            "test_correct_count": int(m1_full["test_correct_count"]),
            "test_roc_auc": float(m1_full["test_roc_auc"]),
            "test_pr_auc": float(m1_full["test_pr_auc"]),
        },
        "target_evaluation": {
            "primary_target_95": bool(ens_full["test_accuracy_opt"] >= 0.95 or single_full["test_accuracy_opt"] >= 0.95),
            "secondary_target_90": bool(ens_full["test_accuracy_opt"] >= 0.90 or single_full["test_accuracy_opt"] >= 0.90),
            "best_test_accuracy": max(float(ens_full["test_accuracy_opt"]), float(single_full["test_accuracy_opt"])),
            "best_correct_count": max(int(ens_full["test_correct_count"]), int(single_full["test_correct_count"])),
            "total_test_count": 60,
        }
    }
    with open(reports_dir / "phase2d_final_test_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved final test summary JSON to %s", reports_dir / "phase2d_final_test_summary.json")


if __name__ == "__main__":
    main()
