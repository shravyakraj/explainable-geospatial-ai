"""Phase 2C Model Improvement Campaign: Controlled architectural experiments evaluated on validation split."""

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
from geospatial_ai.metrics import evaluate_predictions
from geospatial_ai.model import build_siamese_network
from geospatial_ai.train import set_seed, train_siamese_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase2c_campaign")


def main() -> None:
    logger.info("=== Starting Phase 2C Improvement Campaign (Validation Selection) ===")
    manifest_path = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
    experiments_dir = PROJECT_ROOT / "experiments" / "phase2c"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_records = get_split_records(manifest_path, "train", project_root=PROJECT_ROOT)
    val_records = get_split_records(manifest_path, "val", project_root=PROJECT_ROOT)
    logger.info("Loaded Train: %d patches, Val: %d patches", len(train_records), len(val_records))

    val_t1, val_t2, val_y = load_numpy_arrays(val_records)
    batch_size = 8
    seed = 42

    # 2. Experiment Definitions
    experiments = [
        {
            "id": "EXP-1",
            "name": "M1_Baseline_PlainCNN_AbsDiff",
            "backbone_type": "conv",
            "pooling": "gap",
            "fusion_type": "abs_diff",
            "l2_reg": 0.0,
            "lr": 2e-4,
            "desc": "Baseline 4-block ConvNet with Global Average Pooling and Absolute Difference.",
        },
        {
            "id": "EXP-2",
            "name": "M3_PlainCNN_DirectionalFusion",
            "backbone_type": "conv",
            "pooling": "gap",
            "fusion_type": "directional",
            "l2_reg": 0.0,
            "lr": 2e-4,
            "desc": "Plain ConvNet with Directional Fusion (|T1-T2| + (T2-T1)) encoding magnitude and sign of change.",
        },
        {
            "id": "EXP-3",
            "name": "M4_PlainCNN_MultiScalePooling",
            "backbone_type": "conv",
            "pooling": "gap_gmp",
            "fusion_type": "abs_diff",
            "l2_reg": 0.0,
            "lr": 2e-4,
            "desc": "Plain ConvNet with Combined GAP + GMP to preserve localized peak clearing anomalies.",
        },
        {
            "id": "EXP-4",
            "name": "M5_ResSiamese_AbsDiff",
            "backbone_type": "resnet",
            "pooling": "gap",
            "fusion_type": "abs_diff",
            "l2_reg": 1e-4,
            "lr": 2e-4,
            "desc": "4-stage Residual CNN backbone with projection shortcuts, GAP, and Absolute Difference.",
        },
        {
            "id": "EXP-5",
            "name": "M6_ResSiamese_MultiScale_Directional",
            "backbone_type": "resnet",
            "pooling": "gap_gmp",
            "fusion_type": "directional",
            "l2_reg": 1e-4,
            "lr": 2e-4,
            "desc": "Residual CNN backbone with GAP+GMP Multi-Scale Pooling and Directional Fusion.",
        },
    ]

    results = []

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        logger.info("\n>>> Running Experiment %s: %s <<<", exp_id, exp_name)

        set_seed(seed)
        train_ds = create_tf_dataset(
            train_records, batch_size=batch_size, is_training=True, augment=True, seed=seed
        )
        val_ds = create_tf_dataset(
            val_records, batch_size=batch_size, is_training=False, augment=False, seed=seed
        )

        model = build_siamese_network(
            input_shape=(256, 256, 4),
            dense_units=64,
            dropout_rate=0.4,
            backbone_type=exp["backbone_type"],
            pooling=exp["pooling"],
            fusion_type=exp["fusion_type"],
            l2_reg=exp["l2_reg"],
            name=f"siamese_{exp_id.lower()}",
        )

        ckpt_path = experiments_dir / f"checkpoint_{exp_id.lower()}.keras"
        trained_model, history, val_summary = train_siamese_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            val_t1=val_t1,
            val_t2=val_t2,
            val_y=val_y,
            checkpoint_path=ckpt_path,
            log_dir=experiments_dir,
            epochs=50,
            learning_rate=exp["lr"],
            experiment_name=f"{exp_id.lower()}_{exp_name}",
        )

        val_def = val_summary["default_threshold_0_5"]
        val_opt = val_summary["optimal_threshold_metrics"]

        res_entry = {
            "exp_id": exp_id,
            "name": exp_name,
            "description": exp["desc"],
            "parameters": model.count_params(),
            "min_val_loss": round(val_summary["val_loss"], 4),
            "val_pr_auc": round(val_summary["pr_auc"], 4),
            "val_roc_auc": round(val_summary["roc_auc"], 4),
            "val_accuracy_0_5": val_def["accuracy"],
            "val_balanced_acc_0_5": val_def["balanced_accuracy"],
            "val_f1_0_5": val_def["f1_score"],
            "optimal_threshold": val_summary["optimal_threshold"],
            "val_accuracy_opt": val_opt["accuracy"],
            "val_balanced_acc_opt": val_opt["balanced_accuracy"],
            "val_f1_opt": val_opt["f1_score"],
            "val_precision_opt": val_opt["precision"],
            "val_recall_opt": val_opt["recall"],
            "val_specificity_opt": val_opt["specificity"],
            "checkpoint_path": str(ckpt_path),
        }
        results.append(res_entry)

        logger.info(
            "Exp %s Finished: Val Loss=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f, Opt Thresh=%.4f (Acc=%.4f, BalAcc=%.4f, F1=%.4f)",
            exp_id, res_entry["min_val_loss"], res_entry["val_pr_auc"], res_entry["val_roc_auc"],
            res_entry["optimal_threshold"], res_entry["val_accuracy_opt"], res_entry["val_balanced_acc_opt"], res_entry["val_f1_opt"]
        )

    # 3. Compile Comparative Results Table
    df_res = pd.DataFrame(results)
    csv_out = experiments_dir / "campaign_validation_results.csv"
    df_res.to_csv(csv_out, index=False)
    logger.info("\nSaved campaign validation comparison to %s", csv_out)

    with open(experiments_dir / "campaign_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print markdown summary
    print("\n" + "=" * 90)
    print("PHASE 2C VALIDATION EXPERIMENT COMPARISON TABLE")
    print("=" * 90)
    print(df_res[["exp_id", "name", "parameters", "min_val_loss", "val_pr_auc", "val_roc_auc", "val_accuracy_opt", "val_balanced_acc_opt", "val_f1_opt"]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
