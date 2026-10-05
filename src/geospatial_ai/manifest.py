"""Dataset manifest generation and validation."""

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_manifest_files(records: list[dict], output_dir: Path) -> tuple[Path, Path, dict]:
    """Write manifest to JSON and CSV and return validation statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_manifest.json"
    csv_path = output_dir / "split_manifest.csv"

    # Compute validation statistics
    stats = {
        "total_patches": len(records),
        "included_patches": sum(1 for r in records if r["label"] in (0, 1)),
        "excluded_patches": sum(1 for r in records if r["label"] == -1),
        "positive_patches": sum(1 for r in records if r["label"] == 1),
        "negative_patches": sum(1 for r in records if r["label"] == 0),
        "splits": {
            "train": {
                "total": sum(1 for r in records if r["split"] == "train"),
                "positive": sum(1 for r in records if r["split"] == "train" and r["label"] == 1),
                "negative": sum(1 for r in records if r["split"] == "train" and r["label"] == 0),
                "excluded": sum(1 for r in records if r["split"] == "train" and r["label"] == -1),
            },
            "val": {
                "total": sum(1 for r in records if r["split"] == "val"),
                "positive": sum(1 for r in records if r["split"] == "val" and r["label"] == 1),
                "negative": sum(1 for r in records if r["split"] == "val" and r["label"] == 0),
                "excluded": sum(1 for r in records if r["split"] == "val" and r["label"] == -1),
            },
            "test": {
                "total": sum(1 for r in records if r["split"] == "test"),
                "positive": sum(1 for r in records if r["split"] == "test" and r["label"] == 1),
                "negative": sum(1 for r in records if r["split"] == "test" and r["label"] == 0),
                "excluded": sum(1 for r in records if r["split"] == "test" and r["label"] == -1),
            },
            "buffer": {
                "total": sum(1 for r in records if r["split"] == "buffer"),
                "discarded": sum(1 for r in records if r["split"] == "buffer"),
            },
        },
        "exclusion_reasons": {},
    }

    # Calculate class percentages on included patches
    for s_name in ["train", "val", "test"]:
        s_data = stats["splits"][s_name]
        inc = s_data["positive"] + s_data["negative"]
        s_data["included"] = inc
        s_data["positive_pct"] = round((s_data["positive"] / inc * 100.0) if inc > 0 else 0.0, 2)

    total_inc = stats["included_patches"]
    stats["total_positive_pct"] = round((stats["positive_patches"] / total_inc * 100.0) if total_inc > 0 else 0.0, 2)

    # Tally exclusion reasons
    for r in records:
        if r["exclusion_reason"]:
            reason = r["exclusion_reason"]
            stats["exclusion_reasons"][reason] = stats["exclusion_reasons"].get(reason, 0) + 1

    # Write JSON
    manifest_data = {
        "metadata": {
            "study_area": "Rondônia, Brazil (Candeias do Jamari)",
            "sentinel2_tile": "20LMR",
            "relative_orbit": 53,
            "t1_granule": "S2A_MSIL2A_20200808T142741_R053_T20LMR_20200815T111232",
            "t2_granule": "S2B_MSIL2A_20210808T142729_R053_T20LMR_20210809T022211",
            "prodes_layer": "prodes-legal-amz:yearly_deforestation (year=2021)",
            "crs": "EPSG:32720",
            "resolution_m": 10.0,
            "patch_pixels": 256,
            "statistics": stats,
        },
        "patches": records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Write CSV
    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row_copy = r.copy()
                row_copy["patch_bounds_epsg32720"] = json.dumps(row_copy["patch_bounds_epsg32720"])
                writer.writerow(row_copy)

    logger.info("Saved manifest to %s and %s", json_path, csv_path)
    return json_path, csv_path, stats
