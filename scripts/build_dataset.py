#!/usr/bin/env python3
"""Complete Phase 2A Dataset Construction Pipeline.

Orchestrates:
1. Sentinel-2 L2A acquisition for T1 (2020-08-08) and T2 (2021-08-08)
2. INPE TerraBrasilis PRODES 2021 deforestation & exclusion layers acquisition
3. Common grid projection (EPSG:32720, 10m, 6912x5888)
4. SCL cloud/shadow masking & unified exclusion generation
5. 621 patch extraction (256x256) with locked spatial partitioning
6. Split manifest (JSON & CSV) generation
7. Visual quality control artifacts generation
"""

import argparse
import concurrent.futures
import logging
import sys
import time
from pathlib import Path

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from geospatial_ai.acquisition import (
    acquire_sentinel2_window,
    fetch_terrabrasilis_wfs_geojson,
)
from geospatial_ai.geo import LOCKED_GRID
from geospatial_ai.manifest import generate_manifest_files
from geospatial_ai.qc import (
    generate_patch_visual_qc,
    generate_spatial_split_layout,
)
from geospatial_ai.rasterization import (
    combine_exclusion_masks,
    generate_scl_cloud_shadow_mask,
    rasterize_vector_layer,
)
from geospatial_ai.tiling import extract_and_label_patches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_dataset")

# Locked Granule IDs
T1_ITEM_ID = "S2A_MSIL2A_20200808T142741_R053_T20LMR_20200815T111232"
T2_ITEM_ID = "S2B_MSIL2A_20210808T142729_R053_T20LMR_20210809T022211"
BANDS = ["B02", "B03", "B04", "B08", "SCL"]


def step1_acquire_prodes(labels_dir: Path) -> dict[str, Path]:
    """Download PRODES 2021 deforestation and exclusion layers via WFS."""
    logger.info("=== STEP 1: Acquiring INPE TerraBrasilis PRODES Vector Layers ===")
    labels_dir.mkdir(parents=True, exist_ok=True)
    bbox_cql = "BBOX(geom, -9.05, -63.80, -8.50, -63.15)"

    layer_configs = {
        "prodes_2021": (
            "prodes-legal-amz:yearly_deforestation",
            f"year = 2021 AND {bbox_cql}",
            labels_dir / "prodes_2021.geojson",
        ),
        "prior_2007": (
            "prodes-legal-amz:accumulated_deforestation_2007",
            bbox_cql,
            labels_dir / "accumulated_deforestation_2007.geojson",
        ),
        "prior_2008_2020": (
            "prodes-legal-amz:yearly_deforestation",
            f"year <= 2020 AND {bbox_cql}",
            labels_dir / "prior_yearly_deforestation_2008_2020.geojson",
        ),
        "hydrography": (
            "prodes-legal-amz:hydrography",
            bbox_cql,
            labels_dir / "hydrography.geojson",
        ),
        "no_forest": (
            "prodes-legal-amz:no_forest",
            bbox_cql,
            labels_dir / "no_forest.geojson",
        ),
    }

    paths = {}
    for name, (type_name, cql, path) in layer_configs.items():
        logger.info("Fetching %s ...", name)
        paths[name] = fetch_terrabrasilis_wfs_geojson(type_name, cql, path, max_features=15000)
    return paths


def step2_acquire_sentinel2(raw_dir: Path, workers: int = 4) -> tuple[dict[str, Path], dict[str, Path]]:
    """Download windowed Sentinel-2 bands for T1 and T2 in parallel."""
    logger.info("=== STEP 2: Acquiring Windowed Sentinel-2 L2A Bands ===")
    raw_dir.mkdir(parents=True, exist_ok=True)

    t1_paths = {}
    t2_paths = {}

    download_tasks = []
    for b in BANDS:
        download_tasks.append(("t1", T1_ITEM_ID, b))
        download_tasks.append(("t2", T2_ITEM_ID, b))

    def _download_band(task):
        time_prefix, item_id, band = task
        out_file = raw_dir / f"{time_prefix}_{band}.tif"
        if out_file.exists():
            logger.info("Band already exists: %s", out_file.name)
            return time_prefix, band, out_file
        res = acquire_sentinel2_window(item_id, [band], raw_dir, time_prefix, LOCKED_GRID)
        return time_prefix, band, res[band]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_band, t): t for t in download_tasks}
        for future in concurrent.futures.as_completed(futures):
            time_prefix, band, path = future.result()
            if time_prefix == "t1":
                t1_paths[band] = path
            else:
                t2_paths[band] = path

    return t1_paths, t2_paths


def step3_rasterize_and_mask(
    prodes_paths: dict[str, Path],
    t1_paths: dict[str, Path],
    t2_paths: dict[str, Path],
    labels_dir: Path,
) -> tuple[Path, Path, dict]:
    """Rasterize vector layers, build SCL cloud mask, and create unified exclusion mask."""
    logger.info("=== STEP 3: Rasterizing Vectors and Building Exclusion Masks ===")
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Deforestation Ground Truth (2021)
    prodes_tif = labels_dir / "prodes_2021.tif"
    stats_2021 = rasterize_vector_layer(prodes_paths["prodes_2021"], prodes_tif, LOCKED_GRID)

    # 2. Historical & Physical Exclusions
    tif_2007 = labels_dir / "prior_2007.tif"
    rasterize_vector_layer(prodes_paths["prior_2007"], tif_2007, LOCKED_GRID)

    tif_2020 = labels_dir / "prior_2008_2020.tif"
    rasterize_vector_layer(prodes_paths["prior_2008_2020"], tif_2020, LOCKED_GRID)

    tif_hydro = labels_dir / "hydrography.tif"
    rasterize_vector_layer(prodes_paths["hydrography"], tif_hydro, LOCKED_GRID)

    tif_no_forest = labels_dir / "no_forest.tif"
    rasterize_vector_layer(prodes_paths["no_forest"], tif_no_forest, LOCKED_GRID)

    # 3. Cloud/Shadow Mask from SCL
    scl_mask_tif = labels_dir / "scl_cloud_mask.tif"
    cloud_stats = generate_scl_cloud_shadow_mask(t1_paths["SCL"], t2_paths["SCL"], scl_mask_tif, LOCKED_GRID)

    # 4. Combine into Unified Exclusion Mask
    exclusion_tif = labels_dir / "unified_exclusion_mask.tif"
    exclusion_stats = combine_exclusion_masks(
        [tif_2007, tif_2020, tif_hydro, tif_no_forest, scl_mask_tif],
        exclusion_tif,
        LOCKED_GRID,
    )

    summary_stats = {
        "prodes_2021": stats_2021,
        "cloud_shadow": cloud_stats,
        "unified_exclusion": exclusion_stats,
    }
    return prodes_tif, exclusion_tif, summary_stats


def step4_extract_and_manifest(
    t1_paths: dict[str, Path],
    t2_paths: dict[str, Path],
    prodes_tif: Path,
    exclusion_tif: Path,
    processed_dir: Path,
) -> tuple[Path, Path, list[dict], dict]:
    """Extract 621 patches, apply locked spatial split, and write manifest."""
    logger.info("=== STEP 4: Extracting 621 Patches & Creating Manifest ===")
    records = extract_and_label_patches(
        t1_paths,
        t2_paths,
        prodes_tif,
        exclusion_tif,
        processed_dir,
        LOCKED_GRID,
    )
    json_path, csv_path, stats = generate_manifest_files(records, processed_dir)
    return json_path, csv_path, records, stats


def step5_generate_qc(records: list[dict], processed_dir: Path, reports_dir: Path):
    """Generate visual quality control diagnostic figures."""
    logger.info("=== STEP 5: Generating Visual Quality Control Artifacts ===")
    reports_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = processed_dir / "patches"

    # 1. Spatial Split Layout Map
    layout_png = reports_dir / "qc_spatial_split_layout.png"
    generate_spatial_split_layout(records, layout_png, LOCKED_GRID)

    # 2. Sample Patches QC
    # Intact forest patch (clean label=0 in train)
    clean_negs = [r for r in records if r["label"] == 0 and r["split"] == "train"]
    if clean_negs:
        sample_neg = clean_negs[0]["patch_id"]
        generate_patch_visual_qc(
            sample_neg,
            patches_dir,
            reports_dir / "qc_sample_intact_forest.png",
            title_suffix="(Intact Forest, Label=0)",
        )

    # Deforestation patch (strong positive label=1 in train)
    positives = sorted([r for r in records if r["label"] == 1], key=lambda x: x["prodes_positive_ha"], reverse=True)
    if positives:
        sample_pos = positives[0]["patch_id"]
        generate_patch_visual_qc(
            sample_pos,
            patches_dir,
            reports_dir / "qc_sample_deforestation.png",
            title_suffix=f"(Deforestation, Label=1, {positives[0]['prodes_positive_ha']:.1f} ha)",
        )

    # Excluded patch (marginal or high prior loss)
    excluded = [r for r in records if r["label"] == -1 and r["split"] != "buffer"]
    if excluded:
        sample_ex = excluded[0]["patch_id"]
        generate_patch_visual_qc(
            sample_ex,
            patches_dir,
            reports_dir / "qc_sample_excluded_patch.png",
            title_suffix=f"(Excluded: {excluded[0]['exclusion_reason']})",
        )


def main():
    parser = argparse.ArgumentParser(description="Build Phase 2A Dataset")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for band downloads")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting Phase 2A Dataset Construction Pipeline...")

    data_dir = project_root / "data"
    labels_dir = data_dir / "labels"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    reports_dir = project_root / "reports"

    # Step 1: PRODES Vector Acquisition
    prodes_paths = step1_acquire_prodes(labels_dir)

    # Step 2: Sentinel-2 L2A Acquisition
    t1_paths, t2_paths = step2_acquire_sentinel2(raw_dir, workers=args.workers)

    # Step 3: Rasterization & Unified Exclusion
    prodes_tif, exclusion_tif, mask_stats = step3_rasterize_and_mask(
        prodes_paths, t1_paths, t2_paths, labels_dir
    )

    # Step 4: Patch Extraction & Manifest
    json_path, csv_path, records, manifest_stats = step4_extract_and_manifest(
        t1_paths, t2_paths, prodes_tif, exclusion_tif, processed_dir
    )

    # Step 5: Visual QC Figures
    step5_generate_qc(records, processed_dir, reports_dir)

    elapsed = time.time() - start_time
    logger.info("=== Phase 2A Dataset Construction COMPLETE in %.2f seconds ===", elapsed)
    logger.info("Manifest JSON: %s", json_path)
    logger.info("Manifest CSV: %s", csv_path)
    logger.info("Statistics: %s", manifest_stats)


if __name__ == "__main__":
    main()
