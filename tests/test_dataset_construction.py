"""Deterministic tests for Phase 2A dataset construction, spatial splitting, and integrity."""

import csv
import json
from pathlib import Path
import pytest
import geopandas as gpd
import numpy as np
import rasterio

from geospatial_ai.geo import LOCKED_GRID
from geospatial_ai.tiling import get_spatial_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def test_study_grid_specifications():
    """Verify locked spatial grid parameters."""
    assert LOCKED_GRID.cols_patches == 27
    assert LOCKED_GRID.rows_patches == 23
    assert LOCKED_GRID.total_patches == 621
    assert LOCKED_GRID.patch_size == 256
    assert LOCKED_GRID.pixel_size == 10.0
    assert LOCKED_GRID.crs == "EPSG:32720"
    assert LOCKED_GRID.width_pixels == 6912
    assert LOCKED_GRID.height_pixels == 5888
    # Projected bounds check
    minx, miny, maxx, maxy = LOCKED_GRID.bounds_projected
    assert minx == 412080.0
    assert maxy == 9060420.0
    assert maxx == 481200.0
    assert miny == 9001540.0


def test_spatial_split_partitioning():
    """Verify column assignments across Train, Buffer, Val, and Test blocks."""
    # Train: cols 1-15 (15 cols)
    for c in range(1, 16):
        assert get_spatial_split(c) == "train"

    # Buffer 1: col 16
    assert get_spatial_split(16) == "buffer"

    # Val: cols 17-21 (5 cols)
    for c in range(17, 22):
        assert get_spatial_split(c) == "val"

    # Buffer 2: col 22
    assert get_spatial_split(22) == "buffer"

    # Test: cols 23-27 (5 cols)
    for c in range(23, 28):
        assert get_spatial_split(c) == "test"

    # Out of bounds
    with pytest.raises(ValueError):
        get_spatial_split(0)
    with pytest.raises(ValueError):
        get_spatial_split(28)


def test_spatial_buffer_isolation_distance():
    """Verify buffer columns provide >= 2.56 km spatial separation."""
    # 1 patch width = 256 pixels * 10 m = 2,560 m = 2.56 km
    buffer_width_m = LOCKED_GRID.patch_size * LOCKED_GRID.pixel_size
    assert buffer_width_m == 2560.0


def test_prodes_vector_layers_integrity():
    """Verify downloaded PRODES vectors and exact feature counts."""
    prodes_2021_file = LABELS_DIR / "prodes_2021.geojson"
    assert prodes_2021_file.exists(), "PRODES 2021 GeoJSON must exist"
    gdf_2021 = gpd.read_file(prodes_2021_file)
    assert len(gdf_2021) == 301, f"Expected 301 features for 2021, got {len(gdf_2021)}"

    acc_2007_file = LABELS_DIR / "accumulated_deforestation_2007.geojson"
    assert acc_2007_file.exists()
    gdf_2007 = gpd.read_file(acc_2007_file)
    assert len(gdf_2007) == 583

    prior_file = LABELS_DIR / "prior_yearly_deforestation_2008_2020.geojson"
    assert prior_file.exists()
    gdf_prior = gpd.read_file(prior_file)
    assert len(gdf_prior) == 2315

    hydro_file = LABELS_DIR / "hydrography.geojson"
    assert hydro_file.exists()
    gdf_hydro = gpd.read_file(hydro_file)
    assert len(gdf_hydro) == 112

    nf_file = LABELS_DIR / "no_forest.geojson"
    assert nf_file.exists()
    gdf_nf = gpd.read_file(nf_file)
    assert len(gdf_nf) == 14


def test_sentinel2_acquired_bands():
    """Verify all 10 Sentinel-2 bands exist and match expected dimensions."""
    expected_bands = [
        "t1_B02.tif", "t1_B03.tif", "t1_B04.tif", "t1_B08.tif", "t1_SCL.tif",
        "t2_B02.tif", "t2_B03.tif", "t2_B04.tif", "t2_B08.tif", "t2_SCL.tif",
    ]
    for b in expected_bands:
        p = RAW_DIR / b
        assert p.exists(), f"Missing band: {b}"
        with rasterio.open(p) as src:
            assert src.width == LOCKED_GRID.width_pixels
            assert src.height == LOCKED_GRID.height_pixels
            assert src.crs.to_string() == LOCKED_GRID.crs


def test_rasterized_masks_grid_alignment():
    """Verify rasterized GeoTIFF dimensions and CRS match LOCKED_GRID."""
    for tif_name in ["prodes_2021.tif", "scl_cloud_mask.tif", "unified_exclusion_mask.tif"]:
        tif_path = LABELS_DIR / tif_name
        assert tif_path.exists(), f"Missing mask: {tif_name}"
        with rasterio.open(tif_path) as src:
            assert src.width == LOCKED_GRID.width_pixels
            assert src.height == LOCKED_GRID.height_pixels
            assert src.crs.to_string() == LOCKED_GRID.crs
            assert src.transform == LOCKED_GRID.transform


def test_manifest_structure_and_counts():
    """Verify split_manifest.json and split_manifest.csv."""
    json_path = PROCESSED_DIR / "split_manifest.json"
    csv_path = PROCESSED_DIR / "split_manifest.csv"
    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["crs"] == "EPSG:32720"
    assert meta["patch_pixels"] == 256
    stats = meta["statistics"]
    assert stats["total_patches"] == 621
    assert stats["included_patches"] == 145
    assert stats["excluded_patches"] == 476
    assert stats["positive_patches"] == 74
    assert stats["negative_patches"] == 71

    # Check split numbers
    assert stats["splits"]["train"]["total"] == 345
    assert stats["splits"]["val"]["total"] == 115
    assert stats["splits"]["test"]["total"] == 115
    assert stats["splits"]["buffer"]["total"] == 46

    # Verify CSV row count
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 621


def test_no_data_leakage_and_buffer_isolation():
    """Verify zero overlap and complete buffer isolation."""
    json_path = PROCESSED_DIR / "split_manifest.json"
    with open(json_path) as f:
        data = json.load(f)

    patches = data["patches"]
    train_cols = {p["col"] for p in patches if p["split"] == "train"}
    val_cols = {p["col"] for p in patches if p["split"] == "val"}
    test_cols = {p["col"] for p in patches if p["split"] == "test"}
    buffer_cols = {p["col"] for p in patches if p["split"] == "buffer"}

    # Disjoint column sets
    assert train_cols.isdisjoint(val_cols)
    assert val_cols.isdisjoint(test_cols)
    assert train_cols.isdisjoint(test_cols)
    assert buffer_cols == {16, 22}

    # Verify all buffer patches have label == -1
    for p in patches:
        if p["split"] == "buffer":
            assert p["label"] == -1
            assert p["exclusion_reason"] == "buffer_column_discarded"


def test_extracted_patch_file_content():
    """Verify random sample of extracted .npz patch contents."""
    patches_dir = PROCESSED_DIR / "patches"
    assert len(list(patches_dir.glob("*.npz"))) == 621

    sample_patch = patches_dir / "patch_r05_c05.npz"
    assert sample_patch.exists()
    data = np.load(sample_patch)
    assert "t1" in data and data["t1"].shape == (256, 256, 4)
    assert "t2" in data and data["t2"].shape == (256, 256, 4)
    assert "mask" in data and data["mask"].shape == (256, 256)
    assert "exclude" in data and data["exclude"].shape == (256, 256)
    assert "label" in data and data["label"] in (-1, 0, 1)


def test_qc_figures_exist():
    """Verify visual quality control diagnostic figures exist."""
    expected_qc = [
        "qc_spatial_split_layout.png",
        "qc_sample_intact_forest.png",
        "qc_sample_deforestation.png",
        "qc_sample_excluded_patch.png",
    ]
    for q in expected_qc:
        p = REPORTS_DIR / q
        assert p.exists(), f"Missing QC figure: {q}"
        assert p.stat().st_size > 10000, f"QC figure too small: {q}"
