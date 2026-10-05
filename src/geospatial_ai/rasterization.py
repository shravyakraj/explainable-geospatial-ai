"""Vector rasterization and exclusion mask generation."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from geospatial_ai.geo import StudyGrid, LOCKED_GRID

logger = logging.getLogger(__name__)


def rasterize_vector_layer(
    vector_path: Path,
    output_tif: Path,
    grid: StudyGrid = LOCKED_GRID,
    fill_value: int = 0,
    burn_value: int = 1,
) -> dict:
    """Rasterize a vector GeoJSON onto the locked 10m UTM grid."""
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(vector_path)
    logger.info("Loaded %d features from %s", len(gdf), vector_path.name)

    stats = {
        "feature_count": len(gdf),
        "total_area_km2": 0.0,
        "min_area_ha": 0.0,
        "max_area_ha": 0.0,
        "rasterized_positive_pixels": 0,
        "rasterized_positive_ha": 0.0,
    }

    if len(gdf) == 0:
        logger.warning("Empty vector file: %s", vector_path)
        mask = np.zeros((grid.height_pixels, grid.width_pixels), dtype=np.uint8)
    else:
        # Reproject to target grid CRS
        if gdf.crs is None or gdf.crs.to_string() != grid.crs:
            gdf = gdf.to_crs(grid.crs)

        # Calculate geometric areas in EPSG:32720
        areas_ha = gdf.geometry.area / 10000.0  # m2 to ha
        stats["total_area_km2"] = float((gdf.geometry.area.sum()) / 1e6)
        stats["min_area_ha"] = float(areas_ha.min()) if len(areas_ha) > 0 else 0.0
        stats["max_area_ha"] = float(areas_ha.max()) if len(areas_ha) > 0 else 0.0

        # Filter valid geometries
        shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        mask = rasterize(
            shapes=shapes,
            out_shape=(grid.height_pixels, grid.width_pixels),
            transform=grid.transform,
            fill=fill_value,
            default_value=burn_value,
            dtype=np.uint8,
        )

    stats["rasterized_positive_pixels"] = int(np.count_nonzero(mask == burn_value))
    stats["rasterized_positive_ha"] = float(stats["rasterized_positive_pixels"] * (grid.pixel_size ** 2) / 10000.0)

    profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width_pixels,
        "height": grid.height_pixels,
        "compress": "deflate",
        "nodata": 0,
    }

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(mask, 1)

    logger.info("Saved %s (positive pixels: %d, area: %.2f ha)", output_tif.name, stats["rasterized_positive_pixels"], stats["rasterized_positive_ha"])
    return stats


def generate_scl_cloud_shadow_mask(
    t1_scl_path: Path,
    t2_scl_path: Path,
    output_tif: Path,
    grid: StudyGrid = LOCKED_GRID,
) -> dict:
    """Extract invalid cloud, shadow, and nodata pixels from SCL layers."""
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(t1_scl_path) as s1:
        scl1 = s1.read(1)
    with rasterio.open(t2_scl_path) as s2:
        scl2 = s2.read(1)

    # Sentinel-2 SCL classification:
    # 0 = NO_DATA, 1 = SATURATED_OR_DEFECTIVE, 2 = DARK_AREA_PIXELS,
    # 3 = CLOUD_SHADOWS, 8 = CLOUD_MEDIUM_PROBABILITY, 9 = CLOUD_HIGH_PROBABILITY, 10 = THIN_CIRRUS
    invalid_classes = {0, 1, 3, 8, 9, 10}
    mask_scl1 = np.isin(scl1, list(invalid_classes))
    mask_scl2 = np.isin(scl2, list(invalid_classes))
    cloud_mask = (mask_scl1 | mask_scl2).astype(np.uint8)

    profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width_pixels,
        "height": grid.height_pixels,
        "compress": "deflate",
        "nodata": 0,
    }

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(cloud_mask, 1)

    total_px = grid.width_pixels * grid.height_pixels
    cloud_px = int(np.count_nonzero(cloud_mask))
    return {
        "cloud_shadow_pixels": cloud_px,
        "cloud_shadow_pct": (cloud_px / total_px) * 100.0,
    }


def combine_exclusion_masks(
    mask_paths: list[Path],
    output_tif: Path,
    grid: StudyGrid = LOCKED_GRID,
) -> dict:
    """Combine prior deforestation, water, non-forest, and cloud/shadow masks."""
    combined = np.zeros((grid.height_pixels, grid.width_pixels), dtype=np.uint8)

    for p in mask_paths:
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            data = src.read(1)
            combined = combined | (data > 0).astype(np.uint8)

    profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width_pixels,
        "height": grid.height_pixels,
        "compress": "deflate",
        "nodata": 0,
    }

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(combined, 1)

    total_px = grid.width_pixels * grid.height_pixels
    excluded_px = int(np.count_nonzero(combined))
    return {
        "total_excluded_pixels": excluded_px,
        "total_excluded_pct": (excluded_px / total_px) * 100.0,
    }
