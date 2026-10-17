"""Geospatial region extraction, coordinate transformation, and candidate deforestation area inventory.

Implements connected component analysis with 8-connectivity to segment
spatially discrete deforestation events, converts pixel coordinates to
real-world UTM and WGS84 coordinates, and exports GeoJSON features.
"""

from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyproj
import scipy.ndimage
from shapely.geometry import box, mapping, Polygon

logger = logging.getLogger(__name__)

# Official master grid parameters for Rondônia (MGRS 20LMR) benchmark
BENCHMARK_ORIGIN_X = 412080.0
BENCHMARK_ORIGIN_Y = 9060420.0
BENCHMARK_PIXEL_SIZE = 10.0
BENCHMARK_PATCH_SIZE = 256
BENCHMARK_CRS = "EPSG:32720"


@dataclass
class CandidateRegion:
    """Individual suspected deforestation event identified via connected component segmentation."""

    region_id: str  # "Area 01", "Area 02", ...
    rank: int
    pixel_count: int
    area_ha: float
    bbox_pixel: Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    centroid_pixel: Tuple[float, float]  # (r_c, c_c)
    mean_probability: float
    max_probability: float
    valid_fraction: float
    ref_overlap_pixels: int
    ref_overlap_ha: float
    ref_overlap_fraction: float
    ref_status: str
    georeferenced: bool
    crs: Optional[str] = None
    centroid_utm: Optional[Tuple[float, float]] = None  # (Easting, Northing)
    centroid_wgs84: Optional[Tuple[float, float]] = None  # (Longitude, Latitude)
    bbox_utm: Optional[Tuple[float, float, float, float]] = None  # (min_e, min_n, max_e, max_n)
    bbox_wgs84: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)
    dates: str = "2020-08-08 to 2021-08-08"
    sensor: str = "Sentinel-2 MSI Level-2A"

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidateRegion to dictionary representation."""
        return asdict(self)


def get_wgs84_transformer(source_crs_str: str) -> Optional[pyproj.Transformer]:
    """Build a pyproj Transformer from source CRS to WGS84 (EPSG:4326) with (lon, lat) ordering."""
    try:
        return pyproj.Transformer.from_crs(source_crs_str, "EPSG:4326", always_xy=True)
    except Exception as e:
        logger.warning("Could not construct CRS transformer from %s: %s", source_crs_str, e)
        return None


def extract_candidate_deforestation_areas(
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    w_mask: Optional[np.ndarray] = None,
    y_ref: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    pixel_resolution_m: float = 10.0,
    min_pixel_size: int = 1,
) -> List[CandidateRegion]:
    """Extract individual suspected deforestation areas using 8-connectivity component analysis.

    Args:
        prob_map: 2D array of deforestation probabilities in [0.0, 1.0], shape (H, W).
        binary_mask: 2D boolean or float array of thresholded predictions.
        w_mask: 2D or 3D validity mask (1 = valid, 0 = excluded).
        y_ref: 2D or 3D reference mask (1 = loss, 0 = no loss).
        metadata: Geospatial metadata dict containing bounds, CRS, or patch information.
        pixel_resolution_m: Spatial resolution in meters (default 10m for Sentinel-2).
        min_pixel_size: Minimum pixel threshold for reporting (default 1, retains all predictions).

    Returns:
        List of CandidateRegion objects sorted by area in hectares descending.
    """
    if metadata is None:
        metadata = {}

    w_2d = (w_mask[..., 0] > 0.5) if (w_mask is not None and w_mask.ndim == 3) else (
        (w_mask > 0.5) if w_mask is not None else np.ones(binary_mask.shape[:2], dtype=bool)
    )
    b_2d = (binary_mask[..., 0] > 0.5) if binary_mask.ndim == 3 else (binary_mask > 0.5)

    # Restrict segmentation to valid observation area
    eval_mask = b_2d & w_2d

    # 8-connectivity structuring element
    structure_8 = np.ones((3, 3), dtype=int)
    labeled_array, num_features = scipy.ndimage.label(eval_mask, structure=structure_8)

    if num_features == 0:
        return []

    # Check for georeferencing metadata
    georeferenced = False
    source_crs = None
    transformer = None
    top_left_x = None
    top_left_y = None

    # Benchmark dataset patch bounds (e.g. from split manifest)
    bounds = metadata.get("patch_bounds_epsg32720")
    if bounds and len(bounds) == 4:
        georeferenced = True
        source_crs = BENCHMARK_CRS
        # [minx, miny, maxx, maxy] -> top-left is (minx, maxy)
        top_left_x = float(bounds[0])
        top_left_y = float(bounds[3])
        transformer = get_wgs84_transformer(BENCHMARK_CRS)
    elif "affine" in metadata and "crs" in metadata:
        # Custom GeoTIFF upload
        try:
            aff = metadata["affine"]
            top_left_x = float(aff[2])
            top_left_y = float(aff[5])
            source_crs = str(metadata["crs"])
            georeferenced = True
            transformer = get_wgs84_transformer(source_crs)
        except Exception as e:
            logger.warning("Error parsing GeoTIFF affine transform: %s", e)

    ha_per_pixel = (pixel_resolution_m * pixel_resolution_m) / 10000.0
    dates_str = metadata.get("observation_dates", "2020-08-08 to 2021-08-08")
    sensor_str = metadata.get("sensor", "Sentinel-2 MSI Level-2A")

    # Collect raw features
    feature_items = []
    for fid in range(1, num_features + 1):
        reg_mask = (labeled_array == fid)
        cnt = int(np.sum(reg_mask))
        if cnt < min_pixel_size:
            continue
        feature_items.append((fid, cnt, reg_mask))

    # Sort descending by size
    feature_items.sort(key=lambda item: item[1], reverse=True)

    candidate_regions: List[CandidateRegion] = []
    for rank, (fid, cnt, reg_mask) in enumerate(feature_items, start=1):
        area_ha = round(float(cnt * ha_per_pixel), 2)
        r_coords, c_coords = np.where(reg_mask)
        r_min, r_max = int(np.min(r_coords)), int(np.max(r_coords))
        c_min, c_max = int(np.min(c_coords)), int(np.max(c_coords))
        bbox_pixel = (r_min, c_min, r_max, c_max)

        r_c = float(np.mean(r_coords))
        c_c = float(np.mean(c_coords))
        centroid_pixel = (round(r_c, 1), round(c_c, 1))

        probs = prob_map[reg_mask]
        p_mean = round(float(np.mean(probs)), 3)
        p_max = round(float(np.max(probs)), 3)

        # Validity overlap
        val_overlap = np.sum(reg_mask & w_2d)
        valid_frac = round(float(val_overlap / cnt), 3)

        # PRODES reference comparison
        ref_overlap_cnt = 0
        ref_overlap_ha = 0.0
        ref_frac = 0.0
        ref_status = "Reference data unavailable"

        if y_ref is not None:
            y_2d = (y_ref[..., 0] > 0.5) if y_ref.ndim == 3 else (y_ref > 0.5)
            ref_overlap_cnt = int(np.sum(reg_mask & y_2d))
            ref_overlap_ha = round(float(ref_overlap_cnt * ha_per_pixel), 2)
            ref_frac = round(float(ref_overlap_cnt / cnt), 3)

            if ref_frac >= 0.50:
                ref_status = f"Confirmed PRODES reference loss ({ref_frac:.1%} overlap)"
            elif ref_frac >= 0.05:
                ref_status = f"Partial overlap with PRODES reference ({ref_frac:.1%})"
            else:
                ref_status = "Model-flagged anomaly not in PRODES 2021 reference (<6.25 ha MMU or unmapped)"

        # Geographic coordinate calculation
        centroid_utm = None
        centroid_wgs84 = None
        bbox_utm = None
        bbox_wgs84 = None

        if georeferenced and top_left_x is not None and top_left_y is not None:
            # Centroid in UTM
            e_c = round(top_left_x + (c_c + 0.5) * pixel_resolution_m, 1)
            n_c = round(top_left_y - (r_c + 0.5) * pixel_resolution_m, 1)
            centroid_utm = (e_c, n_c)

            # Bounding box in UTM
            min_e = round(top_left_x + c_min * pixel_resolution_m, 1)
            max_e = round(top_left_x + (c_max + 1) * pixel_resolution_m, 1)
            max_n = round(top_left_y - r_min * pixel_resolution_m, 1)
            min_n = round(top_left_y - (r_max + 1) * pixel_resolution_m, 1)
            bbox_utm = (min_e, min_n, max_e, max_n)

            # Transform to WGS84
            if transformer is not None:
                lon_c, lat_c = transformer.transform(e_c, n_c)
                centroid_wgs84 = (round(float(lon_c), 5), round(float(lat_c), 5))

                lon1, lat1 = transformer.transform(min_e, min_n)
                lon2, lat2 = transformer.transform(max_e, max_n)
                bbox_wgs84 = (
                    round(min(lon1, lon2), 5),
                    round(min(lat1, lat2), 5),
                    round(max(lon1, lon2), 5),
                    round(max(lat1, lat2), 5),
                )

        candidate_regions.append(
            CandidateRegion(
                region_id=f"Area {rank:02d}",
                rank=rank,
                pixel_count=cnt,
                area_ha=area_ha,
                bbox_pixel=bbox_pixel,
                centroid_pixel=centroid_pixel,
                mean_probability=p_mean,
                max_probability=p_max,
                valid_fraction=valid_frac,
                ref_overlap_pixels=ref_overlap_cnt,
                ref_overlap_ha=ref_overlap_ha,
                ref_overlap_fraction=ref_frac,
                ref_status=ref_status,
                georeferenced=georeferenced,
                crs=source_crs if georeferenced else None,
                centroid_utm=centroid_utm,
                centroid_wgs84=centroid_wgs84,
                bbox_utm=bbox_utm,
                bbox_wgs84=bbox_wgs84,
                dates=dates_str,
                sensor=sensor_str,
            )
        )

    return candidate_regions


def export_regions_to_geojson(
    regions: List[CandidateRegion],
    patch_id: str,
) -> Optional[str]:
    """Export georeferenced candidate regions as a standardized GeoJSON FeatureCollection string.

    Returns None if the regions lack real-world georeferencing.
    """
    if not regions:
        return None

    # Check if first region is georeferenced
    if not regions[0].georeferenced or regions[0].bbox_wgs84 is None:
        return None

    features = []
    for reg in regions:
        if reg.bbox_wgs84 is None:
            continue
        min_lon, min_lat, max_lon, max_lat = reg.bbox_wgs84
        geom = box(min_lon, min_lat, max_lon, max_lat)

        props = {
            "patch_id": patch_id,
            "region_id": reg.region_id,
            "area_ha": reg.area_ha,
            "pixel_count": reg.pixel_count,
            "mean_probability": reg.mean_probability,
            "max_probability": reg.max_probability,
            "reference_status": reg.ref_status,
            "reference_overlap_ha": reg.ref_overlap_ha,
            "observation_dates": reg.dates,
            "sensor": reg.sensor,
            "centroid_wgs84": list(reg.centroid_wgs84) if reg.centroid_wgs84 else None,
            "centroid_utm_epsg32720": list(reg.centroid_utm) if reg.centroid_utm else None,
        }

        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": props,
        })

    feature_collection = {
        "type": "FeatureCollection",
        "name": f"deforestation_candidate_areas_{patch_id}",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    return json.dumps(feature_collection, indent=2)
