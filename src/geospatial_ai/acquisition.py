"""Data acquisition for Sentinel-2 L2A and INPE PRODES datasets."""

import json
import logging
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from geospatial_ai.geo import StudyGrid, LOCKED_GRID

logger = logging.getLogger(__name__)

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TERRABRASILIS_WFS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"

# SSL context for TerraBrasilis INPE GeoServer
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def get_stac_item(item_id: str, collection: str = "sentinel-2-l2a") -> pystac.Item:
    """Retrieve STAC item metadata from Microsoft Planetary Computer."""
    url = f"{STAC_API_URL}/collections/{collection}/items/{item_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "geospatial-ai-deforestation"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return pystac.Item.from_dict(data)


def acquire_sentinel2_window(
    item_id: str,
    bands: list[str],
    output_dir: Path,
    time_prefix: str,
    grid: StudyGrid = LOCKED_GRID,
    max_retries: int = 3,
) -> dict[str, Path]:
    """Download windowed Sentinel-2 bands directly into EPSG:32720 GeoTIFFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    col_off_10m = 1212
    row_off_10m = 3958

    env_config = {
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "500000000",  # 500 MB to prevent cache thrashing
    }

    with rasterio.Env(**env_config):
        for band in bands:
            out_file = output_dir / f"{time_prefix}_{band}.tif"
            if out_file.exists():
                logger.info("Band already exists: %s", out_file)
                saved_paths[band] = out_file
                continue

            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    # Re-fetch item to refresh SAS token if needed
                    item = get_stac_item(item_id)
                    signed_item = pc.sign(item)
                    asset_href = signed_item.assets[band].href
                    logger.info("Fetching %s %s from COG (attempt %d/%d)...", time_prefix, band, attempt, max_retries)

                    with rasterio.open(asset_href) as src:
                        if band == "SCL":
                            win_scl = Window(col_off=col_off_10m // 2, row_off=row_off_10m // 2,
                                             width=grid.width_pixels // 2, height=grid.height_pixels // 2)
                            data = src.read(
                                1,
                                window=win_scl,
                                out_shape=(grid.height_pixels, grid.width_pixels),
                                resampling=Resampling.nearest,
                            )
                            dtype = rasterio.uint8
                        else:
                            win_10m = Window(col_off=col_off_10m, row_off=row_off_10m,
                                             width=grid.width_pixels, height=grid.height_pixels)
                            data = src.read(1, window=win_10m)
                            dtype = rasterio.uint16

                        # Write out common grid GeoTIFF
                        profile = {
                            "driver": "GTiff",
                            "dtype": dtype,
                            "count": 1,
                            "crs": grid.crs,
                            "transform": grid.transform,
                            "width": grid.width_pixels,
                            "height": grid.height_pixels,
                            "compress": "deflate",
                            "nodata": 0 if band != "SCL" else 0,
                        }

                        with rasterio.open(out_file, "w", **profile) as dst:
                            dst.write(data.astype(dtype), 1)

                    logger.info("Saved %s (shape: %s)", out_file, data.shape)
                    saved_paths[band] = out_file
                    success = True
                    break
                except Exception as exc:
                    logger.warning("Error fetching %s %s on attempt %d: %s", time_prefix, band, attempt, exc)
                    if out_file.exists():
                        out_file.unlink()
                    if attempt < max_retries:
                        time.sleep(3 * attempt)
                    else:
                        raise

    return saved_paths


def fetch_terrabrasilis_wfs_geojson(
    type_name: str,
    cql_filter: str,
    output_path: Path,
    max_features: int = 15000,
) -> Path:
    """Query TerraBrasilis GeoServer WFS and save resulting GeoJSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info("GeoJSON already exists: %s", output_path)
        return output_path

    logger.info("Querying WFS layer: %s ...", type_name)
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
        "maxFeatures": str(max_features),
        "cql_filter": cql_filter,
    }
    url = f"{TERRABRASILIS_WFS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "geospatial-ai-deforestation"})

    with urllib.request.urlopen(req, context=SSL_CTX, timeout=120) as resp:
        content = resp.read()

    with open(output_path, "wb") as f:
        f.write(content)

    logger.info("Saved %s (%d bytes)", output_path, len(content))
    return output_path
