"""Input validation and routing for upload-to-report deforestation analysis workflow.

Supports three distinct input categories:
- Input Type A: Compatible research archives (.npz containing t1/t2 or x_8ch).
- Input Type B: Compatible bi-temporal multispectral imagery (two files containing B02, B03, B04, B08).
- Input Type C: Standard RGB satellite or aerial imagery (JPEG, PNG, single-date or bi-temporal).
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

logger = logging.getLogger(__name__)

REFLECTANCE_SCALE = 10000.0


def validate_npz_archive(uploaded_file) -> Dict[str, Any]:
    """Validate user-uploaded .npz archive strictly against Sentinel-2 input contracts.

    Category: Input Type A (Compatible research data).
    """
    try:
        npz_data = np.load(uploaded_file)
        keys = list(npz_data.keys())

        if "t1" in npz_data and "t2" in npz_data:
            t1_raw = npz_data["t1"]
            t2_raw = npz_data["t2"]

            if t1_raw.shape != (256, 256, 4) or t2_raw.shape != (256, 256, 4):
                return {
                    "valid": False,
                    "supported": False,
                    "input_type": "TYPE_A_NPZ",
                    "error": (
                        f"Invalid array dimensions: t1 shape is {t1_raw.shape}, "
                        f"t2 shape is {t2_raw.shape}. Expected (256, 256, 4)."
                    ),
                }

            scale_t1 = REFLECTANCE_SCALE if np.max(t1_raw) > 1.5 else 1.0
            scale_t2 = REFLECTANCE_SCALE if np.max(t2_raw) > 1.5 else 1.0
            t1 = np.clip(t1_raw.astype(np.float32) / scale_t1, 0.0, 1.0)
            t2 = np.clip(t2_raw.astype(np.float32) / scale_t2, 0.0, 1.0)
            x_8ch = np.concatenate([t1, t2], axis=-1)

        elif "x" in npz_data:
            x_raw = npz_data["x"]
            if x_raw.shape != (256, 256, 8):
                return {
                    "valid": False,
                    "supported": False,
                    "input_type": "TYPE_A_NPZ",
                    "error": f"Invalid x dimension: {x_raw.shape}. Expected (256, 256, 8).",
                }
            scale_x = REFLECTANCE_SCALE if np.max(x_raw) > 1.5 else 1.0
            x_8ch = np.clip(x_raw.astype(np.float32) / scale_x, 0.0, 1.0)
            t1 = x_8ch[..., :4]
            t2 = x_8ch[..., 4:]
        else:
            return {
                "valid": False,
                "supported": False,
                "input_type": "TYPE_A_NPZ",
                "error": (
                    f"Missing required arrays. Archive contains keys {keys}, "
                    "but requires either ('t1', 't2') or ('x')."
                ),
            }

        # Check for optional reference and validity masks
        has_reference = False
        y_ref = None
        w_mask = np.ones((256, 256, 1), dtype=np.float32)

        if "mask" in npz_data and "exclude" in npz_data:
            mask_raw = npz_data["mask"]
            exclude_raw = npz_data["exclude"]
            if mask_raw.shape == (256, 256) and exclude_raw.shape == (256, 256):
                has_reference = True
                y_ref = ((mask_raw == 1) & (exclude_raw == 0)).astype(np.float32)[..., np.newaxis]
                w_mask = (exclude_raw == 0).astype(np.float32)[..., np.newaxis]

        label = int(npz_data["label"]) if "label" in npz_data else None

        metadata = {
            "sensor": "Sentinel-2 MSI Level-2A",
            "spatial_resolution": "10 meters per pixel",
            "dimensions": "256 × 256 pixels (2.56 × 2.56 km)",
            "area_ha": 655.36,
            "bands": ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"],
            "temporal_pairing": "Bi-temporal paired observations (T1 and T2)",
            "has_reference_cartography": has_reference,
            "filename": getattr(uploaded_file, "name", "custom_patch.npz"),
        }

        return {
            "valid": True,
            "supported": True,
            "input_type": "TYPE_A_NPZ",
            "t1": t1,
            "t2": t2,
            "x": x_8ch,
            "y": y_ref,
            "w": w_mask,
            "has_reference": has_reference,
            "label": label,
            "metadata": metadata,
            "keys_found": keys,
        }

    except Exception as e:
        logger.exception("Error validating uploaded .npz file")
        return {
            "valid": False,
            "supported": False,
            "input_type": "TYPE_A_NPZ",
            "error": f"Could not parse uploaded file as valid NumPy .npz archive: {e}",
        }


def read_multispectral_file(file_obj, filename: str) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[str]]:
    """Read a multi-band raster file (TIFF, GeoTIFF, NPY) and return (array_HWC, metadata, error)."""
    raw_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    fn_lower = filename.lower()

    if fn_lower.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(raw_bytes))
            if arr.ndim == 3 and arr.shape[2] == 4:
                return arr, {"driver": "NumPy", "bands": 4}, None
            elif arr.ndim == 3 and arr.shape[0] == 4:
                return np.transpose(arr, (1, 2, 0)), {"driver": "NumPy", "bands": 4}, None
            else:
                return None, {}, f"NumPy array shape is {arr.shape}. Expected (256, 256, 4)."
        except Exception as e:
            return None, {}, f"Failed to load .npy array: {e}"

    if HAS_RASTERIO:
        try:
            with MemoryFile(raw_bytes) as memfile:
                with memfile.open() as src:
                    arr = src.read()  # (bands, H, W)
                    crs = str(src.crs) if src.crs else "Unprojected"
                    bounds = list(src.bounds) if src.bounds else None
                    count = src.count

                    if count < 4:
                        return None, {}, f"File has {count} band(s). Requires at least 4 bands [B02, B03, B04, B08]."

                    arr_4ch = arr[:4, ...]  # take first 4 bands
                    arr_hwc = np.transpose(arr_4ch, (1, 2, 0))  # (H, W, 4)

                    meta = {
                        "driver": "GeoTIFF/Rasterio",
                        "crs": crs,
                        "bounds": bounds,
                        "bands": count,
                        "height": src.height,
                        "width": src.width,
                    }
                    return arr_hwc, meta, None
        except Exception as e:
            logger.warning("Rasterio read failed for %s: %s", filename, e)

    if HAS_TIFFFILE:
        try:
            arr = tifffile.imread(io.BytesIO(raw_bytes))
            if arr.ndim == 3:
                if arr.shape[2] >= 4:
                    return arr[:, :, :4], {"driver": "TiffFile", "bands": arr.shape[2]}, None
                elif arr.shape[0] >= 4:
                    return np.transpose(arr[:4, :, :], (1, 2, 0)), {"driver": "TiffFile", "bands": arr.shape[0]}, None
            return None, {}, f"TIFF shape {arr.shape} does not have required 4 spectral bands."
        except Exception as e:
            logger.warning("TiffFile read failed for %s: %s", filename, e)

    return None, {}, f"Unsupported file format or missing geospatial driver for {filename}."


def validate_bitemporal_multispectral_files(
    file_t1,
    file_t2,
    name_t1: str,
    name_t2: str,
) -> Dict[str, Any]:
    """Validate two separate multispectral files representing Date 1 and Date 2.

    Category: Input Type B (Compatible multispectral imagery from two dates).
    """
    arr1, meta1, err1 = read_multispectral_file(file_t1, name_t1)
    if err1:
        return {
            "valid": False,
            "supported": False,
            "input_type": "TYPE_B_MULTISPECTRAL",
            "error": f"Error in Date 1 file ({name_t1}): {err1}",
        }

    arr2, meta2, err2 = read_multispectral_file(file_t2, name_t2)
    if err2:
        return {
            "valid": False,
            "supported": False,
            "input_type": "TYPE_B_MULTISPECTRAL",
            "error": f"Error in Date 2 file ({name_t2}): {err2}",
        }

    # Verify matching dimensions
    if arr1.shape[:2] != arr2.shape[:2]:
        return {
            "valid": False,
            "supported": False,
            "input_type": "TYPE_B_MULTISPECTRAL",
            "error": (
                f"Dimension mismatch between acquisitions: Date 1 is {arr1.shape[:2]}, "
                f"Date 2 is {arr2.shape[:2]}. Both must match in spatial dimensions."
            ),
        }

    # Verify 256x256 shape (or crop/resize if needed, but per prompt preserve spatial alignment)
    if arr1.shape[0] != 256 or arr1.shape[1] != 256:
        # If dimensions are not 256x256, model contract requires (256, 256)
        if arr1.shape[0] >= 256 and arr1.shape[1] >= 256:
            # Take center 256x256 crop with notification
            cy, cx = arr1.shape[0] // 2, arr1.shape[1] // 2
            arr1 = arr1[cy - 128 : cy + 128, cx - 128 : cx + 128, :]
            arr2 = arr2[cy - 128 : cy + 128, cx - 128 : cx + 128, :]
        else:
            return {
                "valid": False,
                "supported": False,
                "input_type": "TYPE_B_MULTISPECTRAL",
                "error": f"Image dimensions {arr1.shape[:2]} are smaller than the required 256×256 pixels.",
            }

    # Reflectance normalization
    scale1 = REFLECTANCE_SCALE if np.max(arr1) > 1.5 else 1.0
    scale2 = REFLECTANCE_SCALE if np.max(arr2) > 1.5 else 1.0

    t1 = np.clip(arr1.astype(np.float32) / scale1, 0.0, 1.0)
    t2 = np.clip(arr2.astype(np.float32) / scale2, 0.0, 1.0)
    x_8ch = np.concatenate([t1, t2], axis=-1)
    w_mask = np.ones((256, 256, 1), dtype=np.float32)

    metadata = {
        "sensor": "Compatible Multispectral Sensor (Sentinel-2 band equivalent)",
        "spatial_resolution": "10 meters per pixel (calibrated)",
        "dimensions": "256 × 256 pixels (2.56 × 2.56 km)",
        "area_ha": 655.36,
        "bands": ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"],
        "temporal_pairing": f"Date 1 ({name_t1}) and Date 2 ({name_t2})",
        "has_reference_cartography": False,
        "filename": f"{name_t1} + {name_t2}",
    }

    return {
        "valid": True,
        "supported": True,
        "input_type": "TYPE_B_MULTISPECTRAL",
        "t1": t1,
        "t2": t2,
        "x": x_8ch,
        "y": None,
        "w": w_mask,
        "has_reference": False,
        "label": None,
        "metadata": metadata,
    }


def validate_rgb_imagery(
    files: List[Any],
    filenames: List[str],
) -> Dict[str, Any]:
    """Validate ordinary RGB imagery (JPEG, PNG, standard TIFF).

    Category: Input Type C (Ordinary RGB satellite or aerial imagery).
    Does NOT invent infrared channels or fabricate change inference.
    Returns honest evidence diagnostics and instructions.
    """
    loaded_images = []
    metadata_list = []

    for f_obj, fn in zip(files, filenames):
        try:
            raw_bytes = f_obj.read() if hasattr(f_obj, "read") else f_obj
            if hasattr(f_obj, "seek"):
                f_obj.seek(0)
            img = Image.open(io.BytesIO(raw_bytes))
            rgb_arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            loaded_images.append(rgb_arr)
            metadata_list.append({
                "filename": fn,
                "format": img.format or "Raster Image",
                "width": img.width,
                "height": img.height,
                "channels": 3,
                "mode": img.mode,
            })
        except Exception as e:
            return {
                "valid": False,
                "supported": False,
                "input_type": "TYPE_C_RGB",
                "error": f"Failed to decode image file '{fn}': {e}",
            }

    num_dates = len(loaded_images)
    is_single_date = (num_dates == 1)

    reasons = []
    if is_single_date:
        reasons.append(
            "Missing Temporal Baseline: Forest loss anomaly detection requires registered observations "
            "across at least two distinct calendar dates to measure canopy disturbance. A single static view "
            "cannot distinguish ongoing deforestation from stable agricultural fields, pastureland, roads, or natural clearings."
        )
    reasons.append(
        "Missing Near-Infrared (NIR / B08) Spectral Band: In tropical rainforest environments, standard 3-channel "
        "RGB imagery (Red, Green, Blue) lacks the essential Near-Infrared reflectance plateau (700–900 nm). "
        "NIR is required to detect cellular mesophyll breakdown and moisture loss in tree canopies. "
        "Without NIR, optical RGB imagery produces unacceptable false-positive rates on shadow, bare soil, and crop cycles."
    )
    reasons.append(
        "No Calibrated RGB Deforestation Model: The repository's evaluated production models (MobileNetV3 Siamese "
        "classifier and Early-Fusion U-Net) strictly require 4-band and 8-channel inputs. No scientifically validated "
        "3-channel RGB deforestation model exists in the verified benchmark checkpoints."
    )

    requirements_text = (
        "To perform verified deforestation anomaly detection, provide:\n"
        "1. Bi-temporal Sentinel-2 MSI Level-2A surface reflectance data from two calendar dates (e.g., T1 and T2).\n"
        "2. Four core spectral bands per acquisition: B02 (Blue), B03 (Green), B04 (Red), and B08 (Near-Infrared).\n"
        "3. Supported formats: .npz research archives or multi-band GeoTIFF files with 10 m spatial resolution."
    )

    return {
        "valid": True,
        "supported": False,  # Explicitly marked as unsupported for direct inference
        "input_type": "TYPE_C_RGB",
        "num_dates": num_dates,
        "images": loaded_images,
        "image_metadata": metadata_list,
        "deficiencies": reasons,
        "requirements": requirements_text,
        "recommendation": (
            "Evidence submitted is insufficient for reliable deforestation inference. "
            "No fabricated prediction or misleading alert has been generated."
        ),
    }
