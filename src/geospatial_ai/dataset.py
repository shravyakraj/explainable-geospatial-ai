"""Data pipeline for bi-temporal Sentinel-2 patch loading and augmentation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Standard ESA Sentinel-2 L2A surface reflectance scaling factor
REFLECTANCE_SCALE = 10000.0


def load_split_manifest(manifest_path: Path) -> Dict:
    """Load split manifest JSON from disk."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_records(
    manifest_path: Path,
    split_name: str,
    project_root: Optional[Path] = None,
) -> List[Dict]:
    """Retrieve filtered records for a given split, strictly excluding label == -1."""
    if project_root is None:
        project_root = manifest_path.resolve().parent.parent.parent

    manifest = load_split_manifest(manifest_path)
    records = []

    for p in manifest.get("patches", []):
        if p.get("split") == split_name:
            label = int(p.get("label", -1))
            # Strictly filter out excluded patches (buffers, high water/prior loss, marginal)
            if label in (0, 1):
                rec = dict(p)
                p_file = rec["patch_file"]
                # Resolve relative path to absolute
                full_path = project_root / p_file if not Path(p_file).is_absolute() else Path(p_file)
                rec["absolute_path"] = str(full_path)
                records.append(rec)

    logger.info("Loaded %d usable patches for split '%s' (excluded label=-1)", len(records), split_name)
    return records


def load_npz_patch(file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load T1, T2 arrays and integer label from a compressed .npz patch."""
    data = np.load(file_path)
    t1 = data["t1"].astype(np.float32) / REFLECTANCE_SCALE
    t2 = data["t2"].astype(np.float32) / REFLECTANCE_SCALE
    t1 = np.clip(t1, 0.0, 1.0)
    t2 = np.clip(t2, 0.0, 1.0)
    label = int(data["label"])
    return t1, t2, label


def create_tf_dataset(
    records: List[Dict],
    batch_size: int = 8,
    is_training: bool = False,
    augment: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Construct a streaming tf.data.Dataset from patch records."""
    file_paths = [r["absolute_path"] for r in records]
    labels = [int(r["label"]) for r in records]

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(records), seed=seed, reshuffle_each_iteration=True)

    def _py_load_patch(fpath_tensor, label_tensor):
        def _read(fpath):
            fpath_str = fpath.decode("utf-8") if isinstance(fpath, bytes) else str(fpath)
            data = np.load(fpath_str)
            t1 = data["t1"].astype(np.float32) / REFLECTANCE_SCALE
            t2 = data["t2"].astype(np.float32) / REFLECTANCE_SCALE
            t1 = np.clip(t1, 0.0, 1.0)
            t2 = np.clip(t2, 0.0, 1.0)
            return t1, t2

        t1, t2 = tf.numpy_function(_read, [fpath_tensor], [tf.float32, tf.float32])
        t1.set_shape([256, 256, 4])
        t2.set_shape([256, 256, 4])
        lbl = tf.cast(label_tensor, tf.float32)
        return ({"input_t1": t1, "input_t2": t2}, lbl)

    ds = ds.map(_py_load_patch, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and augment:
        def _augment_pair(inputs, label):
            t1 = inputs["input_t1"]
            t2 = inputs["input_t2"]

            # Random horizontal flip (synchronized)
            do_h_flip = tf.random.uniform([], 0, 1) > 0.5
            t1 = tf.cond(do_h_flip, lambda: tf.image.flip_left_right(t1), lambda: t1)
            t2 = tf.cond(do_h_flip, lambda: tf.image.flip_left_right(t2), lambda: t2)

            # Random vertical flip (synchronized)
            do_v_flip = tf.random.uniform([], 0, 1) > 0.5
            t1 = tf.cond(do_v_flip, lambda: tf.image.flip_up_down(t1), lambda: t1)
            t2 = tf.cond(do_v_flip, lambda: tf.image.flip_up_down(t2), lambda: t2)

            # Random 90-degree rotations (synchronized: 0, 90, 180, 270)
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            t1 = tf.image.rot90(t1, k)
            t2 = tf.image.rot90(t2, k)

            return ({"input_t1": t1, "input_t2": t2}, label)

        ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_numpy_arrays(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all records into memory as numpy arrays for deterministic evaluation."""
    t1_list, t2_list, y_list = [], [], []
    for r in records:
        t1, t2, lbl = load_npz_patch(r["absolute_path"])
        t1_list.append(t1)
        t2_list.append(t2)
        y_list.append(lbl)

    return (
        np.array(t1_list, dtype=np.float32),
        np.array(t2_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )
