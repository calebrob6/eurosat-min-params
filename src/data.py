"""EuroSAT data loading utilities.

The EuroSAT dataset ships as 27,000 64x64 patches with 13 Sentinel-2 bands each
(``uint16`` GeoTIFFs).  The official train/val/test split is provided as three
text files listing ``<Class>_<id>.jpg`` filenames; we map those to the
corresponding 13-band ``.tif`` files.

Band-name lists describe the physical TIFF channels. Historical numeric SWIR
aliases below retain their trained positions independently of those labels.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterator

import numpy as np
import rasterio

# --- constants -------------------------------------------------------------

CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

TIFF_BAND_NAMES = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
    'B08', 'B09', 'B10', 'B11', 'B12', 'B8A',
]
BAND_NAMES = TIFF_BAND_NAMES
# Legacy SWIR aliases index B12 and B8A, respectively. See REPRODUCIBILITY.md.
B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2 = 1, 2, 3, 7, 11, 12

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'EuroSAT')
DATA_ROOT = os.path.abspath(DATA_ROOT)
TIF_ROOT = os.path.join(
    DATA_ROOT, 'ds', 'images', 'remote_sensing',
    'otherDatasets', 'sentinel_2', 'tif',
)
SPLIT_FILES = {
    'train': os.path.join(DATA_ROOT, 'eurosat-train.txt'),
    'val': os.path.join(DATA_ROOT, 'eurosat-val.txt'),
    'test': os.path.join(DATA_ROOT, 'eurosat-test.txt'),
}


def _class_of(filename: str) -> str:
    """Class name is the prefix before the final ``_<id>``."""
    base = filename.rsplit('.', 1)[0]
    return base.rsplit('_', 1)[0]


def list_split(split: str) -> tuple[list[str], np.ndarray]:
    """Return (tif paths, integer labels) for a split ('train'|'val'|'test')."""
    paths: list[str] = []
    labels: list[int] = []
    with open(SPLIT_FILES[split]) as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            cls = _class_of(name)
            stem = name.rsplit('.', 1)[0]
            paths.append(os.path.join(TIF_ROOT, cls, stem + '.tif'))
            labels.append(CLASS_TO_IDX[cls])
    return paths, np.asarray(labels, dtype=np.int64)


def read_tif(path: str) -> np.ndarray:
    """Read a 13-band patch as a ``float32`` array of shape (13, 64, 64)."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def iter_images(split: str, batch_size: int = 128) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream original TIFFs in split-file order; never read NumPy caches."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    paths, labels = list_split(split)
    for start in range(0, len(paths), batch_size):
        images = np.stack([read_tif(path) for path in paths[start:start + batch_size]])
        if images.shape[1:] != (13, 64, 64):
            raise ValueError(f'{split}: expected 13-band 64x64 TIFFs, got {images.shape}')
        yield images, labels[start:start + len(images)]


def load_images(split: str, max_workers: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Load all raw patches for a split as (N, 13, 64, 64) float32 + labels."""
    paths, labels = list_split(split)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        imgs = list(ex.map(read_tif, paths, chunksize=32))
    return np.stack(imgs).astype(np.float32), labels
