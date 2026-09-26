"""EuroSAT download, splits, and feature extraction shared by the experiments."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data' / 'EuroSAT'
TIF_ROOT = DATA_ROOT / 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake',
]  # fmt: skip
SPLITS = ('train', 'val', 'test')
SIZES = (16200, 5400, 5400)
DATA_URL = 'https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c'
CHECKSUMS = {
    'EuroSATallBands.zip': '751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59',
    'eurosat-train.txt': '1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695',
    'eurosat-val.txt': 'b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330',
    'eurosat-test.txt': 'cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592',
    'eurosat-spatial-train.txt': '2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f',
    'eurosat-spatial-val.txt': '6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b',
    'eurosat-spatial-test.txt': 'de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452',
}


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _download(path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urlopen(f'{DATA_URL}/{path.name}', timeout=120) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
        if digest(temporary) != CHECKSUMS[path.name]:
            raise ValueError(f'checksum mismatch downloading {path.name}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_data(download: bool = False) -> None:
    """Check the split files and TIFFs, downloading and extracting them if asked."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, checksum in CHECKSUMS.items():
        path = DATA_ROOT / name
        if name.endswith('.zip'):
            continue
        if not path.exists() and download:
            _download(path)
        if not path.exists():
            raise FileNotFoundError(f'{path}; pass --download to fetch EuroSAT')
        if digest(path) != checksum:
            raise ValueError(f'checksum mismatch: {path}')
    paths = [p for split in SPLITS for p in list_split(split)[0]]
    if all(p.is_file() for p in paths):
        return
    archive = DATA_ROOT / 'EuroSATallBands.zip'
    if not archive.exists():
        if not download:
            raise FileNotFoundError('EuroSAT TIFFs missing; pass --download')
        _download(archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(DATA_ROOT)
    if not all(p.is_file() for p in paths):
        raise ValueError('archive did not supply every split image')


def list_split(split: str) -> tuple[list[Path], np.ndarray]:
    """TIFF paths and integer labels for 'train', 'val', 'test', or 'spatial-{split}'."""
    paths, labels = [], []
    for line in (DATA_ROOT / f'eurosat-{split}.txt').read_text().split():
        stem = line.rsplit('.', 1)[0]
        name = stem.rsplit('_', 1)[0]
        paths.append(TIF_ROOT / name / f'{stem}.tif')
        labels.append(CLASSES.index(name))
    return paths, np.asarray(labels, dtype=np.int64)


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def iter_images(split: str, batch_size: int = 256) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (images (B, 13, 64, 64), labels) batches in split-file order."""
    paths, labels = list_split(split)
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        yield np.stack([read_tif(p) for p in batch]), labels[start : start + len(batch)]


def extract(extractors: dict[str, torch.nn.Module], split: str, device: str, batch_size: int = 256) -> dict[str, np.ndarray]:
    """Run several feature extractors over a split in one pass; returns name -> (N, F) arrays."""
    extractors = {name: e.to(device) for name, e in extractors.items()}
    parts = {name: [] for name in extractors}
    for x, _ in iter_images(split, batch_size):
        x = torch.from_numpy(x).to(device)
        for name, extractor in extractors.items():
            parts[name].append(extractor(x).cpu().numpy())
    return {name: np.concatenate(p) for name, p in parts.items()}


def default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
