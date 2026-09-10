"""Checksum-pinned official dataset downloads; never replace existing data."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
import zipfile
from pathlib import Path
from urllib.request import urlopen

SOURCES = {
    'eurosat': {
        'url': 'https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c',
        'archive': 'EuroSATallBands.zip',
        'sha256': '751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59',
        'images': 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif',
    },
    'resisc45': {
        'url': 'https://hf.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e',
        'archive': 'NWPU-RESISC45.zip',
        'sha256': 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53',
        'images': 'NWPU-RESISC45',
    },
}


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download_file(path: Path, url: str, checksum: str) -> None:
    """Publish verified bytes without replacing a pre-existing file."""
    if path.exists():
        if digest(path) != checksum:
            raise ValueError(f'checksum mismatch in existing file; preserved unchanged: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.download')
    try:
        with urlopen(url, timeout=120) as response, partial.open('xb') as target:
            shutil.copyfileobj(response, target)
        if digest(partial) != checksum:
            raise ValueError(f'download checksum mismatch: {path.name}')
        # Unlike rename/replace, link fails if another process created path.
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def extract_missing(archive: zipfile.ZipFile, root: Path, required: list[Path]) -> None:
    """Reject unsafe members; extract only absent expected images, with x mode."""
    root = root.resolve()
    members = {}
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if not target.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f'unsafe archive member: {member.filename}')
        if member.filename in members:
            raise ValueError(f'duplicate archive member: {member.filename}')
        members[member.filename] = member
    for relative in required:
        target = root / relative
        if not target.resolve().is_relative_to(root):
            raise ValueError(f'image path escapes dataset root: {relative}')
        if target.exists():
            continue
        member = members.get(relative.as_posix())
        if member is None or member.is_dir():
            raise ValueError(f'official archive lacks required image: {relative}')
    for relative in required:
        target = root / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(members[relative.as_posix()]) as source:
            # A concurrent writer cannot be overwritten.
            created = False
            try:
                with target.open('xb') as destination:
                    created = True
                    shutil.copyfileobj(source, destination)
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise


def prepare_dataset(dataset: str, root: Path, split_checksums: tuple[str, ...]) -> None:
    source = SOURCES[dataset]
    required = []
    for split, checksum in zip(('train', 'val', 'test'), split_checksums, strict=True):
        path = root / f'{dataset}-{split}.txt'
        download_file(path, f'{source["url"]}/{path.name}', checksum)
        for name in path.read_text().splitlines():
            name = name.strip()
            if not name:
                continue
            if Path(name).name != name or Path(name).suffix != '.jpg':
                raise ValueError(f'unsafe image filename: {name!r}')
            class_name = Path(name).stem.rsplit('_', 1)[0]
            image = Path(name).with_suffix('.tif').name if dataset == 'eurosat' else name
            required.append(Path(source['images']) / class_name / image)
    if all((root / relative).is_file() for relative in required):
        return
    path = root / source['archive']
    download_file(path, f'{source["url"]}/{path.name}', source['sha256'])
    with zipfile.ZipFile(path) as archive:
        extract_missing(archive, root, required)
