"""Small TIFF-to-tensor feature extraction checks."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from eurosat_features import EuroSATFeatures
from extract_features import input_paths

ROOT = Path(__file__).resolve().parents[1]


class FeatureCLITests(unittest.TestCase):
    def test_input_collection_rejects_empty_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'no TIFF'):
                input_paths([root])
            path = root / 'one.tif'
            path.touch()
            self.assertEqual(input_paths([root]), [path])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                input_paths([path, root])
            with self.assertRaisesRegex(ValueError, 'expected a TIFF'):
                input_paths([root / 'missing.tif'])

    def test_cli_writes_features_and_names_without_overwriting(self) -> None:
        image = np.random.default_rng(51).integers(
            1, 10000, (13, 64, 64), dtype=np.uint16
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / 'patch.tif', root / 'features.npz'
            with rasterio.open(
                path,
                'w',
                driver='GTiff',
                width=64,
                height=64,
                count=13,
                dtype='uint16',
                transform=from_origin(500000, 1000000, 10, 10),
                crs='EPSG:32632',
            ) as handle:
                handle.write(image)
            command = [
                sys.executable,
                str(ROOT / 'extract_features.py'),
                str(path),
                '--features',
                '389',
                '--output',
                str(output),
            ]
            subprocess.run(
                command, cwd=ROOT, check=True, capture_output=True, text=True
            )
            extractor = EuroSATFeatures('389')
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result['features'].shape, (1, 389))
                self.assertEqual(
                    result['feature_names'].tolist(), list(extractor.feature_names)
                )
                self.assertEqual(result['filenames'].tolist(), [str(path)])
                np.testing.assert_allclose(
                    result['features'],
                    extractor(torch.from_numpy(image[None].astype(np.float32))).numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )
            previous = output.read_bytes()
            failed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('output already exists', failed.stderr)
            self.assertEqual(previous, output.read_bytes())
            self.assertEqual(
                sorted(p.name for p in root.iterdir()), ['features.npz', 'patch.tif']
            )


if __name__ == '__main__':
    unittest.main()
