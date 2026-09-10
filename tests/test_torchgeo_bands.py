"""Compare the feature input contract with TorchGeo's EuroSAT dataset."""

import importlib.util
import unittest
from pathlib import Path

import rasterio
import torch

from eurosat_features import TIFF_BAND_NAMES

ROOT = Path(__file__).resolve().parents[1] / 'data/EuroSAT'


@unittest.skipUnless(importlib.util.find_spec('torchgeo'), 'requires optional TorchGeo')
class TorchGeoBandTests(unittest.TestCase):
    def test_physical_band_names_match_torchgeo(self) -> None:
        from torchgeo.datasets import EuroSAT

        self.assertEqual(TIFF_BAND_NAMES, EuroSAT.all_band_names)

    def test_default_dataset_returns_the_original_tiff_channel_order(self) -> None:
        from torchgeo.datasets import EuroSAT

        if (
            not (ROOT / 'eurosat-val.txt').is_file()
            or not (ROOT / EuroSAT.base_dir).is_dir()
        ):
            self.skipTest('original EuroSAT data is not present')
        dataset = EuroSAT(root=ROOT, split='val', download=False)
        self.assertEqual(tuple(dataset.bands), TIFF_BAND_NAMES)
        self.assertEqual(dataset.band_indices.tolist(), list(range(13)))
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            with self.subTest(index=index):
                with rasterio.open(dataset.samples[index][0]) as source:
                    raw = torch.from_numpy(source.read(out_dtype='float32'))
                torch.testing.assert_close(dataset[index]['image'], raw, rtol=0, atol=0)

    def test_explicit_band_selection_is_not_the_default_contract(self) -> None:
        from torchgeo.datasets import EuroSAT

        if (
            not (ROOT / 'eurosat-val.txt').is_file()
            or not (ROOT / EuroSAT.base_dir).is_dir()
        ):
            self.skipTest('original EuroSAT data is not present')
        dataset = EuroSAT(
            root=ROOT, split='val', bands=EuroSAT.rgb_bands, download=False
        )
        self.assertEqual(dataset.band_indices.tolist(), [3, 2, 1])
        self.assertEqual(tuple(dataset[0]['image'].shape), (3, 64, 64))


if __name__ == '__main__':
    unittest.main()
