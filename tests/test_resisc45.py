"""Checks for the two small RESISC45 baselines."""

import contextlib
import io
from pathlib import Path
import stat
import sys
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
from rasterio.enums import Resampling

from experiments.resisc45 import download, rgb, run


class ResiscTests(unittest.TestCase):
    def test_population_statistics_and_feature_order(self):
        images = np.arange(12, dtype=np.uint8).reshape(1, 3, 2, 2)
        expected = [[1.5, np.sqrt(1.25), 0, 3, 5.5, np.sqrt(1.25), 4, 7,
                     9.5, np.sqrt(1.25), 8, 11]]
        np.testing.assert_allclose(run.image_statistics(images), expected)

    def test_fixed_33_recipe(self):
        images = np.random.default_rng(8).integers(0, 256, (2, 3, 64, 64), dtype=np.uint8)
        values, names = rgb.rgb_feature_pool(images)
        self.assertEqual(values.shape, (2, 147))
        self.assertEqual(tuple(names[i] for i in rgb.SELECTED_INDICES), rgb.SELECTED_NAMES)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(44 * (len(rgb.SELECTED_INDICES) + 1), 1496)
        self.assertEqual(44 * (12 + 1), 572)

    def test_jpeg_reader_uses_rasterio_bilinear_uint8(self):
        source = MagicMock(count=3, height=256, width=256)
        source.read.return_value = np.zeros((3, 64, 64), dtype=np.uint8)
        with patch.object(run.rasterio, 'open') as opened:
            opened.return_value.__enter__.return_value = source
            actual = run.read_image(Path('patch.jpg'))
        source.read.assert_called_once_with(out_shape=(3, 64, 64), resampling=Resampling.bilinear)
        self.assertEqual(actual.dtype, np.uint8)

    def test_check_compares_current_reference_counts(self):
        expected = (Path(run.__file__).parent / 'results.csv').read_text()
        with patch.object(Path, 'open', side_effect=[io.StringIO(expected), io.StringIO(expected)]):
            run.check_results(Path('unused'))
        wrong = expected.replace(',3721,', ',3720,')
        with patch.object(Path, 'open', side_effect=[io.StringIO(expected), io.StringIO(wrong)]):
            with self.assertRaisesRegex(ValueError, 'correct differs'):
                run.check_results(Path('unused'))

    def test_output_is_confined_to_new_runtime_directories(self):
        for path in ('models', 'data/cache', 'output'):
            with (
                patch.object(sys, 'argv', ['run', '--output', str(run.ROOT / path)]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                run.main()

    def test_download_never_replaces_invalid_existing_files(self):
        path = MagicMock(spec=Path)
        path.exists.return_value = True
        with patch.object(download, 'digest', return_value='wrong'):
            with patch.object(download, 'urlopen') as request:
                with self.assertRaisesRegex(ValueError, 'preserved unchanged'):
                    download.download_file(path, 'https://example.invalid/file', 'expected')
        request.assert_not_called()
        path.unlink.assert_not_called()

    def test_unsafe_archive_members_are_rejected(self):
        for name, permissions in (('../outside.jpg', 0), ('/absolute.jpg', 0),
                                  ('link.jpg', (stat.S_IFLNK | 0o777) << 16)):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                member = zipfile.ZipInfo(name)
                member.external_attr = permissions
                archive.writestr(member, 'not an image')
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive:
                with self.assertRaisesRegex(ValueError, 'unsafe archive member'):
                    download.extract_missing(archive, run.ROOT / 'output/unwritten', [])


if __name__ == '__main__':
    unittest.main()
