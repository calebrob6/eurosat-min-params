"""Small data-free checks for the article's auxiliary reference commands."""
from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import stat
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from rasterio.enums import Resampling

from experiments.article_baselines import download, rgb, run


class ArticleBaselineTests(unittest.TestCase):
    def test_mosaiks_counts_input_normalization_as_well_as_the_head(self):
        from experiments.article_baselines.other import mosaiks_parameter_counts

        self.assertEqual(
            mosaiks_parameter_counts(np.zeros((9, 512)), np.zeros(9),
                                     np.zeros((1, 13, 1, 1)), np.ones((1, 13, 1, 1))),
            {'learned_parameters': 4643, 'head_parameters': 4617, 'normalization_parameters': 26},
        )

    def test_optional_cnn_rejects_missing_cuda_before_staging_inputs(self):
        from experiments.article_baselines import other

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with (
            patch.dict('sys.modules', {'torch': fake_torch}),
            patch.object(other.sys, 'argv', [
                'other.py', '--method', 'tiny-cnn', '--output-dir', 'output/not-created',
            ]),
            patch.object(other, 'prepare_raw_arrays') as stage,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            other.main()
        stage.assert_not_called()

    def test_statistics_are_band_major_population_statistics(self):
        images = np.array([[[[0, 2], [4, 6]], [[2, 2], [2, 2]]]], dtype=np.uint16)
        values, names = run.statistics(images)
        np.testing.assert_allclose(values, [[3, np.sqrt(5), 0, 6, 2, 0, 2, 2]])
        self.assertEqual(names[:4], ['mean_band0', 'std_band0', 'min_band0', 'max_band0'])
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(run.statistics_indices(2, 'means'), [0, 4])
        np.testing.assert_array_equal(run.statistics_indices(2, 'mean_std'), [0, 1, 4, 5])
        np.testing.assert_array_equal(run.statistics_indices(2, 'imagestats'), np.arange(8))

    def test_archived_rgb_pool_schema_and_batch_independence(self):
        images = np.random.default_rng(8).integers(0, 256, (3, 3, 64, 64), dtype=np.uint8)
        features, names = rgb.rgb_feature_pool(images)
        self.assertEqual(features.shape, (3, 147))
        self.assertEqual(tuple(names[i] for i in rgb.SELECTED_INDICES), rgb.SELECTED_NAMES)
        self.assertEqual(len(set(names)), 147)
        self.assertTrue(np.isfinite(features).all())
        separately = np.concatenate([rgb.rgb_feature_pool(image[None])[0] for image in images])
        np.testing.assert_allclose(features, separately, rtol=1e-6, atol=1e-6)
        self.assertEqual(44 * (12 + 1), 572)
        self.assertEqual(44 * (len(rgb.SELECTED_INDICES) + 1), 1496)

    def test_rgb_constant_and_black_images_are_finite(self):
        images = np.stack([np.zeros((3, 64, 64)), np.full((3, 64, 64), 255)])
        features, _ = rgb.rgb_feature_pool(images)
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaisesRegex(ValueError, 'expected RGB'):
            rgb.rgb_feature_pool(np.zeros((1, 13, 64, 64)))

    def test_selection_uses_validation_only_and_archived_one_se_rule(self):
        sweep = [
            dict(C=100, validation_accuracy=.615714),
            dict(C=30, validation_accuracy=.612698),
            dict(C=10, validation_accuracy=.608),
        ]
        self.assertEqual(run.choose_c(sweep, 6300), 100)
        self.assertEqual(run.choose_c(sweep, 6300, one_se=True), 30)
        self.assertEqual(run.choose_c([
            dict(C=3, validation_accuracy=.5), dict(C=1, validation_accuracy=.5),
        ], 10), 3)
        with self.assertRaises(ValueError):
            run.choose_c([], 10)

    def test_refit_receives_train_and_val_without_test(self):
        train = dict(statistics=np.arange(12).reshape(4, 3), labels=np.array([0, 1, 0, 1]))
        val = dict(statistics=np.arange(6).reshape(2, 3), labels=np.array([0, 1]))
        model = (np.zeros((2, 2)), np.zeros(2), np.array([0, 1]))
        with patch.object(run, 'fit_folded_logreg', return_value=model) as fit:
            with patch.object(run, 'predict', side_effect=[np.array([1, 1]), np.array([0, 1])]):
                _, sweep, c = run.fit_candidates(train, val, 'means', np.array([0, 1]), (1, 3))
        self.assertEqual(c, 3)
        self.assertEqual(len(sweep), 2)
        self.assertIs(fit.call_args_list[0].args[0], train['statistics'])
        self.assertIs(fit.call_args_list[0].args[1], train['labels'])

    def test_jpeg_reader_preserves_archived_rasterio_bilinear_quantization(self):
        source = MagicMock(count=3, height=256, width=256)
        source.read.return_value = np.zeros((3, 64, 64), dtype=np.uint8)
        with patch.object(run.rasterio, 'open') as opened:
            opened.return_value.__enter__.return_value = source
            image = run.read_image(('resisc45', 'original.jpg'))
        source.read.assert_called_once_with(out_shape=(3, 64, 64), resampling=Resampling.bilinear)
        self.assertEqual(image.dtype, np.uint8)
        source.height = 64
        with patch.object(run.rasterio, 'open') as opened:
            opened.return_value.__enter__.return_value = source
            with self.assertRaisesRegex(ValueError, 'original'):
                run.read_image(('resisc45', 'already_resized.jpg'))

    def test_raw_extraction_keeps_order_and_does_not_load_caches(self):
        images = [np.full((13, 64, 64), i, dtype=np.uint16) for i in (3, 1, 2)]
        record = dict(paths=['a', 'b', 'c'], labels=np.array([2, 1, 0]))
        with patch.object(run, 'read_image', side_effect=images) as read:
            with patch.object(np, 'load', side_effect=AssertionError('cache read forbidden')):
                result = run.extract_split('eurosat', record, 2, 1)
        self.assertEqual([call.args[0] for call in read.call_args_list],
                         [('eurosat', name) for name in ('a', 'b', 'c')])
        np.testing.assert_array_equal(result['statistics'][:, 0], [3, 1, 2])
        self.assertEqual(result['decoded_image_sha256'],
                         hashlib.sha256(np.stack(images).tobytes()).hexdigest())

    def test_output_cannot_target_historical_models_or_caches(self):
        for path in ('output', 'submissions/14_reference_class_96', 'data/cache', 'experiments/article_baselines'):
            with self.assertRaisesRegex(ValueError, 'protected'):
                run.output_directory('eurosat', run.ROOT / path)

    def test_official_split_checksum_failure_precedes_image_reads(self):
        with patch.object(Path, 'is_file', return_value=True):
            with patch.object(run, 'file_digest', return_value='not-the-official-checksum'):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    run.dataset_lists('eurosat', Path('data/EuroSAT'))

    def test_archive_manifest_covers_all_quoted_auxiliary_numbers(self):
        path = run.ROOT / 'experiments/article_baselines/archive_manifest.csv'
        with path.open(newline='') as handle:
            rows = {row['claim_id']: row for row in csv.DictReader(handle)}
        expected = {
            'eurosat_means': ('74.8', '126'),
            'eurosat_mean_std': ('87.8', '243'),
            'eurosat_imagestats': ('90.96', '477'),
            'resisc45_imagestats': ('36.63', '572'),
            'resisc45_selected33': ('59.06', '1496'),
            'tiny_cnn': ('89.2', '2058'),
            'mosaiks': ('95', '4617'),
        }
        for claim, (accuracy, parameters) in expected.items():
            self.assertEqual(rows[claim]['article_test_accuracy_percent'], accuracy)
            self.assertEqual(rows[claim]['learned_parameters'], parameters)
            self.assertTrue(rows[claim]['reproduction_status'])
            self.assertTrue(rows[claim]['command'])
            if rows[claim]['source_commit']:
                self.assertEqual(len(rows[claim]['source_commit']), 40)
        self.assertIn('missing', rows['eurosat_means']['reproduction_status'])

    def test_measured_check_rejects_accuracy_count_and_dataset_drift(self):
        expected = (run.ROOT / 'experiments/article_baselines/measured/metrics.csv').read_text()
        lines = expected.splitlines()
        actual = '\n'.join([lines[0]] + [line for line in lines[1:] if line.startswith('eurosat,')])
        with patch.object(Path, 'open', side_effect=[io.StringIO(expected), io.StringIO(actual)]):
            run.check_results(Path('not-read-from-disk'), 'eurosat')
        for wrong in (
            actual.replace(',4123,', ',4122,'),
            actual.replace('0.7635185185185185', '0.8'),
            actual.replace('eurosat,', 'resisc45,'),
        ):
            with patch.object(Path, 'open', side_effect=[io.StringIO(expected), io.StringIO(wrong)]):
                with self.assertRaises(ValueError):
                    run.check_results(Path('not-read-from-disk'), 'eurosat')

    def test_driver_cli_defaults_to_both_datasets(self):
        base = run.ROOT / 'output/staged-test/baselines'
        metrics = 'dataset,model,split,accuracy\neurosat,means,test,0.76\n'
        with (
            patch.object(run.sys, 'argv', ['run', '--output', str(base), '--download', '--check']),
            patch.object(run, 'output_directory', return_value=base),
            patch.object(run, 'run', side_effect=[base / 'eurosat', base / 'resisc45']) as execute,
            patch.object(run, 'check_results') as check,
            patch.object(run, 'write_csv') as write,
            patch.object(run.shutil, 'copyfile') as copy,
            patch.object(download, 'prepare_dataset') as prepare,
            patch.object(Path, 'open', side_effect=[io.StringIO(metrics), io.StringIO(metrics)]),
        ):
            run.main()
        self.assertEqual([call.args[0].dataset for call in execute.call_args_list], ['eurosat', 'resisc45'])
        self.assertEqual([call.args[0].output_dir for call in execute.call_args_list],
                         [base / 'eurosat', base / 'resisc45'])
        self.assertEqual([call.args[0] for call in prepare.call_args_list], ['eurosat', 'resisc45'])
        self.assertEqual(check.call_count, 2)
        self.assertEqual(write.call_args.args[0], base / 'metrics.csv')
        self.assertEqual(copy.call_args.args[1], base / 'archive_manifest.csv')

    def test_download_preserves_invalid_existing_files(self):
        path = MagicMock(spec=Path)
        path.exists.return_value = True
        with patch.object(download, 'digest', return_value='wrong'):
            with patch.object(download, 'urlopen') as request:
                with self.assertRaisesRegex(ValueError, 'preserved unchanged'):
                    download.download_file(path, 'https://example.invalid/not-requested', 'expected')
        request.assert_not_called()
        path.unlink.assert_not_called()

    def test_failed_download_checksum_cannot_publish_data(self):
        path = MagicMock(spec=Path)
        path.name = 'dataset.zip'
        path.exists.return_value = False
        partial = path.with_name.return_value
        partial.open.return_value.__enter__.return_value = io.BytesIO()
        with (
            patch.object(download, 'urlopen', return_value=io.BytesIO(b'wrong payload')),
            patch.object(download, 'digest', return_value='wrong'),
            patch.object(download.os, 'link') as publish,
        ):
            with self.assertRaisesRegex(ValueError, 'download checksum mismatch'):
                download.download_file(path, 'https://example.invalid/mocked', 'expected')
        publish.assert_not_called()
        partial.unlink.assert_called_once_with(missing_ok=True)

    def test_archive_rejects_traversal_and_symlinks_before_writing(self):
        for filename, permissions in (
            ('../outside.jpg', 0), ('/absolute.jpg', 0),
            ('image-link.jpg', (stat.S_IFLNK | 0o777) << 16),
        ):
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w') as archive:
                info = zipfile.ZipInfo(filename)
                info.external_attr = permissions
                archive.writestr(info, 'not an image')
            data.seek(0)
            with zipfile.ZipFile(data) as archive:
                with self.assertRaisesRegex(ValueError, 'unsafe archive member'):
                    download.extract_missing(archive, run.ROOT / 'output/unwritten-test-data', [])

    def test_archive_does_not_replace_existing_images(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('class/image.jpg', b'replacement')
        data.seek(0)
        with zipfile.ZipFile(data) as archive:
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(archive, 'open') as read:
                    download.extract_missing(archive, run.ROOT / 'output/unwritten-test-data',
                                             [Path('class/image.jpg')])
                    read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
