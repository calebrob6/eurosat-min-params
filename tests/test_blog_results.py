"""Data-free checks for the current result-table exporter."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from export_blog_results import (
    BACKBONE_SCOREBOARD, ROOT, SUPPORTING_RESULTS, read_rows,
    representation_tables, scoreboard, training_fractions, validate_backbone,
)


class BlogResultsTests(unittest.TestCase):
    def test_scoreboard_keeps_current_local_results(self):
        rows = scoreboard()
        local = {row['model']: row for row in rows
                 if row['source_kind'] == 'checked_in_local_result'}
        self.assertEqual(set(local), {'Ours 33 features', 'ImageStats'})
        self.assertEqual(local['Ours 33 features']['test_accuracy_percent'], '96.04')
        self.assertEqual(local['ImageStats']['test_accuracy_percent'], '90.96')
        self.assertEqual(local['Ours 33 features']['learned_parameters'], 306)
        self.assertEqual(local['ImageStats']['learned_parameters'], 477)
        for row in rows:
            self.assertEqual(row['learned_parameters'],
                             row['backbone_parameters'] + row['probe_parameters'])
            self.assertNotIn('experiments/imported', row['source_csv'])

    def test_recorded_backbones_match_compact_scoreboard(self):
        original = {row['display_name']: row for row in read_rows(BACKBONE_SCOREBOARD)}
        backbones = [row for row in scoreboard()
                     if row['source_kind'] == 'recorded_backbone_scoreboard']
        self.assertEqual({row['model'] for row in backbones}, set(original))
        for row in backbones:
            self.assertEqual(row['test_accuracy'], original[row['model']]['test_accuracy_reported'])
            self.assertEqual(row['source_csv'], BACKBONE_SCOREBOARD.relative_to(ROOT).as_posix())

    def test_backbone_scoreboard_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('export_blog_results.BACKBONE_SCOREBOARD', Path(directory) / 'absent.csv'):
                self.assertEqual(len(scoreboard()), 2)

    def test_backbone_protocol_mismatch_fails(self):
        original = read_rows(BACKBONE_SCOREBOARD)[0]
        validate_backbone(original)
        for field, value in [('merge_val', 'True'), ('seed', '1'), ('n_train', '8100'),
                             ('dataset', 'resisc45'), ('feature_dim', '0'),
                             ('probe_parameters', '10'), ('test_accuracy_reported', 'nan')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_backbone({**original, field: value})

    def test_curve_coverage_and_uncertainties(self):
        rows = training_fractions()
        self.assertEqual(len(rows), 28)
        self.assertEqual({row['model'] for row in rows}, {'Ours 33 features', 'ImageStats'})
        for row in rows:
            self.assertEqual(row['n_test'], 5400)
            self.assertEqual(row['uncertainty'], 'sample_stdev_over_5_training_seeds')
            self.assertTrue(row['source_csv'].startswith('results/'))
            original = next(r for r in read_rows(ROOT / row['source_csv'])
                            if r['split_protocol'] == row['split_protocol']
                            and int(r['n_train']) == row['n_train'])
            self.assertEqual(row['test_accuracy'], original['test_accuracy_mean'])
            self.assertEqual(row['test_accuracy_stdev'], original['test_accuracy_stdev'])

    def test_missing_duplicate_or_invalid_local_rows_fail(self):
        for function, filename in [
            (scoreboard, 'eurosat_models.csv'),
            (training_fractions, 'imagestats_fractions.csv'),
        ]:
            for change in ('missing', 'duplicate', 'nan'):
                def altered_rows(path):
                    rows = read_rows(path)
                    if path.name == filename:
                        if change == 'missing':
                            rows = rows[:-1]
                        elif change == 'duplicate':
                            rows.append(rows[-1])
                        else:
                            metric = 'accuracy' if function is scoreboard else 'test_accuracy_mean'
                            rows[-1][metric] = 'nan'
                    return rows

                with self.subTest(function=function.__name__, change=change):
                    with patch('export_blog_results.read_rows', side_effect=altered_rows):
                        with self.assertRaises(ValueError):
                            function()

    def test_duplicate_backbone_rows_fail(self):
        def duplicated_rows(path):
            rows = read_rows(path)
            if path == BACKBONE_SCOREBOARD:
                rows.append(rows[0])
            return rows

        with patch('export_blog_results.read_rows', side_effect=duplicated_rows):
            with self.assertRaisesRegex(ValueError, 'distinct recorded backbone'):
                scoreboard()

    def test_cli_requires_only_stdlib_and_recorded_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, '-S', str(ROOT / 'export_blog_results.py'), '--output', directory],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            expected = {
                'scoreboard.csv': len(scoreboard()), 'training_fractions.csv': 28,
                'combined_features.csv': 5, 'decoded_features.csv': 5,
                'explained_variance.csv': 5, 'features_33.csv': 33,
                'class_accuracy.csv': 40, 'models.csv': 4,
            }
            if SUPPORTING_RESULTS.exists():
                expected['supporting_results.csv'] = len(read_rows(SUPPORTING_RESULTS))
            self.assertEqual({path.name for path in Path(directory).iterdir()}, set(expected))
            for filename, count in expected.items():
                with self.subTest(filename=filename):
                    self.assertEqual(len(read_rows(Path(directory) / filename)), count)
            self.assertEqual(read_rows(Path(directory) / 'features_33.csv'),
                             read_rows(ROOT / 'results/eurosat_306_features.csv'))

    def test_representation_tables_match_current_results(self):
        tables = representation_tables()
        self.assertEqual(set(tables), {
            'combined_features.csv', 'decoded_features.csv', 'explained_variance.csv',
        })
        decoded = {row['model']: row for row in tables['decoded_features.csv']}
        self.assertEqual(decoded['ResNet-50']['mean_r2'], '0.757')
        self.assertEqual(decoded['OlmoEarth v1.2 Base']['low_ndvi_anisotropy_r2'], '0.452')
        combined = {row['model']: row for row in tables['combined_features.csv']}
        self.assertEqual(combined['ConvNeXt-Tiny']['plus_377_accuracy_percent'], '98.17')
        self.assertAlmostEqual(combined['DOFA Large']['plus_377_gain_points'], 0.5740740740740713)

    def test_representation_table_missing_or_duplicate_row_fails(self):
        for duplicate in (False, True):
            def changed(path):
                rows = read_rows(path)
                if path.name == 'features.csv':
                    rows = rows + [rows[0]] if duplicate else rows[1:]
                return rows

            with self.subTest(duplicate=duplicate):
                with patch('export_blog_results.read_rows', side_effect=changed):
                    with self.assertRaisesRegex(ValueError, '33 unique decoded features'):
                        representation_tables()


if __name__ == '__main__':
    unittest.main()
