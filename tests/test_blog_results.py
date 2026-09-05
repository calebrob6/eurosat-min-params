"""Data-free checks that exported blog values come from their original sources."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from export_blog_results import ROOT, linear_rows, scoreboard, training_fractions, validate_backbone


class BlogResultsTests(unittest.TestCase):
    def test_scoreboard_matches_original_article_values(self):
        rows = scoreboard()
        self.assertEqual(len(rows), 15)
        scores = {row['model']: row['test_accuracy_percent'] for row in rows}
        self.assertEqual(scores['torchgeo/dofa_large'], '98.33')
        self.assertEqual(scores['OlmoEarth v1.2 Nano'], '96.89')
        self.assertEqual(scores['OlmoEarth v1.2 Small'], '98.65')
        self.assertEqual(scores['OlmoEarth v1.2 Base'], '98.80')
        self.assertEqual(scores['Ours 33 features'], '96.04')
        self.assertEqual(scores['ImageStats'], '90.96')
        for row in rows:
            if row['source_kind'] == 'original_uploaded_benchmark':
                self.assertTrue(row['source_csv'].startswith('experiments/imported/'))

    def test_curve_coverage_and_uncertainties(self):
        rows = training_fractions()
        self.assertEqual(len(rows), 126)
        for row in rows:
            self.assertEqual(row['n_test'], 5400)
            with (ROOT / row['source_csv']).open() as handle:
                source_rows = list(csv.DictReader(handle))
            if row['source_kind'] == 'original_uploaded_benchmark':
                self.assertEqual(row['test_accuracy_stdev'], '')
                self.assertLessEqual(float(row['ci_lower']), float(row['test_accuracy']))
                self.assertLessEqual(float(row['test_accuracy']), float(row['ci_upper']))
                original = next(r for r in source_rows
                                if r['method'] == 'linear' and int(r['n_train']) == row['n_train'])
                self.assertEqual(row['test_accuracy'], original['metric_value'])
                self.assertEqual(row['ci_lower'], original['ci_lower'])
                self.assertEqual(row['ci_upper'], original['ci_upper'])
            else:
                self.assertGreaterEqual(float(row['test_accuracy_stdev']), 0)
                self.assertEqual((row['ci_lower'], row['ci_upper']), ('', ''))
                original = next(r for r in source_rows if r['split_protocol'] == row['split_protocol']
                                and int(r['n_train']) == row['n_train'])
                self.assertEqual(row['test_accuracy'], original['test_accuracy_mean'])
                self.assertEqual(row['test_accuracy_stdev'], original['test_accuracy_stdev'])

    def test_original_random_full_points_come_from_scoreboard(self):
        for row in training_fractions():
            if row['model'] == 'torchgeo/dofa_large' and row['split_protocol'] == 'random':
                if row['n_train'] == 16200:
                    self.assertIn('eurosat-13band-merge-val-false-', row['source_csv'])
                    self.assertAlmostEqual(float(row['test_accuracy']), 0.9833333333333333)
                else:
                    self.assertIn('eurosat-train-fractions-', row['source_csv'])

    def test_original_protocol_mismatch_fails(self):
        source = ROOT / 'experiments/imported/eurosat-13band-merge-val-false-20260901/tgeo_dofa_large.csv'
        with source.open() as handle:
            original = next(row for row in csv.DictReader(handle) if row['method'] == 'linear')
        validate_backbone(original, 'eurosat', 16200)
        for field, value in [('merge_val', 'True'), ('seed', '1'), ('n_train', '8100'),
                             ('metric_name', 'balanced_accuracy'), ('method', 'knn5'),
                             ('ci_lower', 'nan'), ('ci_upper', '0.1')]:
            with self.assertRaises(ValueError):
                validate_backbone({**original, field: value}, 'eurosat', 16200)

    def test_missing_curve_point_fails(self):
        def missing_point(path):
            rows = linear_rows(path)
            rows.pop(162, None)
            return rows

        with patch('export_blog_results.linear_rows', side_effect=missing_point):
            with self.assertRaisesRegex(ValueError, 'missing training count 162'):
                training_fractions()

    def test_duplicate_or_missing_local_rows_fail(self):
        reader = csv.DictReader
        for function, filename in [
            (scoreboard, 'eurosat_models.csv'),
            (training_fractions, 'eval_imagestats_fractions_result.csv'),
        ]:
            for duplicate in (True, False):
                def altered_rows(handle):
                    rows = list(reader(handle))
                    if Path(handle.name).name == filename:
                        rows = rows + [rows[-1]] if duplicate else rows[:-1]
                    return rows

                with self.subTest(function=function.__name__, duplicate=duplicate):
                    with patch('export_blog_results.csv.DictReader', side_effect=altered_rows):
                        with self.assertRaises(ValueError):
                            function()

    def test_duplicate_backbone_rows_fail(self):
        reader = csv.DictReader

        def duplicated_rows(handle):
            rows = list(reader(handle))
            if 'imported' in Path(handle.name).parts:
                rows.append(next(row for row in rows if row['method'] == 'linear'))
            return rows

        with patch('export_blog_results.csv.DictReader', side_effect=duplicated_rows):
            with self.assertRaisesRegex(ValueError, 'duplicate linear training counts'):
                training_fractions()

    def test_cli_requires_no_installed_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, '-S', str(ROOT / 'export_blog_results.py'), '--output', directory],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            for filename, count in [('scoreboard.csv', 15), ('training_fractions.csv', 126)]:
                with (Path(directory) / filename).open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)


if __name__ == '__main__':
    unittest.main()
