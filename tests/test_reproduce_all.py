"""The public reproduction commands must connect fresh outputs to later steps."""
from argparse import Namespace
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_overlap import compare
from reproduce_all import commands, products, ROOT, STEPS


class ReproductionCommandsTests(unittest.TestCase):
    def options(self, steps):
        return Namespace(
            steps=steps, cpu_python=Path('/cpu/python'), benchmark_python=Path('/gpu/python'),
            output=ROOT / 'output/fresh', download=True, check=True, device='cuda:0', exploratory=False,
        )

    def test_fresh_metadata_and_embeddings_feed_overlap(self):
        jobs = dict(commands(self.options(list(STEPS))))
        select = jobs['overlap-select']
        self.assertEqual(select[select.index('--embeddings') + 1], str(ROOT / 'output/fresh/embeddings'))
        self.assertEqual(select[select.index('--metadata') + 1], str(ROOT / 'output/fresh/backbones'))
        self.assertNotIn('overlap-embeddings', jobs)
        self.assertIn('--check', jobs['local'])
        self.assertIn('--check', jobs['overlap-comparison'])
        self.assertIn('--fractions', jobs['backbones-curves'])
        self.assertEqual(jobs['baselines'][0], '/cpu/python')
        self.assertIn('--check', jobs['baselines'])

    def test_overlap_only_extracts_missing_inputs_without_classifiers(self):
        jobs = dict(commands(self.options(['overlap'])))
        self.assertIn('--extract-only', jobs['overlap-embeddings'])
        self.assertNotIn('backbones-scoreboard', jobs)
        self.assertLess(list(jobs).index('overlap-select'), list(jobs).index('overlap-lock'))
        self.assertLess(list(jobs).index('overlap-lock'), list(jobs).index('overlap-evaluate'))

    def test_dry_run_needs_no_installed_dependencies(self):
        result = subprocess.run(
            [sys.executable, '-S', str(ROOT / 'reproduce_all.py'), '--dry-run'],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertIn('overlap-select:', result.stdout)
        self.assertIn('baselines:', result.stdout)

    def test_expensive_historical_reruns_are_explicit(self):
        args = self.options(['local'])
        self.assertNotIn('tiny-cnn', dict(commands(args)))
        args.exploratory = True
        jobs = dict(commands(args))
        self.assertIn('--epochs', jobs['tiny-cnn'])
        self.assertIn('60', jobs['tiny-cnn'])
        self.assertIn('--method', jobs['mosaiks'])
        self.assertIn('mosaiks', jobs['mosaiks'])

    def test_missing_stage_outputs_are_not_recorded_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, 'required output is missing'):
                products('overlap-lock', output)
            (output / 'overlap').mkdir()
            path = output / 'overlap/locked.json'
            path.write_text('{}')
            self.assertEqual(products('overlap-lock', output), [path])
            with self.assertRaisesRegex(ValueError, 'no results were written'):
                products('figures', output)

    def test_reference_comparison_is_exact_and_changes_are_visible(self):
        reference = ROOT / 'experiments/representation_overlap/results'
        rows = compare(reference, reference, 0)
        self.assertTrue(all(row['match'] for row in rows))
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory)
            from compare_overlap import TABLES
            for filename in TABLES:
                with (reference / filename).open() as handle:
                    data = list(csv.DictReader(handle))
                if filename == 'classification.csv':
                    data[0]['accuracy'] = str(float(data[0]['accuracy']) - .01)
                with (fresh / filename).open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
            differences = [row for row in compare(reference, fresh, 1e-6) if not row['match']]
            self.assertEqual(len(differences), 1)
            self.assertEqual(differences[0]['metric'], 'accuracy')


if __name__ == '__main__':
    unittest.main()
