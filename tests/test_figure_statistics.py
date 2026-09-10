"""Data-free checks of the article's fixed training-patch figure samples."""
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from experiments.figure_statistics import check_results, sample_indices, ROOT


class FigureStatisticsTests(unittest.TestCase):
    def test_samples_are_fixed_distinct_and_balanced(self):
        labels = np.repeat(np.arange(10), 150)
        chosen = sample_indices(labels)
        np.testing.assert_array_equal(chosen, sample_indices(labels))
        self.assertEqual(len(np.unique(chosen)), 1200)
        np.testing.assert_array_equal(np.bincount(labels[chosen]), [120] * 10)

    def test_changed_reference_measurement_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            shutil.copyfile(ROOT / 'results/figure_examples.csv', output / 'orientation_examples.csv')
            shutil.copyfile(ROOT / 'results/figure_gradient_medians.csv', output / 'gradient_medians.csv')
            check_results(output)
            (output / 'orientation_examples.csv').write_text('figure,filename\n4,wrong.tif\n')
            with self.assertRaisesRegex(ValueError, 'figure measurements differ'):
                check_results(output)


if __name__ == '__main__':
    unittest.main()
