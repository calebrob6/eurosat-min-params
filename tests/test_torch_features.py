"""Test native PyTorch feature semantics and the frozen public schema.

The independent NumPy/SciPy recipe checks continuous statistics on small,
unbiased random inputs. Discrete descriptors use unambiguous analytic inputs,
not cross-library rounding at thresholds or angle boundaries. Neither library
nor CPU/CUDA results are required to be bit-identical.
"""

import importlib.util
import math
import unittest
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import label, uniform_filter

from eurosat_features import TIFF_BAND_NAMES, EuroSATFeatures, _ops
from experiments.representation_overlap.data import feature_matrices

MODES = (('33', 'frontier33'), ('377', 'pool377'), ('52', 'imagestats52'))
DEVICES = ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',)
# DN reductions, normalized descriptors, and FFT statistics have distinct scales.
CONTINUOUS_TOLERANCES = {
    'spectral_statistics': (3e-7, 2e-5),
    'image_statistics': (3e-7, 2e-5),
    'multiscale_gradients': (6e-7, 2e-5),
    'coherence': (5e-7, 2e-7),
    'spectral_peaks': (1e-6, 3e-6),
    'cross_band': (0, 5e-7),
    'index_texture': (5e-7, 3e-7),
    'harris_corners': (0, 2e-7),
    'spectral_slope': (1e-6, 3e-6),
    'index_coherence': (5e-7, 2e-7),
    'index_texture_scale2': (5e-7, 2e-7),
    'cross_band_correlation': (0, 5e-7),
    'region_shape': (0, 3e-7),
}


class FeatureSchema(TypedDict):
    """Reference feature identifiers and their families."""

    names: list[str]
    families: list[str]


def patterns() -> NDArray[np.float32]:
    """Return constant, ramp, periodic, and impulse inputs.

    Returns:
        Float32 TIFF-order image batch of shape ``(7, 13, 64, 64)``.
    """
    y, x = np.mgrid[:64, :64]
    band = np.arange(1, 14, dtype=np.float32)[:, None, None]
    impulse = np.zeros((64, 64), np.float32)
    impulse[0, 0], impulse[32, 31], impulse[-1, -1] = 100, 500, 700
    return np.stack(
        (
            np.zeros((13, 64, 64), np.float32),
            np.broadcast_to(band * 100, (13, 64, 64)),
            1000 + band * (x + y),
            1000 + band * (x - y),
            1000 + band * ((x + y) % 2) * 100,
            1000 + band * ((x // 8) % 2) * 100,
            1000 + band * impulse,
        )
    ).astype(np.float32)


class TorchFeaturesTests(unittest.TestCase):
    """Exercise fixed feature definitions and ordinary module behavior."""

    old_threads: int

    @classmethod
    def setUpClass(cls) -> None:
        """Limit small CPU tensor tests to one worker thread."""
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the caller's CPU worker count."""
        torch.set_num_threads(cls.old_threads)

    def assert_continuous_columns(
        self,
        actual: NDArray[np.float32],
        expected: NDArray[np.float32],
        schema: FeatureSchema,
        context: str,
    ) -> None:
        """Compare continuous reference columns with family-specific budgets.

        Args:
            actual: Native PyTorch feature matrix.
            expected: Independent NumPy/SciPy feature matrix.
            schema: Ordered column names and feature families.
            context: Device and recipe identifier for assertion messages.
        """
        for i, (name, family) in enumerate(zip(schema['names'], schema['families'])):
            if family not in CONTINUOUS_TOLERANCES or name.startswith('corn2frac'):
                continue
            rtol, atol = CONTINUOUS_TOLERANCES[family]
            if family == 'spectral_statistics' and name.startswith('p'):
                # Float32 quantile ranks differ from NumPy's float64 ranks.
                rtol = 1e-6
            np.testing.assert_allclose(
                actual[:, i],
                expected[:, i],
                rtol=rtol,
                atol=atol,
                err_msg=f'{context}/{family}/{name}',
            )

    def assert_output_contract(self, images: NDArray[np.float32]) -> None:
        """Check dimensions, dtype, device, and finiteness for all recipes.

        Args:
            images: Float32 TIFF-order image batch.
        """
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    result = EuroSATFeatures(mode).to(device)(
                        torch.from_numpy(images).to(device)
                    )
                    self.assertEqual(result.device.type, device)
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.shape, (len(images), int(mode)))
                    self.assertTrue(torch.isfinite(result).all())

    def test_synthetic_patterns_are_finite(self) -> None:
        """Allow native threshold decisions while checking all pattern outputs."""
        self.assert_output_contract(patterns())

    def test_random_continuous_statistics_and_schema(self) -> None:
        """Compare continuous math and every name against the independent recipe."""
        images = (
            np.random.default_rng(41)
            .uniform(1, 10000, (3, 13, 64, 64))
            .astype(np.float32)
        )
        images[1] *= -1
        reference, schemas = feature_matrices(images)
        for device in DEVICES:
            for mode, key in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device)
                    result = module(torch.from_numpy(images).to(device))
                    self.assertEqual(module.feature_names, tuple(schemas[key]['names']))
                    self.assert_continuous_columns(
                        result.cpu().numpy(),
                        reference[key],
                        schemas[key],
                        f'{device}/{mode}',
                    )

    @unittest.skipUnless(
        importlib.util.find_spec('rasterio'), 'raw TIFF checks require rasterio'
    )
    def test_raw_tiff_shape_and_finiteness(self) -> None:
        """Exercise original TIFF pixels without requiring matching discrete counts."""
        import rasterio

        root = Path(__file__).resolve().parents[1] / 'data' / 'EuroSAT'
        tif_root = root / 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        split = root / 'eurosat-val.txt'
        if not split.is_file() or not tif_root.is_dir():
            self.skipTest('original EuroSAT TIFF data not present')
        names_by_class = {}
        for name in split.read_text().splitlines():
            names_by_class.setdefault(name.rsplit('_', 1)[0], name)
        images = []
        for name in names_by_class.values():
            path = tif_root / name.rsplit('_', 1)[0] / (Path(name).stem + '.tif')
            if not path.is_file():
                self.skipTest(f'original TIFF absent: {path.name}')
            with rasterio.open(path) as handle:
                images.append(handle.read().astype(np.float32))
        if not images:
            self.skipTest('validation split is empty')
        self.assert_output_contract(np.stack(images))

    def test_frontier_order_and_duplicate_correlations(self) -> None:
        """Preserve frozen pool offsets and the frontier-only tail descriptor."""
        from src.frontier import POOL_INDICES

        image = torch.from_numpy(
            np.random.default_rng(3)
            .uniform(1, 10000, (1, 13, 64, 64))
            .astype(np.float32)
        )
        for device in DEVICES:
            module = EuroSATFeatures().to(device)
            pool_module = EuroSATFeatures('377').to(device)
            pool = pool_module(image.to(device))
            result = module(image.to(device))
            torch.testing.assert_close(
                result[:, :32], pool[:, POOL_INDICES.tolist()], rtol=2e-6, atol=1e-6
            )
            self.assertEqual(
                module.feature_names[:-1],
                tuple(pool_module.feature_names[i] for i in POOL_INDICES),
            )
            self.assertEqual(module.feature_names[-1], 'tail_aniso_low_ndvi')
            self.assertNotIn(module.feature_names[-1], pool_module.feature_names)
            self.assertEqual(len(set(pool_module.feature_names)), 369)
            for name in set(pool_module.feature_names):
                ids = [
                    i
                    for i, candidate in enumerate(pool_module.feature_names)
                    if candidate == name
                ]
                if len(ids) == 2:
                    torch.testing.assert_close(pool[:, ids[0]], pool[:, ids[1]])

    def test_batch_invariance_noncontiguous_and_empty(self) -> None:
        """Support independent samples, noncontiguous inputs, and empty batches."""
        image = torch.from_numpy(
            np.random.default_rng(4)
            .uniform(1, 10000, (3, 13, 64, 64))
            .astype(np.float32)
        )
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device)
                    batch = image.to(device).transpose(-1, -2)
                    self.assertFalse(batch.is_contiguous())
                    result = module(batch)
                    singles = torch.cat([module(row.unsqueeze(0)) for row in batch])
                    torch.testing.assert_close(result, singles, rtol=2e-6, atol=1e-6)
                    torch.testing.assert_close(result, module(batch.contiguous()))
                    empty = module(batch[:0])
                    self.assertEqual(empty.shape, (0, int(mode)))
                    self.assertEqual(empty.device.type, device)
                    self.assertEqual(empty.dtype, torch.float32)

    def test_uint16_double_and_autocast(self) -> None:
        """Keep extraction float32 despite input dtype or an outer autocast context."""
        image = torch.from_numpy(
            np.random.default_rng(7).integers(
                0, 10000, (1, 13, 64, 64), dtype=np.uint16
            )
        )
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device).double()
                    x = image.to(device)
                    expected = module(x.float())
                    for converted in (x, x.double(), x.to(torch.bfloat16)):
                        actual = module(converted)
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual, module(converted.float()))
                    with torch.autocast(device_type=device):
                        actual = module(x)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected)
                    with torch.autocast(device_type=device):
                        empty = module(x[:0])
                    self.assertEqual(empty.dtype, torch.float32)
                    self.assertEqual(empty.shape, (0, int(mode)))

    def test_module_contract_and_head_backprop(self) -> None:
        """Keep the extractor parameter-free and nondifferentiable, but train heads."""
        self.assertEqual(
            TIFF_BAND_NAMES,
            (
                'B01',
                'B02',
                'B03',
                'B04',
                'B05',
                'B06',
                'B07',
                'B08',
                'B09',
                'B10',
                'B11',
                'B12',
                'B8A',
            ),
        )
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode)
                    self.assertIsInstance(module.feature_names, tuple)
                    self.assertEqual(module.num_features, int(mode))
                    self.assertIn(f'num_features={mode}', repr(module))
                    self.assertEqual(list(module.parameters()), [])
                    self.assertEqual(dict(module.state_dict()), {})
                    with self.assertRaises(AttributeError):
                        module.num_features = 4
                    with self.assertRaises(AttributeError):
                        module.feature_names = ('wrong',)
                    module.to(device)
                    self.assertTrue(
                        all(buffer.device.type == device for buffer in module.buffers())
                    )
                    head = torch.nn.Linear(int(mode), 2).to(device)
                    model = torch.nn.Sequential(module, head)
                    image = (
                        torch.rand(1, 13, 64, 64, device=device)
                        .add_(1)
                        .requires_grad_()
                    )
                    self.assertFalse(module(image).requires_grad)
                    model(image).square().sum().backward()
                    self.assertIsNone(image.grad)
                    self.assertIsNotNone(head.weight.grad)
                    self.assertTrue(torch.isfinite(head.weight.grad).all())
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device):
                        prediction = model(image)
                        loss = prediction.square().mean()
                    self.assertEqual(
                        prediction.dtype,
                        torch.float16 if device == 'cuda' else torch.bfloat16,
                    )
                    loss.backward()
                    self.assertIsNone(image.grad)
                    self.assertIsNotNone(head.weight.grad)
                    self.assertTrue(torch.isfinite(head.weight.grad).all())
                    torch.testing.assert_close(
                        EuroSATFeatures(mode)(image), module(image)
                    )

    def test_state_dict_round_trip(self) -> None:
        """Recreate nonpersistent geometry when loading a standard state dict."""
        image = torch.from_numpy(
            np.random.default_rng(13)
            .uniform(1, 10000, (1, 13, 64, 64))
            .astype(np.float32)
        )
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device)
                    stream = BytesIO()
                    torch.save(module.state_dict(), stream)
                    stream.seek(0)
                    restored = EuroSATFeatures(mode).to(device)
                    restored.load_state_dict(
                        torch.load(stream, map_location=device, weights_only=True),
                        strict=True,
                    )
                    self.assertEqual(module.feature_names, restored.feature_names)
                    x = image.to(device)
                    torch.testing.assert_close(module(x), restored(x))

    def test_invalid_inputs(self) -> None:
        """Reject unsupported recipes, shapes, layouts, and undefined arithmetic."""
        for value in ('bad', 'frontier33', 33, None, []):
            with self.subTest(feature_set=value), self.assertRaises(ValueError):
                EuroSATFeatures(value)
        module = EuroSATFeatures()
        for value in (None, [], np.zeros((1, 13, 64, 64))):
            with self.assertRaises(TypeError):
                module(value)
        for shape in ((13, 64, 64), (1, 12, 64, 64), (1, 13, 32, 64), (0, 13, 64, 63)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                module(torch.zeros(shape))
        for dtype in (torch.bool, torch.complex64):
            with self.assertRaises(TypeError):
                module(torch.zeros(1, 13, 64, 64, dtype=dtype))
        with self.assertRaises(TypeError):
            module(torch.zeros(1, 13, 64, 64).to_sparse())
        with self.assertRaises(TypeError):
            module(
                torch.quantize_per_tensor(torch.ones(1, 13, 64, 64), 1, 0, torch.quint8)
            )
        with self.assertRaisesRegex(ValueError, 'CPU and CUDA'):
            module(torch.empty(1, 13, 64, 64, device='meta'))
        for value in (float('nan'), float('inf'), -float('inf')):
            image = torch.ones(1, 13, 64, 64)
            image[0, 0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'finite'):
                module(image)
        with self.assertRaisesRegex(ValueError, 'float32'):
            module(torch.full((1, 13, 64, 64), 1e100, dtype=torch.float64))
        singular = torch.zeros(1, 13, 64, 64)
        singular[:, 3] = -_ops.EPS
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            module(singular)

    def test_population_standard_deviation(self) -> None:
        """Use population rather than sample normalization, including singletons."""
        for device in DEVICES:
            x = torch.tensor([[1.0, 3.0], [7.0, 7.0]], device=device)
            mean, std = _ops.mean_std(x)
            torch.testing.assert_close(mean, torch.tensor([2.0, 7.0], device=device))
            torch.testing.assert_close(std, torch.tensor([1.0, 0.0], device=device))
            single_mean, single_std = _ops.mean_std(x[:, :1])
            torch.testing.assert_close(single_mean, x[:, 0])
            torch.testing.assert_close(single_std, torch.zeros(2, device=device))
            offset = torch.tensor([[100000.0, 100001.0, 100002.0]], device=device)
            expected_std, expected_mean = torch.std_mean(offset, dim=-1, correction=0)
            actual_mean, actual_std = _ops.mean_std(offset)
            torch.testing.assert_close(actual_mean, expected_mean)
            torch.testing.assert_close(actual_std, expected_std)

    def test_percentiles_and_interpolated_median(self) -> None:
        """Follow native quantile semantics and preserve scalar versus vector axes."""
        for device in DEVICES:
            x = torch.tensor(
                [[[0.0, 2.0, 8.0, 10.0], [10.0, 8.0, 2.0, 0.0]]], device=device
            )
            q = [0, 25, 50, 75, 100]
            expected = torch.tensor([0.0, 1.5, 5.0, 8.5, 10.0], device=device)
            result = _ops.percentile(x, q)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.shape, (1, 2, 5))
            torch.testing.assert_close(result, expected.expand(1, 2, 5))
            torch.testing.assert_close(
                result, torch.quantile(x, x.new_tensor(q) / 100, dim=-1).movedim(0, -1)
            )
            for q_scalar in (25, 85.0):
                scalar = _ops.percentile(x, q_scalar)
                self.assertEqual(scalar.shape, (1, 2))
                self.assertEqual(scalar.dtype, torch.float32)
                torch.testing.assert_close(
                    scalar, _ops.percentile(x, [q_scalar])[..., 0]
                )
                torch.testing.assert_close(
                    scalar, torch.quantile(x, x.new_tensor(q_scalar) / 100, dim=-1)
                )
            torch.testing.assert_close(_ops.median(x), x.new_full((1, 2), 5))
            torch.testing.assert_close(
                _ops.median(x[..., :3]), x.new_tensor([[2.0, 8.0]])
            )
            torch.testing.assert_close(_ops.median(x[..., :1]), x[..., 0])
            self.assertEqual(_ops.percentile(x[0, 0], (25, 75)).shape, (2,))
            self.assertEqual(_ops.percentile(x[0, 0], 25).shape, ())

    def test_native_pan_and_historical_index_maps(self) -> None:
        """Average bands natively and preserve B12/B8A historical SWIR choices."""
        for device in DEVICES:
            x = torch.arange(1, 14, device=device, dtype=torch.float32).reshape(
                1, 13, 1, 1
            )
            x = x.expand(2, 13, 64, 64)
            torch.testing.assert_close(_ops.pan(x), x.mean(dim=1))
            maps = _ops.index_maps(x)
            expected = x.new_tensor(
                [
                    (8 - 4) / (8 + 4 + _ops.EPS),
                    (3 - 8) / (3 + 8 + _ops.EPS),
                    (12 - 8) / (12 + 8 + _ops.EPS),
                    (8 - 12) / (8 + 12 + _ops.EPS),
                    (8 - 13) / (8 + 13 + _ops.EPS),
                    ((12 + 4) - (8 + 2)) / ((12 + 4) + (8 + 2) + _ops.EPS),
                ]
            )
            torch.testing.assert_close(
                maps, expected[None, :, None, None].expand_as(maps)
            )
            random = torch.from_numpy(
                np.random.default_rng(203)
                .uniform(-1e4, 1e4, (2, 13, 64, 64))
                .astype(np.float32)
            ).to(device)
            torch.testing.assert_close(_ops.pan(random), random.mean(dim=1))

    def test_constant_gradients_and_native_magnitude(self) -> None:
        """Recover analytic gradient magnitudes and zero constant-gradient spread."""
        for device in DEVICES:
            y, x = torch.meshgrid(
                torch.arange(64, device=device),
                torch.arange(64, device=device),
                indexing='ij',
            )
            ramp = (3 * x + 4 * y).float()[None, None]
            gx, gy = _ops.gradients(ramp)
            torch.testing.assert_close(gx, gx.new_full(gx.shape, 3))
            torch.testing.assert_close(gy, gy.new_full(gy.shape, 4))
            mag = _ops.magnitude(ramp)
            torch.testing.assert_close(mag, mag.new_full(mag.shape, 5))
            mean, std = _ops.mean_std(mag.flatten(-2))
            torch.testing.assert_close(mean, mean.new_full((1, 1), 5))
            torch.testing.assert_close(std, std.new_zeros((1, 1)))
            for scale in range(3):
                actual = _ops.magnitude(ramp)
                torch.testing.assert_close(
                    actual, actual.new_full(actual.shape, 5 * 2**scale)
                )
                ramp = _ops.pool2(ramp)
            random = torch.from_numpy(
                np.random.default_rng(21).normal(size=(2, 3, 7, 9)).astype(np.float32)
            ).to(device)
            gx, gy = _ops.gradients(random)
            torch.testing.assert_close(
                _ops.magnitude(random), (gx * gx + gy * gy).sqrt()
            )

    def test_native_atan2_orientation_bins(self) -> None:
        """Let torch.atan2 choose diagonal bins rather than copying NumPy rounding."""
        for device in DEVICES:
            y, x = torch.meshgrid(
                torch.arange(64, device=device),
                torch.arange(64, device=device),
                indexing='ij',
            )
            directions = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
                device=device,
            )
            ramps = (
                directions[:, 0, None, None] * x + directions[:, 1, None, None] * y
            )[:, None]
            gx = directions[:, 0, None, None, None].expand(-1, 1, 63, 63).contiguous()
            gy = directions[:, 1, None, None, None].expand(-1, 1, 63, 63).contiguous()
            angles = torch.atan2(gy, gx).remainder(math.pi).flatten(-2)
            weights = (gx.square() + gy.square()).sqrt().flatten(-2)
            for bins in (4, 8):
                indices = (angles / (math.pi / bins)).long().clamp(max=bins - 1)
                expected = torch.zeros(len(directions), 1, bins, device=device)
                expected.scatter_add_(-1, indices, weights)
                expected /= expected.sum(-1, keepdim=True) + _ops.EPS
                histogram = _ops.orientation_histogram(ramps, bins)
                self.assertEqual(histogram.dtype, torch.float32)
                torch.testing.assert_close(histogram, expected, rtol=0, atol=2e-7)
            entropy = _ops.orientation_entropy(ramps)
            torch.testing.assert_close(
                entropy,
                -(expected * (expected + _ops.EPS).log()).sum(-1),
                rtol=0,
                atol=2e-7,
            )
            zeros = ramps.new_zeros(1, 1, 64, 64)
            torch.testing.assert_close(
                _ops.orientation_histogram(zeros, 4), zeros.new_zeros(1, 1, 4)
            )
            torch.testing.assert_close(
                _ops.orientation_entropy(zeros), zeros.new_zeros(1, 1)
            )

    def test_radius_one_replicated_boundary(self) -> None:
        """Match the mathematical SciPy boundary, without its accumulation order."""
        image = np.random.default_rng(28).normal(size=(2, 3, 7, 9)).astype(np.float32)
        image[:, :, 0, 0] = 30
        expected = uniform_filter(image, size=(1, 1, 3, 3), mode='reflect')
        for device in DEVICES:
            x = torch.from_numpy(image).to(device)
            result = _ops.uniform_filter3(x)
            np.testing.assert_allclose(
                result.cpu().numpy(), expected, rtol=3e-7, atol=3e-7
            )
            impulse = x.new_zeros(1, 1, 3, 3)
            impulse[0, 0, 0, 0] = 9
            analytic = x.new_tensor([[[[4, 2, 0], [2, 1, 0], [0, 0, 0]]]])
            torch.testing.assert_close(_ops.uniform_filter3(impulse), analytic)
            native = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1
            )
            torch.testing.assert_close(result, native)

    def test_hough_constant_gradient_and_straight_edge(self) -> None:
        """Detect a clear straight edge but not a constant gradient magnitude."""
        bins, nrho, diag = _ops.hough_geometry()
        self.assertEqual(bins.shape, (60, 63 * 63))
        self.assertEqual(bins.dtype, torch.int64)
        self.assertTrue(((bins >= 0) & (bins < nrho)).all())
        first_angle = (
            ((torch.arange(63, dtype=torch.float32) + diag) / 2).floor().long()
        )
        torch.testing.assert_close(bins[0].reshape(63, 63), first_angle.expand(63, 63))
        for device in DEVICES:
            y, x = torch.meshgrid(
                torch.arange(64, device=device),
                torch.arange(64, device=device),
                indexing='ij',
            )
            ramp = (3 * x + 4 * y).float()[None, None]
            result = _ops.hough(ramp, bins.to(device), nrho, diag)
            torch.testing.assert_close(result, ramp.new_zeros(1, 3))
            edge = (x >= 32).float()[None, None]
            result = _ops.hough(edge, bins.to(device), nrho, diag)
            torch.testing.assert_close(result[0, :2], edge.new_tensor([1, 63 / diag]))
            self.assertGreaterEqual(result[0, 2].item(), 1)
            self.assertLessEqual(result[0, 2].item(), 3)

    def test_harris_constant_and_one_direction(self) -> None:
        """Give no Harris corner response for constant or one-direction gradients."""
        for device in DEVICES:
            x = torch.arange(64, dtype=torch.float32, device=device).expand(64, 64)
            channels = torch.stack((torch.ones_like(x), x), 0)[None]
            torch.testing.assert_close(_ops.harris(channels), channels.new_zeros(1, 4))

    def test_eight_connected_components(self) -> None:
        """Converge on diagonal, hollow, winding, isolated, and empty components."""
        masks = np.zeros((7, 64, 64), dtype=bool)
        masks[0] = np.eye(64, dtype=bool)
        masks[1, ::3, ::3] = True
        masks[2] = True
        masks[3, (0, -1), :] = True
        masks[3, :, (0, -1)] = True
        masks[4, 2:5, 3:6] = True
        masks[4, 40:46, 41:47] = True
        masks[5, ::3, :] = True
        for row in range(0, 61, 3):
            masks[5, row : row + 4, -1 if row % 6 == 0 else 0] = True
        for device in DEVICES:
            tensor = torch.from_numpy(masks).to(device)
            result = _ops.component_sizes(tensor)
            self.assertEqual(result.device.type, device)
            sizes = result.cpu().numpy()
            for i, mask in enumerate(masks):
                labels, _ = label(mask, structure=np.ones((3, 3)))
                expected = np.bincount(labels.ravel())[1:]
                np.testing.assert_array_equal(
                    np.sort(sizes[i][sizes[i] > 0]), np.sort(expected)
                )
            supplied = _ops.component_sizes(tensor, _ops.component_edges().to(device))
            torch.testing.assert_close(supplied, result)

    def test_blob_component_proportions(self) -> None:
        """Normalize explicit component counts and sizes in native float32."""
        for device in DEVICES:
            image = torch.zeros(2, 1, 64, 64, device=device)
            image[0, 0, 2:5, 3:6] = 1
            image[0, 0, 40:46, 41:47] = 1
            expected = image.new_tensor(
                [[36 / 4096, math.log(3), 22.5 / 4096], [0, 0, 0]]
            )
            result = _ops.blobs(image, _ops.component_edges().to(device))
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, expected, rtol=2e-7, atol=1e-8)

    def test_lbp_uniform_and_nonuniform_patterns(self) -> None:
        """Count strict neighbor comparisons and circular transitions analytically."""
        for device in DEVICES:
            images = torch.zeros(4, 1, 3, 3, device=device)
            images[0, 0, 1, 1] = 1
            images[1] = 1
            images[1, 0, 1, 1] = 0
            images[2, 0, 0, 0] = 1
            images[3, 0, ::2, ::2] = 1
            result = _ops.lbp(images)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(
                result[:, 0], images.new_zeros(4), rtol=0, atol=3e-10
            )
            torch.testing.assert_close(result[:, 1], images.new_tensor([1, 1, 1, 0]))
            y, x = torch.meshgrid(
                torch.arange(4, device=device),
                torch.arange(4, device=device),
                indexing='ij',
            )
            checkerboard = ((x + y) % 2).float()[None, None]
            torch.testing.assert_close(
                _ops.lbp(checkerboard),
                images.new_tensor([[math.log(2), 0.5]]),
                rtol=2e-7,
                atol=1e-8,
            )

    def test_low_tail_anisotropy(self) -> None:
        """Distinguish a low-valued line from an isotropic square or constant field."""
        for device in DEVICES:
            ndvi = torch.ones(3, 64, 64, device=device)
            ndvi[0, 32, 8:56] = 0
            ndvi[1, 24:40, 24:40] = 0
            result = _ops.tail_anisotropy(ndvi)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result[1:], ndvi.new_zeros(2), rtol=0, atol=2e-7)
            coords = torch.linspace(-1, 1, 64, dtype=torch.float32, device=device)[8:56]
            variance = coords.var(correction=0)
            expected = variance / (variance + _ops.EPS)
            torch.testing.assert_close(result[0], expected, rtol=3e-7, atol=0)

    def test_spectral_slope_radial_definition(self) -> None:
        """Check the log-radius regression against direct radial averages."""
        image = np.random.default_rng(91).normal(size=(2, 3, 64, 64)).astype(np.float32)
        for device in DEVICES:
            chan = torch.from_numpy(image).to(device)
            freq = torch.fft.fftfreq(64, device=device) * 64
            radii = (
                (freq[:, None].square() + freq[None, :].square()).sqrt().round().long()
            )
            power = (
                torch.fft.fft2(chan - chan.mean((-2, -1), keepdim=True)).abs().square()
            )
            radial = torch.stack(
                [power[..., radii == radius].mean(-1) for radius in range(1, 33)], -1
            )
            logr = torch.arange(1, 33, device=device, dtype=torch.float32).log()
            centered = logr - logr.mean()
            logpower = radial.add(_ops.EPS).log()
            logpower = logpower - logpower.mean(-1, keepdim=True)
            expected = (logpower * centered).sum(-1) / centered.square().sum()
            result = EuroSATFeatures._spectral_slope(chan)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=3e-6)
            constant = chan.new_ones(2, 3, 64, 64)
            torch.testing.assert_close(
                EuroSATFeatures._spectral_slope(constant),
                chan.new_zeros(2, 3),
                rtol=0,
                atol=1e-7,
            )


if __name__ == '__main__':
    unittest.main()
