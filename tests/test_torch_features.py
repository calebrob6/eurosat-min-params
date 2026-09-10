"""Native tensor parity against the independent, original NumPy/SciPy recipe.

Run: python -m unittest discover -s tests -p test_torch_features.py

Tolerances are per family: DN reductions allow a few float32 ULPs, continuous
normalized statistics allow sub-micro absolute error, FFT summaries allow a few
micro-units for different FFT/reduction implementations. Integer-count families
(Hough, LBP, components) must agree exactly. Discrete bin/threshold errors are
not covered by broad relative tolerances. A nearly constant gradient's two-pass
standard deviation inherits rounding in its mean: its absolute budget is four
float32 epsilons of that mean, rather than a relative error in the tiny std.
"""
from pathlib import Path
from io import BytesIO
import importlib.util
import unittest

import numpy as np
import torch
from scipy.ndimage import label, uniform_filter

from eurosat_features import EuroSATFeatures, TIFF_BAND_NAMES
from eurosat_features import _ops
from experiments.representation_overlap.data import feature_matrices


MODES = (("33", "frontier33"), ("377", "pool377"), ("52", "imagestats52"))
DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
# (relative tolerance, absolute tolerance), applied to every named column.
TOLERANCES = {
    "spectral_statistics": (3e-7, 2e-5),
    "image_statistics": (3e-7, 2e-5),
    "multiscale_gradients": (6e-7, 2e-5),
    "coherence": (5e-7, 2e-7),
    "orientation_entropy": (0, 4e-7),
    "orientation_histogram": (0, 1e-7),
    "spectral_peaks": (1e-6, 3e-6),
    "cross_band": (0, 5e-7),
    "index_texture": (5e-7, 3e-7),
    "hough_lines": (0, 0),
    "harris_corners": (0, 2e-7),
    "lbp": (0, 0),
    "blobs": (0, 0),
    "spectral_slope": (1e-6, 3e-6),
    "index_coherence": (5e-7, 2e-7),
    "index_texture_scale2": (5e-7, 2e-7),
    "cross_band_correlation": (0, 5e-7),
    "orientation_entropy_scale2": (0, 4e-7),
    "region_shape": (0, 3e-7),
}


def patterns():
    y, x = np.mgrid[:64, :64]
    band = np.arange(1, 14, dtype=np.float32)[:, None, None]
    impulse = np.zeros((64, 64), np.float32)
    impulse[0, 0], impulse[32, 31], impulse[-1, -1] = 100, 500, 700
    return np.stack((
        np.zeros((13, 64, 64), np.float32),
        np.broadcast_to(band * 100, (13, 64, 64)),
        1000 + band * (x + y),
        1000 + band * (x - y),
        1000 + band * ((x + y) % 2) * 100,
        1000 + band * ((x // 8) % 2) * 100,
        1000 + band * impulse,
    )).astype(np.float32)


class TorchFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_columns(self, actual, expected, schema, context):
        names = schema["names"]
        for i, (name, family) in enumerate(zip(names, schema["families"])):
            rtol, atol = TOLERANCES[family]
            if family == "multiscale_gradients" and "std" in name:
                mean_index = names.index(name.replace("std", "mean"))
                budget = np.maximum(atol, 4 * np.finfo(np.float32).eps * np.abs(expected[:, mean_index]))
                error = np.abs(actual[:, i] - expected[:, i])
                self.assertTrue(np.all(error <= budget + rtol * np.abs(expected[:, i])),
                                f"{context}/{name}: errors {error} exceed budgets {budget}")
            else:
                np.testing.assert_allclose(
                    actual[:, i], expected[:, i], rtol=rtol, atol=atol,
                    err_msg=f"{context}/{family}/{name}",
                )

    def assert_reference(self, images):
        reference, schemas = feature_matrices(images)
        for device in DEVICES:
            for mode, key in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device)
                    result = module(torch.from_numpy(images).to(device))
                    self.assertEqual(result.device.type, device)
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.shape, (len(images), int(mode)))
                    self.assertEqual(module.feature_names, tuple(schemas[key]["names"]))
                    actual = result.cpu().numpy()
                    self.assert_columns(actual, reference[key], schemas[key], f"{device}/{mode}")

    def test_synthetic_pattern_parity_all_columns(self):
        self.assert_reference(patterns())

    def test_random_real_dn_parity_all_columns(self):
        rng = np.random.default_rng(41)
        images = rng.uniform(1, 10000, (3, 13, 64, 64)).astype(np.float32)
        images[1] *= -1  # No reflectance-range or nonnegative-DN assumption.
        self.assert_reference(images)

    @unittest.skipUnless(importlib.util.find_spec("rasterio"), "raw TIFF parity requires rasterio")
    def test_raw_tiff_parity_all_columns(self):
        import rasterio

        root = Path(__file__).resolve().parents[1] / "data" / "EuroSAT"
        tif_root = root / "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
        split = root / "eurosat-val.txt"
        if not split.is_file() or not tif_root.is_dir():
            self.skipTest("original EuroSAT TIFF data not present")
        names_by_class = {}
        for name in split.read_text().splitlines():
            names_by_class.setdefault(name.rsplit("_", 1)[0], name)
        images = []
        for name in names_by_class.values():
            path = tif_root / name.rsplit("_", 1)[0] / (Path(name).stem + ".tif")
            if not path.is_file():
                self.skipTest(f"original TIFF absent: {path.name}")
            with rasterio.open(path) as handle:
                images.append(handle.read().astype(np.float32))
        if not images:
            self.skipTest("validation split is empty")
        self.assert_reference(np.stack(images))

    def test_frontier_is_not_pool_subset(self):
        from src.frontier import POOL_INDICES

        image = torch.from_numpy(patterns()[2:3])
        module = EuroSATFeatures()
        pool = EuroSATFeatures("377")(image)
        result = module(image)
        torch.testing.assert_close(result[:, :32], pool[:, POOL_INDICES.tolist()], rtol=0, atol=0)
        self.assertEqual(module.feature_names[-1], "tail_aniso_low_ndvi")
        self.assertNotIn(module.feature_names[-1], EuroSATFeatures("377").feature_names)

    def test_batch_invariance_noncontiguous_and_empty(self):
        image = torch.from_numpy(np.random.default_rng(4).integers(
            1, 10000, (3, 13, 64, 64), dtype=np.uint16).astype(np.float32))
        _, schemas = feature_matrices(image.numpy())
        for device in DEVICES:
            for mode, key in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device)
                    batch = image.to(device).transpose(-1, -2)
                    self.assertFalse(batch.is_contiguous())
                    result = module(batch)
                    singles = torch.cat([module(row.unsqueeze(0)) for row in batch])
                    # CUDA may choose different reduction trees by batch size;
                    # count-valued families must still be exactly invariant.
                    self.assert_columns(result.cpu().numpy(), singles.cpu().numpy(),
                                        schemas[key], f"batch invariance/{device}/{mode}")
                    torch.testing.assert_close(result, module(batch.contiguous()), rtol=0, atol=0)
                    empty = module(batch[:0])
                    self.assertEqual(empty.shape, (0, int(mode)))
                    self.assertEqual(empty.device.type, device)
                    self.assertEqual(empty.dtype, torch.float32)

    def test_uint16_double_and_autocast(self):
        image = torch.from_numpy(np.random.default_rng(7).integers(
            0, 10000, (1, 13, 64, 64), dtype=np.uint16))
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device).double()
                    x = image.to(device)
                    expected = module(x.float())
                    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
                    torch.testing.assert_close(module(x.double()), expected, rtol=0, atol=0)
                    with torch.autocast(device_type=device):
                        actual = module(x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_module_contract_and_head_backprop(self):
        self.assertEqual(TIFF_BAND_NAMES, (
            "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B09", "B10", "B11", "B12", "B8A"))
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode)
                    self.assertIsInstance(module.feature_names, tuple)
                    self.assertEqual(module.num_features, int(mode))
                    self.assertEqual(list(module.parameters()), [])
                    self.assertEqual(dict(module.state_dict()), {})
                    with self.assertRaises(AttributeError):
                        module.num_features = 4
                    with self.assertRaises(AttributeError):
                        module.feature_names = ("wrong",)
                    module.to(device)
                    self.assertTrue(all(buffer.device.type == device for buffer in module.buffers()))
                    model = torch.nn.Sequential(module, torch.nn.Linear(int(mode), 2)).to(device)
                    image = torch.rand(1, 13, 64, 64, device=device).add_(1).requires_grad_()
                    self.assertFalse(module(image).requires_grad)
                    model(image).square().sum().backward()
                    self.assertIsNone(image.grad)
                    self.assertIsNotNone(model[1].weight.grad)
                    self.assertTrue(torch.isfinite(model[1].weight.grad).all())
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device):
                        prediction = model(image)
                        loss = prediction.square().mean()
                    self.assertEqual(prediction.dtype, torch.float16 if device == "cuda" else torch.bfloat16)
                    loss.backward()
                    self.assertIsNone(image.grad)
                    self.assertIsNotNone(model[1].weight.grad)
                    self.assertTrue(torch.isfinite(model[1].weight.grad).all())
                    # Fixed geometry can also follow input without .to().
                    torch.testing.assert_close(EuroSATFeatures(mode)(image), module(image), rtol=0, atol=0)

    def test_state_dict_round_trip(self):
        image = torch.from_numpy(np.random.default_rng(13).uniform(
            1, 10000, (1, 13, 64, 64)).astype(np.float32))
        for device in DEVICES:
            for mode, _ in MODES:
                with self.subTest(device=device, mode=mode):
                    module = EuroSATFeatures(mode).to(device)
                    stream = BytesIO()
                    torch.save(module.state_dict(), stream)
                    stream.seek(0)
                    restored = EuroSATFeatures(mode).to(device)
                    restored.load_state_dict(torch.load(stream, map_location=device, weights_only=True), strict=True)
                    self.assertEqual(module.feature_names, restored.feature_names)
                    x = image.to(device)
                    torch.testing.assert_close(module(x), restored(x), rtol=0, atol=0)

    def test_invalid_inputs(self):
        for value in ("bad", "frontier33", 33, None, []):
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
        for value in (float("nan"), float("inf"), -float("inf")):
            image = torch.ones(1, 13, 64, 64)
            image[0, 0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                module(image)
        with self.assertRaisesRegex(ValueError, "float32"):
            module(torch.full((1, 13, 64, 64), 1e100, dtype=torch.float64))

    def test_scipy_edge_padding(self):
        rng = np.random.default_rng(28)
        image = rng.normal(size=(2, 3, 7, 9)).astype(np.float32)
        image[:, :, 0, 0] = 30
        expected = uniform_filter(image, size=(1, 1, 3, 3), mode="reflect")
        for device in DEVICES:
            actual = _ops.uniform_filter3(torch.from_numpy(image).to(device)).cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
        # PyTorch's reflect excludes the edge sample and is not equivalent.
        reflected = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(
            torch.from_numpy(image), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        self.assertNotEqual(reflected[0, 0, 0, 0].item(), float(expected[0, 0, 0, 0]))

    def test_eight_connected_components(self):
        masks = np.zeros((7, 64, 64), dtype=bool)
        masks[0] = np.eye(64, dtype=bool)  # Diagonal adjacency is connected.
        masks[1, ::3, ::3] = True
        masks[2] = True
        masks[3, (0, -1), :] = True
        masks[3, :, (0, -1)] = True  # Hollow square, path longer than diameter.
        masks[4, 2:5, 3:6] = True
        masks[4, 40:46, 41:47] = True
        # Long serpentine component tests convergence, not a fixed 64 steps.
        masks[5, ::3, :] = True
        for row in range(0, 61, 3):
            masks[5, row:row + 4, -1 if row % 6 == 0 else 0] = True
        for device in DEVICES:
            sizes = _ops.component_sizes(torch.from_numpy(masks).to(device)).cpu().numpy()
            for i, mask in enumerate(masks):
                labels, _ = label(mask, structure=np.ones((3, 3)))
                expected = np.bincount(labels.ravel())[1:]
                np.testing.assert_array_equal(np.sort(sizes[i][sizes[i] > 0]), np.sort(expected))

    def test_percentiles_and_even_median(self):
        image = np.random.default_rng(62).uniform(-1e4, 1e4, (2, 3, 64)).astype(np.float32)
        q = [10, 25, 50, 75, 85, 90]
        expected = np.percentile(image, q, axis=-1).transpose(1, 2, 0)
        for device in DEVICES:
            x = torch.from_numpy(image).to(device)
            np.testing.assert_array_equal(_ops.percentile(x, q).cpu().numpy(), expected)
            np.testing.assert_array_equal(_ops.median(x).cpu().numpy(), np.median(image, axis=-1))
            scalar = _ops.percentile(x, 85)
            self.assertEqual(scalar.dtype, torch.float32)
            np.testing.assert_array_equal(scalar.cpu().numpy(), np.percentile(image, 85, axis=-1))

    def test_scalar_percentile_rounding_and_gradient_magnitude(self):
        # The 85th percentile falls 80% between adjacent floats, so rounding the
        # scalar threshold excludes the upper value whereas float64 includes it.
        values = np.zeros(3969, dtype=np.float32)
        lower = np.float32(0.05)
        upper = np.nextafter(lower, np.float32(np.inf))
        values[3372], values[3373:] = lower, upper
        scalar = np.percentile(values, 85)
        vector = np.percentile(values, [85])[0]
        self.assertEqual(scalar, upper)
        self.assertLess(vector, float(upper))
        rng = np.random.default_rng(203)
        channels = rng.uniform(-1, 1, (3, 5, 64, 64)).astype(np.float32)
        gx = np.diff(channels, axis=-1)[..., :-1, :]
        gy = np.diff(channels, axis=-2)[..., :, :-1]
        expected_mag = np.sqrt(gx * gx + gy * gy)
        for device in DEVICES:
            tensor = torch.from_numpy(values).to(device)
            threshold = _ops.percentile(tensor, 85)
            np.testing.assert_array_equal((tensor > threshold).cpu().numpy(), values > scalar)
            magnitude = _ops.magnitude(torch.from_numpy(channels).to(device))
            np.testing.assert_array_equal(magnitude.cpu().numpy(), expected_mag)

    def test_constant_hough_and_diagonal_orientation_ties(self):
        from src.features import _hough_line_stats, orientation_histogram_features

        y, x = np.mgrid[:64, :64]
        ramp = np.broadcast_to((x + y).astype(np.float32), (1, 13, 64, 64)).copy()
        expected, _ = orientation_histogram_features(ramp)
        bins, nrho, diag = _ops.hough_geometry()
        pan = ramp.mean(1)
        hough_expected = np.column_stack(_hough_line_stats(pan))
        for device in DEVICES:
            tensor = torch.from_numpy(ramp).to(device)
            histogram = _ops.orientation_histogram(tensor, 4).flatten(1).float()
            np.testing.assert_array_equal(histogram.cpu().numpy(), expected)
            native_pan = _ops.pan(tensor)
            # A constant gradient must produce no pixels above its percentile.
            result = _ops.hough(native_pan[:, None], bins.to(device), nrho, diag)
            np.testing.assert_array_equal(result.cpu().numpy(), hough_expected)


if __name__ == "__main__":
    unittest.main()
