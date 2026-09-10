"""Composable, zero-learned-parameter EuroSAT descriptors."""
import torch
from torch import nn

from . import _ops as ops
from ._schema import FRONTIER_NAMES, GROUPS, PAIRS, POOL_NAMES, STATS_NAMES


class EuroSATFeatures(nn.Module):
    """Extract fixed features from raw EuroSAT TIFF-order tensors.

    Args:
        feature_set: ``'33'`` (default), ``'377'``, or ``'52'``. The 33-feature
            frontier consists of 32 pool columns and a separate region-shape
            statistic, NOT a subset of the 377 pool. The 52 features are
            per-band mean, population standard deviation, minimum, maximum.

    Input is a real numeric tensor of shape ``(N, 13, 64, 64)`` in original,
    unnormalized DN units. Physical band order is ``TIFF_BAND_NAMES``. Historical
    names/arithmetic are preserved: SWIR1 means index 11 (B12), SWIR2 means
    index 12 (B8A). Do not reorder those channels to match the legacy aliases.

    Arithmetic and output stay on the input CPU/CUDA device, with float32
    output regardless of input dtype. Reference float64 intermediates are
    preserved where needed. No NumPy, SciPy, rasterio, TorchGeo, or sklearn is
    used. Normal ``.to(device)`` moves fixed buffers; input-device use also
    works without moving the module first. Buffers are derived constants and
    intentionally absent from the state dict; there are no learned Parameters.

    Extraction runs under ``no_grad`` and is not differentiable (thresholds,
    binning, and connected components). A subsequent trainable ``nn.Linear``
    works normally. The numeric reference is NumPy 2.5 (including weak promotion
    for scalar percentiles). CPU/CUDA reductions, FFTs, and angle binning are
    not bit-identical across libraries; near-boundary inputs can change discrete
    features. NumPy versions with different scalar-promotion or transcendental
    implementations need not produce identical counts.
    No input range is imposed, but undefined/nonfinite float32 arithmetic
    (e.g. singular index denominators or overflow) raises ``ValueError``.
    Empty batches return ``(0, num_features)``.
    """

    def __init__(self, feature_set="33"):
        super().__init__()
        if not isinstance(feature_set, str) or feature_set not in ("33", "377", "52"):
            raise ValueError("feature_set must be '33', '377', or '52'")
        self._feature_set = feature_set
        self._feature_names = {"33": FRONTIER_NAMES, "377": POOL_NAMES, "52": STATS_NAMES}[feature_set]
        bins = None
        if feature_set != "52":
            bins, self._nrho, self._diagonal = ops.hough_geometry()
        self.register_buffer("_hough_bins", bins, persistent=False)
        self.register_buffer("_component_edges", ops.component_edges() if feature_set != "52" else None,
                             persistent=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Immutable ordered identifiers (the 377 pool includes duplicate correlations)."""
        return self._feature_names

    @property
    def num_features(self) -> int:
        return len(self._feature_names)

    def extra_repr(self):
        return f"feature_set={self._feature_set!r}, num_features={self.num_features}"

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.layout != torch.strided or images.is_quantized:
            raise TypeError("images must be a dense, unquantized real numeric tensor")
        if images.is_complex() or images.dtype == torch.bool:
            raise TypeError("images must have a real numeric dtype, not complex or bool")
        if images.ndim != 4 or tuple(images.shape[1:]) != (13, 64, 64):
            raise ValueError("expected images of shape (N, 13, 64, 64) in physical TIFF band order")
        if images.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA tensors are supported")
        # Ignore outer autocast contexts: these frozen descriptors require float32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            x = images.to(dtype=torch.float32).contiguous()
            if not torch.isfinite(x).all():
                raise ValueError("images must be finite and representable in float32")
            if len(x) == 0:
                return x.new_empty((0, self.num_features))
            if self._feature_set == "52":
                flat = x.flatten(2)
                mean, std = ops.mean_std(flat)
                result = torch.stack((mean, std, flat.amin(-1), flat.amax(-1)), -1).flatten(1)
            else:
                result = self._extract(x)
            if not torch.isfinite(result).all():
                raise ValueError("feature arithmetic produced nonfinite values; check DN inputs and index denominators")
            return result.float()

    def _extract(self, x):
        wanted = set(self.feature_names)
        values, cache = {}, {"scale0": x}

        def scale(s):
            key = f"scale{s}"
            if key not in cache:
                cache[key] = ops.pool2(scale(s - 1))
            return cache[key]

        def maps(s=0):
            key = f"maps{s}"
            if key not in cache:
                cache[key] = ops.index_maps(scale(s))
            return cache[key]

        def channels(ids):
            if 0 in ids and "pan" not in cache:
                cache["pan"] = ops.pan(x)
            return torch.stack([cache["pan"] if i == 0 else maps()[:, (0, 2)[i - 1]]
                                for i in ids], 1)

        def correlation():
            if "corr" not in cache:
                flat = x.flatten(2)
                z = flat - flat.mean(-1, keepdim=True)
                sd = (z * z).mean(-1).sqrt() + ops.EPS
                cache["corr"] = torch.stack([(z[:, a] * z[:, b]).mean(-1) / (sd[:, a] * sd[:, b])
                                             for a, b in PAIRS], -1)
            return cache["corr"]

        for group, names in GROUPS:
            selected = [i for i, name in enumerate(names) if name in wanted]
            if not selected:
                continue
            if group == "spectral_statistics":
                mean_bands = [i for i in selected if i < 13]
                std_bands = [i - 13 for i in selected if 13 <= i < 26]
                if mean_bands:
                    means = x[:, mean_bands].flatten(2).mean(-1)
                    for j, band in enumerate(mean_bands):
                        values[names[band]] = means[:, j]
                if std_bands:
                    _, std = ops.mean_std(x[:, std_bands].flatten(2))
                    for j, band in enumerate(std_bands):
                        values[names[13 + band]] = std[:, j]
                percentile_ids = [i for i in selected if i >= 26]
                if percentile_ids:
                    bands = sorted({i % 13 for i in percentile_ids})
                    levels = sorted({i // 13 - 2 for i in percentile_ids})
                    q = [(10, 25, 50, 75, 90)[level] for level in levels]
                    pc = ops.percentile(x[:, bands].flatten(2), q).float()
                    for i in percentile_ids:
                        values[names[i]] = pc[:, bands.index(i % 13), levels.index(i // 13 - 2)]
                continue
            elif group == "multiscale_gradients":
                for s in sorted({i // 26 for i in selected}):
                    ids = [i for i in selected if i // 26 == s]
                    bands = sorted({i % 13 for i in ids})
                    g = ops.magnitude(scale(s)[:, bands]).flatten(2)
                    if any(i % 26 >= 13 for i in ids):
                        mean, std = ops.mean_std(g)
                    else:
                        mean, std = g.mean(-1), None
                    for i in ids:
                        stat = mean if i % 26 < 13 else std
                        values[names[i]] = stat[:, bands.index(i % 13)]
                continue
            elif group == "coherence":
                result = torch.cat([ops.coherence(scale(s)) for s in range(2)], -1)
            elif group in ("orientation_entropy", "orientation_entropy_scale2"):
                bands = selected
                result = ops.orientation_entropy(scale(int(group.endswith("scale2")))[:, bands])
                for i, col in enumerate(bands):
                    values[names[col]] = result[:, i]
                continue
            elif group == "orientation_histogram":
                result = ops.orientation_histogram(x, 4).flatten(1).float()
            elif group == "spectral_peaks":
                centered = x - x.mean((-2, -1), keepdim=True)
                power = torch.fft.fftshift(torch.fft.fft2(centered).abs().square(), dim=(-2, -1))
                grid = torch.arange(64, device=x.device) - 32
                radius2 = grid[:, None].square() + grid[None, :].square()
                p = power[..., (radius2 >= 16) & (radius2 <= 576)]
                result = (p.amax(-1) / (p.mean(-1) + ops.EPS)).log1p()
            elif group in ("cross_band", "cross_band_correlation"):
                result = correlation()
            elif group in ("index_texture", "index_texture_scale2"):
                m = maps(int(group.endswith("scale2")))
                _, std = ops.mean_std(m.flatten(2))
                gm, gs = ops.mean_std(ops.magnitude(m).flatten(2))
                parts = [std, gm]
                if group == "index_texture":
                    quantiles = ops.percentile(m.flatten(2), [10, 90])
                    parts.extend((gs, (quantiles[..., 1] - quantiles[..., 0]).float()))
                result = torch.cat(parts, -1)
            elif group == "index_coherence":
                result = ops.coherence(maps())
            elif group in ("hough_lines", "harris_corners", "lbp", "blobs", "spectral_slope"):
                width = {"hough_lines": 3, "harris_corners": 2, "lbp": 2, "blobs": 3,
                         "spectral_slope": 1}[group]
                ids = sorted({i // width for i in selected})
                chan = channels(ids)
                if group == "hough_lines":
                    result = ops.hough(chan, self._hough_bins.to(x.device), self._nrho, self._diagonal)
                elif group == "harris_corners":
                    result = ops.harris(chan)
                elif group == "lbp":
                    result = ops.lbp(chan)
                elif group == "blobs":
                    result = ops.blobs(chan, self._component_edges.to(x.device))
                else:
                    result = self._spectral_slope(chan)
                for j, channel in enumerate(ids):
                    for k in range(width):
                        values[names[channel * width + k]] = result[:, j * width + k]
                continue
            else:
                raise AssertionError(group)
            for i in selected:
                values[names[i]] = result[:, i]
        if "tail_aniso_low_ndvi" in wanted:
            values["tail_aniso_low_ndvi"] = ops.tail_anisotropy(maps()[:, 0])
        return torch.stack([values[name] for name in self.feature_names], -1)

    @staticmethod
    def _spectral_slope(chan):
        freq = torch.fft.fftfreq(64, device=chan.device, dtype=torch.float64) * 64
        rbin = (freq[:, None].square() + freq[None, :].square()).sqrt().round().long().flatten()
        onehot = (rbin[None] == torch.arange(1, 33, device=chan.device)[:, None]).float()
        onehot = onehot / onehot.sum(-1, keepdim=True)
        logr = torch.arange(1, 33, device=chan.device, dtype=torch.float64).log()
        lc = logr - logr.mean()
        centered = chan - chan.mean((-2, -1), keepdim=True)
        power = torch.fft.fft2(centered).abs().square().flatten(-2)
        # Explicit reductions avoid TF32/autocast-dependent matrix products.
        radial = torch.stack([(power * row).sum(-1) for row in onehot], -1)
        return (((radial + ops.EPS).log().double() * lc).sum(-1) / (lc @ lc)).float()
