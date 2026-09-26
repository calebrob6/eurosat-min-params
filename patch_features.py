"""Fixed spectral and spatial image features as a drop-in PyTorch module.

Copy this file into your project; it only needs PyTorch.

    >>> from patch_features import PatchFeatures
    >>> extractor = PatchFeatures(bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])
    >>> features = extractor(images)  # (N, 6, H, W) -> (N, extractor.num_features)

Nothing is learned: every feature is a fixed computation on the pixels, so the
module can sit in front of any classifier, e.g. ``nn.Sequential(extractor,
nn.Linear(extractor.num_features, num_classes))``.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6

# Band names (case-insensitive) that are recognized for each spectral role:
# plain names, Sentinel-2, and Landsat 8/9 surface reflectance.
ROLE_ALIASES = {
    'blue': ('blue', 'b', 'b02', 'sr_b2'),
    'green': ('green', 'g', 'b03', 'sr_b3'),
    'red': ('red', 'r', 'b04', 'sr_b4'),
    'nir': ('nir', 'b08', 'sr_b5'),
    'swir1': ('swir1', 'swir16', 'b11', 'sr_b6'),
    'swir2': ('swir2', 'swir22', 'b12', 'sr_b7'),
}


def _nd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) / (a + b + EPS)


def _saturation(r: dict[str, torch.Tensor]) -> torch.Tensor:
    rgb = torch.stack((r['red'], r['green'], r['blue']))
    high = rgb.amax(0)
    return (high - rgb.amin(0)) / (high + EPS)


def _lightness(r: dict[str, torch.Tensor]) -> torch.Tensor:
    rgb = torch.stack((r['red'], r['green'], r['blue']))
    return (rgb.amax(0) + rgb.amin(0)) / 2


# Derived per-pixel channels: name -> (required roles, function of those bands).
DERIVED: dict[str, tuple[tuple[str, ...], Callable]] = {
    'ndvi': (('nir', 'red'), lambda r: _nd(r['nir'], r['red'])),
    'ndwi': (('green', 'nir'), lambda r: _nd(r['green'], r['nir'])),
    'ndbi': (('swir1', 'nir'), lambda r: _nd(r['swir1'], r['nir'])),
    'ndmi': (('nir', 'swir1'), lambda r: _nd(r['nir'], r['swir1'])),
    'nbr': (('nir', 'swir2'), lambda r: _nd(r['nir'], r['swir2'])),
    'bsi': (
        ('swir1', 'red', 'nir', 'blue'),
        lambda r: ((r['swir1'] + r['red']) - (r['nir'] + r['blue']))
        / ((r['swir1'] + r['red']) + (r['nir'] + r['blue']) + EPS),
    ),
    'exg': (
        ('red', 'green', 'blue'),
        lambda r: (2 * r['green'] - r['red'] - r['blue'])
        / (r['red'] + r['green'] + r['blue'] + EPS),
    ),
    'ngrdi': (('green', 'red'), lambda r: _nd(r['green'], r['red'])),
    'ngbdi': (('green', 'blue'), lambda r: _nd(r['green'], r['blue'])),
    'nrbdi': (('red', 'blue'), lambda r: _nd(r['red'], r['blue'])),
    'saturation': (('red', 'green', 'blue'), _saturation),
    'lightness': (('red', 'green', 'blue'), _lightness),
}
# Channels that get the spatial-structure families in the default feature list.
STRUCTURAL = ('pan', 'ndvi', 'ndbi', 'exg', 'saturation')
CORRELATION_PAIRS = (
    ('red', 'nir'),
    ('green', 'nir'),
    ('blue', 'nir'),
    ('swir1', 'nir'),
    ('red', 'swir1'),
    ('green', 'red'),
    ('nir', 'swir2'),
    ('swir1', 'swir2'),
    ('red', 'blue'),
    ('green', 'blue'),
)
GROUPS = {
    'moments': ('mean', 'std', 'min', 'max'),
    'percentiles': ('p10', 'p25', 'p50', 'p75', 'p90', 'spread'),
    'gradient': ('grad_mean', 'grad_std'),
    'coherence': ('coherence',),
    'entropy': ('orient_entropy',),
    'histogram': tuple(f'orient_hist{i}' for i in range(4)),
    'fft_peak': ('fft_peak',),
    'hough': ('line_peak_frac', 'line_peak_len', 'line_top3'),
    'harris': ('corner_frac', 'corner_mag'),
    'lbp': ('lbp_entropy', 'lbp_uniform'),
    'blobs': ('blob_largest', 'blob_count', 'blob_mean'),
    'fft_slope': ('fft_slope',),
    'tail': ('tail_aniso_low', 'tail_spread_low', 'tail_aniso_high', 'tail_spread_high'),
}
MEASURE_GROUP = {m: group for group, measures in GROUPS.items() for m in measures}


class PatchFeatures(nn.Module):
    """Compute fixed spectral and spatial features from multiband image patches.

    Feature names are ``{measure}_{channel}`` or ``{measure}_s{k}_{channel}``,
    where ``s{k}`` means the channel was average-pooled 2x2 ``k`` times first.
    Channels are the input bands, ``pan`` (the mean of all bands), and derived
    maps such as ``ndvi``; correlations are ``corr_{band}_{band}``. Any valid
    combination can be requested with ``features=``. Measures:

    - ``mean``, ``std``, ``min``, ``max``, ``p10``...``p90``, ``spread`` (p90 - p10)
    - ``grad_mean``, ``grad_std``: gradient magnitude statistics
    - ``coherence``: structure-tensor anisotropy of the gradients
    - ``orient_entropy``, ``orient_hist0``...``orient_hist3``: gradient directions
    - ``fft_peak``, ``fft_slope``: Fourier power peak and radial power-law slope
    - ``line_*`` (Hough lines), ``corner_*`` (Harris corners), ``lbp_*`` (local
      binary patterns), ``blob_*`` (connected above-median regions), and
      ``tail_*`` (shape of the lowest/highest-quartile pixels)

    Band roles (blue, green, red, nir, swir1, swir2) enable the derived maps and
    are recognized from common names (``'red'``, ``'B04'``, ``'SR_B4'``, ...);
    pass ``roles`` to set or override them. Maps whose bands are missing are
    unavailable, e.g. RGB input has no ``ndvi``.

    Inputs are ``(N, C, H, W)`` tensors in any consistent units (no
    normalization needed), with ``min(H, W) >= 16``. Output is float32 on the
    input device. Extraction runs without gradients.

    Args:
        bands: Name of each input channel, in order.
        features: Feature names to compute, in output order. Defaults to a
            broad list covering every family for the available channels.
        roles: Optional mapping from role to band name, e.g. ``{'nir': 'B8A'}``.
            Map a role to ``None`` to disable it.
    """

    def __init__(
        self,
        bands: Sequence[str],
        features: Sequence[str] | None = None,
        roles: Mapping[str, str | None] | None = None,
    ) -> None:
        """Resolve band roles and parse the requested features."""
        super().__init__()
        self.bands = tuple(bands)
        if len(set(self.bands)) != len(self.bands):
            raise ValueError('band names must be unique')
        clashes = set(self.bands) & ({'pan'} | set(DERIVED))
        if clashes:
            raise ValueError(f'band names clash with derived channels: {sorted(clashes)}')
        self.roles = _resolve_roles(self.bands, roles or {})
        derived = [
            name for name, (need, _) in DERIVED.items() if all(r in self.roles for r in need)
        ]
        self._band_index = {band: i for i, band in enumerate(self.bands)}
        self._channels = set(self.bands) | {'pan'} | set(derived)
        if features is None:
            features = _default_features(self.bands, self.roles, derived)
        self._feature_names = tuple(features)
        self._specs = [self._parse(name) for name in self._feature_names]
        self._plan: dict[tuple[str, int], tuple[list[str], set[str]]] = {}
        for measure, scale, channel in self._specs:
            key = (MEASURE_GROUP.get(measure, 'corr'), scale)
            channels, measures = self._plan.setdefault(key, ([], set()))
            for c in channel if measure == 'corr' else (channel,):
                if c not in channels:
                    channels.append(c)
            measures.add(measure)
        self._max_scale = max((s for _, s, _ in self._specs), default=0)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of the output columns, in order."""
        return self._feature_names

    @property
    def num_features(self) -> int:
        """Number of output columns."""
        return len(self._feature_names)

    @property
    def feature_families(self) -> tuple[str, ...]:
        """Family that computes each output column, e.g. ``'gradient'`` or ``'hough'``."""
        return tuple(MEASURE_GROUP.get(m, 'correlation') for m, _, _ in self._specs)

    def extra_repr(self) -> str:
        """Summarize the configuration."""
        return f'bands={len(self.bands)}, num_features={self.num_features}'

    def _parse(self, name: str) -> tuple[str, int, str | tuple[str, str]]:
        if name.startswith('corr_'):
            rest = name[5:]
            for i, char in enumerate(rest):
                if char == '_' and rest[:i] in self._channels and rest[i + 1 :] in self._channels:
                    return 'corr', 0, (rest[:i], rest[i + 1 :])
        for measure in sorted(MEASURE_GROUP, key=len, reverse=True):
            if not name.startswith(measure + '_'):
                continue
            rest = name[len(measure) + 1 :]
            scaled = re.fullmatch(r's(\d+)_(.+)', rest)
            if scaled and scaled[2] in self._channels:
                return measure, int(scaled[1]), scaled[2]
            if rest in self._channels:
                return measure, 0, rest
        raise ValueError(f'unknown feature or unavailable channel: {name!r}')

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute the features.

        Args:
            images: Tensor of shape ``(N, len(bands), H, W)``.

        Returns:
            Float32 tensor of shape ``(N, num_features)``.
        """
        if images.ndim != 4 or images.shape[1] != len(self.bands):
            raise ValueError(f'expected images of shape (N, {len(self.bands)}, H, W)')
        size = min(images.shape[-2:])
        if size < 16 or size >> self._max_scale < 4:
            raise ValueError('images are too small for the requested features')
        with torch.autocast(images.device.type, enabled=False):
            x = images.float()
            if len(x) == 0:
                return x.new_empty((0, self.num_features))
            if not torch.isfinite(x).all():
                raise ValueError('images must be finite')
            values = self._compute(x)
            out = torch.stack([values[(m, s, c)] for m, s, c in self._specs], -1)
            if not torch.isfinite(out).all():
                raise ValueError('features are not finite; check for invalid pixel values')
            return out

    def _compute(self, x: torch.Tensor) -> dict:
        pyramid = [x]
        cache: dict[tuple[str, int], torch.Tensor] = {}

        def channel(name: str, scale: int) -> torch.Tensor:
            if (name, scale) not in cache:
                while len(pyramid) <= scale:
                    pyramid.append(_pool2(pyramid[-1]))
                level = pyramid[scale]
                if name in self._band_index:
                    value = level[:, self._band_index[name]]
                elif name == 'pan':
                    value = level.mean(1)
                else:
                    need, fn = DERIVED[name]
                    value = fn({r: level[:, self._band_index[self.roles[r]]] for r in need})
                cache[name, scale] = value
            return cache[name, scale]

        values = {}
        for (group, scale), (channels, measures) in self._plan.items():
            stack = torch.stack([channel(c, scale) for c in channels], 1)
            if group == 'corr':
                z = stack.flatten(2)
                z = z - z.mean(-1, keepdim=True)
                sd = (z * z).mean(-1).sqrt() + EPS
                for m, s, pair in self._specs:
                    if m == 'corr':
                        a, b = (channels.index(c) for c in pair)
                        values[m, s, pair] = (z[:, a] * z[:, b]).mean(-1) / (sd[:, a] * sd[:, b])
                continue
            result = _GROUP_FUNCTIONS[group](stack, measures)
            for measure in measures:
                for i, c in enumerate(channels):
                    values[measure, scale, c] = result[measure][:, i]
        return values


class EuroSATFeatures(PatchFeatures):
    """Features for 13-band EuroSAT patches in TorchGeo's band order.

    ``'33'`` is the feature set of the 306-parameter EuroSAT classifier,
    ``'52'`` is the ImageStats baseline, and ``'377'``/``'389'`` are the
    candidate pools from the article. Like the trained model, these use B12 as
    ``swir1`` and B8A as ``swir2``.
    """

    def __init__(self, feature_set: str = '33') -> None:
        """Select one of the EuroSAT feature sets."""
        sets = {'33': EUROSAT_33, '52': EUROSAT_52, '377': EUROSAT_377, '389': EUROSAT_389}
        if feature_set not in sets:
            raise ValueError(f'feature_set must be one of {sorted(sets)}')
        super().__init__(EUROSAT_BANDS, sets[feature_set], EUROSAT_ROLES)


def _resolve_roles(bands: tuple[str, ...], overrides: Mapping) -> dict[str, str]:
    lower = {band.lower(): band for band in bands}
    roles = {}
    for role, aliases in ROLE_ALIASES.items():
        match = next((lower[a] for a in aliases if a in lower), None)
        if match is not None:
            roles[role] = match
    for role, band in overrides.items():
        if role not in ROLE_ALIASES:
            raise ValueError(f'unknown role {role!r}; expected one of {sorted(ROLE_ALIASES)}')
        if band is None:
            roles.pop(role, None)
        elif band not in bands:
            raise ValueError(f'role {role!r} refers to unknown band {band!r}')
        else:
            roles[role] = band
    return roles


def _name(measure: str, scale: int, channel: str) -> str:
    return f'{measure}_s{scale}_{channel}' if scale else f'{measure}_{channel}'


def _default_features(
    bands: tuple[str, ...], roles: dict[str, str], derived: list[str]
) -> list[str]:
    names = [
        f'{m}_{b}'
        for m in ('mean', 'std', 'min', 'max', 'p10', 'p25', 'p50', 'p75', 'p90')
        for b in bands
    ]
    names += [_name(m, s, b) for s in range(3) for m in GROUPS['gradient'] for b in bands]
    names += [_name(m, s, b) for m in ('coherence', 'orient_entropy') for s in range(2) for b in bands]
    names += [f'{m}_{b}' for b in bands for m in GROUPS['histogram']]
    names += [f'fft_peak_{b}' for b in bands]
    names += [
        f'corr_{roles[a]}_{roles[b]}'
        for a, b in CORRELATION_PAIRS
        if a in roles and b in roles and roles[a] != roles[b]
    ]
    for d in derived:
        names += [f'{m}_{d}' for m in ('std', 'grad_mean', 'grad_std', 'spread', 'coherence')]
        names += [f'std_s1_{d}', f'grad_mean_s1_{d}']
    for c in STRUCTURAL:
        if c == 'pan' or c in derived:
            names += [
                f'{m}_{c}'
                for group in ('hough', 'harris', 'lbp', 'blobs', 'fft_slope', 'tail')
                for m in GROUPS[group]
            ]
    return list(dict.fromkeys(names))


# --- Feature families. Each maps (N, C, H, W) to {measure: (N, C)}. ---


def _pool2(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    x = x[..., : h // 2 * 2, : w // 2 * 2]
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def _gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences on the shared interior grid, (H - 1, W - 1)."""
    return x[..., :-1, 1:] - x[..., :-1, :-1], x[..., 1:, :-1] - x[..., :-1, :-1]


def _magnitude(x: torch.Tensor) -> torch.Tensor:
    gx, gy = _gradients(x)
    return (gx * gx + gy * gy).sqrt()


def _quantile(x: torch.Tensor, q: Sequence[float]) -> torch.Tensor:
    """Percentiles along the last axis, stacked on a new last axis."""
    return torch.quantile(x, torch.tensor(q, dtype=x.dtype, device=x.device) / 100, dim=-1).movedim(0, -1)


def _moments(x: torch.Tensor, measures: set[str]) -> dict:
    flat = x.flatten(2)
    fns = {
        'mean': lambda: flat.mean(-1),
        'std': lambda: flat.std(-1, correction=0),
        'min': lambda: flat.amin(-1),
        'max': lambda: flat.amax(-1),
    }
    return {m: fns[m]() for m in measures}


def _percentiles(x: torch.Tensor, measures: set[str]) -> dict:
    qs = {int(m[1:]) for m in measures if m != 'spread'}
    if 'spread' in measures:
        qs |= {10, 90}
    qs = sorted(qs)
    values = _quantile(x.flatten(2), qs)
    out = {f'p{q}': values[..., i] for i, q in enumerate(qs)}
    if 'spread' in measures:
        out['spread'] = out['p90'] - out['p10']
    return out


def _gradient(x: torch.Tensor, measures: set[str]) -> dict:
    g = _magnitude(x).flatten(2)
    return {'grad_mean': g.mean(-1), 'grad_std': g.std(-1, correction=0)}


def _coherence(x: torch.Tensor, measures: set[str]) -> dict:
    gx, gy = _gradients(x)
    sxx, syy, sxy = ((a * b).mean((-2, -1)) for a, b in ((gx, gx), (gy, gy), (gx, gy)))
    return {'coherence': ((sxx - syy).square() + 4 * sxy * sxy).sqrt() / (sxx + syy + EPS)}


def _orientation_histogram(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Magnitude-weighted histogram of unsigned gradient direction, (N, C, bins)."""
    gx, gy = _gradients(x)
    mag = (gx * gx + gy * gy).sqrt().flatten(-2)
    angle = torch.remainder(torch.atan2(gy, gx), math.pi)
    idx = (angle / (math.pi / bins)).long().clamp(max=bins - 1).flatten(-2)
    hist = torch.stack([torch.where(idx == b, mag, 0).sum(-1) for b in range(bins)], -1)
    return hist / (hist.sum(-1, keepdim=True) + EPS)


def _entropy(x: torch.Tensor, measures: set[str]) -> dict:
    hist = _orientation_histogram(x, 8)
    return {'orient_entropy': -(hist * (hist + EPS).log()).sum(-1)}


def _histogram(x: torch.Tensor, measures: set[str]) -> dict:
    hist = _orientation_histogram(x, 4)
    return {f'orient_hist{i}': hist[..., i] for i in range(4)}


def _frequencies(h: int, w: int, device: torch.device, shift: bool = False) -> torch.Tensor:
    """Radial frequency of each FFT coefficient, in cycles per min(h, w) pixels."""
    n = min(h, w)
    fy, fx = torch.fft.fftfreq(h, device=device) * n, torch.fft.fftfreq(w, device=device) * n
    if shift:
        fy, fx = torch.fft.fftshift(fy), torch.fft.fftshift(fx)
    return (fy[:, None].square() + fx[None, :].square()).sqrt()


def _fft_peak(x: torch.Tensor, measures: set[str]) -> dict:
    """Log ratio of peak to mean power at mid frequencies (n/16 to 3n/8 cycles)."""
    n = min(x.shape[-2:])
    centered = x - x.mean((-2, -1), keepdim=True)
    power = torch.fft.fftshift(torch.fft.fft2(centered).abs().square(), dim=(-2, -1))
    radius = _frequencies(*x.shape[-2:], x.device, shift=True)
    p = power[..., (radius >= n / 16) & (radius <= 3 * n / 8)]
    return {'fft_peak': (p.amax(-1) / (p.mean(-1) + EPS)).log1p()}


def _fft_slope(x: torch.Tensor, measures: set[str]) -> dict:
    """Least-squares slope of log radial power against log frequency."""
    h, w = x.shape[-2:]
    radii = torch.arange(1, min(h, w) // 2 + 1, device=x.device)
    rbin = _frequencies(h, w, x.device).round().long().flatten()
    onehot = (rbin[None] == radii[:, None]).float()
    onehot = onehot / onehot.sum(-1, keepdim=True)
    logr = radii.float().log()
    lc = logr - logr.mean()
    centered = x - x.mean((-2, -1), keepdim=True)
    radial = torch.fft.fft2(centered).abs().square().flatten(-2) @ onehot.T
    logpower = (radial + EPS).log()
    logpower = logpower - logpower.mean(-1, keepdim=True)
    return {'fft_slope': (logpower @ lc) / (lc @ lc)}


@functools.lru_cache(maxsize=8)
def _hough_geometry(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, int, float]:
    """Rho-bin index of every gradient pixel for 60 angles, with 2-pixel rho bins."""
    diag = math.hypot(h, w)
    nrho = math.ceil(diag) + 1
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij'
    )
    theta = torch.arange(60, dtype=torch.float32) * (math.pi / 60)
    rho = theta.cos()[:, None] * x.flatten() + theta.sin()[:, None] * y.flatten()
    bins = ((rho + diag) / 2).floor().long().clamp(0, nrho - 1)
    return bins.to(device), nrho, diag


def _hough(x: torch.Tensor, measures: set[str]) -> dict:
    """Straight-line evidence among the strongest 15% of gradients."""
    mag = _magnitude(x)
    bins, nrho, diag = _hough_geometry(*mag.shape[-2:], x.device)
    mag = mag.flatten(-2)
    edge = (mag > _quantile(mag, [85])).float()
    flat = edge.reshape(-1, edge.shape[-1])
    peaks = []
    for chunk in bins.split(10):  # bound memory for large batches
        acc = flat.new_zeros((len(flat), len(chunk), nrho))
        acc.scatter_add_(
            2,
            chunk.unsqueeze(0).expand(len(flat), -1, -1),
            flat.unsqueeze(1).expand(-1, len(chunk), -1),
        )
        peaks.append(acc.amax(-1))
    per_angle = torch.cat(peaks, -1)
    peak = per_angle.amax(-1).reshape(edge.shape[:-1])
    top3 = per_angle.topk(3, dim=-1).values.sum(-1).reshape(edge.shape[:-1])
    count = edge.sum(-1) + EPS
    return {'line_peak_frac': peak / count, 'line_peak_len': peak / diag, 'line_top3': top3 / count}


def _harris(x: torch.Tensor, measures: set[str]) -> dict:
    """Share of strong-gradient pixels with a positive Harris response, and its magnitude."""
    gy, gx = torch.gradient(x, dim=(-2, -1), edge_order=1)

    def box(v: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode='replicate'), 3, stride=1)

    jxx, jyy, jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    trace = jxx + jyy
    response = ((jxx * jyy - jxy * jxy) - 0.05 * trace * trace).flatten(-2)
    trace = trace.flatten(-2)
    active = trace > torch.quantile(trace, 0.5, dim=-1).unsqueeze(-1)
    frac = ((response > 0) & active).sum(-1).float() / (active.sum(-1).float() + EPS)
    mag = response.clamp_min(0).sqrt().sum(-1) / (trace.sum(-1) + EPS)
    return {'corner_frac': frac, 'corner_mag': mag}


def _lbp(x: torch.Tensor, measures: set[str]) -> dict:
    """Entropy and uniform share of rotation-invariant local binary patterns."""
    h, w = x.shape[-2:]
    center = x[..., 1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    bits = torch.stack(
        [x[..., 1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx] > center for dy, dx in offsets]
    )
    transitions = (bits != bits.roll(1, 0)).sum(0)
    riu = torch.where(transitions <= 2, bits.sum(0), 9).flatten(-2)
    hist = torch.stack([(riu == i).sum(-1) for i in range(10)], -1).float() / riu.shape[-1]
    p = hist.clamp_min(1e-12)
    return {'lbp_entropy': -(p * p.log()).sum(-1), 'lbp_uniform': hist[..., :9].sum(-1)}


@functools.lru_cache(maxsize=8)
def _neighbor_edges(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Eight-neighbor pixel pairs using one-based labels, (2, E)."""
    nodes = torch.arange(1, h * w + 1).reshape(h, w)
    left, right = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        xa = slice(max(0, -dx), min(w, w - dx))
        xb = slice(max(0, dx), min(w, w + dx))
        left.append(nodes[: h - dy, xa].flatten())
        right.append(nodes[dy:, xb].flatten())
    return torch.stack((torch.cat(left), torch.cat(right))).to(device)


def _component_sizes(mask: torch.Tensor) -> torch.Tensor:
    """Pixel count per root label of eight-connected regions, (..., H * W + 1)."""
    h, w = mask.shape[-2:]
    flat = mask.reshape(-1, h * w)
    valid = F.pad(flat, (1, 0), value=False)
    a, b = (side.expand(len(flat), -1) for side in _neighbor_edges(h, w, mask.device))
    active = valid.gather(1, a) & valid.gather(1, b)
    a, b = torch.where(active, a, 0), torch.where(active, b, 0)
    parents = torch.arange(h * w + 1, device=mask.device).expand(len(flat), -1).clone()
    while True:  # union by minimum label with pointer jumping, until stable
        old = parents.clone()
        ra, rb = parents.gather(1, a), parents.gather(1, b)
        parents.scatter_reduce_(1, torch.maximum(ra, rb), torch.minimum(ra, rb), reduce='amin')
        parents = parents.gather(1, parents)
        if torch.equal(old, parents):
            break
    sizes = torch.zeros_like(parents)
    sizes.scatter_add_(1, parents[:, 1:], flat.long())
    return sizes.reshape(*mask.shape[:-2], h * w + 1)


def _blobs(x: torch.Tensor, measures: set[str]) -> dict:
    """Largest share, log count, and mean share of connected above-median regions."""
    flat = x.flatten(-2)
    sizes = _component_sizes(x > torch.quantile(flat, 0.5, dim=-1)[..., None, None])
    count = (sizes > 0).sum(-1)
    return {
        'blob_largest': sizes.amax(-1).float() / flat.shape[-1],
        'blob_count': count.float().log1p(),
        'blob_mean': sizes.sum(-1).float() / count.clamp_min(1) / flat.shape[-1],
    }


def _tail(x: torch.Tensor, measures: set[str]) -> dict:
    """Anisotropy and spread of pixels weighted by how far they fall below Q1 or above Q3."""
    h, w = x.shape[-2:]
    flat = x.flatten(-2)
    q = _quantile(flat, [25, 75])
    weights = torch.stack(
        ((q[..., :1] - flat).clamp_min(0), (flat - q[..., 1:]).clamp_min(0)), -2
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing='ij',
    )
    xx, yy = xx.flatten(), yy.flatten()
    mass = weights.sum(-1) + EPS
    mx, my = ((weights * c).sum(-1) / mass for c in (xx, yy))
    dx, dy = xx - mx[..., None], yy - my[..., None]
    sxx, syy, sxy = ((weights * a * b).sum(-1) / mass for a, b in ((dx, dx), (dy, dy), (dx, dy)))
    trace = sxx + syy
    aniso = ((sxx - syy).square() + 4 * sxy * sxy).sqrt() / (trace + EPS)
    spread = trace.sqrt()
    return {
        'tail_aniso_low': aniso[..., 0],
        'tail_spread_low': spread[..., 0],
        'tail_aniso_high': aniso[..., 1],
        'tail_spread_high': spread[..., 1],
    }


_GROUP_FUNCTIONS = {
    'moments': _moments,
    'percentiles': _percentiles,
    'gradient': _gradient,
    'coherence': _coherence,
    'entropy': _entropy,
    'histogram': _histogram,
    'fft_peak': _fft_peak,
    'hough': _hough,
    'harris': _harris,
    'lbp': _lbp,
    'blobs': _blobs,
    'fft_slope': _fft_slope,
    'tail': _tail,
}

# --- EuroSAT presets ---

EUROSAT_BANDS = (
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A',
)  # fmt: skip
EUROSAT_ROLES = {'swir1': 'B12', 'swir2': 'B8A'}
EUROSAT_33 = (
    'lbp_uniform_ndbi', 'grad_std_nbr', 'corner_frac_ndvi', 'grad_std_ndwi',
    'grad_mean_s2_B02', 'grad_mean_B04', 'grad_mean_s2_B03', 'std_s1_nbr',
    'orient_entropy_B12', 'orient_entropy_B01', 'grad_std_bsi', 'p75_B11', 'p10_B12',
    'corner_mag_ndvi', 'grad_mean_ndvi', 'p90_B09', 'grad_mean_nbr', 'lbp_uniform_ndvi',
    'grad_mean_B05', 'std_B05', 'line_top3_ndvi', 'p10_B04', 'p50_B12',
    'orient_entropy_B03', 'p75_B04', 'blob_largest_ndvi', 'corner_frac_pan',
    'grad_mean_ndbi', 'p75_B01', 'p10_B07', 'orient_entropy_s1_B02', 'mean_B03',
    'tail_aniso_low_ndvi',
)  # fmt: skip
EUROSAT_52 = tuple(f'{m}_{b}' for b in EUROSAT_BANDS for m in ('mean', 'std', 'min', 'max'))


def _eurosat_pool() -> tuple[str, ...]:
    bands, indices = EUROSAT_BANDS, ('ndvi', 'ndwi', 'ndbi', 'ndmi', 'nbr', 'bsi')
    pairs = (
        ('B04', 'B08'), ('B03', 'B08'), ('B02', 'B08'), ('B12', 'B08'),
        ('B04', 'B12'), ('B03', 'B04'), ('B08', 'B8A'), ('B12', 'B8A'),
    )  # fmt: skip
    corr = [f'corr_{a}_{b}' for a, b in pairs]
    names = [f'{m}_{b}' for m in ('mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90') for b in bands]
    names += [_name(m, s, b) for s in range(3) for m in GROUPS['gradient'] for b in bands]
    names += [_name('coherence', s, b) for s in range(2) for b in bands]
    names += [f'orient_entropy_{b}' for b in bands]
    names += [f'{m}_{b}' for b in bands for m in GROUPS['histogram']]
    names += [f'fft_peak_{b}' for b in bands]
    names += corr
    names += [f'{m}_{i}' for m in ('std', 'grad_mean', 'grad_std', 'spread') for i in indices]
    for group in ('hough', 'harris', 'lbp', 'blobs', 'fft_slope'):
        names += [f'{m}_{c}' for c in STRUCTURAL[:3] for m in GROUPS[group]]
    names += [f'coherence_{i}' for i in indices]
    names += [f'{m}_s1_{i}' for m in ('std', 'grad_mean') for i in indices]
    names += corr  # the historical pool repeats these eight columns
    names += [f'orient_entropy_s1_{b}' for b in bands]
    return tuple(names)


EUROSAT_377 = _eurosat_pool()
EUROSAT_389 = EUROSAT_377 + tuple(
    f'tail_{stat}_{tail}_{c}'
    for c in STRUCTURAL[:3]
    for tail in ('low', 'high')
    for stat in ('aniso', 'spread')
)
