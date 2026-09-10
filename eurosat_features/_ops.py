"""Native tensor arithmetic shared by the fixed feature families."""

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

EPS = 1e-6


def pool2(x: torch.Tensor) -> torch.Tensor:
    """Average nonoverlapping 2-by-2 spatial blocks.

    Args:
        x: Tensor of shape ``(N, C, H, W)`` with even spatial dimensions.

    Returns:
        Tensor of shape ``(N, C, H // 2, W // 2)``.
    """
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def mean_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and population standard deviation along the last axis.

    Args:
        x: Floating-point tensor whose last axis contains observations.

    Returns:
        Mean and standard deviation, each with the last axis removed.
    """
    std, mean = torch.std_mean(x, dim=-1, correction=0)
    return mean, std


def percentile(x: torch.Tensor, q: float | Sequence[float]) -> torch.Tensor:
    """Compute linearly interpolated percentiles with native tensor arithmetic.

    Args:
        x: Floating-point tensor whose last axis contains observations.
        q: A percentile in ``[0, 100]`` or a sequence of percentiles.

    Returns:
        Tensor with the observation axis removed for a scalar percentile, or
        replaced by a final percentile axis for a sequence.
    """
    scalar = isinstance(q, (int, float))
    quantiles = torch.tensor(q, dtype=x.dtype, device=x.device) / 100
    result = torch.quantile(x, quantiles, dim=-1)
    return result if scalar else result.movedim(0, -1)


def median(x: torch.Tensor) -> torch.Tensor:
    """Compute the median, interpolating between even-sized central pairs.

    Args:
        x: Floating-point tensor whose last axis contains observations.

    Returns:
        Median along the last axis, with that axis removed.
    """
    return torch.quantile(x, 0.5, dim=-1)


def gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward differences on the shared interior pixel grid.

    Args:
        x: Tensor with spatial dimensions in the last two axes.

    Returns:
        Horizontal and vertical gradients with one fewer row and column.
    """
    return x[..., :-1, 1:] - x[..., :-1, :-1], x[..., 1:, :-1] - x[..., :-1, :-1]


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean magnitude of forward spatial differences.

    Args:
        x: Tensor with spatial dimensions in the last two axes.

    Returns:
        Gradient magnitudes with one fewer row and column.
    """
    gx, gy = gradients(x)
    return (gx * gx + gy * gy).sqrt()


def index_maps(x: torch.Tensor) -> torch.Tensor:
    """Compute six spectral indices with the frozen historical band choices.

    Args:
        x: TIFF-order tensor of shape ``(N, 13, H, W)``. Historical SWIR1 and
            SWIR2 refer to indices 11 (B12) and 12 (B8A), respectively.

    Returns:
        Tensor of shape ``(N, 6, H, W)`` ordered NDVI, NDWI, NDBI, NDMI, NBR, BSI.
    """
    nir, red, green, blue, sw1, sw2 = (x[:, i] for i in (7, 3, 2, 1, 11, 12))
    pairs = ((nir, red), (green, nir), (sw1, nir), (nir, sw1), (nir, sw2))
    maps = [(a - b) / (a + b + EPS) for a, b in pairs]
    maps.append(((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue) + EPS))
    return torch.stack(maps, 1)


def pan(x: torch.Tensor) -> torch.Tensor:
    """Compute the panchromatic proxy by averaging all input bands.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Band mean of shape ``(N, H, W)``.
    """
    return x.mean(dim=1)


def coherence(x: torch.Tensor) -> torch.Tensor:
    """Measure global structure-tensor anisotropy for each channel.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Coherence values of shape ``(N, C)``.
    """
    gx, gy = gradients(x)
    sxx, syy, sxy = ((a * b).mean((-2, -1)) for a, b in ((gx, gx), (gy, gy), (gx, gy)))
    return ((sxx - syy).square() + 4 * sxy * sxy).sqrt() / (sxx + syy + EPS)


def orientation_histogram(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Build magnitude-weighted unsigned gradient-orientation histograms.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.
        bins: Number of equal-width angle bins over ``[0, pi)``.

    Returns:
        Normalized histograms of shape ``(N, C, bins)`` using native
        ``torch.atan2`` angles. Zero-gradient channels have zero histograms.
    """
    gx, gy = gradients(x)
    mag = (gx * gx + gy * gy).sqrt().flatten(-2)
    angle = torch.atan2(gy, gx)
    angle = torch.remainder(angle, math.pi)
    idx = (angle / (math.pi / bins)).long().clamp(max=bins - 1).flatten(-2)
    hist = torch.stack([torch.where(idx == b, mag, 0).sum(-1) for b in range(bins)], -1)
    return hist / (hist.sum(-1, keepdim=True) + EPS)


def orientation_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute the Shannon entropy of eight-bin orientation histograms.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Entropy values of shape ``(N, C)``.
    """
    hist = orientation_histogram(x, 8)
    return -(hist * (hist + EPS).log()).sum(-1)


def uniform_filter3(x: torch.Tensor) -> torch.Tensor:
    """Average 3-by-3 neighborhoods with replicated boundary samples.

    For radius one this is the same boundary extension as SciPy's half-sample
    symmetric ``reflect`` mode, not PyTorch's ``reflect`` padding.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Locally averaged tensor with the same shape as the input.
    """
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)


def harris(x: torch.Tensor) -> torch.Tensor:
    """Summarize positive Harris responses on strong-gradient pixels.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Tensor of shape ``(N, 2 * C)`` with positive-response fraction among
        above-median-trace pixels and normalized corner magnitude per channel.
    """
    gy, gx = torch.gradient(x, dim=(-2, -1), edge_order=1)
    jxx, jyy, jxy = (uniform_filter3(a * b) for a, b in ((gx, gx), (gy, gy), (gx, gy)))
    trace = jxx + jyy
    response = (jxx * jyy - jxy * jxy) - 0.05 * trace * trace
    trace, response = trace.flatten(-2), response.flatten(-2)
    active = trace > median(trace).unsqueeze(-1)
    frac = ((response > 0) & active).sum(-1).float() / (active.sum(-1).float() + EPS)
    mag = response.clamp_min(0).sqrt().sum(-1) / (trace.sum(-1) + EPS)
    return torch.stack((frac, mag), -1).flatten(1)


def hough_geometry() -> tuple[torch.Tensor, int, float]:
    """Build fixed Hough projection indices for the 63-by-63 gradient grid.

    Returns:
        Rho-bin indices of shape ``(60, 63 * 63)``, the number of rho bins,
        and the grid diagonal length. Angles span ``[0, pi)`` with two-pixel
        rho bins.
    """
    size, ntheta = 63, 60
    diag = math.hypot(size, size)
    nrho = math.ceil(diag) + 1
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing='ij',
    )
    theta = torch.arange(ntheta, dtype=torch.float32) * (math.pi / ntheta)
    rho = theta.cos()[:, None] * x.flatten() + theta.sin()[:, None] * y.flatten()
    bins = ((rho + diag) / 2).floor().long().clamp(0, nrho - 1)
    return bins, nrho, diag


def hough(x: torch.Tensor, bins: torch.Tensor, nrho: int, diag: float) -> torch.Tensor:
    """Summarize straight lines among above-85th-percentile gradients.

    Args:
        x: Tensor of shape ``(N, C, 64, 64)``.
        bins: Fixed Hough projection indices on the input device.
        nrho: Number of rho bins.
        diag: Gradient-grid diagonal length.

    Returns:
        Tensor of shape ``(N, 3 * C)`` containing peak votes divided by edge
        count, peak votes divided by the diagonal, and the sum of the three
        strongest angle peaks divided by edge count, in channel-major order.
    """
    mag = magnitude(x).flatten(-2)
    threshold = percentile(mag, 85)
    edge = (mag > threshold.unsqueeze(-1)).float()
    flat = edge.reshape(-1, edge.shape[-1])
    # Chunk angles to bound working memory even for large image batches.
    peaks = []
    for chunk in bins.split(10):
        acc = torch.zeros(
            (len(flat), len(chunk), nrho), dtype=torch.float32, device=x.device
        )
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
    return torch.stack((peak / count, peak / diag, top3 / count), -1).flatten(1)


def component_edges(height: int = 64, width: int = 64) -> torch.Tensor:
    """Enumerate undirected eight-neighbor edges using one-based pixel labels.

    Args:
        height: Number of pixel rows.
        width: Number of pixel columns.

    Returns:
        Integer tensor of shape ``(2, E)`` holding both endpoints of each edge.
    """
    nodes = torch.arange(1, height * width + 1).reshape(height, width)
    left, right = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ya, yb = slice(0, height - dy), slice(dy, height)
        xa = slice(max(0, -dx), min(width, width - dx))
        xb = slice(max(0, dx), min(width, width + dx))
        left.append(nodes[ya, xa].flatten())
        right.append(nodes[yb, xb].flatten())
    return torch.stack((torch.cat(left), torch.cat(right)))


def component_sizes(
    mask: torch.Tensor, edges: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute sizes indexed by root label for eight-connected binary masks.

    Parallel union-by-min with pointer jumping runs until convergence, with no
    image-diameter iteration cap (which would fail on winding/hollow regions).
    Only convergence scalars synchronize; all labels stay on the input device.

    Args:
        mask: Boolean tensor with spatial dimensions in the last two axes.
        edges: Optional precomputed neighbor edges on the mask device.

    Returns:
        Integer sizes of shape ``(*mask.shape[:-2], H * W + 1)``. Only root
        labels have nonzero counts; index zero is reserved for background.
    """
    height, width = mask.shape[-2:]
    flat = mask.reshape(-1, height * width)
    if edges is None:
        edges = component_edges(height, width).to(mask.device)
    valid = F.pad(flat, (1, 0), value=False)
    a, b = (side.unsqueeze(0).expand(len(flat), -1) for side in edges)
    active = valid.gather(1, a) & valid.gather(1, b)
    a, b = torch.where(active, a, 0), torch.where(active, b, 0)
    parents = (
        torch.arange(height * width + 1, device=mask.device)
        .expand(len(flat), -1)
        .clone()
    )
    while True:
        old = parents.clone()
        ra, rb = parents.gather(1, a), parents.gather(1, b)
        parents.scatter_reduce_(
            1,
            torch.maximum(ra, rb),
            torch.minimum(ra, rb),
            reduce='amin',
            include_self=True,
        )
        parents = parents.gather(1, parents)
        if torch.equal(old, parents):
            break
    sizes = torch.zeros_like(parents)
    sizes.scatter_add_(1, parents[:, 1:], flat.long())
    return sizes.reshape(*mask.shape[:-2], height * width + 1)


def blobs(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Summarize eight-connected regions strictly above each channel median.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.
        edges: Precomputed neighbor edges on the input device.

    Returns:
        Tensor of shape ``(N, 3 * C)`` holding largest-region share, log-one-plus
        region count, and average-region share for each channel.
    """
    flat = x.flatten(-2)
    mask = x > median(flat)[..., None, None]
    sizes = component_sizes(mask, edges)
    count = (sizes > 0).sum(-1)
    largest = sizes.amax(-1).float() / flat.shape[-1]
    average = sizes.sum(-1).float() / count.clamp_min(1) / flat.shape[-1]
    return torch.stack((largest, count.float().log1p(), average), -1).flatten(1)


def lbp(x: torch.Tensor) -> torch.Tensor:
    """Summarize rotation-invariant uniform local binary patterns.

    Args:
        x: Tensor of shape ``(N, C, H, W)``. A neighbor bit is set only when
            that neighbor is strictly greater than the center pixel.

    Returns:
        Tensor of shape ``(N, 2 * C)`` with ten-bin pattern entropy and the
        fraction of patterns having at most two circular bit transitions.
    """
    h, w = x.shape[-2:]
    center = x[..., 1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    bits = torch.stack(
        [
            x[..., 1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx] > center
            for dy, dx in offsets
        ],
        0,
    )
    transitions = (bits != bits.roll(1, 0)).sum(0)
    riu = torch.where(transitions <= 2, bits.sum(0), 9).flatten(-2)
    hist = (
        torch.stack([(riu == i).sum(-1) for i in range(10)], -1).float() / riu.shape[-1]
    )
    p = hist.clamp_min(1e-12)
    entropy = -(p * p.log()).sum(-1)
    return torch.stack((entropy, hist[..., :9].sum(-1)), -1).flatten(1)


def tail_anisotropy(ndvi: torch.Tensor) -> torch.Tensor:
    """Measure spatial anisotropy weighted by the low NDVI quartile deficit.

    Args:
        ndvi: NDVI maps of shape ``(N, 64, 64)``.

    Returns:
        Weighted spatial-covariance anisotropy of shape ``(N,)``.
    """
    flat = ndvi.flatten(-2)
    q25 = percentile(flat, 25)
    weights = (q25[..., None] - flat).clamp_min(0)
    coords = torch.linspace(-1, 1, 64, dtype=torch.float32, device=flat.device)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    xx, yy = xx.flatten(), yy.flatten()
    mass = weights.sum(-1) + EPS
    mean_x, mean_y = ((weights * coord).sum(-1) / mass for coord in (xx, yy))
    dx, dy = xx - mean_x[..., None], yy - mean_y[..., None]
    sxx, syy, sxy = (
        (weights * a * b).sum(-1) / mass for a, b in ((dx, dx), (dy, dy), (dx, dy))
    )
    return ((sxx - syy).square() + 4 * sxy * sxy).sqrt() / (sxx + syy + EPS)
